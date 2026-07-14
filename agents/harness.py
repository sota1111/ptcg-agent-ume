"""Candidate-move evaluation harness — the final decision layer (SOT-1691).

The last stage of the PPO track (SOT-1683): every decision now goes through an
explicit **generate → validate → score → decide** pipeline over a small set of
candidate moves, instead of trusting any single component's output directly.

Pipeline per selection
----------------------
1. **Generate** candidates from every decision source:

   * ``policy_sample`` — the PPO policy's masked-softmax sample
     (:func:`agents.policy_net.sample_action`), the trained distribution that
     SOT-1689/1690 measured;
   * ``mcts`` — at critical positions only, the determinized-MCTS re-evaluation
     (:meth:`agents.mcts.DeterminizedMCTS.maybe_search`), present only when the
     search beat the policy pick by its conservative ``deviate_margin``;
   * ``policy_argmax`` / ``policy_top`` — deterministic policy alternatives
     (argmax action; per-option top ranks on single-selects), kept as ranked
     backups in case a higher-priority candidate is malformed;
   * ``fallback`` — a :func:`agents.protocol.legal_random_action`, legal by
     construction, so the harness always has a decidable candidate.

2. **Validate** every candidate's shape against the engine-supplied ``select``
   (:func:`agents.protocol.validate_selection`: index range, duplicates,
   count) — an illegal candidate is dropped and counted, never played.

3. **Score** every valid candidate, recorded per decision (auditable):

   * ``policy_logp`` — mean masked-softmax log-probability of the candidate's
     scored indices (the 方策確率 axis);
   * ``mcts`` — 1.0 for the search-refined candidate (its MCTS-value advantage
     over the policy pick was already established inside ``maybe_search``,
     which only returns margin-beating moves — the MCTS価値 axis);
   * ``coverage`` / ``acts`` — immediate observation-side indicators (盤面即時
     指標): the fraction of the candidate's indices the policy head actually
     scored, and whether the candidate acts at all when acting is optional.
     (The R4 post-state board evaluation needed a one-ply engine search per
     option; the MCTS layer supersedes it, so it is not revived here.)

4. **Decide.** The MCTS candidate wins when present and valid (it carries the
   only cross-candidate *value* evidence); otherwise the policy sample plays
   (keeping the exact behaviour the SOT-1690 benchmark validated); if either is
   malformed the best-scored remaining candidate plays; and when everything
   else fails the decision defers to the :class:`~agents.protocol.SafeAgent`
   legal-random fallback. 違法出力0 therefore holds unconditionally.

:class:`HarnessAgent` is the submission agent built on this pipeline (used by
``main.py``); :class:`HarnessStats` records how often each source actually
decided, so benches can report the harness' behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .mcts import MCTSConfig
from .policy_net import forward, masked_log_softmax, sample_action
from .ppo_agent import DEFAULT_POLICY_PATH, PPOAgent
from .protocol import (
    InvalidSelectionError,
    legal_random_action,
    validate_selection,
)

__all__ = [
    "Candidate",
    "HarnessConfig",
    "HarnessStats",
    "DecisionHarness",
    "HarnessAgent",
]


@dataclass
class HarnessConfig:
    """Tunables for the candidate harness (the MCTS knobs live in MCTSConfig)."""

    #: Extra per-option candidates on single-selects (top policy ranks).
    top_alternatives: int = 3
    #: Score weight of the scored-slot coverage indicator.
    coverage_weight: float = 0.05
    #: Score bonus for acting at all when the engine allows an empty action.
    act_bonus: float = 0.02


@dataclass
class Candidate:
    """One candidate move flowing through validate → score → decide."""

    action: list[int]
    source: str  # "policy_sample" | "mcts" | "policy_argmax" | "policy_top" | "fallback"
    valid: Optional[bool] = None
    reject_reason: Optional[str] = None
    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0


@dataclass
class HarnessStats:
    """Per-source decision tally + validation counters (JSON-able report)."""

    decisions: int = 0
    candidates: int = 0
    invalid_candidates: int = 0
    decided_by: dict[str, int] = field(default_factory=dict)

    def record(self, chosen: str, generated: list[Candidate]) -> None:
        self.decisions += 1
        self.candidates += len(generated)
        self.invalid_candidates += sum(1 for c in generated if c.valid is False)
        self.decided_by[chosen] = self.decided_by.get(chosen, 0) + 1

    def report(self) -> dict:
        return {
            "decisions": self.decisions,
            "candidates": self.candidates,
            "invalid_candidates": self.invalid_candidates,
            "decided_by": dict(sorted(self.decided_by.items())),
        }

    def merge(self, other: "HarnessStats") -> None:
        """Accumulate another agent instance's counters (bench aggregation)."""
        self.decisions += other.decisions
        self.candidates += other.candidates
        self.invalid_candidates += other.invalid_candidates
        for k, v in other.decided_by.items():
            self.decided_by[k] = self.decided_by.get(k, 0) + v


class DecisionHarness:
    """The generate → validate → score → decide pipeline for one agent.

    Owns no state beyond its collaborators: the loaded policy dict, an optional
    :class:`~agents.mcts.DeterminizedMCTS`, and the agent's RNG (shared so a
    seeded agent stays reproducible).
    """

    def __init__(
        self,
        policy: dict,
        mcts,  # Optional[DeterminizedMCTS]; untyped to keep engine-free imports
        rng: random.Random,
        config: Optional[HarnessConfig] = None,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        stats: Optional[HarnessStats] = None,
    ) -> None:
        self._policy = policy
        self._mcts = mcts
        self._rng = rng
        self.config = config or HarnessConfig()
        self._deterministic = deterministic
        self._temperature = temperature
        self.stats = stats if stats is not None else HarnessStats()
        #: The full candidate list of the most recent decision (debug/audit).
        self.last_candidates: list[Candidate] = []

    # -- pipeline entry --------------------------------------------------------
    def decide(self, obs: dict, parsed, select) -> Optional[list[int]]:
        """Run the pipeline; ``None`` defers to the SafeAgent fallback."""
        from .features import featurize  # engine-free; local to mirror sibling modules

        features = featurize(obs)
        logits, value = forward(self._policy, features)  # one forward pass, reused everywhere
        logp = masked_log_softmax(logits, len(select.option))

        candidates = self._generate(obs, parsed, select, features, logits, value, logp)
        self._validate(candidates, select)
        self._score(candidates, select, logp)
        chosen = self._choose(candidates)
        self.last_candidates = candidates
        self.stats.record(chosen.source if chosen else "safe_fallback", candidates)
        return chosen.action if chosen else None

    # -- 1. generation ---------------------------------------------------------
    def _generate(
        self,
        obs: dict,
        parsed,
        select,
        features: list[float],
        logits: list[float],
        value: float,
        logp: list[float],
    ) -> list[Candidate]:
        n = len(select.option)
        min_count, max_count = int(select.minCount), int(select.maxCount)

        sampled = sample_action(
            self._policy, features, n, min_count, max_count, self._rng,
            deterministic=self._deterministic,
            temperature=self._temperature,
            logits=logits,
        )
        candidates = [Candidate(action=list(sampled), source="policy_sample")]

        if self._mcts is not None:
            refined = self._mcts.maybe_search(obs, parsed, select, logits, value, sampled)
            if refined is not None:
                # maybe_search never raises and only returns margin-beating moves.
                candidates.insert(0, Candidate(action=list(refined), source="mcts"))

        argmax = sample_action(
            self._policy, features, n, min_count, max_count, self._rng,
            deterministic=True, logits=logits,
        )
        if argmax != sampled:
            candidates.append(Candidate(action=list(argmax), source="policy_argmax"))

        if min_count <= 1 <= max_count:  # per-option ranks only make sense single-pick
            seen = {tuple(c.action) for c in candidates}
            order = sorted(range(len(logp)), key=lambda i: logp[i], reverse=True)
            for i in order[: max(0, self.config.top_alternatives)]:
                if (i,) not in seen:
                    candidates.append(Candidate(action=[i], source="policy_top"))
                    seen.add((i,))

        candidates.append(
            Candidate(action=legal_random_action(select, self._rng), source="fallback")
        )
        return candidates

    # -- 2. validation ---------------------------------------------------------
    @staticmethod
    def _validate(candidates: list[Candidate], select) -> None:
        for c in candidates:
            try:
                c.action = validate_selection(c.action, select)
                c.valid = True
            except InvalidSelectionError as exc:
                c.valid = False
                c.reject_reason = str(exc)

    # -- 3. scoring ------------------------------------------------------------
    def _score(self, candidates: list[Candidate], select, logp: list[float]) -> None:
        cfg = self.config
        acting_optional = int(select.minCount) == 0
        for c in candidates:
            if not c.valid:
                continue
            scored = [logp[i] for i in c.action if 0 <= i < len(logp)]
            c.scores["policy_logp"] = (sum(scored) / len(scored)) if scored else 0.0
            c.scores["coverage"] = (len(scored) / len(c.action)) if c.action else 0.0
            c.scores["acts"] = 1.0 if (acting_optional and c.action) else 0.0
            c.scores["mcts"] = 1.0 if c.source == "mcts" else 0.0
            c.total = (
                c.scores["policy_logp"]
                + cfg.coverage_weight * c.scores["coverage"]
                + cfg.act_bonus * c.scores["acts"]
            )

    # -- 4. decision -----------------------------------------------------------
    @staticmethod
    def _choose(candidates: list[Candidate]) -> Optional[Candidate]:
        valid = [c for c in candidates if c.valid]
        if not valid:
            return None
        for source in ("mcts", "policy_sample"):
            for c in valid:
                if c.source == source:
                    return c
        non_fallback = [c for c in valid if c.source != "fallback"]
        if non_fallback:
            return max(non_fallback, key=lambda c: c.total)
        return valid[0]  # the legal-random fallback candidate


class HarnessAgent(PPOAgent):
    """The submission agent: PPO policy + critical-position MCTS + harness.

    A :class:`~agents.ppo_agent.PPOAgent` (same artifact loading, same MCTS
    wiring, same SafeAgent skeleton underneath) whose per-decision policy is
    the :class:`DecisionHarness` pipeline instead of the bare sampled action.
    With no usable policy artifact it defers everywhere, i.e. degrades to the
    legal-random SafeAgent — never crashes, never emits an illegal action.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        *,
        policy: Optional[dict] = None,
        policy_path: Optional[str] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        time_budget_s: Optional[float] = None,
        mcts: bool = True,
        mcts_config: Optional[MCTSConfig] = None,
        harness_config: Optional[HarnessConfig] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            rng=rng,
            policy=policy,
            policy_path=policy_path or DEFAULT_POLICY_PATH,
            deterministic=deterministic,
            temperature=temperature,
            time_budget_s=time_budget_s,
            mcts=mcts,
            mcts_config=mcts_config,
        )
        self.name = "harness"
        self.harness_stats = HarnessStats()
        self._harness: Optional[DecisionHarness] = None
        if self._policy is not None:
            self._harness = DecisionHarness(
                self._policy,
                self._mcts,
                self._rng,
                harness_config,
                deterministic=deterministic,
                temperature=temperature,
                stats=self.harness_stats,
            )

    def policy(self, obs: dict, parsed, select) -> Optional[list[int]]:
        if self._harness is None or not select.option:
            return None
        return self._harness.decide(obs, parsed, select)
