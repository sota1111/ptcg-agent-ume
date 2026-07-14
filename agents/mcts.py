"""Determinized MCTS reinforcement at critical positions for the PPO agent (SOT-1690).

The third stage of the PPO track (SOT-1683): the PPO policy (SOT-1689) plays
every decision, but at **critical positions only** the choice is re-evaluated
with a determinized Monte-Carlo tree search over the engine's official search
API (``search_begin`` / ``search_step`` / ``search_release`` / ``search_end``,
see :mod:`cg.api`) and overridden when the search finds a clearly better move.

Design contract
---------------
* **Critical positions only.** A decision qualifies when the PPO policy itself
  is uncertain — the masked-softmax **entropy** is high — or the game hangs in
  the balance — the PPO **value head** is near 0 (win probability ≈ 0.5). The
  thresholds are calibration-measured, not guessed: over 432 real decisions
  (12×2 matches of the committed ``data/policy.json`` vs Random/Rule,
  ``/tmp``-calibration for SOT-1690) ``entropy >= 1.9 or |value| <= 0.06``
  activates on ≈25% of all decisions (≈30% of eligible single-select ones).
* **Determinized root search.** Hidden zones are sampled per determinization
  with the merged :class:`~agents.search_agent.UniformDeckPredictor`; each
  determinization opens ONE engine search session and runs simulations from the
  shared root: pick a root candidate by PUCT (PPO priors), step it, then roll
  out with the PPO policy for a bounded depth and score the leaf with the PPO
  value head (a terminal position scores exactly ±1/0). Root action statistics
  are pooled **across determinizations** (複数determinizationの平均).
* **Hard per-decision time cap.** ``time_limit_s`` bounds the whole search
  (all determinizations); the remaining budget is split over the remaining
  determinizations so the cap holds whatever the engine costs. Kaggle持ち時間
  is therefore never threatened — the cap is per *decision* and measured
  (:class:`MCTSStats` records mean/max elapsed).
* **Conservative override.** The searched best move replaces the policy's own
  sampled pick only when its pooled mean return beats the pick's by
  ``deviate_margin`` (the SOT-1672 lesson: free deviation from a decent policy
  is a net loss). The policy pick is always simulated first so the comparison
  is never one-sided.
* **Fail-closed, never illegal.** Any failure — an unloadable deck prior, a
  rejected determinization (``search_begin`` refusing a sampled hidden state),
  a raising rollout, an exhausted budget — simply keeps the PPO policy's own
  action. Search sessions are torn down in ``finally`` (``search_end`` per
  determinization, ``search_release`` per simulation state), so no engine
  search state ever leaks, and the :class:`~agents.protocol.SafeAgent`
  skeleton still validates whatever comes out.

Engine imports are deferred to call time so this module (and the engine-free
half of its tests) imports without the gitignored ``cg/`` engine.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .features import featurize
from .policy_net import forward, masked_log_softmax, sample_action

__all__ = [
    "MCTSConfig",
    "MCTSStats",
    "policy_entropy",
    "is_critical",
    "DeterminizedMCTS",
]


@dataclass
class MCTSConfig:
    """Tunables for the critical-position determinized MCTS."""

    #: Hard wall-clock cap for one decision's whole search (all determinizations).
    time_limit_s: float = 0.5
    #: Hidden-state samples; root statistics are pooled across all of them.
    n_determinizations: int = 3
    #: Root options searched (top PPO-prior ones; the policy pick always joins).
    max_candidates: int = 8
    #: PPO-policy rollout length before the value head scores the leaf.
    rollout_depth: int = 6
    #: PUCT exploration weight on the PPO prior.
    ucb_c: float = 1.0
    #: Override the policy pick only when beaten by at least this mean return.
    deviate_margin: float = 0.1
    #: Critical when the masked-softmax entropy (nats) reaches this…
    entropy_threshold: float = 1.9
    #: …or the |value-head| output is at most this (game in the balance).
    value_threshold: float = 0.06
    #: Only single-select decisions with at least this many options qualify.
    min_options: int = 2
    #: Forwarded to ``search_begin`` (fix coin flips during the lookahead).
    manual_coin: bool = False
    #: Deck list used as the hidden-info prior for both players.
    deck_path: str = "deck.csv"


@dataclass
class MCTSStats:
    """発動率/思考時間 measurement for the acceptance report (JSON-able)."""

    decisions: int = 0            # every decision the PPO policy scored
    eligible: int = 0             # single-select, >=min_options, searchable
    activations: int = 0          # judged critical -> search attempted
    searched: int = 0             # search produced comparable statistics
    overrides: int = 0            # searched move replaced the policy pick
    failures: int = 0             # activated but no usable statistics
    determinizations_ok: int = 0
    determinizations_failed: int = 0
    simulations: int = 0
    elapsed_ms_total: float = 0.0
    elapsed_ms_max: float = 0.0

    def add_elapsed(self, ms: float) -> None:
        self.elapsed_ms_total += ms
        if ms > self.elapsed_ms_max:
            self.elapsed_ms_max = ms

    @property
    def activation_rate(self) -> float:
        """MCTS発動率: activated decisions over ALL policy decisions."""
        return (self.activations / self.decisions) if self.decisions else 0.0

    def report(self) -> dict:
        return {
            "decisions": self.decisions,
            "eligible": self.eligible,
            "activations": self.activations,
            "activation_rate": self.activation_rate,
            "searched": self.searched,
            "overrides": self.overrides,
            "failures": self.failures,
            "determinizations_ok": self.determinizations_ok,
            "determinizations_failed": self.determinizations_failed,
            "simulations": self.simulations,
            "search_ms_mean": (
                self.elapsed_ms_total / self.activations if self.activations else 0.0
            ),
            "search_ms_max": self.elapsed_ms_max,
        }

    def merge(self, other: "MCTSStats") -> None:
        """Accumulate another agent instance's counters (bench aggregation)."""
        for f in (
            "decisions", "eligible", "activations", "searched", "overrides",
            "failures", "determinizations_ok", "determinizations_failed",
            "simulations", "elapsed_ms_total",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.elapsed_ms_max = max(self.elapsed_ms_max, other.elapsed_ms_max)


def policy_entropy(logp: list[float]) -> float:
    """Shannon entropy (nats) of a masked log-softmax distribution."""
    return -sum(math.exp(lp) * lp for lp in logp)


def is_critical(entropy: float, value: float, config: MCTSConfig) -> bool:
    """Critical-position judgement on the PPO policy's own signals.

    High entropy = the policy cannot separate its options; value near 0 = the
    value head sees a coin-flip game. Either alone qualifies (thresholds are
    calibration-measured — see the module docstring).
    """
    return entropy >= config.entropy_threshold or abs(value) <= config.value_threshold


class DeterminizedMCTS:
    """Root-level determinized MCTS driven by the PPO policy/value heads.

    Owned by a :class:`~agents.ppo_agent.PPOAgent`; shares its RNG so a seeded
    agent stays reproducible. All engine access happens inside
    :meth:`maybe_search`, guarded and torn down per determinization.
    """

    def __init__(
        self,
        policy: dict,
        config: Optional[MCTSConfig] = None,
        *,
        rng,
        stats: Optional[MCTSStats] = None,
    ) -> None:
        self._policy = policy
        self.config = config or MCTSConfig()
        self._rng = rng
        self.stats = stats if stats is not None else MCTSStats()
        self._deck_ids: Optional[list[int]] = None

    # -- public entry point ---------------------------------------------------
    def maybe_search(
        self, obs: dict, parsed, select, logits: list[float], value: float, proposed: list[int]
    ) -> Optional[list[int]]:
        """Re-evaluate a critical decision; ``None`` keeps the policy's action.

        Counts every call as a decision (the 発動率 denominator), applies the
        eligibility gate (single-select, enough options, a searchable
        observation), the criticality judgement, and — when critical — the
        time-capped determinized search. Never raises.
        """
        cfg = self.config
        self.stats.decisions += 1
        if select is None or not select.option:
            return None
        n = len(select.option)
        if n < cfg.min_options or not (select.minCount <= 1 <= select.maxCount):
            return None
        if (
            parsed is None
            or parsed.current is None
            or getattr(parsed, "search_begin_input", None) is None
        ):
            return None
        self.stats.eligible += 1

        logp = masked_log_softmax(logits, n)
        if not logp or not is_critical(policy_entropy(logp), value, cfg):
            return None
        self.stats.activations += 1

        t0 = time.perf_counter()
        policy_pick = proposed[0] if proposed else None
        try:
            means = self._search(parsed, logp, policy_pick)
        except Exception:  # noqa: BLE001 - the search must never crash the agent
            means = None
        finally:
            self.stats.add_elapsed((time.perf_counter() - t0) * 1000.0)

        if not means:
            self.stats.failures += 1
            return None
        self.stats.searched += 1
        best = max(means, key=lambda i: means[i])
        if (
            policy_pick is not None
            and best != policy_pick
            and policy_pick in means
            and means[best] >= means[policy_pick] + cfg.deviate_margin
        ):
            self.stats.overrides += 1
            return [best]
        return None  # search agrees / not confidently better — keep the policy pick

    # -- search core ------------------------------------------------------------
    def _search(
        self, parsed, logp: list[float], policy_pick: Optional[int]
    ) -> Optional[dict[int, float]]:
        """Pooled mean returns per root candidate, or ``None`` when unsearchable.

        The caller applies the ``deviate_margin`` override rule; here the job is
        only to produce comparable statistics across determinizations.
        """
        cfg = self.config
        deck = self._deck()
        if not deck:
            return None

        scored = len(logp)  # slots the policy actually scored (<= n options)
        order = sorted(range(scored), key=lambda i: logp[i], reverse=True)
        candidates: list[int] = []
        if policy_pick is not None and 0 <= policy_pick < scored:
            candidates.append(policy_pick)  # always simulated, and simulated first
        for i in order:
            if len(candidates) >= cfg.max_candidates:
                break
            if i not in candidates:
                candidates.append(i)
        if len(candidates) < 2:
            return None  # nothing to re-rank

        prior = {i: math.exp(logp[i]) for i in candidates}
        visits = {i: 0 for i in candidates}
        returns = {i: 0.0 for i in candidates}
        deadline = time.perf_counter() + max(0.0, cfg.time_limit_s)
        your_index = parsed.current.yourIndex

        for d in range(max(1, cfg.n_determinizations)):
            now = time.perf_counter()
            if now >= deadline:
                break
            # Split what's left evenly over the remaining determinizations.
            det_deadline = now + (deadline - now) / (cfg.n_determinizations - d)
            try:
                self._run_determinization(
                    parsed, your_index, candidates, prior, visits, returns, det_deadline
                )
                self.stats.determinizations_ok += 1
            except Exception:  # noqa: BLE001 - a rejected sampled world is just skipped
                self.stats.determinizations_failed += 1

        means = {i: returns[i] / visits[i] for i in candidates if visits[i] > 0}
        return means or None

    def _run_determinization(
        self,
        parsed,
        your_index: int,
        candidates: list[int],
        prior: dict[int, float],
        visits: dict[int, int],
        returns: dict[int, float],
        det_deadline: float,
    ) -> None:
        """One sampled hidden world: a shared root, PUCT-driven simulations.

        Root statistics accumulate into the pooled ``visits``/``returns``. The
        engine session is always closed (``search_end`` in ``finally``).
        """
        from cg.api import search_begin, search_end

        from .search_agent import UniformDeckPredictor

        cfg = self.config
        predictor = UniformDeckPredictor(self._deck(), self._rng)
        hidden = predictor.predict(parsed, your_index)
        root = search_begin(parsed, *hidden, manual_coin=cfg.manual_coin)
        try:
            root_select = root.observation.select
            n_root = len(root_select.option) if root_select and root_select.option else 0
            live = [i for i in candidates if i < n_root]
            dead: set[int] = set()
            while live and time.perf_counter() < det_deadline:
                idx = self._pick_candidate(live, prior, visits, returns)
                try:
                    value = self._simulate(root.searchId, idx, your_index, det_deadline)
                except Exception:  # noqa: BLE001 - engine rejected this branch
                    dead.add(idx)
                    live = [i for i in live if i not in dead]
                    continue
                visits[idx] += 1
                returns[idx] += value
                self.stats.simulations += 1
        finally:
            try:
                search_end()
            except Exception:  # noqa: BLE001 - teardown must never propagate
                pass

    def _pick_candidate(
        self,
        live: list[int],
        prior: dict[int, float],
        visits: dict[int, int],
        returns: dict[int, float],
    ) -> int:
        """Next root candidate: first visit each once (in order), then PUCT."""
        for i in live:
            if visits[i] == 0:
                return i
        total = sum(visits[i] for i in live)
        sqrt_total = math.sqrt(total + 1)

        def puct(i: int) -> float:
            mean = returns[i] / visits[i]
            return mean + self.config.ucb_c * prior[i] * sqrt_total / (1 + visits[i])

        return max(live, key=puct)

    def _simulate(self, root_id: int, idx: int, your_index: int, deadline: float) -> float:
        """Step ``idx`` from the root, roll out with the PPO policy, score the leaf.

        Return in ``[-1, 1]`` from our perspective: exact ±1/0 at a terminal,
        else the clamped PPO value head. Every created search state is released
        in ``finally``. A first-step rejection propagates (the caller retires
        the candidate for this determinization).
        """
        from cg.api import search_release, search_step

        cfg = self.config
        created: list[int] = []
        try:
            state = search_step(root_id, [idx])
            created.append(state.searchId)
            for _ in range(max(0, cfg.rollout_depth)):
                terminal = self._terminal_value(state.observation, your_index)
                if terminal is not None:
                    return terminal
                sel = state.observation.select
                if sel is None or not sel.option:
                    break
                if time.perf_counter() >= deadline:
                    break
                feats = featurize(dataclasses.asdict(state.observation))
                action = sample_action(
                    self._policy,
                    feats,
                    len(sel.option),
                    int(sel.minCount),
                    int(sel.maxCount),
                    self._rng,
                )
                try:
                    state = search_step(state.searchId, action)
                except Exception:  # noqa: BLE001 - stop the rollout, score what we have
                    break
                created.append(state.searchId)
            terminal = self._terminal_value(state.observation, your_index)
            if terminal is not None:
                return terminal
            _, value = forward(self._policy, featurize(dataclasses.asdict(state.observation)))
            return max(-1.0, min(1.0, float(value)))
        finally:
            for sid in created:
                try:
                    search_release(sid)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _terminal_value(observation, your_index: int) -> Optional[float]:
        """±1/0 when the position is decided, else ``None`` (explicit ``is None``
        checks — ``result == 0`` is a real win for player 0, never falsy-skipped)."""
        current = getattr(observation, "current", None)
        if current is None:
            return None
        result = getattr(current, "result", None)
        if result is None:
            return None
        result = int(result)
        if result in (0, 1):
            return 1.0 if result == your_index else -1.0
        if result == 2:
            return 0.0
        return None

    def _deck(self) -> list[int]:
        """Load and cache the deck-id hidden-info prior (empty ⇒ search declines)."""
        if self._deck_ids is None:
            try:
                with open(self.config.deck_path) as f:
                    self._deck_ids = [int(x) for x in f.read().split() if x.strip()][:60]
            except Exception:  # noqa: BLE001 - missing deck ⇒ fail-closed, no search
                self._deck_ids = []
        return list(self._deck_ids)
