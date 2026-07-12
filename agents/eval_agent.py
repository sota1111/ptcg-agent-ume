"""EvalAgent — R4 rule-line agent with board-eval-driven resource management (SOT-1649).

R4 of the parent plan (SOT-1631 案B/2). The R3 champion
(:class:`agents.rule_agent.RuleAgent`) chooses a MAIN sub-action from a *static category
ordering* (KO > energy > evolve > develop > energy-bench > non-KO attack > retreat > end;
:mod:`agents.rule_scoring`). That ordering encodes a sound *turn structure* — build the
board first, attack last — and R4 keeps it. What R3 does **not** do is choose *within* a
category: when several options tie at the top rank (which energy to attach, which card to
play, which of several equal KO attacks), R3 breaks the tie with its RNG.

R4 fills exactly that gap with the unified board evaluation
:func:`agents.board_eval.score_state`, framed as the issue's **資源管理 (resource
management)**: at each MAIN decision it takes R3's top-ranked category, and when more than
one option ties there it plays each tied option on a leak-safe one-ply search copy (reusing
the SOT-1650 machinery) and keeps the one whose *resulting board* scores best. So R4 makes
the identical decision to R3 whenever the top choice is unique, and an **informed** choice
(instead of a random one) whenever R3 would have guessed — a strict improvement on R3's
tie-break that never disturbs the proven turn structure and so cannot walk R4 into the
myopic "chip-and-end-the-turn" trap a flat board-eval search falls into.

Design contract
---------------
* **RuleAgent (R3 champion) untouched (受け入れ条件③).** This subclasses
  :class:`agents.search_agent.SearchAgent` (itself a :class:`RuleAgent`), reusing its
  hidden-info predictor / search-copy stepping / leak accounting, but drives its own
  category-gated tie-break loop. Every non-MAIN selection and the safety skeleton are
  inherited unchanged, so R3 stays the independently-verified champion baseline.
* **Board-eval driven, fail-closed (探索リーク・クラッシュ0).** The evaluator is
  :func:`agents.board_eval.score_state`. Any search failure — a rejected hidden-info
  prediction, an unknown enum, a raising evaluator, a spent budget — degrades to R3's exact
  RNG tie-break, and every search session is torn down in a ``finally`` (inherited
  :meth:`SearchAgent._score_candidate`), so no engine search state ever leaks and R4 is
  never worse-behaved than :class:`RuleAgent`.
* **Traceable (判断ごとに score 内訳と reason code).** The evaluator keeps the highest-total
  :class:`~agents.board_eval.BoardEval` seen during a decision — the *chosen* option's
  breakdown — exposed on :attr:`last_eval` and appended to the bounded :attr:`eval_trace`.
* **Configurable / ablatable weights.** ``weights`` and ``disabled`` are forwarded to
  :func:`score_state` unchanged, so the ablation runner drives this one agent with the
  full-weight baseline and each single-component-disabled variant.
"""

from __future__ import annotations

import collections
import time
from typing import Optional

from cg.api import Observation, OptionType

from .board_eval import (
    DEFAULT_WEIGHTS,
    LOSS_SCORE,
    WIN_SCORE,
    BoardEval,
    EvalWeights,
    score_state,
)
from .rule_agent import RuleAgent, _card_index
from .rule_scoring import CardIndex, score_main_options
from .search_agent import SearchAgent, UniformDeckPredictor

__all__ = ["EvalAgent"]


class EvalAgent(SearchAgent):
    """R4 agent: R3's category ordering, with board-eval breaking the top-rank ties.

    Reuses :class:`SearchAgent`'s one-ply machinery (predictor, search-copy stepping, leak
    accounting) but only to pick among the options R3 ranks equal-best, scoring each by the
    unified :func:`agents.board_eval.score_state`. Falls back to R3's RNG tie-break on any
    search failure, so it is never worse-behaved than :class:`RuleAgent` and never leaks.
    """

    name = "eval"
    version = "1"

    def __init__(
        self,
        *args,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        disabled: frozenset[str] = frozenset(),
        trace_capacity: int = 64,
        **kwargs,
    ) -> None:
        """Args (beyond :class:`SearchAgent`'s ``deck_path`` / ``search_budget_s`` / …):
        weights: per-component :class:`~agents.board_eval.EvalWeights` for the evaluation.
        disabled: component names to zero out (ablation); forwarded to :func:`score_state`.
        trace_capacity: max recent per-decision board evals kept on :attr:`eval_trace`.
        """
        self.eval_weights = weights
        self.eval_disabled = frozenset(disabled)
        # Running-best BoardEval of the current decision (reset each MAIN decision); the
        # tie-break takes the arg-max total, so this ends up being the *chosen* eval.
        self._best_eval: Optional[BoardEval] = None
        self.last_eval: Optional[BoardEval] = None
        self.eval_trace: collections.deque = collections.deque(maxlen=trace_capacity)
        # Plug the board evaluation in as SearchAgent's pluggable evaluator (used by the
        # inherited :meth:`SearchAgent._score_candidate`).
        super().__init__(*args, evaluate=self._board_evaluate, **kwargs)

    # -- the board evaluation, tracking the running-best for the trace ---------
    def _board_evaluate(self, observation: Observation, your_index: int, cards: CardIndex) -> float:
        """Score a resulting position with :func:`score_state`; track the running best.

        A decided terminal (``current.result``) is honoured first — the prize-clock
        short-circuit in :func:`score_state` covers the common win, but an explicit engine
        result (deck-out, no-Active loss) is decisive here too. Never raises: a malformed
        observation degrades to a neutral 0, so the tie-break only ever falls back for a
        genuine search failure, not merely to evaluate.
        """
        state = getattr(observation, "current", None)
        result = getattr(state, "result", -1) if state is not None else -1
        if result in (0, 1):
            ev = BoardEval(
                total=WIN_SCORE if result == your_index else LOSS_SCORE,
                reasons=["terminal_win" if result == your_index else "terminal_loss"],
            )
        elif result == 2:  # draw
            ev = BoardEval(total=0.0, reasons=["draw"])
        else:
            ev = score_state(state, your_index, cards, self.eval_weights, self.eval_disabled)
        if self._best_eval is None or ev.total > self._best_eval.total:
            self._best_eval = ev
        return ev.total

    # -- MAIN tactic: R3 ordering, board-eval tie-break within the top rank ----
    def _main_policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        """R3's ranking picks the category; board-eval breaks a top-rank tie (資源管理).

        Scores the options with R3's :func:`~agents.rule_scoring.score_main_options`, takes
        the equal-best set, and — when it has more than one member — chooses among them by
        the one-ply board evaluation; a unique best is R3's exact move. Any failure to score
        the MAIN selection defers to the inherited R3 rule policy so behaviour is unchanged.
        """
        self._best_eval = None
        scored = score_main_options(parsed, select, _card_index())
        if not scored:
            return super()._main_policy(obs, parsed, select)

        best_score = max(s.score for s in scored)
        tie = [s.index for s in scored if s.score == best_score]
        if len(tie) == 1:
            return [tie[0]]

        chosen = self._resource_tiebreak(parsed, select, tie)
        if chosen is None:  # search failed → R3's stable RNG tie-break (unchanged behaviour)
            self.search_stats["fallbacks"] += 1
            chosen = self._rng.choice(tie)
        elif self._best_eval is not None:
            self.last_eval = self._best_eval
            self.eval_trace.append(
                {
                    "total": self._best_eval.total,
                    "reasons": list(self._best_eval.reasons),
                    "components": dict(self._best_eval.components),
                }
            )
        return [chosen]

    def _resource_tiebreak(self, parsed: Observation, select, tie: list[int]) -> Optional[int]:
        """One-ply board-eval search over the tied option indices → best index, or ``None``.

        Predicts the hidden zones once (so every tied option sees the same sampled world),
        steps each on a fresh search copy (inherited leak-safe :meth:`_score_candidate`), and
        keeps the one whose resulting board scores highest. Only real sub-actions are stepped
        — a turn-ending END in the tie set is never searched (it would forfeit the turn); if
        the tie is END-only or nothing scores, returns ``None`` to defer to the RNG tie-break.
        """
        if parsed is None or parsed.current is None or select is None:
            return None
        options = select.option or []
        candidates = [
            i for i in tie
            if 0 <= i < len(options) and int(options[i].type) != int(OptionType.END)
        ][: self.max_candidates]
        if not candidates:
            return None

        your_index = parsed.current.yourIndex
        cards = _card_index()
        predictor = UniformDeckPredictor(self._deck(), self._rng)
        hidden = predictor.predict(parsed, your_index)

        self.search_stats["attempts"] += 1
        deadline = time.perf_counter() + max(0.0, self.search_budget_s)
        best_idx: Optional[int] = None
        best_score = float("-inf")
        scored_any = False
        for idx in candidates:
            if scored_any and time.perf_counter() >= deadline:
                break  # budget spent and we already have something to return
            score = self._score_candidate(parsed, hidden, idx, your_index, cards)
            if score is None:
                continue
            scored_any = True
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is not None:
            self.search_stats["chosen"] += 1
        return best_idx
