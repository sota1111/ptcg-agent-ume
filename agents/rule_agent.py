"""RuleAgent — rule-based policy agent (SOT-1646 skeleton; SOT-1647 R2: MAIN turn).

The agent is a :class:`~agents.protocol.SafeAgent` with a per-``(SelectType,
SelectContext)`` handler table: a registered handler supplies the tactic for its
context; every other selection defers to the safety fallback (legal random) and is
recorded as *unsupported*. So the agent stays legal / never crashes, and each handler
added drops the 未対応率 for its context (observable via :class:`SafeAgent`'s
measurement).

* **R1 (SOT-1646)** shipped the skeleton — an empty table (未対応率 = 1.0).
* **R2 (SOT-1647)** registers the **MAIN turn** handler: the minimal winning policy
  in :mod:`agents.rule_scoring`. The decision logic itself is a pure function of the
  observation + static card data (kept in ``rule_scoring``, engine/I-O free); this
  class only wires it in — builds the card index from the engine once, runs the
  scorer for a MAIN selection, and returns the best option (ties broken by the
  agent's seeded RNG).
* **R3 (SOT-1648)** covers every *other* selection — the setup / forced-selection
  contexts (initial active/bench, go-first, mulligan, promote-after-KO, attach,
  discard, search, prize, special-condition, …). Their tactics live in
  :mod:`agents.setup_scoring` (also pure) and are dispatched by ``SelectContext`` so
  any encountered non-MAIN context is handled by a dedicated policy rather than
  falling through to legal-random. Multi-selection is treated as a combination that
  weighs future resources. An unmapped/unknown context still defers to the safety
  fallback (違法出力0).

Keeping the policy separate from the ``protocol`` skeleton is the SOT-1631 案B split:
the safety骨格 is developed and verified independently of the tactics grown here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cg.api import Observation, SelectContext, SelectType

from .protocol import SafeAgent, SelectKey
from .rule_scoring import CardIndex, pick_best_option, score_main_options
from .setup_scoring import select_for_context

__all__ = ["RuleAgent", "Handler"]

# A per-context tactic: ``handler(obs, parsed, select) -> option indices | None``.
# Returning ``None`` defers to the SafeAgent fallback exactly like an absent handler.
Handler = Callable[[dict, Observation, object], Optional[list[int]]]

# The MAIN turn selection this agent's R2 policy handles.
_MAIN_KEY: SelectKey = (int(SelectType.MAIN), int(SelectContext.MAIN))


@lru_cache(maxsize=1)
def _card_index() -> CardIndex:
    """Static card/attack reference data, loaded from the engine once per process.

    The scorer needs card/attack data (damage, weakness, …) but must stay pure, so
    the single engine read is confined here and cached — the scoring functions in
    :mod:`agents.rule_scoring` only ever receive the resulting :class:`CardIndex`.
    """
    from cg.api import all_attack, all_card_data

    return CardIndex.from_engine(all_card_data(), all_attack())


class RuleAgent(SafeAgent):
    """Rule-based agent: minimal winning MAIN-turn policy (R2), else safe fallback.

    Tactics are registered per ``(SelectType, SelectContext)`` key; the class is the
    stable seam later rounds extend without touching the safety skeleton. R2 registers
    the MAIN handler (:mod:`agents.rule_scoring`); other contexts still fall back.
    """

    name = "rule"
    version = "3"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (type, context) -> handler. Populated by _register_handlers.
        self._handlers: dict[SelectKey, Handler] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register the rule-based tactics. R2: the MAIN turn policy."""
        self.register(_MAIN_KEY, self._main_policy)

    def _main_policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        """MAIN turn tactic: score the legal options and take the best (or defer).

        Pure scoring lives in :mod:`agents.rule_scoring`; this only supplies the card
        index and the RNG for tie-breaking. Returns ``None`` (defer to the safety
        fallback) when there is no board state to reason about, so a degenerate/
        unexpected MAIN selection still resolves safely.
        """
        scored = score_main_options(parsed, select, _card_index())
        if not scored:
            return None
        best = pick_best_option(scored, self._rng)
        return None if best is None else [best]

    def register(self, key: SelectKey, handler: Handler) -> None:
        """Bind ``handler`` to a ``(SelectType, SelectContext)`` key (R2+ tactics)."""
        self._handlers[key] = handler

    def policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        # 1. An exact (SelectType, SelectContext) handler wins — R2's MAIN policy.
        handler = self._handlers.get((int(select.type), int(select.context)))
        if handler is not None:
            proposed = handler(obs, parsed, select)
            if proposed is not None:
                return proposed
            # handler declined (e.g. degenerate MAIN with no board) → try the context
            # router below, then finally the SafeAgent legal-random fallback.
        # 2. R3: a per-SelectContext tactic for setup / forced selections. Returns None
        #    for an unmapped/unknown context, so the SafeAgent fallback keeps us legal.
        return select_for_context(parsed, select, _card_index(), self._rng)
