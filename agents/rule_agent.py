"""RuleAgent — rule-based policy agent (SOT-1646, R1: skeleton only).

R1 delivers only the **skeleton**: a :class:`~agents.protocol.SafeAgent` with an empty
per-``(SelectType, SelectContext)`` handler table. Every selection therefore defers to
the safety fallback (legal random) and is recorded as *unsupported*, so the agent
starts at 未対応率 = 1.0 and stays legal / never crashes. From R2 on, tactics are added
by registering handlers (via :meth:`register` / the ``_handlers`` table); each handler
added drops the unsupported rate for its context, and the measurement in
:class:`SafeAgent` makes that progress observable.

Keeping the policy here — on top of, but separate from, the ``protocol`` skeleton —
is the SOT-1631 案B split: the safety骨格 is developed and verified independently of
whatever tactics the rule-based agent grows.
"""

from __future__ import annotations

from typing import Callable, Optional

from cg.api import Observation

from .protocol import SafeAgent, SelectKey

__all__ = ["RuleAgent", "Handler"]

# A per-context tactic: ``handler(obs, parsed, select) -> option indices | None``.
# Returning ``None`` defers to the SafeAgent fallback exactly like an absent handler.
Handler = Callable[[dict, Observation, object], Optional[list[int]]]


class RuleAgent(SafeAgent):
    """Rule-based agent skeleton (R1). No tactics yet — all selections fall back.

    Tactics are registered per ``(SelectType, SelectContext)`` key. R1 ships with an
    empty table on purpose (方策は R2 以降); the class is the stable seam those rounds
    extend without touching the safety skeleton.
    """

    name = "rule"
    version = "0"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (type, context) -> handler. Empty in R1; populated from R2 on.
        self._handlers: dict[SelectKey, Handler] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register the rule-based tactics. Intentionally empty in R1 (skeleton)."""

    def register(self, key: SelectKey, handler: Handler) -> None:
        """Bind ``handler`` to a ``(SelectType, SelectContext)`` key (R2+ tactics)."""
        self._handlers[key] = handler

    def policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        handler = self._handlers.get((int(select.type), int(select.context)))
        if handler is None:
            return None  # no tactic for this context yet → SafeAgent falls back
        return handler(obs, parsed, select)
