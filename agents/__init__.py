"""Competition agents for the PTCG AI Battle challenge (SOT-1631 案B / SOT-1646).

This package is the *competition agent*, developed **separately from the eval harness'
own reference agents** (:mod:`eval.agents`). Everything is built on one no-stop,
always-legal safety skeleton:

* :mod:`agents.protocol` — the :class:`~agents.protocol.SafeAgent` skeleton, the
  selection-count validator, the legal-random fallback, and the per-context
  encounter / 未対応率 measurement.
* :mod:`agents.random_agent` — :class:`~agents.random_agent.RandomAgent`, the legal
  random baseline.
* :mod:`agents.rule_agent` — :class:`~agents.rule_agent.RuleAgent`, the rule-based
  policy (R1: skeleton only; tactics from R2 on).

Every agent exposes the Kaggle submission entry point ``act(obs) -> list[int]`` and so
plugs directly into :func:`eval.match.play_match` / :func:`eval.arena.run_arena`.
"""

from .protocol import (
    Agent,
    ContextStats,
    FallbackReason,
    InvalidSelectionError,
    SafeAgent,
    SelectKey,
    legal_random_action,
    validate_selection,
)
from .random_agent import RandomAgent
from .rule_agent import RuleAgent
from .search_agent import SearchAgent

__all__ = [
    "Agent",
    "SafeAgent",
    "SelectKey",
    "FallbackReason",
    "InvalidSelectionError",
    "ContextStats",
    "validate_selection",
    "legal_random_action",
    "RandomAgent",
    "RuleAgent",
    "SearchAgent",
]
