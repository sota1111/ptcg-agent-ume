"""RandomAgent — the reference legal-random competition agent (SOT-1646, R1).

A thin :class:`~agents.protocol.SafeAgent` whose *policy* is "pick a uniformly random
legal action for every selection". Because the policy handles every ``(SelectType,
SelectContext)``, its 未対応率 is 0 — unlike a bare :class:`SafeAgent`, which has no
policy and reaches the same behaviour purely through the fallback path. Both are
always-legal and never crash; ``RandomAgent`` is the honest "random baseline" the
Arena measures rule/learned agents against.

This lives in the competition ``agents/`` package (separate from the rule-based
policy, per SOT-1631 案B). It is distinct from :class:`eval.agents.base.RandomAgent`,
which is the eval harness's own reference agent; this one is built on the SafeAgent
safety skeleton and carries its measurement.
"""

from __future__ import annotations

from typing import Optional

from cg.api import Observation

from .protocol import SafeAgent, legal_random_action

__all__ = ["RandomAgent"]


class RandomAgent(SafeAgent):
    """Selects a uniformly-random legal action for every selection."""

    name = "random"
    version = "1"

    def policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        # A real (always-applicable) policy: legal random. Handled, not a fallback,
        # so RandomAgent's unsupported_rate stays 0 across every context.
        return legal_random_action(select, self._rng)
