"""SafeAgent skeleton — the no-stop, always-legal core of the competition agent (SOT-1646, R1).

This module is the **safety骨格** every competition agent is built on. It is kept
deliberately separate from any rule-based *policy* (that lives in
:mod:`agents.rule_agent` and, from R2 on, fills in the tactics): the skeleton's one
job is to guarantee that an agent **never crashes** the match and **never emits an
illegal action**, whatever the policy on top of it does.

Design contract
---------------
* **The engine is the sole authority on legality.** As in :mod:`eval.environment`,
  ``obs.select.option`` is the only enumeration of legal moves; an agent only ever
  returns *option indices* into it. This module never re-implements game rules — it
  only checks an action's *shape* against the engine-supplied ``select``
  (:func:`validate_selection`, the "選択数validator": range + duplicate + count).
* **Unknown-safe dispatch.** :class:`SafeAgent` looks up a policy for the pending
  ``(SelectType, SelectContext)``. If the type/context is unknown (a value the
  engine appended after this code was written), the policy defers, raises, times
  out, or returns a malformed action, the skeleton falls back to a **contract-
  satisfying legal random** action (:func:`legal_random_action`) and records *why*
  (:class:`FallbackReason`). The match therefore continues no matter what.
* **Measurement.** Per ``(SelectType, SelectContext)`` the skeleton counts how often
  it was encountered and how often it had no real policy (未対応率), so later rounds
  can see exactly which contexts still need tactics.

The public surface — ``act(obs) -> list[int]`` — is byte-for-byte the Kaggle
submission entry point, so a :class:`SafeAgent` subclass is submission-ready as-is
and also plugs straight into :func:`eval.match.play_match` / :func:`eval.arena.run_arena`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from cg.api import Observation, SelectContext, SelectType, to_observation_class

__all__ = [
    "Agent",
    "SelectKey",
    "FallbackReason",
    "InvalidSelectionError",
    "validate_selection",
    "legal_random_action",
    "ContextStats",
    "SafeAgent",
    "KNOWN_SELECT_TYPES",
    "KNOWN_SELECT_CONTEXTS",
]

# The engine's currently-known selection type/context values. A value outside these
# sets is an *unknown* selection (the enums may gain members mid-competition), which
# the skeleton must still handle without crashing — see ``FallbackReason.UNKNOWN_*``.
KNOWN_SELECT_TYPES = frozenset(int(t) for t in SelectType)
KNOWN_SELECT_CONTEXTS = frozenset(int(c) for c in SelectContext)

# A dispatch key: the raw ``(type, context)`` int pair of a pending selection.
SelectKey = tuple[int, int]


@runtime_checkable
class Agent(Protocol):
    """Structural Agent Protocol: anything with a compatible ``act`` is an Agent.

    ``act(obs) -> list[int]`` returns option indices into ``obs["select"]["option"]``
    (the engine-supplied legal moves). Mirrors :class:`eval.agents.base.Agent` so the
    two agent families are interchangeable in the eval harness.
    """

    def act(self, obs: dict) -> list[int]:
        ...


class FallbackReason(str, Enum):
    """Why the skeleton fell back to a legal-random action instead of the policy's.

    Recorded per fallback so the safety behaviour is fully auditable. The first three
    mean *the agent has no real policy here* (they drive the 未対応率); the rest are
    runtime safety catches on a policy that does exist.
    """

    NO_POLICY = "no_policy"          # policy declined (returned None) — no tactic yet
    UNKNOWN_TYPE = "unknown_type"    # SelectType value not in KNOWN_SELECT_TYPES
    UNKNOWN_CONTEXT = "unknown_context"  # SelectContext value not in KNOWN_SELECT_CONTEXTS
    POLICY_EXCEPTION = "policy_exception"  # policy raised
    POLICY_TIMEOUT = "policy_timeout"      # policy exceeded the soft time budget
    INVALID_OUTPUT = "invalid_output"      # policy returned an illegal/malformed action

    @property
    def is_unsupported(self) -> bool:
        """True for the reasons that mean *no policy exists* for this context."""
        return self in (
            FallbackReason.NO_POLICY,
            FallbackReason.UNKNOWN_TYPE,
            FallbackReason.UNKNOWN_CONTEXT,
        )


class InvalidSelectionError(ValueError):
    """Raised by :func:`validate_selection` when an action does not conform to the
    engine-supplied ``select`` (wrong shape, out-of-range index, duplicate, or a
    count outside ``[minCount, maxCount]``)."""


def validate_selection(action, select) -> list[int]:
    """Validate an action's *shape* against the engine-supplied ``select``.

    This is the agent-side "選択数validator": a pre-submit check that mirrors the
    engine's own authority (:func:`eval.environment.validate_action`) so the skeleton
    can reject a bad policy output *before* it ever reaches the engine. It never
    re-implements game rules — legality is defined solely by ``select.option`` /
    ``select.minCount`` / ``select.maxCount``.

    Checks: ``action`` is a ``list[int]``; every index is in ``range(len(option))``;
    no duplicates; and ``minCount <= len(action) <= maxCount``.

    Returns the validated action as a plain ``list[int]`` or raises
    :class:`InvalidSelectionError`. ``bool`` is rejected (a ``bool`` is an ``int`` in
    Python, but never a valid option index).
    """
    if select is None:
        # No selection pending: the only legal action is "do nothing".
        if action in (None, []):
            return []
        raise InvalidSelectionError("no selection pending but action is non-empty")

    if not isinstance(action, list) or any(
        isinstance(i, bool) or not isinstance(i, int) for i in action
    ):
        raise InvalidSelectionError(f"action must be list[int], got {action!r}")

    n = len(select.option)
    for i in action:
        if i < 0 or i >= n:
            raise InvalidSelectionError(
                f"option index {i} out of range [0, {n}) for select type={select.type}"
            )
    if len(set(action)) != len(action):
        raise InvalidSelectionError(f"duplicate option indices in action {action!r}")
    if not (select.minCount <= len(action) <= select.maxCount):
        raise InvalidSelectionError(
            f"action length {len(action)} outside [{select.minCount}, "
            f"{select.maxCount}] for select type={select.type}"
        )
    return action


def legal_random_action(select, rng: random.Random) -> list[int]:
    """A uniformly-random action that is *guaranteed legal* for ``select``.

    Picks a count ``k`` in ``[minCount, min(maxCount, len(option))]`` and returns ``k``
    distinct option indices sampled with ``rng``. Correct by construction — the result
    always passes :func:`validate_selection` — so this is the skeleton's safe fallback
    when no policy applies. Returns ``[]`` when nothing is selectable (``select`` is
    ``None`` or has no options), which is the only legal action in that case.
    """
    if select is None:
        return []
    n = len(select.option)
    if n == 0:
        return []
    lo = max(0, int(select.minCount))
    hi = min(int(select.maxCount), n)
    if hi < lo:  # defensive: engine guarantees lo <= hi, but never trust blindly
        hi = lo
    k = rng.randint(lo, hi)
    return rng.sample(range(n), k)


@dataclass
class ContextStats:
    """Per-``(SelectType, SelectContext)`` encounter/fallback tally (the 計測)."""

    encounters: int = 0            # real decisions seen for this key
    handled: int = 0               # policy produced a valid action
    fallbacks: int = 0             # decisions that fell back to legal random
    unsupported: int = 0           # fallbacks meaning *no policy exists* (未対応)
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def unsupported_rate(self) -> float:
        """Fraction of encounters with no real policy (0.0 when never encountered)."""
        return (self.unsupported / self.encounters) if self.encounters else 0.0

    @property
    def fallback_rate(self) -> float:
        """Fraction of encounters that fell back for any reason."""
        return (self.fallbacks / self.encounters) if self.encounters else 0.0


class SafeAgent:
    """No-stop, always-legal agent skeleton. Subclass and override :meth:`policy`.

    ``act`` runs the safety pipeline for every decision:

    1. Parse the raw ``obs`` (guarded). No pending selection, or an empty option list,
       resolves to the trivially-legal empty action ``[]``.
    2. Look up the pending ``(SelectType, SelectContext)``. An unknown type/context
       skips the policy entirely and falls back.
    3. Call :meth:`policy` under an exception guard and a soft time budget. A policy
       that declines (returns ``None``), raises, or overruns triggers a fallback.
    4. Validate the policy's action (:func:`validate_selection`). A malformed/illegal
       action triggers a fallback.
    5. Fallback = a :func:`legal_random_action`, with the :class:`FallbackReason`
       recorded per ``(type, context)``.

    A bare ``SafeAgent`` has no policy, so it behaves as a fully-legal random agent
    that records every decision as ``NO_POLICY`` (未対応率 = 1.0) — useful as a
    baseline and to exercise the measurement. Subclasses that implement real tactics
    (see :mod:`agents.rule_agent`) drive that rate down.
    """

    #: Human-readable name (used by the eval harness for reports/labels).
    name: str = "safe"
    #: Bumped by policy subclasses; stamped into eval traces.
    version: str = "0"

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        *,
        time_budget_s: Optional[float] = None,
    ) -> None:
        """Args:
        seed: seed for a private RNG (used for fallbacks). Ignored if ``rng`` given.
        rng: an explicit ``random.Random`` to share a stream; overrides ``seed``.
        time_budget_s: soft per-decision budget (seconds). A :meth:`policy` call that
            runs longer is discarded and the skeleton falls back (``POLICY_TIMEOUT``).
            ``None`` disables the soft budget. This is *in addition* to the hard,
            thread-based per-move timeout enforced by :func:`eval.match.play_match`.
        """
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()
        self._time_budget_s = time_budget_s
        self.stats: dict[SelectKey, ContextStats] = {}
        #: Trivially-legal no-op decisions, split by cause (not counted as encounters).
        self.trivial_counts: dict[str, int] = {"no_selection": 0, "empty_options": 0}
        #: The most recent (key, reason) fallback — handy for debugging/logging.
        self.last_fallback: Optional[tuple[SelectKey, FallbackReason]] = None

    # -- policy hook ---------------------------------------------------------
    def policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        """Return option indices for this selection, or ``None`` to defer.

        The base skeleton always defers (``None``) — it has no tactics. Override in a
        subclass to implement real decisions for the ``(select.type, select.context)``
        the subclass supports, and return ``None`` for the rest so the skeleton falls
        back safely. A returned action is still validated before use, so a policy need
        not (but may) pre-validate.
        """
        return None

    # -- lifecycle hooks (no-ops; stateful agents may override) --------------
    def on_match_start(self, player_index: int) -> None:  # noqa: D401
        """Called once before a match with this agent's seat (0/1)."""

    def on_match_end(self, result) -> None:
        """Called once after a match with the structured result."""

    # -- the safety pipeline -------------------------------------------------
    def act(self, obs: dict) -> list[int]:
        """Return a guaranteed-legal action for ``obs`` (never raises)."""
        try:
            parsed = to_observation_class(obs)
            select = parsed.select
        except Exception:  # noqa: BLE001 - a malformed obs must never crash the agent
            self.trivial_counts["no_selection"] += 1
            return []

        if select is None:
            self.trivial_counts["no_selection"] += 1
            return []
        if not select.option:  # nothing selectable → empty action is the legal move
            self.trivial_counts["empty_options"] += 1
            return []

        key: SelectKey = (int(select.type), int(select.context))
        stat = self.stats.setdefault(key, ContextStats())
        stat.encounters += 1

        reason: Optional[FallbackReason] = None
        proposed: Optional[list[int]] = None

        if key[0] not in KNOWN_SELECT_TYPES:
            reason = FallbackReason.UNKNOWN_TYPE
        elif key[1] not in KNOWN_SELECT_CONTEXTS:
            reason = FallbackReason.UNKNOWN_CONTEXT
        else:
            t0 = time.perf_counter()
            try:
                proposed = self.policy(obs, parsed, select)
            except Exception:  # noqa: BLE001 - untrusted policy code
                proposed, reason = None, FallbackReason.POLICY_EXCEPTION
            else:
                if (
                    self._time_budget_s is not None
                    and (time.perf_counter() - t0) > self._time_budget_s
                ):
                    proposed, reason = None, FallbackReason.POLICY_TIMEOUT
                elif proposed is None:
                    reason = FallbackReason.NO_POLICY

        if proposed is not None:
            try:
                action = validate_selection(proposed, select)
            except InvalidSelectionError:
                reason = FallbackReason.INVALID_OUTPUT
            else:
                stat.handled += 1
                return action

        return self._fallback(select, stat, reason or FallbackReason.NO_POLICY, key)

    def _fallback(
        self, select, stat: ContextStats, reason: FallbackReason, key: SelectKey
    ) -> list[int]:
        """Record ``reason`` and return a guaranteed-legal random action."""
        stat.fallbacks += 1
        stat.fallback_reasons[reason.value] = stat.fallback_reasons.get(reason.value, 0) + 1
        if reason.is_unsupported:
            stat.unsupported += 1
        self.last_fallback = (key, reason)
        return legal_random_action(select, self._rng)

    # -- measurement accessors ----------------------------------------------
    def encounter_counts(self) -> dict[SelectKey, int]:
        """``{(type, context): encounters}`` for every selection seen."""
        return {key: s.encounters for key, s in self.stats.items()}

    def unsupported_rate(self, key: Optional[SelectKey] = None) -> float:
        """未対応率 for one ``key``, or overall across all encounters when ``key`` is None."""
        if key is not None:
            stat = self.stats.get(key)
            return stat.unsupported_rate if stat else 0.0
        enc = sum(s.encounters for s in self.stats.values())
        uns = sum(s.unsupported for s in self.stats.values())
        return (uns / enc) if enc else 0.0

    def stats_report(self) -> dict:
        """A JSON-serialisable snapshot of the encounter/未対応 measurement."""
        return {
            "totals": {
                "encounters": sum(s.encounters for s in self.stats.values()),
                "handled": sum(s.handled for s in self.stats.values()),
                "fallbacks": sum(s.fallbacks for s in self.stats.values()),
                "unsupported": sum(s.unsupported for s in self.stats.values()),
                "unsupported_rate": self.unsupported_rate(),
                "trivial": dict(self.trivial_counts),
            },
            "by_context": {
                f"{key[0]},{key[1]}": {
                    "type": _name(SelectType, key[0]),
                    "context": _name(SelectContext, key[1]),
                    "encounters": s.encounters,
                    "handled": s.handled,
                    "fallbacks": s.fallbacks,
                    "unsupported": s.unsupported,
                    "unsupported_rate": s.unsupported_rate,
                    "fallback_reasons": dict(s.fallback_reasons),
                }
                for key, s in sorted(self.stats.items())
            },
        }


def _name(enum_cls, value: int) -> str:
    """Enum member name for ``value``, or ``"unknown(<value>)"`` if not a member."""
    try:
        return enum_cls(value).name
    except ValueError:
        return f"unknown({value})"
