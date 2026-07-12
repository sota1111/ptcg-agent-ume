"""Engine boundary for the PTCG AI Battle eval environment.

This module confines the cabt engine's process-global state and its ``ctypes``
boundary behind a single ``Environment`` object. Everything that touches
``cg.game`` / ``cg.sim.Battle`` (the native ``battle_ptr``) lives here; the rest
of the eval stack (agents, arena, reporting) only ever talks to ``Environment``.

Key invariants
--------------
* **One process, one match.** The cabt engine keeps the live battle in a single
  process-global pointer (``cg.sim.Battle.battle_ptr``). Two concurrent battles
  in the same process would clobber each other, so ``Environment`` enforces a
  module-level single-active guard: starting a second battle while one is live
  raises ``EngineError``. Run parallel matches in separate processes.
* **finish() runs exactly once.** ``Environment`` is a context manager; on
  ``__exit__`` it always calls the engine's ``battle_finish()`` (freeing native
  memory) even when the body raised — and it is idempotent, so an explicit
  ``finish()`` followed by the context-manager exit still frees exactly once.
* **The engine is the sole source of legal moves.** ``Observation.select.option``
  is the authoritative legal-move enumeration. We never re-implement the rules;
  ``validate_action`` only checks an action's *shape* against that enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cg import game
from cg.api import Observation, SelectData, to_observation_class

__all__ = [
    "EngineError",
    "IllegalActionError",
    "EndReason",
    "MatchResult",
    "Environment",
    "validate_action",
]


class EngineError(RuntimeError):
    """Raised for engine-boundary violations (start failure, single-active guard,
    native ``Select`` rejection)."""


class IllegalActionError(ValueError):
    """Raised by :func:`validate_action` when an action does not conform to the
    engine-supplied ``obs.select`` (out-of-range index, wrong count, duplicate)."""


class EndReason(str, Enum):
    """Why a match ended."""

    # Normal engine terminations (winner decided by the engine).
    NORMAL = "normal"
    DRAW = "draw"
    # Abnormal terminations attributed to one agent (that agent loses).
    ILLEGAL_MOVE = "illegal_move"
    TIMEOUT = "timeout"
    AGENT_EXCEPTION = "agent_exception"
    # Safety valve: no result within the step budget.
    MAX_STEPS = "max_steps"


@dataclass
class MatchResult:
    """Structured outcome of a single match.

    ``winner`` is the player index (0 or 1) that won, or ``None`` for a draw.
    ``faulted_player`` is set only for abnormal terminations
    (illegal move / timeout / agent exception) and names the agent that lost by
    fault; the ``winner`` is then the other player. ``detail`` carries a
    human-readable note (exception text, engine reason code, ...).
    """

    winner: Optional[int]
    reason: EndReason
    steps: int
    faulted_player: Optional[int] = None
    detail: Optional[str] = None

    @property
    def is_fault(self) -> bool:
        return self.faulted_player is not None

    @property
    def is_draw(self) -> bool:
        return self.winner is None and self.reason is EndReason.DRAW


def validate_action(action, select: Optional[SelectData]) -> list[int]:
    """Validate an agent action against the engine-supplied legal-move set.

    The engine's ``select.option`` list is the *only* source of legal moves; this
    function never re-implements game rules. It only checks that ``action`` is a
    well-formed selection over that option list:

    * a ``list`` of ``int`` option indices,
    * every index in ``range(len(option))``,
    * no duplicates,
    * length within ``[minCount, maxCount]``.

    Returns the validated action (as a plain ``list[int]``) or raises
    :class:`IllegalActionError`.
    """
    if select is None:
        # No selection is pending; the only valid action is the empty list.
        if action in (None, []):
            return []
        raise IllegalActionError("no selection pending but action is non-empty")

    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise IllegalActionError(f"action must be list[int], got {action!r}")

    n = len(select.option)
    for i in action:
        if i < 0 or i >= n:
            raise IllegalActionError(
                f"option index {i} out of range [0, {n}) for select {select.type}"
            )
    if len(set(action)) != len(action):
        raise IllegalActionError(f"duplicate option indices in action {action!r}")
    if not (select.minCount <= len(action) <= select.maxCount):
        raise IllegalActionError(
            f"action length {len(action)} outside [{select.minCount}, "
            f"{select.maxCount}] for select {select.type}"
        )
    return action


# Module-level single-active guard: the engine's battle pointer is process-global,
# so at most one Environment may hold a live battle at a time.
_ACTIVE: Optional["Environment"] = None


class Environment:
    """A single cabt battle, with the engine's global/native state confined here.

    Typical use::

        with Environment() as env:
            obs = env.start(deck0, deck1)
            while not env.is_over:
                action = agents[env.current_player].act(env.observation_dict)
                obs = env.step(action)
        # battle_finish() has run exactly once here, even on exception.

    Prefer :func:`eval.match.play_match` for the full agent-vs-agent loop; this
    class is the low-level boundary.
    """

    def __init__(self) -> None:
        self._obs_dict: Optional[dict] = None
        self._obs: Optional[Observation] = None
        self._started = False
        self._finished = False

    # -- lifecycle -----------------------------------------------------------
    def start(self, deck0: list[int], deck1: list[int]) -> Observation:
        """Begin a battle with the two 60-card decks. Returns the first
        :class:`Observation`. Raises :class:`EngineError` on start failure or if
        another :class:`Environment` already holds a live battle."""
        global _ACTIVE
        if self._started:
            raise EngineError("Environment already started (one match per Environment)")
        if _ACTIVE is not None:
            raise EngineError(
                "another Environment holds a live battle; the cabt engine allows "
                "only one battle per process (use separate processes for parallel matches)"
            )

        obs_dict, start_data = game.battle_start(deck0, deck1)
        if obs_dict is None:
            raise EngineError(
                f"battle_start failed: errorPlayer={start_data.errorPlayer} "
                f"errorType={start_data.errorType}"
            )
        _ACTIVE = self
        self._started = True
        self._finished = False
        self._set_obs(obs_dict)
        return self._obs

    def step(self, action: list[int]) -> Observation:
        """Apply a validated action and return the next :class:`Observation`.

        The action is validated against the current ``obs.select`` before it
        reaches the engine; a native ``Select`` rejection is surfaced as
        :class:`EngineError`. Callers that want an illegal action attributed to an
        agent should catch :class:`IllegalActionError` / :class:`EngineError`.
        """
        if not self._started:
            raise EngineError("Environment not started")
        if self._finished:
            raise EngineError("Environment already finished")
        validate_action(action, self._obs.select)
        try:
            obs_dict = game.battle_select(action)
        except (ValueError, IndexError) as e:
            raise EngineError(f"engine rejected action {action!r}: {e}") from e
        self._set_obs(obs_dict)
        return self._obs

    def finish(self) -> None:
        """Free the native battle resources. Idempotent: safe to call more than
        once (only the first call reaches the engine), so an explicit ``finish()``
        plus the context-manager exit still frees exactly once."""
        global _ACTIVE
        if self._finished or not self._started:
            # Never started, or already finished: nothing native to free. Still
            # release the guard if we happen to hold it.
            if _ACTIVE is self:
                _ACTIVE = None
            return
        self._finished = True
        try:
            game.battle_finish()
        finally:
            if _ACTIVE is self:
                _ACTIVE = None

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Always free native resources, even on exception. Return False so any
        # in-flight exception propagates.
        self.finish()
        return False

    # -- observation / state -------------------------------------------------
    def _set_obs(self, obs_dict: dict) -> None:
        self._obs_dict = obs_dict
        self._obs = to_observation_class(obs_dict)

    @property
    def observation(self) -> Observation:
        """The current observation as a typed :class:`Observation`."""
        if self._obs is None:
            raise EngineError("no observation; start() the Environment first")
        return self._obs

    @property
    def observation_dict(self) -> dict:
        """The current observation as the raw dict — the exact shape a Kaggle
        submission's ``agent(obs_dict)`` receives."""
        if self._obs_dict is None:
            raise EngineError("no observation; start() the Environment first")
        return self._obs_dict

    @property
    def select(self) -> Optional[SelectData]:
        """The engine's current selection — the sole source of legal moves
        (``select.option``). ``None`` when no selection is pending."""
        return self.observation.select

    @property
    def current_player(self) -> Optional[int]:
        """Index (0/1) of the player the engine is currently asking to select."""
        state = self.observation.current
        return None if state is None else state.yourIndex

    @property
    def is_over(self) -> bool:
        """True once the engine has decided a result for the current state."""
        state = self._obs.current if self._obs is not None else None
        return state is not None and state.result != -1

    @property
    def result(self) -> Optional[MatchResult]:
        """The normal-termination result, or ``None`` if the match is not over.
        Abnormal (fault) terminations are produced by the match runner, not here."""
        if not self.is_over:
            return None
        code = self._obs.current.result  # 0/1 winner, 2 draw
        reason_code = self._result_reason_code()
        if code == 2:
            return MatchResult(
                winner=None,
                reason=EndReason.DRAW,
                steps=0,
                detail=f"engine reason={reason_code}",
            )
        return MatchResult(
            winner=code,
            reason=EndReason.NORMAL,
            steps=0,
            detail=f"engine reason={reason_code}",
        )

    def _result_reason_code(self) -> Optional[int]:
        """Best-effort: the ``reason`` field from the engine RESULT log, if the
        last observation carried one (1:no prizes, 2:empty deck, 3:no active,
        4:card effect)."""
        for log in self._obs.logs or []:
            if getattr(log, "reason", None) is not None:
                return log.reason
        return None
