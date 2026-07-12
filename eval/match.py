"""Agent-vs-agent match loop over the :class:`~eval.environment.Environment`.

``play_match`` drives one full match: it asks the current player's agent for an
action, validates and applies it through the ``Environment``, and terminates with
a structured :class:`~eval.environment.MatchResult`. Any agent fault — an illegal
move (rejected against the engine's ``obs.select``), a per-move timeout, or an
exception raised inside the agent — is caught and reported as *that agent's loss*
rather than crashing the match. The engine's native resources are always freed
(``Environment`` is used as a context manager).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional, Sequence

from eval.environment import (
    EndReason,
    EngineError,
    Environment,
    IllegalActionError,
    MatchResult,
)

__all__ = ["play_match"]


def _fault_result(faulted_player: int, reason: EndReason, steps: int,
                  detail: str) -> MatchResult:
    """A match lost by ``faulted_player``; the other player wins."""
    return MatchResult(
        winner=1 - faulted_player,
        reason=reason,
        steps=steps,
        faulted_player=faulted_player,
        detail=detail,
    )


def _call_agent(agent, obs_dict, timeout: Optional[float], executor):
    """Invoke ``agent.act`` with an optional hard per-move timeout.

    When ``timeout`` is set the call runs in a worker thread and is abandoned on
    timeout. This is safe because agents are pure functions of the observation —
    they never touch the engine's global/native state (only the parent thread
    calls ``Environment``), so an abandoned agent thread cannot corrupt it.
    """
    if timeout is None:
        return agent.act(obs_dict)
    future = executor.submit(agent.act, obs_dict)
    return future.result(timeout=timeout)


def play_match(
    deck0: list[int],
    deck1: list[int],
    agents: Sequence,
    *,
    max_steps: int = 100_000,
    per_move_timeout: Optional[float] = None,
    env: Optional[Environment] = None,
) -> MatchResult:
    """Play one match between ``agents[0]`` and ``agents[1]``.

    Args:
        deck0, deck1: the two players' 60-card decks.
        agents: a pair of agents implementing ``act(obs) -> list[int]`` (and,
            optionally, the ``on_match_start`` / ``on_match_end`` hooks).
        max_steps: safety cap on selection steps (unresolved -> ``MAX_STEPS``).
        per_move_timeout: optional hard per-move wall-clock budget (seconds); a
            slower move is the current agent's loss (``TIMEOUT``).
        env: an existing (unstarted) :class:`Environment` to use; a fresh one is
            created otherwise. Either way it is always finished before returning.

    Returns:
        A structured :class:`MatchResult`. Normal engine terminations set
        ``winner`` (or a draw); faults set ``faulted_player`` and award the win to
        the other player.
    """
    if len(agents) != 2:
        raise ValueError("play_match needs exactly 2 agents")

    environment = env if env is not None else Environment()
    executor = ThreadPoolExecutor(max_workers=1) if per_move_timeout else None

    for seat, agent in enumerate(agents):
        hook = getattr(agent, "on_match_start", None)
        if callable(hook):
            hook(seat)

    steps = 0
    result: MatchResult
    try:
        with environment as e:
            e.start(deck0, deck1)
            while True:
                if e.is_over:
                    result = e.result
                    result.steps = steps
                    break
                if steps >= max_steps:
                    result = MatchResult(
                        winner=None, reason=EndReason.MAX_STEPS, steps=steps,
                        detail=f"no result within {max_steps} steps",
                    )
                    break

                player = e.current_player
                agent = agents[player]
                try:
                    action = _call_agent(agent, e.observation_dict,
                                         per_move_timeout, executor)
                except FutureTimeout:
                    result = _fault_result(
                        player, EndReason.TIMEOUT, steps,
                        f"agent {player} exceeded {per_move_timeout}s",
                    )
                    break
                except Exception as ex:  # noqa: BLE001 - agent code is untrusted
                    result = _fault_result(
                        player, EndReason.AGENT_EXCEPTION, steps,
                        f"agent {player} raised {type(ex).__name__}: {ex}",
                    )
                    break

                try:
                    e.step(action)
                except (IllegalActionError, EngineError) as ex:
                    # An action the engine (the sole legal-move authority) refuses
                    # is an illegal move by the current agent.
                    result = _fault_result(
                        player, EndReason.ILLEGAL_MOVE, steps,
                        f"agent {player} illegal action {action!r}: {ex}",
                    )
                    break
                steps += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    for agent in agents:
        hook = getattr(agent, "on_match_end", None)
        if callable(hook):
            hook(result)
    return result
