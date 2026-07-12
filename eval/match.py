"""Agent-vs-agent match loop over the :class:`~eval.environment.Environment`.

``play_match`` drives one full match: it asks the current player's agent for an
action, validates and applies it through the ``Environment``, and terminates with
a structured :class:`~eval.environment.MatchResult`. Any agent fault — an illegal
move (rejected against the engine's ``obs.select``), a per-move timeout, or an
exception raised inside the agent — is caught and reported as *that agent's loss*
rather than crashing the match. The engine's native resources are always freed
(``Environment`` is used as a context manager).

Recording (SOT-1624). Pass a :class:`~eval.trace.TraceWriter` to ``play_match`` (or
use :func:`record_match`) to capture a versioned JSONL trace of the match — meta,
one decision per agent choice, and the terminal result — for faithful record-based
replay (see :mod:`eval.trace`). Recording is opt-in and never changes the match
outcome. :func:`replay_in_engine` provides the L3 engine re-simulation, which is
deliberately *not* faithful (the engine takes no seed).
"""

from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Optional, Sequence

from eval.environment import (
    EndReason,
    EngineError,
    Environment,
    IllegalActionError,
    MatchResult,
)
from eval.trace import (
    FAIL_AGENT_EXCEPTION,
    FAIL_ILLEGAL_MOVE,
    FAIL_TIMEOUT,
    FAIL_TRUNCATED,
    RecordLevel,
    Trace,
    TraceWriter,
    load_trace,
)

__all__ = ["play_match", "record_match", "replay_in_engine"]


# EndReason → trace failure category (the trace is self-describing without importing
# the enum). Only fault terminations map here; NORMAL/DRAW carry no failure.
_FAILURE_CATEGORY = {
    EndReason.ILLEGAL_MOVE: FAIL_ILLEGAL_MOVE,
    EndReason.AGENT_EXCEPTION: FAIL_AGENT_EXCEPTION,
    EndReason.TIMEOUT: FAIL_TIMEOUT,
    EndReason.MAX_STEPS: FAIL_TRUNCATED,
}


def _select_player() -> Optional[int]:
    """Best-effort ``SerialData.selectPlayer`` for the current pending selection.

    ``cg.game`` discards this field, so read it straight from the engine. Returns
    ``None`` if unavailable — the decision's ``your_index`` (State.yourIndex) still
    records the selecting player.
    """
    try:
        from cg.sim import Battle, lib  # type: ignore
        return int(lib.GetBattleData(Battle.battle_ptr).selectPlayer)
    except Exception:
        return None


def _agent_meta(agent: Any, index: int) -> dict:
    """Identity metadata for an agent, stamped into the trace's meta record."""
    return {
        "index": index,
        "name": getattr(agent, "name", type(agent).__name__),
        "version": str(getattr(agent, "version", "0")),
        "type": type(agent).__name__,
    }


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
    writer: Optional[TraceWriter] = None,
    agent_meta: Optional[list[dict]] = None,
    trace_id: Optional[str] = None,
    created_at: Optional[str] = None,
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
        writer: optional :class:`~eval.trace.TraceWriter`. When given, a full
            versioned trace (meta / decision列 / result) is recorded — this never
            changes the outcome. See :func:`record_match` for the convenience wrapper.
        agent_meta / trace_id / created_at: trace-meta fields (only used with
            ``writer``); sensible defaults are derived from ``agents`` / the clock.

    Returns:
        A structured :class:`MatchResult`. Normal engine terminations set
        ``winner`` (or a draw); faults set ``faulted_player`` and award the win to
        the other player.
    """
    if len(agents) != 2:
        raise ValueError("play_match needs exactly 2 agents")

    environment = env if env is not None else Environment()
    executor = ThreadPoolExecutor(max_workers=1) if per_move_timeout else None

    if writer is not None:
        if agent_meta is None:
            agent_meta = [_agent_meta(a, i) for i, a in enumerate(agents)]
        if trace_id is None or created_at is None:
            now = datetime.datetime.now(datetime.timezone.utc)
            trace_id = trace_id or now.strftime("%Y%m%dT%H%M%S%fZ")
            created_at = created_at or now.isoformat()

    for seat, agent in enumerate(agents):
        hook = getattr(agent, "on_match_start", None)
        if callable(hook):
            hook(seat)

    steps = 0
    result: MatchResult
    t0 = time.perf_counter()
    first_player: Optional[int] = None
    final_turn: Optional[int] = None
    try:
        with environment as e:
            obs0 = e.start(deck0, deck1)
            if writer is not None:
                cur0 = e.observation_dict.get("current") or {}
                fp = cur0.get("firstPlayer")
                first_player = fp if fp != -1 else None
                writer.write_meta(
                    trace_id=trace_id,
                    created_at=created_at,
                    agents=agent_meta,
                    decks=[deck0, deck1],
                    first_player=first_player,
                )
            while True:
                current = e.observation_dict.get("current") or {}
                if current.get("firstPlayer", -1) not in (-1, None):
                    first_player = current.get("firstPlayer")
                final_turn = current.get("turn", final_turn)

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
                obs_dict = e.observation_dict
                ts = time.perf_counter()
                try:
                    action = _call_agent(agent, obs_dict,
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
                thinking_ms = (time.perf_counter() - ts) * 1000

                if writer is not None:
                    writer.write_decision(obs_dict, action, _select_player(), thinking_ms)

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

            if writer is not None:
                _write_result(writer, e, result, first_player, final_turn, t0)
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    for agent in agents:
        hook = getattr(agent, "on_match_end", None)
        if callable(hook):
            hook(result)
    return result


def _write_result(
    writer: TraceWriter,
    env: Environment,
    result: MatchResult,
    first_player: Optional[int],
    final_turn: Optional[int],
    t0: float,
) -> None:
    """Translate a :class:`MatchResult` into the terminal trace ``result`` record."""
    final_logs = env.observation_dict.get("logs", [])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    failure: Optional[dict] = None
    if result.is_fault:
        failure = {
            "player": result.faulted_player,
            "category": _FAILURE_CATEGORY.get(result.reason, result.reason.value),
            "error": result.detail,
        }
        code = -1
    elif result.reason is EndReason.MAX_STEPS:
        failure = {"player": None, "category": FAIL_TRUNCATED, "error": result.detail}
        code = -1
    elif result.is_draw:
        code = 2
    elif result.winner in (0, 1):
        code = result.winner
    else:
        code = -1

    writer.write_result(
        result=code,
        final_logs=final_logs,
        first_player=first_player,
        final_turn=final_turn,
        elapsed_ms=elapsed_ms,
        failure=failure,
    )


def record_match(
    deck0: list[int],
    deck1: list[int],
    agents: Sequence,
    *,
    out_path: str = "eval/traces/match.jsonl",
    level: RecordLevel = RecordLevel.LOGS,
    max_steps: int = 100_000,
    per_move_timeout: Optional[float] = None,
    trace_id: Optional[str] = None,
) -> MatchResult:
    """Play one match and write a complete versioned JSONL trace of it.

    A thin wrapper over :func:`play_match`: it opens a :class:`~eval.trace.TraceWriter`
    at ``out_path`` / ``level``, stamps the agents into the trace meta, runs the
    match, and always closes the writer (even on error). Returns the same
    :class:`MatchResult` as :func:`play_match`; read the trace back with
    :func:`eval.trace.load_trace` / :class:`eval.trace.Replay`.
    """
    agent_meta = [_agent_meta(a, i) for i, a in enumerate(agents)]
    with TraceWriter(out_path, level) as writer:
        return play_match(
            deck0, deck1, agents,
            max_steps=max_steps, per_move_timeout=per_move_timeout,
            writer=writer, agent_meta=agent_meta, trace_id=trace_id,
        )


def replay_in_engine(
    trace: "str | Trace",
    *,
    deck0: Optional[list[int]] = None,
    deck1: Optional[list[int]] = None,
    max_steps: int = 100_000,
) -> dict:
    """L3 engine re-simulation of a recorded trace — deliberately **not faithful**.

    Re-runs the cabt engine from the trace's decks while feeding the *recorded*
    option indices back in order. Because the engine takes **no seed**, its hidden
    shuffles differ from the recorded match, so the very same indices soon denote
    different (or illegal) moves. This function therefore only *detects* how far the
    engine follows the recorded line before diverging — it never reproduces the
    match. Faithful reproduction is L2 (:class:`eval.trace.Replay`), not this.

    Returns a report::

        {"faithful": False, "diverged": bool, "diverged_at": int|None,
         "steps_matched": int, "recorded_decisions": int, "note": str}

    ``diverged`` is True if the engine rejected a recorded choice or the pending
    selection shape no longer matched the record; it stays False only in the rare
    case the whole recorded prefix happened to stay legal.
    """
    if isinstance(trace, str):
        trace = load_trace(trace)
    meta = trace.meta or {}
    decks = meta.get("decks") or []
    if deck0 is None:
        deck0 = decks[0] if len(decks) > 0 else None
    if deck1 is None:
        deck1 = decks[1] if len(decks) > 1 else None
    if deck0 is None or deck1 is None:
        raise ValueError("replay_in_engine needs decks (from the trace meta or args)")

    note = ("engine has no seed API: internal shuffles differ from the recorded "
            "match, so this L3 re-simulation is not a faithful reproduction")
    decisions = trace.decisions
    steps_matched = 0
    diverged = False
    diverged_at: Optional[int] = None

    with Environment() as e:
        e.start(deck0, deck1)
        for i, dec in enumerate(decisions):
            if e.is_over or steps_matched >= max_steps:
                break
            choice = dec.get("choice")
            recorded_select = dec.get("select") or {}
            live_select = e.observation_dict.get("select") or {}
            # Shape drift (different number of legal moves) already means the engine
            # is on a different line than the recording — flag it before stepping.
            if len(recorded_select.get("option") or []) != len(live_select.get("option") or []):
                diverged = True
                diverged_at = i
                break
            try:
                e.step(choice if isinstance(choice, list) else [])
            except (IllegalActionError, EngineError):
                diverged = True
                diverged_at = i
                break
            steps_matched += 1

    return {
        "faithful": False,
        "diverged": diverged,
        "diverged_at": diverged_at,
        "steps_matched": steps_matched,
        "recorded_decisions": len(decisions),
        "note": note,
    }
