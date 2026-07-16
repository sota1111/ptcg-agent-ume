"""Versioned match trace schema + faithful replay for the eval environment (SOT-1624).

The cabt engine takes **no seed argument**: re-running a match does not reproduce
it (its internal shuffles are non-deterministic). So the *runtime recording* is
the sole faithful means of reproduction. This module defines that recording — a
versioned JSONL trace of one match — plus a reader (save→load→save round-trip),
a compatibility check, and a **record-based Replay API**.

Trace file layout (one JSON object per line)::

    line 1     : one ``meta`` record (schema/engine/git/python stamp, agents, decks)
    next N     : one ``decision`` record per agent decision (legal moves = the full
                 ``SelectData.option`` + the chosen index/indices + search_begin_input
                 + the event logs seen since the previous decision)
    last line  : one ``result`` record (engine result code + reason, winner, turn /
                 decision counts, elapsed time, and any failure category)

Reproducibility levels (E1: the engine has no seed API — this is stamped in every
meta as ``repro`` so a trace never *claims* full reproducibility it cannot deliver)::

    L1  agent decision   — given the exact recorded observation, a *deterministic*
                           agent re-derives the same action. Verified by
                           :meth:`Replay.verify_agent`. Holds only for agents that
                           are pure functions of the observation.
    L2  faithful replay  — the recorded (obs → choice → logs → result) stream is
                           replayed 1:1 straight from the trace, no engine. This is
                           the canonical, always-exact reproduction
                           (:meth:`Replay.faithful_stream` / :meth:`Replay.regenerate`).
    L3  engine re-sim    — re-running the engine while feeding the recorded actions.
                           NOT faithful: with no seed the engine's hidden shuffles
                           diverge, so this only *detects* divergence; it never
                           guarantees the same line of play. Implemented in
                           :func:`eval.match.replay_in_engine`, clearly labelled.

Record verbosity is controlled by :class:`RecordLevel` because dumping the full
observation JSON at every decision is IO-dominated.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Iterator, Optional, TextIO

# Bump when the trace record shape changes (stamped into every trace's meta so a
# reader can refuse / flag an incompatible trace).
SCHEMA_VERSION = "1.1.0"


def _learning_state(current: dict) -> dict:
    """Compact, stable state features useful to rule/PPO/CABT analysis."""
    players = current.get("players") if isinstance(current.get("players"), list) else []
    summaries = []
    for player in players[:2]:
        player = player if isinstance(player, dict) else {}
        summaries.append({
            "hand_count": len(player.get("hand") or []),
            "bench_count": len(player.get("bench") or []),
            "prize_count": len(player.get("prize") or []),
            "deck_count": player.get("deckCount", player.get("deck_count")),
            "active": player.get("active"),
        })
    return {
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "first_player": current.get("firstPlayer"),
        "players": summaries,
    }


class RecordLevel(IntEnum):
    """How much per-decision detail to persist.

    RESULT   — meta + result only (decisions are counted but not emitted). Smallest.
    LOGS     — RESULT plus one ``decision`` record per decision, carrying the full
               SelectData (all legal moves), the chosen index/indices, thinking
               time, ``search_begin_input``, and the event logs. This is the
               default and satisfies the acceptance criteria.
    FULL_OBS — LOGS plus the full raw observation dict at each decision. IO-heavy,
               but the only level at which the board state (hidden-info handling
               included) can be audited / replayed in full.
    """

    RESULT = 0
    LOGS = 1
    FULL_OBS = 2


# Reproducibility-level labels (documented above; stamped into meta).
REPRO_LEVELS = {
    "L1": "agent decision (deterministic agent reproduces the choice from the recorded obs)",
    "L2": "faithful replay (recorded stream replayed 1:1 from the trace; canonical)",
    "L3": "engine re-simulation (NOT faithful — engine has no seed; divergence only)",
}

# Failure categories recorded in a ``result`` record's ``failure.category``.
# These mirror :class:`eval.environment.EndReason` so the trace is self-describing.
FAIL_START_ERROR = "start_error"          # battle_start reported errorPlayer/errorType
FAIL_ILLEGAL_MOVE = "illegal_move"        # action rejected against obs.select
FAIL_AGENT_EXCEPTION = "agent_exception"  # the agent callable raised
FAIL_TIMEOUT = "timeout"                  # agent exceeded its per-move budget
FAIL_TRUNCATED = "truncated"              # hit max_steps without a RESULT


def engine_hash(lib_path: Optional[str] = None) -> dict:
    """Return a sha256 stamp of the loaded engine shared library.

    Defaults to the exact library ``cg.sim`` loaded for the current platform.
    Best-effort: never raises — records an ``error`` field instead. The 64-hex
    ``sha256`` lets a reader detect an engine-binary mismatch (acceptance: "engine
    hash 不一致を検知").
    """
    if lib_path is None:
        try:
            from cg.sim import lib_path as _resolved  # type: ignore
            lib_path = _resolved
        except Exception as exc:  # pragma: no cover - engine not importable
            return {"path": "", "sha256": None, "size": None, "error": repr(exc)}

    info: dict[str, Any] = {"path": os.path.basename(lib_path), "sha256": None, "size": None}
    try:
        digest = hashlib.sha256()
        size = 0
        with open(lib_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        info["sha256"] = digest.hexdigest()
        info["size"] = size
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def git_sha(repo_dir: Optional[str] = None) -> Optional[str]:
    """Best-effort current git commit SHA (``None`` outside a repo / on error)."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def deck_hash(deck: list[int]) -> str:
    """A stable sha256 over a deck's exact card-id list (order-sensitive)."""
    payload = json.dumps(list(deck), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_meta(
    *,
    trace_id: str,
    created_at: str,
    level: RecordLevel,
    agents: list[dict],
    decks: list[list[int]],
    first_player: Optional[int],
    start_error: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build the ``meta`` record (first line of a trace).

    Stamps everything needed to judge compatibility and provenance: schema version,
    engine binary hash, git SHA, python version, per-deck hashes, and the wall
    clock. The reproducibility-level table is embedded so the trace is honest about
    what it can and cannot reproduce.
    """
    meta = {
        "kind": "meta",
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "created_at": created_at,
        "record_level": int(level),
        "repro": REPRO_LEVELS,
        "engine": engine_hash(),
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "agents": agents,
        "decks": decks,
        "deck_hashes": [deck_hash(d) for d in decks],
        "first_player": first_player,
        "start_error": start_error,
    }
    if extra:
        meta["extra"] = extra
    return meta


def build_decision(
    *,
    index: int,
    obs: dict,
    choice: Any,
    select_player: Optional[int],
    thinking_time_ms: float,
    level: RecordLevel,
) -> dict:
    """Build one ``decision`` record from a raw observation dict.

    Carries the full ``SelectData`` (``option`` = the complete legal-move list), the
    chosen index/indices, thinking time, ``search_begin_input``, and the event logs
    emitted since the previous decision. The full observation dict is included only
    at ``FULL_OBS`` level.

    The observation is stored exactly as the engine handed it to the acting agent,
    so hidden information stays hidden: the opponent's ``hand`` is already ``None``
    and face-down cards are ``None`` in the engine's view (see
    :func:`hidden_info_violations`).
    """
    current = obs.get("current") or {}
    record = {
        "kind": "decision",
        "index": index,
        "select_player": select_player,
        "your_index": current.get("yourIndex"),
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "select": obs.get("select"),          # full SelectData = all legal moves
        "choice": choice,
        "thinking_time_ms": round(thinking_time_ms, 3),
        "search_begin_input": obs.get("search_begin_input"),
        "logs": obs.get("logs", []),          # events since the last decision
        "learning": {
            "actor": current.get("yourIndex"),
            "legal_actions": (obs.get("select") or {}).get("option", []),
            "chosen_action": choice,
            "state": _learning_state(current),
        },
    }
    if level >= RecordLevel.FULL_OBS:
        record["obs"] = obs
    return record


def _extract_result_log(logs: list) -> Optional[dict]:
    """Return the RESULT log (LogType 23) from an event-log list, if present."""
    for log in logs or []:
        if isinstance(log, dict) and log.get("type") == 23:  # LogType.RESULT
            return log
    return None


def build_result(
    *,
    result: int,
    final_logs: list,
    first_player: Optional[int],
    final_turn: Optional[int],
    total_decisions: int,
    elapsed_ms: float,
    failure: Optional[dict] = None,
    start_error: Optional[dict] = None,
) -> dict:
    """Build the terminal ``result`` record.

    Derives ``reason`` (1-4) and ``winner`` from the RESULT log / engine result.
    ``result == -1`` marks a truncated/aborted match (distinguished from a real
    win/draw). On an agent/engine fault the faulting player is scored as the loser.
    """
    result_log = _extract_result_log(final_logs)
    reason = result_log.get("reason") if result_log else None
    truncated = result == -1 and (failure is None or failure.get("category") == FAIL_TRUNCATED)

    if failure and failure.get("category") in (
        FAIL_AGENT_EXCEPTION, FAIL_ILLEGAL_MOVE, FAIL_TIMEOUT,
    ):
        loser = failure.get("player")
        winner = (1 - loser) if loser in (0, 1) else None
    elif result in (0, 1):
        winner = result
    else:  # 2 == draw, -1 == truncated/undecided
        winner = None

    return {
        "kind": "result",
        "result": result,
        "reason": reason,
        "winner": winner,
        "truncated": truncated,
        "first_player": first_player,
        "final_turn": final_turn,
        "total_decisions": total_decisions,
        "elapsed_ms": round(elapsed_ms, 3),
        "failure": failure,
        "start_error": start_error,
        "final_logs": final_logs,
        "learning": {
            "winner": winner,
            "outcome_by_player": [
                "draw" if winner is None and result == 2 else
                "undecided" if winner is None else
                ("win" if winner == player else "loss")
                for player in (0, 1)
            ],
            "reward_by_player": [
                0.0 if winner is None else (1.0 if winner == player else -1.0)
                for player in (0, 1)
            ],
            "termination_reason": reason,
        },
    }


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

class TraceWriter:
    """Streams trace records to a JSONL file, one JSON object per line.

    Flushes after every record so a partial trace survives a crash mid-match.
    """

    def __init__(self, path: str, level: RecordLevel = RecordLevel.LOGS):
        self.path = path
        self.level = RecordLevel(level)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._fh: TextIO = open(path, "w", encoding="utf-8")
        self._closed = False
        self.n_decisions = 0

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()

    def write_meta(self, **kwargs: Any) -> dict:
        rec = build_meta(level=self.level, **kwargs)
        self._write(rec)
        return rec

    def write_decision(
        self,
        obs: dict,
        choice: Any,
        select_player: Optional[int],
        thinking_time_ms: float,
    ) -> Optional[dict]:
        """Record one decision. Always counted; only emitted at LOGS or above."""
        idx = self.n_decisions
        self.n_decisions += 1
        if self.level < RecordLevel.LOGS:
            return None
        rec = build_decision(
            index=idx,
            obs=obs,
            choice=choice,
            select_player=select_player,
            thinking_time_ms=thinking_time_ms,
            level=self.level,
        )
        self._write(rec)
        return rec

    def write_result(self, **kwargs: Any) -> dict:
        rec = build_result(total_decisions=self.n_decisions, **kwargs)
        self._write(rec)
        return rec

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #

@dataclass
class Trace:
    """A parsed match trace: its meta, ordered decisions, and terminal result.

    Built by :func:`load_trace` / :func:`parse_records`. :meth:`to_records` returns
    the exact record list that, re-written, reproduces the file (save→load→save
    round-trip).
    """

    meta: Optional[dict]
    decisions: list[dict] = field(default_factory=list)
    result: Optional[dict] = None

    @property
    def schema_version(self) -> Optional[str]:
        return (self.meta or {}).get("schema_version")

    @property
    def record_level(self) -> Optional[int]:
        return (self.meta or {}).get("record_level")

    def to_records(self) -> list[dict]:
        """The meta / decision / result records in file order."""
        records: list[dict] = []
        if self.meta is not None:
            records.append(self.meta)
        records.extend(self.decisions)
        if self.result is not None:
            records.append(self.result)
        return records

    def write(self, path: str) -> None:
        """Re-serialize this trace to ``path`` (one JSON object per line)."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.to_records():
                fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
                fh.write("\n")


def parse_records(records: list[dict]) -> Trace:
    """Split already-parsed trace records into a :class:`Trace`."""
    meta: Optional[dict] = None
    decisions: list[dict] = []
    result: Optional[dict] = None
    for rec in records:
        kind = rec.get("kind")
        if kind == "meta":
            meta = rec
        elif kind == "decision":
            decisions.append(rec)
        elif kind == "result":
            result = rec
    if meta is None and not decisions and result is None:
        raise ValueError("trace contains no meta/decision/result records")
    # Keep decisions in their recorded order (defensive: sort by index if present).
    decisions.sort(key=lambda d: d.get("index", 0))
    return Trace(meta=meta, decisions=decisions, result=result)


def load_trace(path: str) -> Trace:
    """Parse a trace JSONL file into a :class:`Trace`.

    Raises ``FileNotFoundError`` on a missing file and ``ValueError`` on an empty /
    unparseable trace.
    """
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records:
        raise ValueError(f"empty trace: {path}")
    return parse_records(records)


# --------------------------------------------------------------------------- #
# Compatibility check
# --------------------------------------------------------------------------- #

@dataclass
class CompatReport:
    """Result of checking a trace's meta against the current environment.

    ``compatible`` is True only when both the schema version and the engine binary
    hash match. ``notes`` explains any mismatch.
    """

    schema_ok: bool
    engine_ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return self.schema_ok and self.engine_ok


def check_compatibility(
    trace: Trace,
    *,
    schema_version: str = SCHEMA_VERSION,
    engine: Optional[dict] = None,
) -> CompatReport:
    """Judge whether a trace is compatible with the current schema + engine.

    Compares the trace's ``schema_version`` and engine ``sha256`` against the
    current values (``engine`` defaults to :func:`engine_hash` of the loaded lib).
    A missing meta or a hash mismatch is reported, never raised — this is the
    "schema/version/hash で互換性を判定できる" and "engine hash 不一致を検知"
    acceptance surface.
    """
    notes: list[str] = []
    meta = trace.meta or {}

    trace_schema = meta.get("schema_version")
    schema_ok = trace_schema == schema_version
    if not schema_ok:
        notes.append(f"schema mismatch: trace={trace_schema!r} current={schema_version!r}")

    if engine is None:
        engine = engine_hash()
    trace_sha = (meta.get("engine") or {}).get("sha256")
    cur_sha = (engine or {}).get("sha256")
    if trace_sha is None or cur_sha is None:
        engine_ok = False
        notes.append(f"engine hash unavailable: trace={trace_sha!r} current={cur_sha!r}")
    else:
        engine_ok = trace_sha == cur_sha
        if not engine_ok:
            notes.append(
                f"engine hash mismatch: trace={trace_sha[:12]}… current={cur_sha[:12]}…"
            )
    return CompatReport(schema_ok=schema_ok, engine_ok=engine_ok, notes=notes)


# --------------------------------------------------------------------------- #
# Hidden-information guard
# --------------------------------------------------------------------------- #

def hidden_info_violations(obs: dict) -> list[str]:
    """Return descriptions of any hidden-info leaks in a recorded observation.

    An observation is written from the acting player's viewpoint. The engine
    already hides the opponent, so a *correct* record must preserve:

    * the opponent's ``hand`` is ``None`` (only ``handCount`` is public);
    * every face-down card (opponent's face-down ``active`` / any ``prize`` entry)
      is recorded as ``None``, not as a concrete card.

    Returns an empty list when the observation leaks nothing. Best-effort over the
    (possibly partial) recorded state — a missing ``players`` block yields no
    violations rather than a false positive.
    """
    violations: list[str] = []
    current = obs.get("current") or {}
    players = current.get("players")
    your_index = current.get("yourIndex")
    if not isinstance(players, list) or your_index not in (0, 1):
        return violations
    opp = players[1 - your_index] if len(players) == 2 else None
    if not isinstance(opp, dict):
        return violations

    if opp.get("hand") is not None:
        violations.append("opponent hand is exposed (should be None)")
    prize = opp.get("prize")
    if isinstance(prize, list) and any(card is not None for card in prize):
        violations.append("opponent face-down prize card exposed (should be None)")
    return violations


# --------------------------------------------------------------------------- #
# Record-based Replay API (L1 / L2)
# --------------------------------------------------------------------------- #

@dataclass
class ReplayVerdict:
    """Outcome of replaying a trace against a candidate agent (L1).

    ``matches`` counts decisions where the agent reproduced the recorded choice;
    ``mismatches`` lists ``(index, recorded, produced)`` triples for the rest.
    ``consistent`` is True when every replayable decision matched.
    """

    total: int
    matches: int
    mismatches: list[tuple[int, Any, Any]] = field(default_factory=list)
    skipped: int = 0

    @property
    def consistent(self) -> bool:
        return self.total > 0 and self.matches == self.total


class Replay:
    """Record-based replay over a parsed :class:`Trace` (no engine required).

    This is the L2 faithful replay: it re-emits the recorded observation/action
    stream exactly as captured, and can regenerate the full decision列 + result
    reason (acceptance: "1試合の全意思決定と結果理由を再生成可能"). :meth:`verify_agent`
    adds the L1 check (a deterministic agent reproduces the recorded choices).

    It never re-runs the engine, so it is inherently deterministic and exposes no
    hidden information beyond what the trace already recorded.
    """

    def __init__(self, trace: Trace):
        self.trace = trace

    @classmethod
    def from_file(cls, path: str) -> "Replay":
        return cls(load_trace(path))

    # -- reconstruction ---------------------------------------------------- #
    @staticmethod
    def reconstruct_obs(decision: dict) -> dict:
        """The observation the acting agent saw at ``decision``.

        At ``FULL_OBS`` the exact recorded ``obs`` is returned. Otherwise a partial
        observation is rebuilt from the decision fields — enough for an agent that
        reads ``obs["select"]`` (the sole legal-move source), plus ``logs`` and a
        minimal ``current`` for viewpoint. Never invents hidden information.
        """
        if isinstance(decision.get("obs"), dict):
            return decision["obs"]
        return {
            "select": decision.get("select"),
            "logs": decision.get("logs", []),
            "current": {
                "yourIndex": decision.get("your_index"),
                "turn": decision.get("turn"),
                "turnActionCount": decision.get("turn_action_count"),
            },
            "search_begin_input": decision.get("search_begin_input"),
        }

    # -- L2 faithful replay ------------------------------------------------ #
    def faithful_stream(self) -> Iterator[tuple[int, dict, Any]]:
        """Yield ``(index, obs, recorded_choice)`` for each recorded decision.

        The canonical, always-exact reproduction of the match's decision points.
        """
        for dec in self.trace.decisions:
            yield dec.get("index"), self.reconstruct_obs(dec), dec.get("choice")

    def regenerate(self) -> dict:
        """Regenerate the full decision列 and terminal result reason from the trace.

        Returns ``{"decisions": [...], "result": {...}}`` where each decision is a
        compact ``{index, player, turn, choice}`` and the result carries
        ``{winner, reason, result, truncated, final_turn, failure}``. This is the
        acceptance-required "全意思決定と結果理由を再生成可能" surface: it works purely
        from the recorded stream, with no engine.
        """
        decisions = [
            {
                "index": d.get("index"),
                "player": d.get("your_index"),
                "turn": d.get("turn"),
                "choice": d.get("choice"),
            }
            for d in self.trace.decisions
        ]
        res = self.trace.result or {}
        result = {
            "winner": res.get("winner"),
            "reason": res.get("reason"),
            "result": res.get("result"),
            "truncated": res.get("truncated"),
            "final_turn": res.get("final_turn"),
            "failure": res.get("failure"),
        }
        return {"decisions": decisions, "result": result}

    # -- L1 agent-decision reproducibility --------------------------------- #
    def verify_agent(self, agent: Callable[[dict], Any] | Any) -> ReplayVerdict:
        """Replay the recorded observations through ``agent`` and compare choices.

        For each recorded decision, feed the reconstructed observation to the agent
        (a callable ``act(obs)`` or a bare ``agent(obs)``) and compare its action to
        the recorded ``choice``. Decisions whose choice was not recorded (RESULT-level
        traces) are skipped. Returns a :class:`ReplayVerdict`.

        A *deterministic* agent that only reads ``obs["select"]`` yields
        ``consistent == True`` when replayed against its own trace (L1). A divergence
        means the agent is non-deterministic or reads state the partial obs omits.

        Note: an agent that parses the *whole* observation (e.g. via
        ``to_observation_class``) needs the full obs, so verify it against a
        ``FULL_OBS`` trace; a ``LOGS`` trace reconstructs only ``select`` + ``logs``.
        """
        act = getattr(agent, "act", None)
        if not callable(act):
            act = agent  # a bare callable

        total = 0
        matches = 0
        skipped = 0
        mismatches: list[tuple[int, Any, Any]] = []
        for idx, obs, recorded in self.faithful_stream():
            if recorded is None:
                skipped += 1
                continue
            total += 1
            produced = act(obs)
            if produced == recorded:
                matches += 1
            else:
                mismatches.append((idx, recorded, produced))
        return ReplayVerdict(total=total, matches=matches, mismatches=mismatches, skipped=skipped)

    # -- hidden-info audit ------------------------------------------------- #
    def hidden_info_violations(self) -> list[tuple[int, list[str]]]:
        """Per-decision hidden-info leaks across the trace (empty = clean).

        Only meaningful for FULL_OBS traces (partial obs carry no opponent state, so
        they cannot leak). Returns ``(index, [violation, …])`` for offending decisions.
        """
        out: list[tuple[int, list[str]]] = []
        for dec in self.trace.decisions:
            obs = dec.get("obs")
            if not isinstance(obs, dict):
                continue
            v = hidden_info_violations(obs)
            if v:
                out.append((dec.get("index"), v))
        return out
