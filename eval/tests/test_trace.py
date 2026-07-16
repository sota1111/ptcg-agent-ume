"""Tests for the versioned trace schema, writer, reader and compatibility (SOT-1624).

Split into pure-unit tests (no engine — crafted records) and engine-backed tests
(record a real match, then read it back). The cabt engine is gitignored/absent in
CI, so engine tests skip cleanly.

Covers the issue's verification points:
* save → load → save round-trip is lossless;
* schema / version / hash are recorded and drive a compatibility judgment;
* an engine-binary hash mismatch is detected;
* hidden (non-public) information is not exposed — opponent hand None, face-down
  cards None — and a leak is caught.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from eval.trace import (
    SCHEMA_VERSION,
    CompatReport,
    RecordLevel,
    Replay,
    Trace,
    build_decision,
    build_result,
    check_compatibility,
    deck_hash,
    engine_hash,
    hidden_info_violations,
    load_trace,
    parse_records,
)


def _engine_available() -> bool:
    try:
        import cg.game  # noqa: F401
        return True
    except Exception:
        return False


requires_engine = pytest.mark.skipif(
    not _engine_available(), reason="cabt engine (cg/) not installed"
)


# --------------------------------------------------------------------------- #
# crafted-record helpers (engine-free)
# --------------------------------------------------------------------------- #

def _select(n: int, minc: int = 1, maxc: int = 1) -> dict:
    return {
        "type": 0, "context": 0, "minCount": minc, "maxCount": maxc,
        "remainDamageCounter": 0, "remainEnergyCost": 0,
        "option": [{"type": 14} for _ in range(n)],
        "deck": None, "contextCard": None, "effect": None,
    }


def _crafted_records(engine_sha: str = "a" * 64) -> list[dict]:
    meta = {
        "kind": "meta", "schema_version": SCHEMA_VERSION, "trace_id": "t0",
        "created_at": "2026-07-12T00:00:00+00:00", "record_level": 1,
        "engine": {"path": "libcg.so", "sha256": engine_sha, "size": 1},
        "git_sha": "deadbeef", "python_version": "3.12.0",
        "agents": [{"index": 0, "name": "a0"}, {"index": 1, "name": "a1"}],
        "decks": [[1] * 60, [2] * 60], "deck_hashes": [deck_hash([1] * 60), deck_hash([2] * 60)],
        "first_player": 0, "start_error": None,
    }
    dec0 = {
        "kind": "decision", "index": 0, "select_player": 0, "your_index": 0,
        "turn": 1, "turn_action_count": 0, "select": _select(2, 1, 1),
        "choice": [0], "thinking_time_ms": 0.1, "search_begin_input": "sbi", "logs": [],
    }
    dec1 = {
        "kind": "decision", "index": 1, "select_player": 1, "your_index": 1,
        "turn": 2, "turn_action_count": 0, "select": _select(3, 1, 1),
        "choice": [2], "thinking_time_ms": 0.2, "search_begin_input": "sbi2",
        "logs": [{"type": 23, "result": 0, "reason": 1}],
    }
    result = {
        "kind": "result", "result": 0, "reason": 1, "winner": 0, "truncated": False,
        "first_player": 0, "final_turn": 2, "total_decisions": 2, "elapsed_ms": 1.0,
        "failure": None, "start_error": None,
        "final_logs": [{"type": 23, "result": 0, "reason": 1}],
    }
    return [meta, dec0, dec1, result]


# --------------------------------------------------------------------------- #
# pure-unit: round-trip, parse, compatibility, hidden info
# --------------------------------------------------------------------------- #

def test_parse_records_splits_by_kind():
    trace = parse_records(_crafted_records())
    assert isinstance(trace, Trace)
    assert trace.meta["kind"] == "meta"
    assert [d["index"] for d in trace.decisions] == [0, 1]
    assert trace.result["winner"] == 0
    assert trace.schema_version == SCHEMA_VERSION
    assert trace.record_level == 1


def test_parse_records_rejects_empty():
    with pytest.raises(ValueError):
        parse_records([{"kind": "other"}])  # no meta/decision/result


def test_save_load_save_roundtrip(tmp_path=None):
    records = _crafted_records()
    trace = parse_records(records)
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.jsonl")
        p2 = os.path.join(tmp, "b.jsonl")
        trace.write(p1)
        reloaded = load_trace(p1)
        reloaded.write(p2)
        # Byte-for-byte identical re-serialization, and record-equal round-trip.
        assert open(p1).read() == open(p2).read()
        assert reloaded.to_records() == records


def test_to_records_preserves_order():
    trace = parse_records(_crafted_records())
    kinds = [r["kind"] for r in trace.to_records()]
    assert kinds == ["meta", "decision", "decision", "result"]


def test_decisions_sorted_by_index_even_if_shuffled():
    recs = _crafted_records()
    meta, dec0, dec1, result = recs
    trace = parse_records([meta, dec1, dec0, result])  # decisions out of order
    assert [d["index"] for d in trace.decisions] == [0, 1]


def test_deck_hash_is_stable_and_order_sensitive():
    assert deck_hash([1, 2, 3]) == deck_hash([1, 2, 3])
    assert deck_hash([1, 2, 3]) != deck_hash([3, 2, 1])


def test_compatibility_all_match():
    trace = parse_records(_crafted_records(engine_sha="b" * 64))
    report = check_compatibility(trace, engine={"sha256": "b" * 64})
    assert isinstance(report, CompatReport)
    assert report.compatible
    assert report.schema_ok and report.engine_ok
    assert report.notes == []


def test_compatibility_detects_engine_hash_mismatch():
    trace = parse_records(_crafted_records(engine_sha="b" * 64))
    report = check_compatibility(trace, engine={"sha256": "c" * 64})
    assert not report.compatible
    assert report.schema_ok and not report.engine_ok
    assert any("engine hash mismatch" in n for n in report.notes)


def test_compatibility_detects_schema_mismatch():
    trace = parse_records(_crafted_records(engine_sha="b" * 64))
    report = check_compatibility(trace, schema_version="9.9.9", engine={"sha256": "b" * 64})
    assert not report.compatible
    assert not report.schema_ok
    assert any("schema mismatch" in n for n in report.notes)


def test_compatibility_flags_missing_hash():
    trace = parse_records(_crafted_records(engine_sha="b" * 64))
    report = check_compatibility(trace, engine={"sha256": None})
    assert not report.engine_ok
    assert any("unavailable" in n for n in report.notes)


# --- hidden information -------------------------------------------------------

def _obs(your_index: int, opp: dict) -> dict:
    me = {"hand": [{"id": 1, "serial": 1, "playerIndex": your_index}], "handCount": 1,
          "prize": [None, None, None]}
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {"current": {"yourIndex": your_index, "players": players}}


def test_hidden_info_clean_observation_has_no_violations():
    opp = {"hand": None, "handCount": 5, "prize": [None, None, None, None, None, None]}
    assert hidden_info_violations(_obs(0, opp)) == []


def test_hidden_info_detects_exposed_opponent_hand():
    opp = {"hand": [{"id": 9}], "handCount": 1, "prize": [None]}
    violations = hidden_info_violations(_obs(1, opp))
    assert any("hand" in v for v in violations)


def test_hidden_info_detects_exposed_facedown_prize():
    opp = {"hand": None, "handCount": 3, "prize": [{"id": 7}, None]}
    violations = hidden_info_violations(_obs(0, opp))
    assert any("prize" in v for v in violations)


def test_hidden_info_partial_obs_yields_no_false_positive():
    # A LOGS-level (partial) obs has no players block; must not flag a leak.
    assert hidden_info_violations({"current": {"yourIndex": 0}}) == []
    assert hidden_info_violations({"select": _select(1)}) == []


# --- build_decision level gate (pure) ----------------------------------------

def test_build_decision_omits_obs_below_full_obs():
    obs = {"select": _select(2), "logs": [], "current": {"yourIndex": 0, "turn": 1}}
    at_logs = build_decision(index=0, obs=obs, choice=[0], select_player=0,
                             thinking_time_ms=1.0, level=RecordLevel.LOGS)
    at_full = build_decision(index=0, obs=obs, choice=[0], select_player=0,
                             thinking_time_ms=1.0, level=RecordLevel.FULL_OBS)
    assert "obs" not in at_logs
    assert at_full["obs"] is obs
    assert at_logs["learning"]["actor"] == 0
    assert at_logs["learning"]["chosen_action"] == [0]
    assert at_logs["select"] is obs["select"]


def test_result_has_player_outcomes_and_rewards():
    result = build_result(result=1, final_logs=[], first_player=0,
                          final_turn=4, total_decisions=2, elapsed_ms=1.0)
    assert result["learning"]["winner"] == 1
    assert result["learning"]["outcome_by_player"] == ["loss", "win"]
    assert result["learning"]["reward_by_player"] == [-1.0, 1.0]


# --------------------------------------------------------------------------- #
# engine-backed: record a real match, read it back
# --------------------------------------------------------------------------- #

@requires_engine
def test_recorded_trace_full_contents(deck):
    from eval.agents import RandomAgent
    from eval.match import record_match

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        result = record_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)],
                              out_path=out, level=RecordLevel.LOGS)
        trace = load_trace(out)

    meta = trace.meta
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["engine"]["sha256"] and len(meta["engine"]["sha256"]) == 64
    assert meta["python_version"]
    assert len(meta["decks"]) == 2 and len(meta["decks"][0]) == 60
    assert meta["deck_hashes"][0] == deck_hash(deck)
    assert len(meta["agents"]) == 2

    assert trace.decisions, "at least one decision recorded"
    d0 = trace.decisions[0]
    assert d0["select"] is not None and "option" in d0["select"]
    assert isinstance(d0["choice"], list)
    assert d0["search_begin_input"], "search_begin_input recorded"
    assert "logs" in d0

    res = trace.result
    assert res["total_decisions"] == len(trace.decisions)
    assert res["result"] in (0, 1, 2), res["result"]
    assert res["reason"] in (1, 2, 3, 4), res["reason"]
    assert any(l.get("type") == 23 for l in res["final_logs"]), "RESULT log captured"
    # trace winner agrees with the returned MatchResult
    assert res["winner"] == result.winner


@requires_engine
def test_record_levels_switch(deck):
    from eval.agents import RandomAgent
    from eval.match import record_match

    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, "r.jsonl")
        lp = os.path.join(tmp, "l.jsonl")
        fp = os.path.join(tmp, "f.jsonl")
        record_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)],
                     out_path=rp, level=RecordLevel.RESULT)
        record_match(deck, deck, [RandomAgent(seed=3), RandomAgent(seed=4)],
                     out_path=lp, level=RecordLevel.LOGS)
        record_match(deck, deck, [RandomAgent(seed=5), RandomAgent(seed=6)],
                     out_path=fp, level=RecordLevel.FULL_OBS)

        r = [json.loads(x) for x in open(rp) if x.strip()]
        l_trace = load_trace(lp)
        f_trace = load_trace(fp)

    # RESULT level: no decision rows, but a real decision count on the result.
    assert [x["kind"] for x in r] == ["meta", "result"]
    assert r[-1]["total_decisions"] > 0
    # LOGS: decision rows without the raw obs dump.
    assert l_trace.decisions and "obs" not in l_trace.decisions[0]
    # FULL_OBS: decision rows carry the full observation.
    assert f_trace.decisions and "obs" in f_trace.decisions[0]
    assert f_trace.decisions[0]["obs"].get("current") is not None


@requires_engine
def test_full_obs_trace_has_no_hidden_info_leak(deck):
    """A FULL_OBS recording of a real match must not expose hidden information."""
    from eval.agents import RandomAgent
    from eval.match import record_match

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        record_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)],
                     out_path=out, level=RecordLevel.FULL_OBS)
        trace = load_trace(out)

    leaks = Replay(trace).hidden_info_violations()
    assert leaks == [], f"hidden info leaked in recorded obs: {leaks[:3]}"


@requires_engine
def test_recorded_trace_is_compatible_with_current_engine(deck):
    from eval.agents import RandomAgent
    from eval.match import record_match

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        record_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)],
                     out_path=out, level=RecordLevel.LOGS)
        trace = load_trace(out)

    report = check_compatibility(trace, engine=engine_hash())
    assert report.compatible, report.notes


@requires_engine
def test_battle_finish_once_when_agent_raises_while_recording(deck):
    import cg.game as game
    from eval.agents import RandomAgent

    calls = {"n": 0}
    real = game.battle_finish

    def counted():
        calls["n"] += 1
        return real()

    class Boom(RandomAgent):
        def act(self, obs):
            raise ValueError("boom")

    game.battle_finish = counted
    try:
        from eval.match import record_match
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "m.jsonl")
            result = record_match(deck, deck, [Boom(), RandomAgent(seed=1)],
                                  out_path=out, level=RecordLevel.LOGS)
            trace = load_trace(out)
    finally:
        game.battle_finish = real

    assert calls["n"] == 1, f"battle_finish must run exactly once, got {calls['n']}"
    res = trace.result
    assert res["failure"] is not None
    assert res["failure"]["category"] == "agent_exception"
    assert res["failure"]["player"] == 0
    assert res["winner"] == 1
    assert result.faulted_player == 0
