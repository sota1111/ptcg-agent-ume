"""Tests for the record-based Replay API (SOT-1624).

Covers the acceptance surface:
* the recorded decision列 and result reason can be regenerated from a trace (L2);
* a deterministic agent replayed against its own trace reproduces every choice (L1);
* a diverging agent is reported as inconsistent;
* the L3 engine re-simulation never claims to be faithful (the engine has no seed).

Pure-unit tests use crafted records; the engine-backed tests record a real
deterministic self-play match and skip when the engine is absent.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from eval.trace import (
    SCHEMA_VERSION,
    Replay,
    ReplayVerdict,
    deck_hash,
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


def _select(n: int, minc: int = 1, maxc: int = 1) -> dict:
    return {
        "type": 0, "context": 0, "minCount": minc, "maxCount": maxc,
        "remainDamageCounter": 0, "remainEnergyCost": 0,
        "option": [{"type": 14} for _ in range(n)],
        "deck": None, "contextCard": None, "effect": None,
    }


def _records(choices, obs_full=None):
    """Build a minimal trace: one decision per entry in ``choices``.

    ``choices`` is a list of ``(n_options, chosen_list)``. When ``obs_full`` is set,
    each decision also carries a full ``obs`` (FULL_OBS style).
    """
    meta = {
        "kind": "meta", "schema_version": SCHEMA_VERSION, "trace_id": "t",
        "created_at": "2026-07-12T00:00:00+00:00", "record_level": 1,
        "engine": {"sha256": "a" * 64}, "git_sha": None, "python_version": "3.12.0",
        "agents": [{"index": 0, "name": "a0"}, {"index": 1, "name": "a1"}],
        "decks": [[1] * 60, [2] * 60], "deck_hashes": [deck_hash([1] * 60), deck_hash([2] * 60)],
        "first_player": 0, "start_error": None,
    }
    decisions = []
    for i, (n, chosen) in enumerate(choices):
        sel = _select(n, 1, len(chosen or []) or 1)
        dec = {
            "kind": "decision", "index": i, "select_player": i % 2, "your_index": i % 2,
            "turn": i + 1, "turn_action_count": 0, "select": sel, "choice": chosen,
            "thinking_time_ms": 0.0, "search_begin_input": f"sbi{i}", "logs": [],
        }
        if obs_full:
            dec["obs"] = {"select": sel, "logs": [], "current": {"yourIndex": i % 2, "turn": i + 1}}
        decisions.append(dec)
    result = {
        "kind": "result", "result": 0, "reason": 2, "winner": 0, "truncated": False,
        "first_player": 0, "final_turn": len(choices), "total_decisions": len(choices),
        "elapsed_ms": 0.0, "failure": None, "start_error": None,
        "final_logs": [{"type": 23, "result": 0, "reason": 2}],
    }
    return [meta] + decisions + [result]


# --------------------------------------------------------------------------- #
# L2 faithful replay + regenerate
# --------------------------------------------------------------------------- #

def test_faithful_stream_yields_decisions_in_order():
    replay = Replay(parse_records(_records([(2, [0]), (3, [2]), (1, [0])])))
    stream = list(replay.faithful_stream())
    assert [idx for idx, _, _ in stream] == [0, 1, 2]
    assert [choice for _, _, choice in stream] == [[0], [2], [0]]
    # each reconstructed obs exposes the legal-move select
    for _, obs, _ in stream:
        assert "select" in obs and "option" in obs["select"]


def test_regenerate_recreates_decisions_and_result_reason():
    replay = Replay(parse_records(_records([(2, [0]), (3, [2])])))
    regen = replay.regenerate()
    assert [d["choice"] for d in regen["decisions"]] == [[0], [2]]
    assert [d["player"] for d in regen["decisions"]] == [0, 1]
    # result reason regenerated (2 == empty-deck loss)
    assert regen["result"]["winner"] == 0
    assert regen["result"]["reason"] == 2
    assert regen["result"]["truncated"] is False


def test_reconstruct_obs_prefers_full_obs_when_present():
    recs = _records([(2, [0])], obs_full=True)
    replay = Replay(parse_records(recs))
    dec = replay.trace.decisions[0]
    assert Replay.reconstruct_obs(dec) is dec["obs"]


def test_reconstruct_obs_builds_partial_when_no_obs():
    replay = Replay(parse_records(_records([(2, [1])])))
    dec = replay.trace.decisions[0]
    obs = Replay.reconstruct_obs(dec)
    assert obs["select"]["option"]
    assert obs["current"]["yourIndex"] == 0
    assert obs["search_begin_input"] == "sbi0"


# --------------------------------------------------------------------------- #
# L1 agent-decision reproducibility (verify_agent)
# --------------------------------------------------------------------------- #

class _FixedAgent:
    """Deterministic agent returning ``range(k)`` — mirrors FirstOptionAgent."""

    def act(self, obs):
        select = obs.get("select")
        if not select:
            return []
        n = len(select["option"])
        k = max(select["minCount"], min(select["maxCount"], n))
        return list(range(k))


def test_verify_agent_consistent_on_first_option_trace():
    # recorded choices are all range(k) == [0], reproducible by _FixedAgent (L1)
    replay = Replay(parse_records(_records([(2, [0]), (3, [0]), (4, [0])])))
    verdict = replay.verify_agent(_FixedAgent())
    assert isinstance(verdict, ReplayVerdict)
    assert verdict.total == 3
    assert verdict.consistent
    assert verdict.mismatches == []


def test_verify_agent_reports_mismatch():
    # recorded choice [2] cannot be reproduced by an agent that always picks [0]
    replay = Replay(parse_records(_records([(3, [2]), (3, [0])])))
    verdict = replay.verify_agent(_FixedAgent())
    assert not verdict.consistent
    assert verdict.matches == 1                     # the [0] decision matches
    assert len(verdict.mismatches) == 1
    idx, recorded, produced = verdict.mismatches[0]
    assert idx == 0 and recorded == [2] and produced == [0]


def test_verify_agent_accepts_bare_callable():
    replay = Replay(parse_records(_records([(2, [0])])))
    verdict = replay.verify_agent(lambda obs: [0])
    assert verdict.consistent


def test_verify_agent_skips_unrecorded_choices():
    recs = _records([(2, [0]), (2, None)])
    verdict = Replay(parse_records(recs)).verify_agent(_FixedAgent())
    assert verdict.total == 1 and verdict.skipped == 1


# --------------------------------------------------------------------------- #
# engine-backed: end-to-end L1 + L3
# --------------------------------------------------------------------------- #

@requires_engine
def test_first_option_selfplay_replays_consistently(deck):
    """Record a deterministic FirstOptionAgent self-play, then replay it against a
    fresh FirstOptionAgent — every choice must reproduce (L1).

    Recorded at FULL_OBS: the reference agents parse the *whole* observation
    (``to_observation_class``), so faithful L1 replay of a parsing agent needs the
    full obs, which only FULL_OBS captures (see ``Replay.verify_agent``)."""
    from eval.agents import FirstOptionAgent
    from eval.match import record_match
    from eval.trace import RecordLevel, load_trace

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        record_match(deck, deck, [FirstOptionAgent(), FirstOptionAgent()],
                     out_path=out, level=RecordLevel.FULL_OBS)
        trace = load_trace(out)

    verdict = Replay(trace).verify_agent(FirstOptionAgent())
    assert verdict.total >= 1
    assert verdict.consistent, verdict.mismatches[:3]


@requires_engine
def test_l3_engine_resim_is_not_faithful(deck):
    """The L3 engine re-simulation must never claim faithful reproduction."""
    from eval.agents import RandomAgent
    from eval.match import record_match, replay_in_engine
    from eval.trace import RecordLevel, load_trace

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.jsonl")
        record_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)],
                     out_path=out, level=RecordLevel.LOGS)
        trace = load_trace(out)

    report = replay_in_engine(trace)
    assert report["faithful"] is False
    assert "seed" in report["note"]
    assert report["steps_matched"] <= report["recorded_decisions"]
