"""Self-play data pipeline acceptance tests (SOT-1688).

Pin the PPO-facing contract of :mod:`eval.selfplay`:

* every emitted JSONL line is schema-valid (:func:`eval.selfplay.validate_record`),
* the feature vector has the fixed :data:`agents.features.FEATURE_DIM` length,
* rewards are terminal ±1/0 from the deciding player's perspective and are
  consistent within a ``(game, player)`` trajectory,
* mixed pairings (rule vs random) alternate seats and record both agents,
* the run summary reports fault / invalid counts (expected 0 with SafeAgents).

Real matches need the gitignored cabt engine, hence the importorskip; the pure
``validate_record`` negative cases run first and engine-free semantics are kept
in :mod:`eval.tests.test_features`.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from agents.features import FEATURE_DIM  # noqa: E402
from eval.selfplay import (  # noqa: E402
    AGENT_KINDS,
    RECORD_FIELDS,
    SCHEMA,
    _main,
    run_selfplay,
    validate_record,
)

N_GAMES = 4


@pytest.fixture(scope="module")
def selfplay_run(tmp_path_factory):
    """One small mixed-pairing run shared by the assertions below."""
    out = tmp_path_factory.mktemp("selfplay") / "records.jsonl"
    summary = run_selfplay(
        N_GAMES, str(out), agents=("rule", "random"), base_seed=123,
    )
    with open(out, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return summary, records


# --------------------------------------------------------------------------- #
# validate_record: the machine-checkable schema contract
# --------------------------------------------------------------------------- #
def _valid_record() -> dict:
    return {
        "schema": SCHEMA, "feature_version": 1, "game": 0, "decision": 0,
        "player": 0, "agent": "rule", "features": [0.0] * FEATURE_DIM,
        "action": [1], "action_index": 1, "n_options": 3, "min_count": 1,
        "max_count": 1, "select_type": 0, "select_context": 0, "reward": 1.0,
        "result": "win", "winner": 0, "reason": "normal", "steps": 42,
    }


def test_validate_record_accepts_a_valid_record():
    assert validate_record(_valid_record()) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: r.pop("features"), "missing field"),
    (lambda r: r.update(features=[0.0] * (FEATURE_DIM - 1)), "features length"),
    (lambda r: r.update(action=[5]), "outside [0, 3)"),
    (lambda r: r.update(action=[0, 0], max_count=2), "duplicates"),
    (lambda r: r.update(action=[0, 1], max_count=1), "outside [1, 1]"),
    (lambda r: r.update(reward=0.5), "reward"),
    (lambda r: r.update(result="ok"), "result"),
    (lambda r: r.update(player=2), "player"),
])
def test_validate_record_flags_violations(mutate, fragment):
    record = _valid_record()
    mutate(record)
    errors = validate_record(record)
    assert errors and any(fragment in e for e in errors)


# --------------------------------------------------------------------------- #
# The pipeline run: schema-valid, fault-free, PPO-consistent records
# --------------------------------------------------------------------------- #
def test_run_is_fault_free_and_schema_valid(selfplay_run):
    summary, records = selfplay_run
    assert summary["games"] == N_GAMES
    assert summary["faults"] == 0
    assert summary["invalid_records"] == 0
    assert summary["feature_dim"] == FEATURE_DIM
    assert records, "a match must yield decision records"
    assert len(records) == summary["decisions"]
    for record in records:
        assert validate_record(record) == [], record.get("schema_errors")
        assert set(RECORD_FIELDS) <= set(record)


def test_features_have_fixed_dimension(selfplay_run):
    _, records = selfplay_run
    assert all(len(r["features"]) == FEATURE_DIM for r in records)


def test_rewards_are_terminal_and_consistent_per_trajectory(selfplay_run):
    _, records = selfplay_run
    for record in records:
        assert record["reward"] in (1.0, -1.0, 0.0)
        if record["result"] == "win":
            assert record["winner"] == record["player"] and record["reward"] == 1.0
        elif record["result"] == "loss":
            assert record["winner"] == 1 - record["player"] and record["reward"] == -1.0
        else:
            assert record["reward"] == 0.0
    # Within one (game, player) trajectory the terminal signal is constant.
    trajectories: dict[tuple, set] = {}
    for r in records:
        trajectories.setdefault((r["game"], r["player"]), set()).add(
            (r["reward"], r["result"], r["winner"])
        )
    assert all(len(v) == 1 for v in trajectories.values())


def test_mixed_pairing_alternates_seats_and_labels(selfplay_run):
    _, records = selfplay_run
    assert {r["game"] for r in records} == set(range(N_GAMES))
    assert {r["agent"] for r in records} == {"rule", "random"}
    assert {r["player"] for r in records} <= {0, 1}
    # Seat alternation: rule sits seat 0 in even games, seat 1 in odd games.
    for r in records:
        expected_seat = r["game"] % 2 if r["agent"] == "rule" else 1 - (r["game"] % 2)
        assert r["player"] == expected_seat


def test_decision_indices_are_dense_per_trajectory(selfplay_run):
    _, records = selfplay_run
    by_traj: dict[tuple, list] = {}
    for r in records:
        by_traj.setdefault((r["game"], r["player"]), []).append(r["decision"])
    for indices in by_traj.values():
        assert sorted(indices) == list(range(len(indices)))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_generates_jsonl(tmp_path, capsys):
    out = tmp_path / "cli.jsonl"
    rc = _main(["--games", "2", "--out", str(out), "--agents", "rule,random",
                "--seed", "7"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["games"] == 2 and summary["faults"] == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert lines and all(validate_record(r) == [] for r in lines)


def test_cli_rejects_unknown_agent(tmp_path):
    with pytest.raises(SystemExit):
        _main(["--games", "1", "--out", str(tmp_path / "x.jsonl"),
               "--agents", "rule,alien"])
    assert "alien" not in AGENT_KINDS
