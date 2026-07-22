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
    _engine_reason_code,
    _main,
    _prize_counts,
    load_deck,
    load_deck_dir,
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
# Deck rotation (SOT-1695)
# --------------------------------------------------------------------------- #
def test_load_deck_dir_reads_sorted_csv_decks():
    decks = load_deck_dir("decks/rotation_baseline")
    assert len(decks) == 25
    assert [name for name, _ in decks] == sorted(name for name, _ in decks)
    assert all(len(deck) == 60 for _, deck in decks)


def test_deck_rotation_mirrors_and_stamps_deck_field(tmp_path):
    decks = load_deck_dir("decks/rotation_baseline")[:2]
    out = tmp_path / "rotation.jsonl"
    summary = run_selfplay(
        4, str(out), agents=("rule", "rule"), decks=decks, base_seed=11,
    )
    assert summary["faults"] == 0 and summary["invalid_records"] == 0
    assert summary["decks"] == [name for name, _ in decks]
    assert summary["per_deck"] == {
        decks[0][0]: {"games": 2, "faults": 0},
        decks[1][0]: {"games": 2, "faults": 0},
    }
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert records and all(validate_record(r) == [] for r in records)
    # Game g mirrors decks[g % len(decks)] and every record carries its name.
    for r in records:
        assert r["deck"] == decks[r["game"] % 2][0]


def test_deck_rotation_is_exclusive_with_explicit_decks(tmp_path):
    decks = [("deck.csv", load_deck("deck.csv"))]
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_selfplay(
            1, str(tmp_path / "x.jsonl"), decks=decks, deck0=decks[0][1],
        )


# --------------------------------------------------------------------------- #
# Reward-shaping signal capture (SOT-1699)
# --------------------------------------------------------------------------- #
def test_prize_counts_uses_deciding_player_perspective():
    obs = {"current": {"yourIndex": 1, "players": [
        {"prize": [0] * 4},   # seat 0 has 4 prizes left
        {"prize": [0] * 2},   # seat 1 (me) has 2 prizes left
    ]}}
    assert _prize_counts(obs) == (2, 4)   # (own, opp) from seat 1's view


def test_prize_counts_falls_back_to_six_when_absent():
    assert _prize_counts({}) == (6, 6)
    assert _prize_counts({"current": {"yourIndex": 0, "players": [{}, {}]}}) == (6, 6)


class _FakeResult:
    def __init__(self, detail):
        self.detail = detail


@pytest.mark.parametrize("detail, code", [
    ("engine reason=2", 2),
    ("engine reason=3", 3),
    ("engine reason=None", None),
    (None, None),
    ("no code here", None),
])
def test_engine_reason_code_parses_detail(detail, code):
    assert _engine_reason_code(_FakeResult(detail)) == code


def test_selfplay_records_carry_prize_and_reason_fields(selfplay_run):
    _, records = selfplay_run
    for r in records:
        assert isinstance(r["own_prizes"], int) and 0 <= r["own_prizes"] <= 6
        assert isinstance(r["opp_prizes"], int) and 0 <= r["opp_prizes"] <= 6
        assert r["end_reason_code"] is None or isinstance(r["end_reason_code"], int)
    # A normal engine termination stamps one of the RESULT reason codes.
    assert any(r["end_reason_code"] in (1, 2, 3, 4) for r in records)


def test_record_labels_keeps_only_the_learner_side(tmp_path):
    out = tmp_path / "league.jsonl"
    summary = run_selfplay(
        2, str(out), agents=("ppo", "rule"), base_seed=5, record_labels={"ppo"},
    )
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert records, "the learner side must still be recorded"
    assert {r["agent"] for r in records} == {"ppo"}   # opponent records filtered out
    assert summary["faults"] == 0
    assert all(validate_record(r) == [] for r in records)


def test_collect_selfplay_league_records_only_the_learner(tmp_path):
    """SOT-1699: league games spar the learner vs a past snapshot, learner-only."""
    import random

    pytest.importorskip("numpy", reason="train.ppo is numpy-only")
    from train.ppo import _collect_selfplay, init_params, params_to_policy

    params = init_params(hidden=8, seed=0)
    snapshot = params_to_policy(init_params(hidden=8, seed=1))
    out = tmp_path / "iter.jsonl"
    summary = _collect_selfplay(
        params, games=2, base_seed=0, out_path=str(out),
        league_frac=1.0, snapshots=[snapshot], rng=random.Random(0),
    )
    assert summary["league_games"] == 2 and summary["mirror_games"] == 0
    assert summary["faults"] == 0 and summary["invalid_records"] == 0
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert records and {r["agent"] for r in records} == {"ppo"}
    # No pool -> league falls back to the mirror (both seats are the learner).
    out2 = tmp_path / "iter2.jsonl"
    s2 = _collect_selfplay(params, games=2, base_seed=0, out_path=str(out2),
                           league_frac=1.0, snapshots=[])
    assert s2["league_games"] == 0 and s2["mirror_games"] == 2


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
