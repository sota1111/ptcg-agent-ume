"""Tests for the KPI recording/reporting pipeline (SOT-1710) — engine-free."""
from __future__ import annotations

import pytest

from eval.kpi import (append_history, build_record, load_history,
                      record_from_bench_result, shard_sizes, wilson_ci)
from eval.kpi_report import compare_last_two


def _bench_result(opponent="rule", wins=30, n=48, faults=0, mean_ms=200.0):
    losses = n - wins
    lo, hi = wilson_ci(wins, n)
    return {
        "opponent": opponent, "n": n, "seed": 7,
        "temperature": 0.25, "time_limit_s": 0.4, "per_move_timeout_s": 5.0,
        "final_wins": wins, "opponent_wins": losses, "draws": 0,
        "undecided": 0, "final_win_rate": wins / n,
        "ci95_low": lo, "ci95_high": hi,
        "final_faults": faults, "final_fault_categories": {},
        "latency_final": {"n_decisions": 900, "mean_ms": mean_ms,
                          "max_ms": 410.0},
    }


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(30, 48)
    assert 0.0 <= lo < 30 / 48 < hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_shard_sizes_are_balanced_and_complete():
    assert shard_sizes(200, 4) == [50, 50, 50, 50]
    assert shard_sizes(10, 3) == [4, 3, 3]
    assert shard_sizes(2, 8) == [1, 1]
    assert shard_sizes(0, 4) == []


def test_shard_sizes_reject_invalid_values():
    with pytest.raises(ValueError):
        shard_sizes(-1, 2)
    with pytest.raises(ValueError):
        shard_sizes(10, 0)


def test_build_record_full():
    rec = build_record(_bench_result("rule"), _bench_result("random", wins=44,
                                                            n=48),
                       issue="SOT-TEST")
    assert rec["schema"] == "ume-kpi-v1"
    assert rec["issue"] == "SOT-TEST"
    assert rec["n_rule"] == 48 and rec["n_random"] == 48
    kpis = rec["kpis"]
    assert kpis["winrate_vs_rule"]["value"] == pytest.approx(30 / 48, abs=1e-4)
    assert kpis["winrate_vs_random"]["value"] == pytest.approx(44 / 48,
                                                               abs=1e-4)
    assert kpis["winrate_vs_rule"]["ci95"][0] < 30 / 48
    assert kpis["fault_total"]["value"] == 0
    assert kpis["decision_time_mean_ms"]["value"] == 200.0
    assert kpis["decision_time_mean_ms"]["timing_opponent"] == "rule"


def test_build_record_requires_one_result():
    with pytest.raises(ValueError):
        build_record(None, None)


def test_record_from_bench_result_routes_by_opponent():
    rec = record_from_bench_result(_bench_result("random"), issue="SOT-X")
    assert rec["source"] == "bench_final_vs_rule"
    assert rec["kpis"]["winrate_vs_rule"]["value"] is None
    assert rec["kpis"]["winrate_vs_random"]["value"] is not None
    assert rec["n_rule"] is None and rec["n_random"] == 48


def test_record_from_aggregate_shape_latency():
    result = _bench_result("rule")
    del result["latency_final"]
    result["latency_final_mean_ms_chunks"] = [180.0, 220.0]
    result["latency_final_max_ms"] = 430.0
    rec = record_from_bench_result(result, issue="SOT-X")
    kpi = rec["kpis"]["decision_time_mean_ms"]
    assert kpi["value"] == 200.0
    assert kpi["max_ms"] == 430.0


def test_history_roundtrip(tmp_path):
    path = str(tmp_path / "hist.jsonl")
    rec = build_record(_bench_result(), issue="SOT-X")
    append_history(rec, path)
    append_history(rec, path)
    hist = load_history(path)
    assert len(hist) == 2 and hist[0]["issue"] == "SOT-X"
    with open(path) as f:
        assert len(f.readlines()) == 2  # 1 measurement = 1 line


def test_history_env_override(tmp_path, monkeypatch):
    path = str(tmp_path / "env_hist.jsonl")
    monkeypatch.setenv("UME_KPI_HISTORY", path)
    append_history(build_record(_bench_result(), issue="SOT-X"))
    assert len(load_history()) == 1


def test_compare_last_two_verdicts():
    prev = build_record(_bench_result("rule", wins=15, n=48),
                        _bench_result("random", wins=40, n=48),
                        issue="A")
    cur = build_record(_bench_result("rule", wins=30, n=48, mean_ms=300.0,
                                     faults=1),
                       issue="B")  # no random this time
    rows = {r["kpi"]: r for r in compare_last_two([prev, cur])}
    assert rows["winrate_vs_rule"]["verdict"] == "改善"
    assert rows["winrate_vs_rule"]["ci_disjoint"] is True
    assert rows["winrate_vs_random"]["verdict"] == "n/a"  # null skipped
    assert rows["fault_total"]["verdict"] == "NG"
    assert rows["decision_time_mean_ms"]["verdict"] == "悪化"


def test_compare_flat_band():
    prev = build_record(_bench_result("rule", wins=30, n=1000), issue="A")
    cur = build_record(_bench_result("rule", wins=31, n=1000), issue="B")
    rows = {r["kpi"]: r for r in compare_last_two([prev, cur])}
    assert rows["winrate_vs_rule"]["verdict"] == "横ばい"
    assert rows["fault_total"]["verdict"] == "OK"


def test_compare_needs_two():
    assert compare_last_two([build_record(_bench_result(), issue="A")]) == []
    assert compare_last_two([]) == []
