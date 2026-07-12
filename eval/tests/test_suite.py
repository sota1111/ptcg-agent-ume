"""Tests for the baseline suite + promotion pipeline (SOT-1626).

Split into a pure-unit part (the arena-record → candidate-row converter, no engine)
and engine-backed parts that run a real suite and check the four acceptance criteria:
2+ distinct agents complete a side-swapped run, the 95% CI is regenerable from the
recorded results, an abnormal agent is isolated and its 例外率 reported, and the
baseline promotion verdict is produced automatically.
"""
from __future__ import annotations

import json

from eval.agents import BaseAgent, RandomAgent
from eval.arena import MatchRecord
from eval.config import AgentSpec, DeckSpec, RunConfig
from eval.report import aggregate_run
from eval.suite import candidate_row, run_suite

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Pure-unit: candidate-centric conversion (candidate is always agent A)
# --------------------------------------------------------------------------- #
def _match_record(**kw) -> MatchRecord:
    base = dict(
        match_index=0, pair_index=0, seat_of_a=1, first_player=0,
        label_a="cand", label_b="base", seed_a=0, seed_b=1,
        winner_seat=1, winner_label="cand", a_won=True, b_won=False,
        draw=False, undecided=False, reason="normal",
        faulted_seat=None, faulted_label=None, fault_category=None,
        steps=12, a_decisions=6, b_decisions=6, a_decision_ms=3.0,
        b_decision_ms=4.0, trace_path=None,
    )
    base.update(kw)
    return MatchRecord(**base)


def test_candidate_row_maps_agent_a_to_candidate():
    row = candidate_row(_match_record(), "cand_vs_base")
    assert row["matchup"] == "cand_vs_base"
    assert row["candidate"] == "cand" and row["opponent"] == "base"
    assert row["candidate_won"] is True
    assert row["candidate_seat"] == 1
    assert row["candidate_decisions"] == 6
    assert row["candidate_decision_ms"] == 3.0  # agent A's ms, not B's
    assert row["candidate_faulted"] is False


def test_candidate_row_attributes_candidate_fault_only():
    # Fault by agent B is NOT the candidate's fault.
    b_fault = candidate_row(
        _match_record(faulted_seat=0, faulted_label="base",
                      fault_category="illegal_move", reason="illegal_move"),
        "cand_vs_base",
    )
    assert b_fault["candidate_faulted"] is False
    assert b_fault["fault_category"] is None
    # Fault by agent A (the candidate) IS.
    a_fault = candidate_row(
        _match_record(a_won=False, winner_seat=0, winner_label="base",
                      faulted_seat=1, faulted_label="cand",
                      fault_category="agent_exception", reason="agent_exception"),
        "cand_vs_base",
    )
    assert a_fault["candidate_faulted"] is True
    assert a_fault["fault_category"] == "agent_exception"


# --------------------------------------------------------------------------- #
# Engine-backed: a real suite
# --------------------------------------------------------------------------- #
def _config(deck, **kw) -> RunConfig:
    base = dict(
        candidate=AgentSpec(kind="random", name="cand"),
        baselines=[AgentSpec(kind="first", name="first"),
                   AgentSpec(kind="random", name="best")],
        deck0=DeckSpec(cards=tuple(deck)),
        n_matches=6, side_swap=True, agent_seed=0, time_limit_s=60.0,
    )
    base.update(kw)
    return RunConfig(**base)


@requires_engine
def test_suite_runs_all_baselines_with_gate(deck, tmp_path):
    cfg = _config(deck, out_dir=str(tmp_path))
    res = run_suite(cfg)
    # one summary per baseline, keyed by matchup
    assert set(res.summaries) == {"cand_vs_first", "cand_vs_best"}
    for s in res.summaries.values():
        assert s.n == 6
        assert s.seat_counts == {0: 3, 1: 3}  # side-swap balances seats
    # the gate targets the last baseline ("直前best" convention) automatically
    assert res.gate_matchup == "cand_vs_best"
    assert isinstance(res.gate.promote, bool)


@requires_engine
def test_suite_ci_regenerable_from_results(deck, tmp_path):
    """受け入れ条件: 結果から95% CIを再生成可能."""
    cfg = _config(deck, out_dir=str(tmp_path), label="regen")
    res = run_suite(cfg)
    rows = [json.loads(l) for l in open(res.results_path)]
    regen = aggregate_run(rows)
    assert set(regen) == set(res.summaries)
    for k, s in res.summaries.items():
        assert regen[k].to_dict() == s.to_dict()  # incl. ci_low / ci_high


@requires_engine
def test_suite_writes_manifest_and_artifacts(deck, tmp_path):
    cfg = _config(deck, out_dir=str(tmp_path), label="artifacts")
    res = run_suite(cfg)
    manifest = json.loads(open(f"{res.suite_dir}/manifest.json").read())
    assert manifest["candidate"]["name"] == "cand"
    assert manifest["gate_baseline_index"] == -1
    gate = json.loads(open(res.gate_path).read())
    assert gate["gate_matchup"] == "cand_vs_best"
    assert "verdict" in gate


class _BadAgent(BaseAgent):
    name = "bad"

    def act(self, obs: dict) -> list[int]:
        raise RuntimeError("boom")


@requires_engine
def test_suite_isolates_abnormal_candidate_and_reports_exception_rate(deck, tmp_path):
    """受け入れ条件: 異常agentを隔離し例外率を報告できる (batch does not crash)."""
    from eval.config import register_agent

    register_agent("bad", lambda params, seed: _BadAgent())
    cfg = RunConfig(
        candidate=AgentSpec(kind="bad", name="bad"),
        baselines=[AgentSpec(kind="random", name="best")],
        deck0=DeckSpec(cards=tuple(deck)),
        n_matches=4, side_swap=True, out_dir=str(tmp_path),
    )
    res = run_suite(cfg)
    s = res.summaries["bad_vs_best"]
    assert s.n == 4
    assert s.exceptions == 4          # every match faulted the candidate
    assert s.exception_rate == 1.0
    assert s.wins == 0                # a faulting agent never wins
    # gate must HOLD: exceptions > 0
    assert res.gate.promote is False
    assert any("exceptions" in r for r in res.gate.reasons)
