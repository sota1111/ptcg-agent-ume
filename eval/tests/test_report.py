"""Tests for arena statistics + the promotion gate (SOT-1626).

All pure — no engine. Covers the Wilson CI known value, candidate-centric matchup
aggregation (win/draw/loss, decisive win rate, 手数, 意思決定時間, 例外率, 席数),
deterministic re-aggregation from recorded rows (受け入れ条件: 結果から95%CIを
再生成可能), and every branch of the promotion gate (受け入れ条件: baseline昇格判定が
自動化される).
"""
from __future__ import annotations

import pytest

from eval.report import (
    MatchupSummary,
    aggregate_run,
    promotion_gate,
    summarize_matchup,
    wilson_ci,
)


# --------------------------------------------------------------------------- #
# Wilson CI
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_value():
    low, high = wilson_ci(50, 100)
    assert round(low, 4) == 0.4038
    assert round(high, 4) == 0.5962


def test_wilson_ci_no_evidence():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_rejects_out_of_range():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_wilson_ci_all_wins_upper_is_one():
    low, high = wilson_ci(100, 100)
    assert high == 1.0
    assert low < 1.0  # still uncertain from below


# --------------------------------------------------------------------------- #
# summarize_matchup over crafted candidate-centric rows
# --------------------------------------------------------------------------- #
def _row(**kw) -> dict:
    base = dict(
        matchup="cand_vs_base", candidate="cand", opponent="base",
        candidate_seat=0, candidate_won=True, draw=False, undecided=False,
        candidate_faulted=False, fault_category=None, reason="normal",
        steps=10, candidate_decisions=4, candidate_decision_ms=2.0,
    )
    base.update(kw)
    return base


def test_summarize_empty_is_maximally_uncertain():
    s = summarize_matchup([])
    assert s.n == 0
    assert (s.ci_low, s.ci_high) == (0.0, 1.0)
    assert s.decisive_win_rate is None


def test_summarize_win_draw_loss_rates_and_ci():
    rows = (
        [_row(candidate_won=True) for _ in range(6)]
        + [_row(candidate_won=False) for _ in range(3)]
        + [_row(candidate_won=False, draw=True) for _ in range(1)]
    )
    s = summarize_matchup(rows)
    assert (s.n, s.wins, s.losses, s.draws) == (10, 6, 3, 1)
    assert s.win_rate == 0.6
    assert s.loss_rate == 0.3
    assert s.draw_rate == 0.1
    # decisive win rate excludes the draw: 6 / (6+3)
    assert round(s.decisive_win_rate, 4) == round(6 / 9, 4)
    lo, hi = wilson_ci(6, 10)
    assert (s.ci_low, s.ci_high) == (lo, hi)


def test_summarize_means_and_seat_counts():
    rows = [
        _row(candidate_seat=0, steps=10, candidate_decisions=2, candidate_decision_ms=4.0),
        _row(candidate_seat=1, steps=20, candidate_decisions=2, candidate_decision_ms=6.0),
    ]
    s = summarize_matchup(rows)
    assert s.mean_steps == 15.0
    # total_ms / total_decisions = (4+6)/(2+2) = 2.5 ms per decision
    assert s.mean_decision_ms == 2.5
    assert s.seat_counts == {0: 1, 1: 1}


def test_summarize_exception_rate():
    rows = [
        _row(candidate_won=False, candidate_faulted=True,
             fault_category="agent_exception", reason="agent_exception"),
        _row(candidate_won=True),
    ]
    s = summarize_matchup(rows)
    assert s.exceptions == 1
    assert s.exception_rate == 0.5


def test_summarize_accepts_dataclass_records_too():
    """summarize_matchup reads attrs OR dict keys (works on MatchRecord-likes)."""
    class Rec:
        candidate = "cand"; opponent = "base"; candidate_won = True
        draw = False; candidate_faulted = False; steps = 5
        candidate_decision_ms = 1.0; candidate_decisions = 1; candidate_seat = 0
    s = summarize_matchup([Rec(), Rec()])
    assert s.n == 2 and s.wins == 2


def test_aggregate_run_groups_by_matchup_and_is_deterministic():
    rows = (
        [_row(matchup="cand_vs_a", candidate_won=True) for _ in range(3)]
        + [_row(matchup="cand_vs_b", candidate_won=False) for _ in range(2)]
    )
    g1 = aggregate_run(rows)
    g2 = aggregate_run(rows)
    assert set(g1) == {"cand_vs_a", "cand_vs_b"}
    assert g1["cand_vs_a"].wins == 3
    assert g1["cand_vs_b"].losses == 2
    # pure function of the rows -> byte-identical re-aggregation
    assert {k: v.to_dict() for k, v in g1.items()} == {k: v.to_dict() for k, v in g2.items()}


# --------------------------------------------------------------------------- #
# promotion_gate
# --------------------------------------------------------------------------- #
def _summary(**kw) -> MatchupSummary:
    base = dict(
        candidate="cand", opponent="best", n=1000, wins=560, losses=440, draws=0,
        win_rate=0.56, draw_rate=0.0, loss_rate=0.44, ci_low=0.529, ci_high=0.590,
        decisive_win_rate=0.56, mean_steps=40.0, mean_decision_ms=0.1,
        exceptions=0, exception_rate=0.0,
    )
    base.update(kw)
    return MatchupSummary(**base)


def test_gate_promotes_when_all_conditions_hold():
    v = promotion_gate(_summary(), time_limit_s=100.0, elapsed_s=30.0)
    assert v.promote is True
    assert v.reasons == []


def test_gate_holds_when_ci_low_not_above_half():
    v = promotion_gate(_summary(ci_low=0.5), time_limit_s=100.0, elapsed_s=1.0)
    assert v.promote is False
    assert any("CI lower bound" in r for r in v.reasons)


def test_gate_holds_on_candidate_exceptions():
    v = promotion_gate(_summary(exceptions=1, exception_rate=0.001),
                       time_limit_s=100.0, elapsed_s=1.0)
    assert v.promote is False
    assert any("exceptions" in r for r in v.reasons)


def test_gate_holds_when_over_time_budget():
    v = promotion_gate(_summary(), time_limit_s=10.0, elapsed_s=25.0)
    assert v.promote is False
    assert any("exceeds limit" in r for r in v.reasons)


def test_gate_time_check_skipped_without_limit():
    # No time_limit_s -> time is not a gating condition even if elapsed is large.
    v = promotion_gate(_summary(), elapsed_s=10_000.0)
    assert v.promote is True


def test_gate_reports_multiple_failures():
    v = promotion_gate(_summary(ci_low=0.4, exceptions=2, exception_rate=0.002),
                       time_limit_s=1.0, elapsed_s=9.0)
    assert v.promote is False
    assert len(v.reasons) == 3
