"""Tests for the paired-match evaluation Arena (SOT-1645).

Split into pure-unit tests (no engine — statistics, factory resolution, and the
aggregation over crafted records) and engine-backed tests (real matches through the
cabt engine). The engine (``cg/``) is gitignored/absent in CI, so engine tests skip
cleanly via ``requires_engine``.

Covers the R0 acceptance criteria:
* any two agents can be injected and played paired (RuleAgent vs new agent);
* Random vs Random paired N>=200 completes stably;
* a seed/agent/deck/version-stamped JSONL trace and an aggregation report are produced
  (win rate + Wilson CI, latency p50/p95/p99, safety rates, 先後別 win rates);
* an agent crash/illegal move is isolated as that agent's loss without killing the batch.
"""
from __future__ import annotations

import json
import os

import pytest

from eval.agents import BaseAgent, FirstOptionAgent, RandomAgent
from eval.arena import (
    MatchRecord,
    aggregate,
    percentile,
    run_arena,
    wilson_ci,
    _resolve_factory,
)

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Pure-unit: statistics
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_value():
    low, high = wilson_ci(50, 100)
    assert round(low, 4) == 0.4038
    assert round(high, 4) == 0.5962


def test_wilson_ci_no_evidence_is_maximally_uncertain():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_rejects_out_of_range():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_percentile_linear_interpolation():
    xs = [10, 20, 30, 40]
    assert percentile(xs, 0) == 10
    assert percentile(xs, 100) == 40
    assert percentile(xs, 50) == 25  # midpoint of 20 and 30
    assert percentile([], 95) == 0.0
    assert percentile([7], 95) == 7


def test_percentile_monotonic():
    xs = list(range(1, 1001))
    assert percentile(xs, 50) <= percentile(xs, 95) <= percentile(xs, 99)


# --------------------------------------------------------------------------- #
# Pure-unit: agent injection (instance / 0-arg factory / 1-arg factory)
# --------------------------------------------------------------------------- #
def test_resolve_factory_instance_is_reused():
    inst = FirstOptionAgent()
    make = _resolve_factory(inst)
    assert make(1) is inst and make(2) is inst


def test_resolve_factory_zero_arg_and_one_arg():
    make0 = _resolve_factory(lambda: FirstOptionAgent())
    assert isinstance(make0(5), FirstOptionAgent)

    seeds = []
    make1 = _resolve_factory(lambda s: seeds.append(s) or RandomAgent(seed=s))
    a = make1(7)
    assert isinstance(a, RandomAgent) and seeds == [7]


def test_resolve_factory_class_receives_seed():
    # A class is callable; RandomAgent(seed) must build a seeded agent.
    make = _resolve_factory(RandomAgent)
    assert isinstance(make(3), RandomAgent)


# --------------------------------------------------------------------------- #
# Pure-unit: aggregation over crafted records (no engine)
# --------------------------------------------------------------------------- #
def _record(**kw) -> MatchRecord:
    """A MatchRecord with sensible defaults; override only the fields under test."""
    base = dict(
        match_index=0, pair_index=0, seat_of_a=0, first_player=0,
        label_a="A", label_b="B", seed_a=0, seed_b=1,
        winner_seat=0, winner_label="A", a_won=True, b_won=False,
        draw=False, undecided=False, reason="normal",
        faulted_seat=None, faulted_label=None, fault_category=None,
        steps=10, a_decisions=5, b_decisions=5, a_decision_ms=1.0,
        b_decision_ms=1.0, trace_path=None,
    )
    base.update(kw)
    return MatchRecord(**base)


_CFG = {"label_a": "A", "label_b": "B"}


def test_aggregate_basic_win_rates():
    recs = [_record(match_index=i) for i in range(4)]  # A wins all
    recs.append(_record(match_index=4, winner_seat=1, winner_label="B",
                        a_won=False, b_won=True))
    rep = aggregate(recs, [1.0], [2.0], _CFG)
    assert rep.totals["n"] == 5
    assert rep.totals["a_wins"] == 4
    assert rep.totals["b_wins"] == 1
    assert rep.win_rates["a_win_rate"] == 0.8
    lo, hi = rep.win_rates["a_win_rate_ci95"]
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0


def test_aggregate_seat_winrate_split_is_per_agent():
    """先後別勝率: A and B must use their OWN seat, not A's for both (regression)."""
    recs = [
        # A seat 0 & first, A wins  -> A as_first win, B as_second loss
        _record(match_index=0, seat_of_a=0, first_player=0,
                winner_seat=0, winner_label="A", a_won=True, b_won=False),
        # A seat 1, first_player 0 -> B (seat 0) is first; B wins
        _record(match_index=1, seat_of_a=1, first_player=0,
                winner_seat=0, winner_label="B", a_won=False, b_won=True),
    ]
    rep = aggregate(recs, [], [], _CFG)
    sw = rep.seat_winrate
    # Match 0: A first. Match 1: B first. So each agent was first exactly once.
    assert sw["A"]["as_first"]["n"] == 1
    assert sw["A"]["as_second"]["n"] == 1
    assert sw["B"]["as_first"]["n"] == 1
    assert sw["B"]["as_second"]["n"] == 1
    # The first mover won both matches -> first_player_win_rate == 1.0
    assert sw["first_player_win_rate"] == 1.0
    assert sw["second_player_win_rate"] == 0.0
    # A won when first (match 0), lost when second (match 1).
    assert sw["A"]["as_first"]["win_rate"] == 1.0
    assert sw["A"]["as_second"]["win_rate"] == 0.0
    # B won when first (match 1), lost when second (match 0).
    assert sw["B"]["as_first"]["win_rate"] == 1.0
    assert sw["B"]["as_second"]["win_rate"] == 0.0


def test_aggregate_safety_attribution_and_categories():
    recs = [
        _record(match_index=0),  # clean A win
        _record(match_index=1, winner_seat=0, winner_label="A", a_won=True,
                b_won=False, reason="agent_exception", faulted_seat=1,
                faulted_label="B", fault_category="agent_exception"),
        _record(match_index=2, winner_seat=1, winner_label="B", a_won=False,
                b_won=True, reason="illegal_move", faulted_seat=0,
                faulted_label="A", fault_category="illegal_move"),
        _record(match_index=3, winner_seat=None, winner_label=None, a_won=False,
                b_won=False, draw=False, undecided=True, reason="max_steps"),
    ]
    rep = aggregate(recs, [], [], _CFG)
    assert rep.safety["a_faults"] == 1
    assert rep.safety["b_faults"] == 1
    assert rep.safety["a_fault_categories"] == {"illegal_move": 1}
    assert rep.safety["b_fault_categories"] == {"agent_exception": 1}
    assert rep.safety["undecided"] == 1
    assert rep.reason_counts["max_steps"] == 1


def test_aggregate_is_deterministic_from_records():
    recs = [_record(match_index=i, winner_seat=i % 2,
                    winner_label="A" if i % 2 == 0 else "B",
                    a_won=(i % 2 == 0), b_won=(i % 2 == 1)) for i in range(20)]
    r1 = aggregate(recs, [1, 2, 3], [4, 5, 6], _CFG).to_dict()
    r2 = aggregate(recs, [1, 2, 3], [4, 5, 6], _CFG).to_dict()
    assert r1 == r2  # pure function of the inputs -> report is regenerable


# --------------------------------------------------------------------------- #
# Engine-backed: real matches
# --------------------------------------------------------------------------- #
class _BadAgent(BaseAgent):
    """Always raises inside act — used to verify fault isolation."""

    name = "bad"

    def act(self, obs: dict) -> list[int]:
        raise RuntimeError("boom")


@requires_engine
def test_inject_two_different_agents_paired(deck, tmp_path):
    """Any two distinct agents can be injected and played paired (R0 core)."""
    rep = run_arena(
        lambda s: RandomAgent(seed=s),
        lambda s: FirstOptionAgent(),
        deck0=deck, n_matches=6, side_swap=True, agent_seed=0,
        label_a="random", label_b="first",
        out_dir=str(tmp_path), record_traces=False,
    )
    assert rep.totals["n"] == 6
    assert set(rep.latency) == {"random", "first"}
    assert 0.0 <= rep.win_rates["a_win_rate"] <= 1.0
    assert 0.0 <= rep.win_rates["b_win_rate"] <= 1.0
    # every match is decided (A win + B win + draw + undecided == n)
    t = rep.totals
    assert t["a_wins"] + t["b_wins"] + t["draws"] + t["undecided"] == 6


@requires_engine
def test_side_swap_balances_seats(deck, tmp_path):
    rep = run_arena(
        lambda s: RandomAgent(seed=s), lambda s: RandomAgent(seed=s),
        deck0=deck, n_matches=10, side_swap=True,
        out_dir=str(tmp_path), record_traces=False,
    )
    recs = [json.loads(l) for l in open(rep.results_path)]
    assert len(recs) == 10
    seat0 = sum(1 for r in recs if r["seat_of_a"] == 0)
    assert seat0 == 5  # exactly balanced for an even match count
    assert [r["pair_index"] for r in recs[:2]] == [0, 0]


@requires_engine
def test_trace_stamped_with_seed_agent_deck_version(deck, tmp_path):
    rep = run_arena(
        lambda s: RandomAgent(seed=s), lambda s: RandomAgent(seed=s),
        deck0=deck, n_matches=2, side_swap=True,
        out_dir=str(tmp_path), record_traces=True,
    )
    traces = sorted((tmp_path).glob("**/traces/match_*.jsonl"))
    assert len(traces) == 2
    meta = json.loads(open(traces[0]).readline())
    # agent identity + per-match SEED
    assert len(meta["agents"]) == 2
    assert all("seed" in a for a in meta["agents"])
    assert all("name" in a for a in meta["agents"])
    # decks + deck hashes (deck provenance)
    assert len(meta["decks"]) == 2
    assert len(meta["deck_hashes"]) == 2
    # version/provenance: engine hash + git sha + schema
    assert meta["schema_version"]
    assert "engine" in meta and "git_sha" in meta
    # the manifest also stamps deck hash + engine + git sha
    manifest = json.loads(open(tmp_path / rep.config["run_label"] / "manifest.json").read())
    assert manifest["deck0_hash"] and "engine" in manifest


@requires_engine
def test_fault_isolation_bad_agent(deck, tmp_path):
    """A crashing agent loses its matches; the batch still completes and reports it."""
    rep = run_arena(
        _BadAgent(), lambda s: RandomAgent(seed=s),
        deck0=deck, n_matches=4, side_swap=True,
        label_a="bad", label_b="random",
        out_dir=str(tmp_path), record_traces=False,
    )
    assert rep.totals["n"] == 4
    # every match is a bad-agent fault -> bad never wins, random wins all
    assert rep.safety["a_faults"] == 4
    assert rep.safety["a_fault_categories"].get("agent_exception") == 4
    assert rep.totals["a_wins"] == 0
    assert rep.totals["b_wins"] == 4


@requires_engine
def test_latency_percentiles_present_and_monotonic(deck, tmp_path):
    rep = run_arena(
        lambda s: RandomAgent(seed=s), lambda s: RandomAgent(seed=s),
        deck0=deck, n_matches=6, side_swap=True,
        out_dir=str(tmp_path), record_traces=False,
    )
    for label, stats in rep.latency.items():
        assert stats["n_decisions"] > 0
        assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]


@requires_engine
def test_random_vs_random_paired_n200_completes(deck, tmp_path):
    """R0 acceptance: Random vs Random paired N>=200 runs to completion, stably."""
    rep = run_arena(
        lambda s: RandomAgent(seed=s), lambda s: RandomAgent(seed=s),
        deck0=deck, n_matches=200, side_swap=True, agent_seed=0,
        out_dir=str(tmp_path), record_traces=False,
    )
    t = rep.totals
    assert t["n"] == 200
    assert t["a_wins"] + t["b_wins"] + t["draws"] + t["undecided"] == 200
    assert rep.safety["undecided"] == 0  # random self-play always resolves
    # report is regenerable from the persisted results.jsonl (pure aggregation)
    recs = [MatchRecord(**json.loads(l)) for l in open(rep.results_path)]
    assert len(recs) == 200
