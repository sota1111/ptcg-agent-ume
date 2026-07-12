"""Engine-backed tests for the competition agents package (SOT-1646, R1).

Covers the two live-engine acceptance criteria:
* **Random self-play N>=200 with crash 0** — the ``agents.RandomAgent`` plays 200
  paired matches through the real cabt engine with zero faults (no illegal move, no
  exception, no timeout attributed to either side): the safety skeleton emits only
  legal actions in practice, not just on fixtures.
* **R0 Arena injection** — ``agents.RandomAgent`` and the ``agents.RuleAgent`` skeleton
  are injected into :func:`eval.arena.run_arena` and play to completion (対戦成立),
  and a direct match confirms the RuleAgent skeleton encounters real contexts, records
  them all as unsupported (未対応率 = 1.0), and still never faults.

The engine (``cg/``) is gitignored/absent in CI, so these skip via ``requires_engine``.
"""
from __future__ import annotations

import pytest

from eval.arena import run_arena
from eval.match import play_match

from .conftest import requires_engine

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from agents import RandomAgent, RuleAgent  # noqa: E402


@requires_engine
def test_random_self_play_200_no_crash(deck):
    """200 paired RandomAgent-vs-RandomAgent matches complete with zero faults."""
    report = run_arena(
        lambda s: RandomAgent(seed=s),
        lambda s: RandomAgent(seed=s),
        deck0=deck,
        n_matches=200,
        side_swap=True,
        label_a="randomA",
        label_b="randomB",
        write_outputs=False,
        record_traces=False,
    )
    assert report.totals["n"] == 200
    # A legal-random agent must never fault the engine's legality check.
    assert report.safety["a_faults"] == 0
    assert report.safety["b_faults"] == 0
    # Every match reached a real terminal state (no MAX_STEPS truncation).
    assert report.totals["undecided"] == 0


@requires_engine
def test_arena_injects_random_and_rule(deck):
    """RandomAgent vs the RuleAgent skeleton play a paired arena to completion."""
    report = run_arena(
        lambda s: RandomAgent(seed=s),
        lambda s: RuleAgent(seed=s),
        deck0=deck,
        n_matches=10,
        side_swap=True,
        label_a="random",
        label_b="rule",
        write_outputs=False,
        record_traces=False,
    )
    assert report.totals["n"] == 10
    # Both sides are always-legal skeletons: neither may fault.
    assert report.safety["a_faults"] == 0
    assert report.safety["b_faults"] == 0
    # 対戦成立: the matches produced decisions for both agents.
    assert report.latency["random"]["n_decisions"] > 0
    assert report.latency["rule"]["n_decisions"] > 0


@requires_engine
def test_rule_skeleton_measures_and_stays_legal(deck):
    """A direct match lets us read the RuleAgent skeleton's own measurement."""
    rng_agent = RandomAgent(seed=1)
    rule = RuleAgent(seed=2)
    result = play_match(deck, deck, [rng_agent, rule], max_steps=100_000)

    # The match resolved without attributing a fault to either always-legal agent.
    assert result.faulted_player is None
    # The skeleton actually made decisions and recorded encounters for real contexts.
    assert sum(s.encounters for s in rule.stats.values()) > 0
    # R1 skeleton has no tactics, so every encounter is unsupported and fell back.
    assert rule.unsupported_rate() == 1.0
    assert sum(s.handled for s in rule.stats.values()) == 0
