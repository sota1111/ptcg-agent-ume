"""Tests for the candidate-move decision harness and the submission entry (SOT-1691).

The pipeline core (generate → validate → score → decide) is engine-free — the
selection contract is just ``option``/``minCount``/``maxCount`` attributes — so
the unit half runs without the gitignored ``cg/`` engine. The integration half
(:class:`agents.harness.HarnessAgent` end-to-end, ``main.py``'s ``agent``)
parses real observations and skips cleanly when the engine is absent, matching
the other agent tests.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from agents.harness import Candidate, DecisionHarness, HarnessAgent, HarnessStats

from .test_ppo_agent import tiny_policy


def test_submission_uses_calibrated_ppo_temperature():
    """The SOT-1701 promotion setting must not drift from the entry point."""
    import main

    assert main.PPO_TEMPERATURE == 0.25
    assert main._agent._temperature == main.PPO_TEMPERATURE


def mk_select(n_options: int = 3, *, min_count: int = 1, max_count: int = 1):
    """A minimal engine-shaped selection (the attributes the harness reads)."""
    return SimpleNamespace(
        type=0,
        context=0,
        minCount=min_count,
        maxCount=max_count,
        option=[{"type": 14} for _ in range(n_options)],
    )


def mk_obs_dict(n_options: int = 3) -> dict:
    return {"select": {"option": [{"type": 14}] * n_options}, "current": {}, "logs": []}


# --------------------------------------------------------------------------- #
# pipeline core — engine-free
# --------------------------------------------------------------------------- #
def test_decide_returns_legal_action_and_records_stats():
    harness = DecisionHarness(tiny_policy(), None, random.Random(0))
    select = mk_select(3)
    action = harness.decide(mk_obs_dict(3), None, select)
    assert isinstance(action, list) and len(action) == 1
    assert action[0] in (0, 1, 2)
    assert harness.stats.decisions == 1
    assert sum(harness.stats.decided_by.values()) == 1
    # sample + fallback always generated; single-select adds ranked alternatives
    assert harness.stats.candidates >= 2
    assert all(c.valid is not None for c in harness.last_candidates)


def test_decide_multiselect_respects_count_contract():
    harness = DecisionHarness(tiny_policy(), None, random.Random(1))
    select = mk_select(4, min_count=2, max_count=3)
    action = harness.decide(mk_obs_dict(4), None, select)
    assert 2 <= len(action) <= 3
    assert len(set(action)) == len(action)
    assert all(0 <= i < 4 for i in action)


def test_decide_reproducible_with_seed():
    a1 = DecisionHarness(tiny_policy(), None, random.Random(7)).decide(
        mk_obs_dict(5), None, mk_select(5)
    )
    a2 = DecisionHarness(tiny_policy(), None, random.Random(7)).decide(
        mk_obs_dict(5), None, mk_select(5)
    )
    assert a1 == a2


def test_validate_drops_malformed_candidates_and_choose_falls_through():
    select = mk_select(2)
    candidates = [
        Candidate(action=[99], source="policy_sample"),   # out of range
        Candidate(action=[0, 0], source="policy_argmax"),  # duplicate
        Candidate(action=[1], source="fallback"),
    ]
    DecisionHarness._validate(candidates, select)
    assert candidates[0].valid is False and candidates[0].reject_reason
    assert candidates[1].valid is False
    assert candidates[2].valid is True
    chosen = DecisionHarness._choose(candidates)
    assert chosen is candidates[2]  # only the legal fallback survived


def test_choose_prefers_mcts_then_sample_then_best_alternative():
    mcts = Candidate(action=[1], source="mcts", valid=True)
    sample = Candidate(action=[0], source="policy_sample", valid=True)
    top = Candidate(action=[2], source="policy_top", valid=True, total=-0.1)
    top2 = Candidate(action=[3], source="policy_top", valid=True, total=-2.0)
    fallback = Candidate(action=[4], source="fallback", valid=True)
    assert DecisionHarness._choose([mcts, sample, top, fallback]) is mcts
    assert DecisionHarness._choose([sample, top, fallback]) is sample
    assert DecisionHarness._choose([top2, top, fallback]) is top  # best-scored backup
    assert DecisionHarness._choose([fallback]) is fallback
    assert DecisionHarness._choose([]) is None


def test_choose_skips_invalid_higher_priority_sources():
    bad_mcts = Candidate(action=[99], source="mcts", valid=False)
    sample = Candidate(action=[0], source="policy_sample", valid=True)
    assert DecisionHarness._choose([bad_mcts, sample]) is sample


def test_scores_are_recorded_per_valid_candidate():
    harness = DecisionHarness(tiny_policy(), None, random.Random(3))
    harness.decide(mk_obs_dict(4), None, mk_select(4))
    valid = [c for c in harness.last_candidates if c.valid]
    assert valid
    for c in valid:
        assert {"policy_logp", "coverage", "mcts", "acts"} <= set(c.scores)
        # every scored index is in range, so coverage is full here
        assert c.scores["coverage"] == 1.0


def test_stats_merge_accumulates():
    a, b = HarnessStats(), HarnessStats()
    a.record("policy_sample", [Candidate([0], "policy_sample", valid=True)])
    b.record("mcts", [Candidate([1], "mcts", valid=True), Candidate([9], "fallback", valid=False)])
    a.merge(b)
    assert a.decisions == 2
    assert a.candidates == 3
    assert a.invalid_candidates == 1
    assert a.decided_by == {"policy_sample": 1, "mcts": 1}


# --------------------------------------------------------------------------- #
# HarnessAgent end-to-end (needs cg.api dataclasses to parse real observations)
# --------------------------------------------------------------------------- #
def _engine_obs(n_options: int = 3, *, select=True) -> dict:
    def mk_pokemon(card_id):
        return {"id": card_id, "serial": card_id, "playerIndex": 0, "hp": 100, "maxHp": 100,
                "appearThisTurn": False, "energies": [], "energyCards": [], "tools": [],
                "preEvolution": []}

    def mk_player(card_id):
        return {"active": [mk_pokemon(card_id)], "bench": [], "benchMax": 5, "deckCount": 40,
                "discard": [], "prize": [None] * 6, "handCount": 5, "hand": None,
                "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
                "confused": False}

    sel = {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
           "remainDamageCounter": 0, "remainEnergyCost": 0,
           "option": [{"type": 14}] * n_options, "deck": None, "contextCard": None,
           "effect": None} if select else None
    return {"select": sel, "logs": [],
            "current": {"turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                        "supporterPlayed": False, "stadiumPlayed": False,
                        "energyAttached": False, "retreated": False, "result": -1,
                        "stadium": [], "looking": None,
                        "players": [mk_player(1), mk_player(2)]},
            "search_begin_input": None}


def test_harness_agent_always_legal_across_seeds():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    for seed in range(8):
        agent = HarnessAgent(seed=seed, policy=tiny_policy(), mcts=False)
        action = agent.act(_engine_obs(3))
        assert isinstance(action, list) and len(action) == 1
        assert action[0] in (0, 1, 2)
    assert agent.harness_stats.decisions == 1


def test_harness_agent_mcts_wiring_fails_closed_without_session():
    """mcts=True but no search_begin_input: eligibility gate declines, policy plays."""
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    agent = HarnessAgent(seed=0, policy=tiny_policy(), mcts=True)
    assert agent.mcts_stats is not None
    action = agent.act(_engine_obs(4))
    assert isinstance(action, list) and len(action) == 1 and 0 <= action[0] < 4
    assert agent.mcts_stats.activations == 0
    assert agent.harness_stats.decided_by.get("policy_sample", 0) == 1


def test_harness_agent_defers_without_policy_artifact():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    agent = HarnessAgent(seed=0, policy_path="/nonexistent/policy.json")
    assert not agent.policy_loaded
    action = agent.act(_engine_obs(3))
    assert isinstance(action, list) and len(action) == 1 and 0 <= action[0] < 3
    assert agent.unsupported_rate() == 1.0  # SafeAgent legal-random fallback path


# --------------------------------------------------------------------------- #
# main.py — the Kaggle submission entry point
# --------------------------------------------------------------------------- #
def test_main_returns_committed_deck_on_initial_selection():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    import main

    deck = main.agent(_engine_obs(select=False))
    assert len(deck) == 60
    assert all(isinstance(c, int) for c in deck)
    with open(main._DECK_PATH) as fh:
        committed = [int(line) for line in fh.read().split("\n")[:60]]
    assert deck == committed


def test_main_agent_is_harness_configured_and_legal():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    import main

    assert isinstance(main._agent, HarnessAgent)
    assert main._agent.policy_loaded  # the committed data/policy.json loads
    assert main._agent._mcts is not None  # MCTS reinforcement is wired in
    action = main.agent(_engine_obs(5))
    assert isinstance(action, list) and len(action) == 1
    assert 0 <= action[0] < 5
