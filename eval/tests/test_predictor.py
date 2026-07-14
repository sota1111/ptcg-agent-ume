"""Tests for the hidden-information predictor (SOT-1650 R5, rehomed by SOT-1691).

Migrated from ``test_search_agent.py`` when the superseded SearchAgent was
removed: :class:`agents.predictor.UniformDeckPredictor` lives on as the
determinized MCTS' hidden-state sampler. Parsing the observation needs the
engine's ``cg.api`` dataclasses (gitignored / absent in CI) — hence the
importorskip, matching the other agent tests.
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from cg.api import to_observation_class  # noqa: E402

from agents.predictor import UniformDeckPredictor  # noqa: E402

END = 14


def mk_pokemon(card_id, hp=100, max_hp=100):
    return {"id": card_id, "serial": card_id, "playerIndex": 0, "hp": hp, "maxHp": max_hp,
            "appearThisTurn": False, "energies": [], "energyCards": [], "tools": [],
            "preEvolution": []}


def mk_player(active=None, *, deck_count=40, prize=6, hand_count=5, discard=None):
    return {"active": [active] if active is not None else [], "bench": [],
            "benchMax": 5, "deckCount": deck_count, "discard": discard or [],
            "prize": [None] * prize, "handCount": hand_count, "hand": None,
            "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
            "confused": False}


def mk_obs(me, opp, your_index=0):
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {"select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                       "remainDamageCounter": 0, "remainEnergyCost": 0,
                       "option": [{"type": END}], "deck": None, "contextCard": None,
                       "effect": None},
            "logs": [],
            "current": {"turn": 3, "turnActionCount": 0, "yourIndex": your_index,
                        "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
                        "energyAttached": False, "retreated": False, "result": -1,
                        "stadium": [], "looking": None, "players": players},
            "search_begin_input": None}


def test_predictor_returns_exact_counts():
    deck = list(range(1, 61))  # 60 distinct ids
    me = mk_player(mk_pokemon(1), deck_count=40, prize=6, hand_count=5)
    opp = mk_player(mk_pokemon(2), deck_count=38, prize=6, hand_count=4)
    parsed = to_observation_class(mk_obs(me, opp))
    (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active) = \
        UniformDeckPredictor(deck, random.Random(0)).predict(parsed, 0)
    assert len(your_deck) == 40
    assert len(your_prize) == 6
    assert len(opp_deck) == 38
    assert len(opp_prize) == 6
    assert len(opp_hand) == 4
    assert opp_active == []  # opponent Active is face-up (not None) → no prediction


def test_predictor_excludes_visible_cards_from_hidden_pool():
    """Our visible Active card is removed from the deck multiset exactly once."""
    deck = [1] * 2 + list(range(2, 60))  # two copies of card 1, one is our Active
    me = mk_player(mk_pokemon(1), deck_count=59, prize=0, hand_count=0)
    opp = mk_player(mk_pokemon(2), deck_count=40, prize=6, hand_count=4)
    parsed = to_observation_class(mk_obs(me, opp))
    your_deck, *_ = UniformDeckPredictor(deck, random.Random(1)).predict(parsed, 0)
    assert len(your_deck) == 59
    assert your_deck.count(1) == 1  # one copy visible on board → one left hidden


def test_predictor_predicts_facedown_opponent_active():
    deck = list(range(1, 61))
    me = mk_player(mk_pokemon(1))
    opp = mk_player(None, deck_count=38, prize=6, hand_count=4)
    opp["active"] = [None]  # face-down opponent Active
    parsed = to_observation_class(mk_obs(me, opp))
    *_, opp_active = UniformDeckPredictor(deck, random.Random(2)).predict(parsed, 0)
    assert len(opp_active) == 1
