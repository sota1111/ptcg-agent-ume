"""Tests for the one-ply SearchAgent (SOT-1650, R5).

Two things are pinned without a live battle:

* the pure **perspective evaluation** (:func:`agents.search_agent.evaluate_state`) and
  the **hidden-info predictor** (:class:`~agents.search_agent.UniformDeckPredictor`);
* the **fail-closed contract** — on a hand-built MAIN selection there is no real engine
  search session (``search_begin_input is None`` ⇒ ``search_begin`` rejects it), so the
  agent must fall back to the inherited RuleAgent policy, return the *same* legal move,
  never leak a search session, and never crash or emit an illegal action.

Reading the observation needs the engine's ``cg.api`` dataclasses/enums (gitignored /
absent in CI) — hence the importorskip, matching the other agent tests.
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from cg.api import (  # noqa: E402
    Attack,
    CardData,
    EnergyType,
    to_observation_class,
)

from agents.rule_agent import RuleAgent  # noqa: E402
from agents.rule_scoring import CardIndex  # noqa: E402
from agents.search_agent import (  # noqa: E402
    SearchAgent,
    UniformDeckPredictor,
    evaluate_state,
)

PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
ACTIVE, BENCH, HAND = 4, 5, 2
WATER, FIRE = int(EnergyType.WATER), int(EnergyType.FIRE)


# --------------------------------------------------------------------------- #
# fixture builders (mirrors test_rule_policy.py; kept local for self-containment)
# --------------------------------------------------------------------------- #
def mk_card(card_id, *, energy_type=WATER, weakness=None, hp=100, attacks=None):
    return CardData(
        cardId=card_id, name=f"card{card_id}", cardType=0, retreatCost=1, hp=hp,
        weakness=weakness, resistance=None, energyType=energy_type, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False, aceSpec=False,
        evolvesFrom=None, skills=[], attacks=attacks or [],
    )


def mk_attack(attack_id, damage, energies=None):
    return Attack(attackId=attack_id, name=f"atk{attack_id}", text="", damage=damage,
                  energies=energies or [WATER])


def mk_pokemon(card_id, hp, max_hp):
    return {"id": card_id, "serial": card_id, "playerIndex": 0, "hp": hp, "maxHp": max_hp,
            "appearThisTurn": False, "energies": [], "energyCards": [], "tools": [],
            "preEvolution": []}


def mk_player(active=None, bench=None, *, deck_count=40, prize=6, hand_count=5, discard=None):
    return {"active": [active] if active is not None else [], "bench": bench or [],
            "benchMax": 5, "deckCount": deck_count, "discard": discard or [],
            "prize": [None] * prize, "handCount": hand_count, "hand": None,
            "poisoned": False, "burned": False, "asleep": False, "paralyzed": False,
            "confused": False}


def mk_state(me, opp, your_index=0, result=-1):
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {"turn": 3, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
            "retreated": False, "result": result, "stadium": [], "looking": None,
            "players": players}


def mk_select(options, *, min_count=1, max_count=1, type_=0, context=0):
    return {"type": type_, "context": context, "minCount": min_count, "maxCount": max_count,
            "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
            "deck": None, "contextCard": None, "effect": None}


def opt(type_, **kw):
    return {"type": type_, **kw}


def mk_obs(options, *, me=None, opp=None, your_index=0, result=-1, sbi=None, **sel_kw):
    me = me if me is not None else mk_player(mk_pokemon(1, 100, 100))
    opp = opp if opp is not None else mk_player(mk_pokemon(2, 100, 100))
    return {"select": mk_select(options, **sel_kw), "logs": [],
            "current": mk_state(me, opp, your_index, result), "search_begin_input": sbi}


def cards(opp_weakness=None):
    return CardIndex.from_engine(
        [mk_card(1, energy_type=WATER, attacks=[10, 20]),
         mk_card(2, energy_type=FIRE, weakness=opp_weakness)],
        [mk_attack(10, 20), mk_attack(20, 60)],
    )


# --------------------------------------------------------------------------- #
# evaluate_state — pure perspective score
# --------------------------------------------------------------------------- #
def test_evaluate_terminal_win_loss_draw():
    parsed_win = to_observation_class(mk_obs([opt(END)], result=0, your_index=0))
    parsed_loss = to_observation_class(mk_obs([opt(END)], result=1, your_index=0))
    parsed_draw = to_observation_class(mk_obs([opt(END)], result=2, your_index=0))
    ci = cards()
    assert evaluate_state(parsed_win, 0, ci) > 0
    assert evaluate_state(parsed_loss, 0, ci) < 0
    assert evaluate_state(parsed_draw, 0, ci) == 0.0
    # From the opponent's perspective the same win flips sign.
    assert evaluate_state(parsed_win, 1, ci) < 0


def test_evaluate_prefers_fewer_own_prizes_remaining():
    """Taking a prize (fewer remaining for us) strictly raises our score."""
    ci = cards()
    ahead = mk_obs([opt(END)], me=mk_player(mk_pokemon(1, 100, 100), prize=3),
                   opp=mk_player(mk_pokemon(2, 100, 100), prize=6))
    behind = mk_obs([opt(END)], me=mk_player(mk_pokemon(1, 100, 100), prize=6),
                    opp=mk_player(mk_pokemon(2, 100, 100), prize=3))
    assert evaluate_state(to_observation_class(ahead), 0, ci) > \
        evaluate_state(to_observation_class(behind), 0, ci)


def test_evaluate_offense_rewards_ko_threat():
    """A position where our Active threatens a KO scores above a non-threatening one."""
    ci = cards(opp_weakness=WATER)  # Water attacker doubles on a Water-weak Fire target
    lethal = mk_obs([opt(END)], me=mk_player(mk_pokemon(1, 100, 100)),
                    opp=mk_player(mk_pokemon(2, 30, 100)))   # 60*2 >= 30 → KO
    healthy = mk_obs([opt(END)], me=mk_player(mk_pokemon(1, 100, 100)),
                     opp=mk_player(mk_pokemon(2, 100, 100)))
    assert evaluate_state(to_observation_class(lethal), 0, ci) > \
        evaluate_state(to_observation_class(healthy), 0, ci)


def test_evaluate_never_raises_on_malformed():
    class Bad:
        current = object()  # no .result / .players
    assert evaluate_state(Bad(), 0, cards()) == 0.0


# --------------------------------------------------------------------------- #
# UniformDeckPredictor — hidden-zone counts
# --------------------------------------------------------------------------- #
def test_predictor_returns_exact_counts():
    deck = list(range(1, 61))  # 60 distinct ids
    me = mk_player(mk_pokemon(1, 100, 100), deck_count=40, prize=6, hand_count=5)
    opp = mk_player(mk_pokemon(2, 100, 100), deck_count=38, prize=6, hand_count=4)
    parsed = to_observation_class(mk_obs([opt(END)], me=me, opp=opp))
    (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active) = \
        UniformDeckPredictor(deck, random.Random(0)).predict(parsed, 0)
    assert len(your_deck) == 40
    assert len(your_prize) == 6
    assert len(opp_deck) == 38
    assert len(opp_prize) == 6
    assert len(opp_hand) == 4
    assert opp_active == []  # opponent Active is face-up (not None) → no prediction


# --------------------------------------------------------------------------- #
# SearchAgent — fail-closed to the rule policy, always legal, no leak/crash
# --------------------------------------------------------------------------- #
def _main_options():
    # ATTACH-to-active (energy_active) should beat END under the rule policy.
    return [opt(ATTACH, inPlayArea=ACTIVE), opt(END)]


def test_search_falls_back_to_rule_and_matches_when_no_engine_session():
    """No search_begin_input ⇒ search rejected ⇒ identical to RuleAgent, no leak."""
    obs = mk_obs(_main_options())
    rule_move = RuleAgent(seed=1).act(dict(obs))
    agent = SearchAgent(seed=1, deck_path="deck.csv")
    search_move = agent.act(dict(obs))
    assert search_move == rule_move          # fell back to the rule tactic
    assert search_move == [0]                # ATTACH-active over END
    assert agent.search_stats["fallbacks"] >= 1
    assert agent.search_stats["chosen"] == 0
    assert agent.search_stats["leaks"] == 0  # every search session torn down cleanly


def test_search_multiselect_main_defers_to_rule():
    """A multi-select MAIN is never one-ply searched; it defers to the rule policy."""
    obs = mk_obs(_main_options(), min_count=1, max_count=2)
    agent = SearchAgent(seed=2)
    action = agent.act(dict(obs))
    assert isinstance(action, list)
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in action)
    assert 1 <= len(action) <= 2
    assert all(0 <= i < 2 for i in action)


def test_search_agent_output_always_legal_across_seeds():
    """Whatever happens inside, act() returns a validated legal selection, never raises."""
    obs = mk_obs(_main_options())
    for seed in range(8):
        action = SearchAgent(seed=seed).act(dict(obs))
        assert isinstance(action, list) and len(action) == 1
        assert action[0] in (0, 1)


def test_search_agent_is_rule_agent_subclass():
    """受け入れ条件③: SearchAgent is an independent module built ON RuleAgent, not a fork."""
    assert issubclass(SearchAgent, RuleAgent)
    a = SearchAgent(seed=0)
    assert a.name == "search"
    # Non-MAIN behaviour is inherited unchanged (a distinct SelectContext defers to
    # the RuleAgent setup handlers / SafeAgent fallback — never crashes).
    non_main = mk_obs([opt(PLAY)], type_=0, context=99)  # unknown context
    assert isinstance(a.act(dict(non_main)), list)
