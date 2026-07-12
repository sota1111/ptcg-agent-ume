"""Expected-action fixtures for the RuleAgent MAIN-turn policy (SOT-1647, R2).

These pin the *decision logic* — the pure scoring in :mod:`agents.rule_scoring` — on
hand-built MAIN selections with a small injected :class:`~agents.rule_scoring.CardIndex`
and **no live battle**. They cover every category and the priority ordering
(受け入れ条件①: 固定fixtureの期待行動が全て通る), plus the stable tie-break.

The scorer is engine-*call* free, but reading the observation still needs the engine's
``cg.api`` dataclasses/enums, which are gitignored/absent in CI — hence the importorskip.
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

from agents.rule_scoring import (  # noqa: E402
    CardIndex,
    OptionCategory,
    estimate_attack,
    pick_best_option,
    score_main_options,
)

# OptionType / AreaType raw ints (see cg.api).
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
ACTIVE, BENCH, HAND = 4, 5, 2
WATER = int(EnergyType.WATER)
FIRE = int(EnergyType.FIRE)


# --------------------------------------------------------------------------- #
# fixture builders — a MAIN observation the engine would emit, built by hand
# --------------------------------------------------------------------------- #
def mk_card(card_id, *, energy_type=WATER, weakness=None, hp=100, card_type=0):
    return CardData(
        cardId=card_id, name=f"card{card_id}", cardType=card_type, retreatCost=1, hp=hp,
        weakness=weakness, resistance=None, energyType=energy_type, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False, aceSpec=False,
        evolvesFrom=None, skills=[], attacks=[],
    )


def mk_attack(attack_id, damage, energies=None):
    return Attack(attackId=attack_id, name=f"atk{attack_id}", text="", damage=damage,
                  energies=energies or [WATER])


def mk_pokemon(card_id, hp, max_hp, energies=None):
    return {
        "id": card_id, "serial": card_id, "playerIndex": 0, "hp": hp, "maxHp": max_hp,
        "appearThisTurn": False, "energies": energies or [], "energyCards": [],
        "tools": [], "preEvolution": [],
    }


def mk_player(active=None, bench=None):
    return {
        "active": [active] if active is not None else [], "bench": bench or [],
        "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * 6,
        "handCount": 5, "hand": None, "poisoned": False, "burned": False,
        "asleep": False, "paralyzed": False, "confused": False,
    }


def mk_state(me, opp, your_index=0):
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {
        "turn": 3, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
        "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
        "retreated": False, "result": -1, "stadium": [], "looking": None,
        "players": players,
    }


def mk_select(options):
    return {
        "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
        "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
        "deck": None, "contextCard": None, "effect": None,
    }


def opt(type_, **kw):
    return {"type": type_, **kw}


def build(options, *, me=None, opp=None):
    """A parsed (Observation, SelectData) for a MAIN selection with these options."""
    me = me if me is not None else mk_player(mk_pokemon(1, 100, 100))
    opp = opp if opp is not None else mk_player(mk_pokemon(2, 100, 100))
    obs = {"select": mk_select(options), "logs": [], "current": mk_state(me, opp),
           "search_begin_input": None}
    parsed = to_observation_class(obs)
    return parsed, parsed.select


# A card index: my attacker is Water; the opponent card is Water-weak in KO fixtures.
def cards(opp_weakness=None):
    return CardIndex.from_engine(
        [mk_card(1, energy_type=WATER), mk_card(2, energy_type=FIRE, weakness=opp_weakness)],
        [mk_attack(10, 20), mk_attack(20, 60), mk_attack(30, 0)],  # small / big / scaling
    )


def _best(options, *, me=None, opp=None, opp_weakness=None, seed=0):
    parsed, sel = build(options, me=me, opp=opp)
    scored = score_main_options(parsed, sel, cards(opp_weakness))
    return pick_best_option(scored, random.Random(seed)), scored


# --------------------------------------------------------------------------- #
# estimate_attack — damage / weakness-doubling / KO
# --------------------------------------------------------------------------- #
def test_estimate_attack_plain_and_ko():
    atk = mk_attack(10, 20)
    my, opp_card = mk_card(1, energy_type=WATER), mk_card(2, energy_type=FIRE, weakness=None)
    assert estimate_attack(atk, my, 100, opp_card) == (20.0, False)
    assert estimate_attack(atk, my, 20, opp_card) == (20.0, True)   # exactly lethal


def test_estimate_attack_weakness_doubles_and_can_ko():
    atk = mk_attack(10, 20)
    my = mk_card(1, energy_type=WATER)
    weak = mk_card(2, energy_type=FIRE, weakness=WATER)  # weak to our Water attacker
    assert estimate_attack(atk, my, 100, weak) == (40.0, False)
    assert estimate_attack(atk, my, 40, weak) == (40.0, True)      # doubled to lethal


def test_estimate_attack_zero_damage_never_ko():
    atk = mk_attack(30, 0)  # scaling attack: static damage 0
    my = mk_card(1, energy_type=WATER)
    weak = mk_card(2, energy_type=FIRE, weakness=WATER)
    assert estimate_attack(atk, my, 1, weak) == (0.0, False)


# --------------------------------------------------------------------------- #
# Priority ordering of the categories
# --------------------------------------------------------------------------- #
def test_ko_attack_taken_immediately_over_setup():
    # KO available: attack now, even though an energy attach is also offered.
    opp = mk_player(mk_pokemon(2, 50, 90))  # 50 HP <= 60 dmg -> KO
    idx, scored = _best(
        [opt(ATTACH, area=HAND, index=0, inPlayArea=ACTIVE, inPlayIndex=0),
         opt(ATTACK, attackId=20),
         opt(END)],
        opp=opp,
    )
    assert idx == 1
    assert scored[1].category == OptionCategory.WINNING_ATTACK


def test_setup_precedes_non_ko_attack():
    # No KO (100 HP vs 20 dmg): attach energy to the active first, attack later.
    idx, scored = _best(
        [opt(ATTACH, area=HAND, index=0, inPlayArea=ACTIVE, inPlayIndex=0),
         opt(ATTACK, attackId=10),
         opt(END)],
    )
    assert idx == 0
    assert scored[0].category == OptionCategory.ENERGY_ACTIVE


def test_attack_preferred_over_ending_when_no_setup():
    idx, scored = _best([opt(ATTACK, attackId=10), opt(END)])
    assert idx == 0
    assert scored[0].category == OptionCategory.ATTACK


def test_end_is_last_resort():
    idx, _ = _best([opt(END)])
    assert idx == 0


def test_pick_ko_attack_among_several_attacks():
    opp = mk_player(mk_pokemon(2, 50, 90))
    idx, scored = _best(
        [opt(ATTACK, attackId=10),   # 20 dmg, non-lethal
         opt(ATTACK, attackId=20),   # 60 dmg -> KO
         opt(END)],
        opp=opp,
    )
    assert idx == 1
    assert scored[1].category == OptionCategory.WINNING_ATTACK


def test_weakness_turns_small_attack_into_ko():
    opp = mk_player(mk_pokemon(2, 40, 90))  # 40 HP, weak to Water -> 20*2 = 40 KO
    idx, scored = _best(
        [opt(ATTACK, attackId=10), opt(END)],
        opp=opp, opp_weakness=WATER,
    )
    assert idx == 0
    assert scored[0].category == OptionCategory.WINNING_ATTACK


def test_develop_beats_bench_energy_and_non_ko_attack():
    idx, scored = _best(
        [opt(PLAY, index=0),
         opt(ATTACH, area=HAND, index=1, inPlayArea=BENCH, inPlayIndex=0),
         opt(ATTACK, attackId=10),
         opt(END)],
    )
    assert idx == 0
    assert scored[0].category == OptionCategory.DEVELOP


def test_evolve_beats_develop():
    idx, scored = _best([opt(PLAY, index=0), opt(EVOLVE, index=1), opt(END)])
    assert idx == 1
    assert scored[1].category == OptionCategory.EVOLVE


def test_ability_and_discard_are_not_used_proactively():
    # Ability/discard score below END, so the policy ends the turn rather than fire them.
    idx, scored = _best([opt(ABILITY, area=HAND, index=0), opt(DISCARD, area=ACTIVE, index=0),
                         opt(END)])
    assert idx == 2
    assert scored[2].category == OptionCategory.END
    assert scored[0].category == OptionCategory.AVOID


# --------------------------------------------------------------------------- #
# きぜつ回避/交代 — a guarded retreat
# --------------------------------------------------------------------------- #
def test_retreat_when_active_hurt_and_bench_available():
    me = mk_player(mk_pokemon(1, 20, 100), bench=[mk_pokemon(3, 100, 100)])  # 20/100 hurt
    idx, scored = _best([opt(RETREAT), opt(END)], me=me)
    assert idx == 0
    assert scored[0].category == OptionCategory.SWITCH


def test_no_retreat_when_active_healthy():
    me = mk_player(mk_pokemon(1, 100, 100), bench=[mk_pokemon(3, 100, 100)])
    idx, scored = _best([opt(RETREAT), opt(END)], me=me)
    assert idx == 1  # retreat not beneficial -> end
    assert scored[0].category == OptionCategory.AVOID


def test_no_retreat_without_bench():
    me = mk_player(mk_pokemon(1, 20, 100), bench=[])  # hurt but nowhere to retreat to
    idx, scored = _best([opt(RETREAT), opt(END)], me=me)
    assert idx == 1
    assert scored[0].category == OptionCategory.AVOID


def test_attack_preferred_over_retreat():
    # Even when hurt, a KO this turn beats retreating.
    me = mk_player(mk_pokemon(1, 20, 100), bench=[mk_pokemon(3, 100, 100)])
    opp = mk_player(mk_pokemon(2, 50, 90))
    idx, scored = _best([opt(RETREAT), opt(ATTACK, attackId=20), opt(END)], me=me, opp=opp)
    assert idx == 1
    assert scored[1].category == OptionCategory.WINNING_ATTACK


# --------------------------------------------------------------------------- #
# tie-break: 同点のみ安定した乱択
# --------------------------------------------------------------------------- #
def test_ties_broken_stably_within_top_set():
    # Two interchangeable active-energy attaches tie at the top.
    options = [
        opt(ATTACH, area=HAND, index=0, inPlayArea=ACTIVE, inPlayIndex=0),
        opt(ATTACH, area=HAND, index=1, inPlayArea=ACTIVE, inPlayIndex=0),
        opt(END),
    ]
    parsed, sel = build(options)
    scored = score_main_options(parsed, sel, cards())
    top = {0, 1}
    # Always one of the tied top options, and stable for a fixed RNG state.
    for seed in range(5):
        assert pick_best_option(scored, random.Random(seed)) in top
    assert (pick_best_option(scored, random.Random(0))
            == pick_best_option(scored, random.Random(0)))


def test_score_main_options_defers_without_state():
    # No board (current is None) -> nothing to reason about -> empty (caller defers).
    obs = {"select": mk_select([opt(END)]), "logs": [], "current": None,
           "search_begin_input": None}
    parsed = to_observation_class(obs)
    assert score_main_options(parsed, parsed.select, cards()) == []
    assert pick_best_option([], random.Random(0)) is None
