"""Fixtures for the RuleAgent setup / forced-selection context handlers (SOT-1648, R3).

These pin the *decision logic* — the pure tactics in :mod:`agents.setup_scoring` — on
hand-built selections with a small injected :class:`~agents.rule_scoring.CardIndex` and
**no live battle**, mirroring R2's :mod:`test_rule_policy`. They cover:

* every setup / forced-selection context named in the Issue — initial active/bench, go
  first, mulligan, promote-after-KO / switch, attach, discard, search, draw-count, prize,
  special condition — each routed to its dedicated tactic (受け入れ条件①);
* the combination- and future-resource-aware multi-selection helpers (受け入れ条件③);
* that **every** known non-MAIN ``SelectContext`` is mapped to a policy and that every
  policy emits a *legal* action for the engine's ``select`` (受け入れ条件②: 違法出力0).

The tactics are engine-*call* free, but reading the observation still needs the engine's
``cg.api`` dataclasses/enums (gitignored/absent in CI) — hence the importorskip.
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from cg.api import (  # noqa: E402
    Attack,
    CardData,
    EnergyType,
    SelectContext,
    SelectType,
    to_observation_class,
)

from agents.protocol import validate_selection  # noqa: E402
from agents.rule_scoring import CardIndex  # noqa: E402
from agents.setup_scoring import (  # noqa: E402
    CONTEXT_POLICIES,
    board_pokemon_value,
    card_pokemon_value,
    select_beneficial_combo,
    select_cost_combo,
    select_for_context,
    select_max_count,
    select_neutral_min,
)

# OptionType / AreaType raw ints (see cg.api).
CARD, ENERGY, PLAY, ATTACH, RETREAT, END = 3, 6, 7, 8, 12, 14
YES, NO, NUMBER = 1, 2, 0
ACTIVE, BENCH, HAND = 4, 5, 2
WATER, FIRE = int(EnergyType.WATER), int(EnergyType.FIRE)


# --------------------------------------------------------------------------- #
# fixture builders — a setup/forced observation the engine would emit, by hand
# --------------------------------------------------------------------------- #
def mk_card(card_id, *, energy_type=WATER, weakness=None, hp=100, retreat=1, attacks=None):
    return CardData(
        cardId=card_id, name=f"card{card_id}", cardType=0, retreatCost=retreat, hp=hp,
        weakness=weakness, resistance=None, energyType=energy_type, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False, aceSpec=False,
        evolvesFrom=None, skills=[], attacks=attacks or [],
    )


def mk_attack(attack_id, damage):
    return Attack(attackId=attack_id, name=f"atk{attack_id}", text="", damage=damage,
                  energies=[WATER])


def mk_pokemon(card_id, hp, max_hp):
    return {"id": card_id, "serial": card_id, "hp": hp, "maxHp": max_hp,
            "appearThisTurn": False, "energies": [], "energyCards": [], "tools": [],
            "preEvolution": []}


def mk_hand(card_ids):
    return [{"id": cid, "serial": 100 + i, "playerIndex": 0} for i, cid in enumerate(card_ids)]


def mk_player(active=None, bench=None, hand=None):
    return {"active": [active] if active is not None else [], "bench": bench or [],
            "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * 6,
            "handCount": len(hand or []), "hand": hand, "poisoned": False, "burned": False,
            "asleep": False, "paralyzed": False, "confused": False}


def mk_state(me, opp):
    players = [me, opp]
    return {"turn": 1, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
            "retreated": False, "result": -1, "stadium": [], "looking": None,
            "players": players}


def opt(type_, **kw):
    return {"type": type_, **kw}


def build(context, options, *, mn=1, mx=1, me=None, opp=None):
    me = me if me is not None else mk_player()
    opp = opp if opp is not None else mk_player(mk_pokemon(2, 100, 100))
    select = {"type": 0, "context": int(context), "minCount": mn, "maxCount": mx,
              "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
              "deck": None, "contextCard": None, "effect": None}
    obs = {"select": select, "logs": [], "current": mk_state(me, opp),
           "search_begin_input": None}
    parsed = to_observation_class(obs)
    return parsed, parsed.select


# A card index: card 1 is a strong attacker, card 2 a weak one, card 3 mid.
def cards():
    return CardIndex.from_engine(
        [mk_card(1, hp=120, attacks=[10]), mk_card(2, hp=60, attacks=[11]),
         mk_card(3, hp=90, attacks=[12])],
        [mk_attack(10, 90), mk_attack(11, 10), mk_attack(12, 40)],
    )


def run(context, options, **kw):
    parsed, sel = build(context, options, **kw)
    return select_for_context(parsed, sel, cards(), random.Random(0)), sel


# --------------------------------------------------------------------------- #
# value functions
# --------------------------------------------------------------------------- #
def test_card_value_ranks_by_attack_then_hp():
    ci = cards()
    strong = card_pokemon_value(ci.card(1), ci)   # 90 dmg, 120 hp
    weak = card_pokemon_value(ci.card(2), ci)      # 10 dmg, 60 hp
    assert strong > weak


def test_board_value_prefers_healthy():
    from types import SimpleNamespace
    ci = cards()
    healthy = board_pokemon_value(SimpleNamespace(id=1, hp=120), ci)
    hurt = board_pokemon_value(SimpleNamespace(id=1, hp=20), ci)
    assert healthy > hurt


# --------------------------------------------------------------------------- #
# placement / promotion
# --------------------------------------------------------------------------- #
def test_setup_active_picks_strongest_hand_pokemon():
    me = mk_player(hand=mk_hand([2, 1, 3]))  # weak, strong, mid at hand indices 0,1,2
    chosen, _ = run(SelectContext.SETUP_ACTIVE_POKEMON,
                    [opt(CARD, area=HAND, index=0), opt(CARD, area=HAND, index=1),
                     opt(CARD, area=HAND, index=2)], me=me)
    assert chosen == [1]  # the strong attacker (card 1)


def test_setup_bench_develops_up_to_cap_best_first():
    me = mk_player(hand=mk_hand([2, 1, 3]))
    chosen, _ = run(SelectContext.SETUP_BENCH_POKEMON,
                    [opt(CARD, area=HAND, index=0), opt(CARD, area=HAND, index=1),
                     opt(CARD, area=HAND, index=2)], mn=0, mx=2, me=me)
    # A combination of the two strongest basics (cards 1 & 3 at option indices 1 & 2).
    assert chosen == [1, 2]


def test_promote_after_ko_picks_healthiest_attacker():
    me = mk_player(bench=[mk_pokemon(1, 20, 120), mk_pokemon(1, 120, 120)])
    chosen, _ = run(SelectContext.TO_ACTIVE,
                    [opt(CARD, area=BENCH, index=0), opt(CARD, area=BENCH, index=1)], me=me)
    assert chosen == [1]  # the full-HP copy


def test_forced_switch_uses_board_promotion():
    me = mk_player(bench=[mk_pokemon(2, 60, 60), mk_pokemon(1, 120, 120)])
    chosen, _ = run(SelectContext.SWITCH,
                    [opt(CARD, area=BENCH, index=0), opt(CARD, area=BENCH, index=1)], me=me)
    assert chosen == [1]  # stronger attacker with more HP


# --------------------------------------------------------------------------- #
# attach / resources / cost
# --------------------------------------------------------------------------- #
def test_attach_prefers_the_active():
    chosen, _ = run(SelectContext.ATTACH_TO,
                    [opt(CARD, area=BENCH, index=0, inPlayArea=BENCH),
                     opt(CARD, area=ACTIVE, inPlayArea=ACTIVE)], mn=0, mx=1)
    assert chosen == [1]  # attach to the active attacker


def test_search_to_hand_takes_maximum_offered():
    # The engine hides searched-card identities, so the cards are interchangeable; the
    # tactic takes the *maximum* count (the cap) — which 3 of the 5 is immaterial.
    chosen, sel = run(SelectContext.TO_HAND,
                      [opt(CARD, area=1, index=i) for i in range(5)], mn=0, mx=3)
    assert len(chosen) == 3  # take the cap (3) — maximise acquired resources
    assert validate_selection(chosen, sel) == chosen


def test_draw_count_takes_the_largest_number():
    chosen, _ = run(SelectContext.DRAW_COUNT,
                    [opt(NUMBER, number=0), opt(NUMBER, number=1),
                     opt(NUMBER, number=3), opt(NUMBER, number=2)])
    assert chosen == [2]  # the "draw 3" option


def test_discard_energy_spends_the_minimum():
    me = mk_player(active=mk_pokemon(1, 120, 120))
    chosen, sel = run(SelectContext.DISCARD_ENERGY,
                      [opt(ENERGY, area=ACTIVE, count=1),
                       opt(ENERGY, area=ACTIVE, energyIndex=1, count=1),
                       opt(ENERGY, area=ACTIVE, energyIndex=2, count=1)], mn=1, mx=1, me=me)
    assert len(chosen) == 1  # give up only what is required
    assert validate_selection(chosen, sel) == chosen


# --------------------------------------------------------------------------- #
# initiative / forced answers
# --------------------------------------------------------------------------- #
def test_go_first_takes_the_initiative():
    chosen, _ = run(SelectContext.IS_FIRST, [opt(YES), opt(NO)])
    assert chosen == [0]  # YES = go first


def test_mulligan_reveals():
    chosen, _ = run(SelectContext.MULLIGAN, [opt(YES), opt(NO)])
    assert chosen == [0]


def test_prize_setup_is_neutral_minimum():
    chosen, _ = run(SelectContext.TO_PRIZE,
                    [opt(CARD, area=1, index=i) for i in range(6)], mn=0, mx=6)
    assert chosen == []  # blind, min 0 -> take nothing extra


# --------------------------------------------------------------------------- #
# combination- & future-resource-aware multi-selection helpers (受け入れ条件③)
# --------------------------------------------------------------------------- #
def test_beneficial_combo_takes_top_set_and_drops_junk():
    rng = random.Random(0)
    # Values: idx0=5, idx1=3, idx2=0, idx3=-1. min1 max3 -> take the two positives only.
    scored = [(0, 5.0), (1, 3.0), (2, 0.0), (3, -1.0)]
    assert select_beneficial_combo(scored, 1, 3, rng) == [0, 1]


def test_beneficial_combo_respects_minimum_even_without_upside():
    rng = random.Random(0)
    scored = [(0, 0.0), (1, 0.0), (2, 0.0)]
    got = select_beneficial_combo(scored, 2, 3, rng)
    assert len(got) == 2  # forced to take the minimum


def test_cost_combo_gives_up_least_valuable_minimum():
    rng = random.Random(0)
    # Keep the strong cards (5,4); part with the two weakest (1,2) when min2.
    scored = [(0, 5.0), (1, 1.0), (2, 2.0), (3, 4.0)]
    assert select_cost_combo(scored, 2, rng) == [1, 2]


def test_cost_combo_minimum_zero_keeps_everything():
    assert select_cost_combo([(0, 1.0), (1, 2.0)], 0, random.Random(0)) == []


def test_max_count_breaks_ties_stably():
    class Sel:
        option = [type("O", (), {"number": 2})(), type("O", (), {"number": 2})()]
    a = select_max_count(Sel(), random.Random(0))
    b = select_max_count(Sel(), random.Random(0))
    assert a == b and a[0] in (0, 1)


# --------------------------------------------------------------------------- #
# coverage (受け入れ条件①) and legality (受け入れ条件②)
# --------------------------------------------------------------------------- #
def test_every_non_main_context_has_a_policy():
    """Every known SelectContext except MAIN maps to a dedicated tactic."""
    missing = [c.name for c in SelectContext
               if c is not SelectContext.MAIN and int(c) not in CONTEXT_POLICIES]
    assert missing == [], f"contexts with no handler: {missing}"
    assert int(SelectContext.MAIN) not in CONTEXT_POLICIES  # MAIN stays with R2


@pytest.mark.parametrize("context", [c for c in SelectContext if c is not SelectContext.MAIN])
def test_every_policy_emits_a_legal_action(context):
    """Every context tactic returns an action that passes the selection-count validator.

    A generic 4-option selection (min 1, max 2) is legal for all contexts here; the point
    is that no tactic ever produces an out-of-range / duplicate / wrong-count action.
    """
    me = mk_player(active=mk_pokemon(1, 120, 120),
                   bench=[mk_pokemon(3, 90, 90), mk_pokemon(2, 60, 60)],
                   hand=mk_hand([1, 2, 3, 1]))
    options = [opt(CARD, area=HAND, index=0), opt(CARD, area=BENCH, index=0),
               opt(CARD, area=ACTIVE), opt(CARD, area=HAND, index=1)]
    parsed, sel = build(context, options, mn=1, mx=2, me=me)
    action = select_for_context(parsed, sel, cards(), random.Random(1))
    assert action is not None, f"{context.name} produced no action"
    # Must be legal for the engine-supplied select (range + no dup + count in [1,2]).
    assert validate_selection(action, sel) == action


def test_unmapped_context_defers_to_fallback():
    """MAIN (owned by R2) and an out-of-enum context both defer (None)."""
    parsed, sel = build(SelectContext.MAIN, [opt(END)])
    assert select_for_context(parsed, sel, cards(), random.Random(0)) is None
    # An empty option list also defers regardless of context.
    parsed2, sel2 = build(SelectContext.TO_HAND, [])
    assert select_for_context(parsed2, sel2, cards(), random.Random(0)) is None


def test_neutral_min_helper_takes_minimum():
    _p, sel = build(SelectContext.LOOK, [opt(CARD, area=1, index=i) for i in range(4)],
                    mn=0, mx=4)
    assert select_neutral_min(sel, random.Random(0)) == []
