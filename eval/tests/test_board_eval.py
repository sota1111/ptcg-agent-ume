"""Pure unit tests for the R4 unified board evaluation ``score(state)`` (SOT-1649).

:func:`agents.board_eval.score_state` is a pure function of a parsed state, the seat index,
a static :class:`~agents.rule_scoring.CardIndex`, and a set of :class:`~agents.board_eval.EvalWeights`,
so it is pinned here on hand-built states with no live battle: symmetry (a mirrored board
scores 0), the prize-clock terminal short-circuit, each component's differential feature,
the weight config, and the ablation ``disabled`` / ``without`` paths.

Reading the observation still needs the engine's ``cg.api`` dataclasses (gitignored / absent
in CI) — hence the importorskip, matching the other agent tests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from cg.api import (  # noqa: E402
    Attack,
    CardData,
    EnergyType,
    to_observation_class,
)

from agents.board_eval import (  # noqa: E402
    COMPONENTS,
    DEFAULT_WEIGHTS,
    LOSS_SCORE,
    WIN_SCORE,
    EvalWeights,
    score_state,
)
from agents.rule_scoring import CardIndex  # noqa: E402

WATER, FIRE = int(EnergyType.WATER), int(EnergyType.FIRE)


# --------------------------------------------------------------------------- #
# fixture builders (mirror the other agent tests; kept local for self-containment)
# --------------------------------------------------------------------------- #
def mk_card(card_id, *, energy_type=WATER, weakness=None, hp=100, retreat_cost=1, attacks=None):
    return CardData(
        cardId=card_id, name=f"card{card_id}", cardType=0, retreatCost=retreat_cost, hp=hp,
        weakness=weakness, resistance=None, energyType=energy_type, basic=True,
        stage1=False, stage2=False, ex=False, megaEx=False, tera=False, aceSpec=False,
        evolvesFrom=None, skills=[], attacks=attacks or [],
    )


def mk_attack(attack_id, damage, energies=None):
    return Attack(attackId=attack_id, name=f"atk{attack_id}", text="", damage=damage,
                  energies=energies or [WATER])


def mk_pokemon(card_id, hp, max_hp, energies=None):
    return {"id": card_id, "serial": card_id, "playerIndex": 0, "hp": hp, "maxHp": max_hp,
            "appearThisTurn": False, "energies": energies or [], "energyCards": [], "tools": [],
            "preEvolution": []}


def mk_player(active=None, bench=None, *, prize=6, hand_count=5):
    return {"active": [active] if active is not None else [], "bench": bench or [],
            "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * prize,
            "handCount": hand_count, "hand": None, "poisoned": False, "burned": False,
            "asleep": False, "paralyzed": False, "confused": False}


def mk_state(me, opp, your_index=0):
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {"turn": 3, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
            "retreated": False, "result": -1, "stadium": [], "looking": None, "players": players}


def state_of(me, opp, your_index=0):
    """A parsed :class:`~cg.api.State` for a hand-built board."""
    obs = {"select": None, "logs": [], "current": mk_state(me, opp, your_index),
           "search_begin_input": None}
    return to_observation_class(obs).current


def cards(*, my_weak_target=False, retreat_cost=1):
    # card 1 = our Water attacker with a 60-dmg attack; card 2 = a Fire target, optionally
    # Water-weak so the attacker's printed 60 already lethals a low-HP defender.
    weakness = WATER if my_weak_target else None
    return CardIndex.from_engine(
        [mk_card(1, energy_type=WATER, retreat_cost=retreat_cost, attacks=[20]),
         mk_card(2, energy_type=FIRE, weakness=weakness, retreat_cost=retreat_cost)],
        [mk_attack(20, 60, energies=[WATER])],
    )


# --------------------------------------------------------------------------- #
# symmetry + terminal
# --------------------------------------------------------------------------- #
def test_mirrored_board_scores_zero():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100)), mk_player(mk_pokemon(1, 100, 100)))
    assert score_state(st, 0, ci).total == 0.0


def test_perspective_flips_sign():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=3),
                  mk_player(mk_pokemon(2, 100, 100), prize=6))
    assert score_state(st, 0, ci).total == pytest.approx(-score_state(st, 1, ci).total)


def test_terminal_win_when_my_prizes_empty():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=0),
                  mk_player(mk_pokemon(2, 100, 100), prize=4))
    ev = score_state(st, 0, ci)
    assert ev.total == WIN_SCORE and ev.reasons == ["win"]


def test_terminal_loss_when_opp_prizes_empty():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=4),
                  mk_player(mk_pokemon(2, 100, 100), prize=0))
    ev = score_state(st, 0, ci)
    assert ev.total == LOSS_SCORE and ev.reasons == ["loss"]


# --------------------------------------------------------------------------- #
# individual component features (isolate one weight at a time)
# --------------------------------------------------------------------------- #
def _only(component: str, value: float = 1.0) -> EvalWeights:
    """Weights with every component zeroed except ``component`` (set to ``value``)."""
    kw = {c: 0.0 for c in COMPONENTS}
    kw[component] = value
    return EvalWeights(**kw)


def test_prize_component_rewards_fewer_own_prizes():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=3),   # we lead by 3 prizes
                  mk_player(mk_pokemon(2, 100, 100), prize=6))
    ev = score_state(st, 0, ci, weights=_only("prize", 1000.0))
    assert ev.total == pytest.approx(3000.0)  # (6 - 3) * 1000


def test_active_survival_rewards_having_active_and_hp():
    ci = cards()
    # We have a full-HP active; opponent has none (empty active slot).
    st = state_of(mk_player(mk_pokemon(1, 100, 100)), mk_player(active=None))
    ev = score_state(st, 0, ci, weights=_only("active_survival", 100.0))
    # 2*(1-0) presence + (1.0 - 0.0) hp fraction = 3.0, * 100.
    assert ev.total == pytest.approx(300.0)


def test_ko_threat_rewards_affordable_lethal():
    ci = cards(my_weak_target=False)
    # Our active has the energy for its 60-dmg attack; opp active sits at 50 HP → lethal.
    me = mk_player(mk_pokemon(1, 100, 100, energies=[WATER]))
    opp = mk_player(mk_pokemon(2, 50, 100))
    st = state_of(me, opp)
    ev = score_state(st, 0, ci, weights=_only("ko_threat", 60.0))
    assert ev.total == pytest.approx(60.0)


def test_hand_value_rewards_more_cards():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), hand_count=7),
                  mk_player(mk_pokemon(2, 100, 100), hand_count=3))
    ev = score_state(st, 0, ci, weights=_only("hand_value", 5.0))
    assert ev.total == pytest.approx(20.0)  # (7 - 3) * 5


def test_energy_tempo_counts_all_in_play_energy():
    ci = cards()
    me = mk_player(mk_pokemon(1, 100, 100, energies=[WATER, WATER]),
                   bench=[mk_pokemon(1, 100, 100, energies=[WATER])])
    opp = mk_player(mk_pokemon(2, 100, 100, energies=[WATER]))
    st = state_of(me, opp)
    ev = score_state(st, 0, ci, weights=_only("energy_tempo", 30.0))
    assert ev.total == pytest.approx(60.0)  # (3 - 1) * 30


def test_retreat_capacity_needs_energy_and_a_bench():
    ci = cards(retreat_cost=1)
    # We can retreat (1 energy, bench present); opponent cannot (no energy).
    me = mk_player(mk_pokemon(1, 100, 100, energies=[WATER]), bench=[mk_pokemon(1, 100, 100)])
    opp = mk_player(mk_pokemon(2, 100, 100), bench=[mk_pokemon(2, 100, 100)])
    st = state_of(me, opp)
    ev = score_state(st, 0, ci, weights=_only("retreat_capacity", 10.0))
    assert ev.total == pytest.approx(10.0)  # (1 - 0) * 10


# --------------------------------------------------------------------------- #
# weights config + ablation
# --------------------------------------------------------------------------- #
def test_without_zeroes_named_component_and_keeps_the_rest():
    w = DEFAULT_WEIGHTS.without("prize")
    assert w.prize == 0.0
    assert w.active_survival == DEFAULT_WEIGHTS.active_survival


def test_without_unknown_component_raises():
    with pytest.raises(KeyError):
        DEFAULT_WEIGHTS.without("nope")


def test_disabled_drops_component_from_total_but_reports_it():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=3),
                  mk_player(mk_pokemon(2, 100, 100), prize=6))
    full = score_state(st, 0, ci)
    ablated = score_state(st, 0, ci, disabled=frozenset({"prize"}))
    assert ablated.components["prize"] == 0.0
    assert ablated.total == pytest.approx(full.total - full.components["prize"])


def test_reasons_are_sorted_by_magnitude():
    ci = cards()
    st = state_of(mk_player(mk_pokemon(1, 100, 100), prize=3, hand_count=7),
                  mk_player(mk_pokemon(2, 100, 100), prize=6, hand_count=3))
    ev = score_state(st, 0, ci)
    # prize (3000) dominates hand_value (20), so it is listed first.
    assert ev.reasons[0].startswith("prize")


def test_never_raises_on_malformed_state():
    assert score_state(None, 0, cards()).total == 0.0

    class Bad:
        players = None
    assert score_state(Bad(), 0, cards()).total == 0.0
