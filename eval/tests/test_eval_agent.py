"""Tests for the R4 EvalAgent — R3 parity + board-eval tie-break contract (SOT-1649).

Without a live engine search session (``search_begin_input is None`` on a hand-built
selection ⇒ ``search_begin`` rejects it), the board-eval tie-break cannot step any
candidate, so R4 must fall back to R3's exact RNG tie-break: it returns a legal move,
matches :class:`~agents.rule_agent.RuleAgent` decision-for-decision, never leaks a search
session, and never crashes / emits an illegal action. The evaluator itself
(:meth:`EvalAgent._board_evaluate`) and the trace bookkeeping are pinned directly.

Reading the observation needs ``cg.api`` (gitignored / absent in CI) — hence the importorskip.
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

from agents.board_eval import WIN_SCORE  # noqa: E402
from agents.eval_agent import EvalAgent  # noqa: E402
from agents.rule_agent import RuleAgent  # noqa: E402
from agents.rule_scoring import CardIndex  # noqa: E402

PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END = 7, 8, 9, 10, 11, 12, 13, 14
ACTIVE, BENCH, HAND = 4, 5, 2
WATER, FIRE = int(EnergyType.WATER), int(EnergyType.FIRE)


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


def mk_player(active=None, bench=None, *, prize=6):
    return {"active": [active] if active is not None else [], "bench": bench or [],
            "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * prize,
            "handCount": 5, "hand": None, "poisoned": False, "burned": False,
            "asleep": False, "paralyzed": False, "confused": False}


def mk_state(me, opp, your_index=0, result=-1):
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return {"turn": 3, "turnActionCount": 0, "yourIndex": your_index, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
            "retreated": False, "result": result, "stadium": [], "looking": None, "players": players}


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


def cards():
    return CardIndex.from_engine(
        [mk_card(1, energy_type=WATER, attacks=[20]), mk_card(2, energy_type=FIRE)],
        [mk_attack(20, 60)],
    )


# --------------------------------------------------------------------------- #
# _board_evaluate — the plugged-in evaluator
# --------------------------------------------------------------------------- #
def test_board_evaluate_terminal_win_loss():
    agent = EvalAgent(seed=0)
    ci = cards()
    win = to_observation_class(mk_obs([opt(END)], result=0, your_index=0))
    loss = to_observation_class(mk_obs([opt(END)], result=1, your_index=0))
    assert agent._board_evaluate(win, 0, ci) == WIN_SCORE
    assert agent._board_evaluate(loss, 0, ci) == -WIN_SCORE  # LOSS_SCORE == -WIN_SCORE
    # From the opponent's seat the same terminal flips.
    assert agent._board_evaluate(win, 1, ci) < 0


def test_board_evaluate_tracks_running_best():
    agent = EvalAgent(seed=0)
    ci = cards()
    agent._best_eval = None
    ahead = to_observation_class(mk_obs([opt(END)],
                                        me=mk_player(mk_pokemon(1, 100, 100), prize=3),
                                        opp=mk_player(mk_pokemon(2, 100, 100), prize=6)))
    behind = to_observation_class(mk_obs([opt(END)],
                                         me=mk_player(mk_pokemon(1, 100, 100), prize=6),
                                         opp=mk_player(mk_pokemon(2, 100, 100), prize=3)))
    agent._board_evaluate(behind, 0, ci)
    agent._board_evaluate(ahead, 0, ci)
    # Running-best keeps the higher-scoring (ahead) position.
    assert agent._best_eval is not None and agent._best_eval.total > 0


# --------------------------------------------------------------------------- #
# fail-closed parity with RuleAgent (no engine session ⇒ RNG tie-break)
# --------------------------------------------------------------------------- #
def _decisions_match(options, *, seed=0, **obs_kw):
    parsed = to_observation_class(mk_obs(options, **obs_kw))
    rule = RuleAgent(seed=seed).policy(mk_obs(options, **obs_kw), parsed, parsed.select)
    ev = EvalAgent(seed=seed).policy(mk_obs(options, **obs_kw), parsed, parsed.select)
    return rule, ev


def test_matches_rule_when_top_choice_unique():
    # ATTACH-to-active is the unique top category → identical single move.
    rule, ev = _decisions_match([opt(ATTACH, inPlayArea=ACTIVE), opt(END)])
    assert ev == rule
    assert ev is not None and all(0 <= i < 2 for i in ev)


def test_matches_rule_on_tie_via_stable_rng_fallback():
    # Two equal ATTACH-to-active options tie at the top; with no engine session the
    # board-eval tie-break defers to R3's seeded RNG, so both agents pick the same one.
    opts = [opt(ATTACH, inPlayArea=ACTIVE), opt(ATTACH, inPlayArea=ACTIVE), opt(END)]
    rule, ev = _decisions_match(opts, seed=7)
    assert ev == rule


def test_never_leaks_or_crashes_without_session():
    agent = EvalAgent(seed=0)
    opts = [opt(ATTACH, inPlayArea=ACTIVE), opt(ATTACH, inPlayArea=ACTIVE), opt(END)]
    parsed = to_observation_class(mk_obs(opts))
    move = agent.policy(mk_obs(opts), parsed, parsed.select)
    assert move is not None
    assert agent.search_stats["leaks"] == 0


def test_multi_select_or_empty_defers_safely():
    # A degenerate MAIN with only END still yields a legal, non-crashing choice.
    agent = EvalAgent(seed=0)
    parsed = to_observation_class(mk_obs([opt(END)]))
    assert agent.policy(mk_obs([opt(END)]), parsed, parsed.select) is not None
