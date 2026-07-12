"""Illegal-output guard fixtures for the SafeAgent skeleton (SOT-1646, R1).

These are the acceptance fixtures for the safety骨格: over a battery of crafted
selections — known *and* unknown ``(SelectType, SelectContext)``, empty candidates,
and hostile policies that raise / overrun / return garbage — every agent built on
:class:`agents.protocol.SafeAgent` must return a **legal** action (or the legal empty
action) and must **never raise**. They also pin the validator, the legal-random
fallback, and the per-context encounter / 未対応率 measurement.

They construct ``obs`` dicts directly (no live battle), but still need the engine's
``cg.api`` dataclasses/enums, which are gitignored/absent in CI — hence
``requires_engine``.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")

from cg.api import SelectContext, SelectType  # noqa: E402

from agents import (  # noqa: E402
    FallbackReason,
    InvalidSelectionError,
    RandomAgent,
    RuleAgent,
    SafeAgent,
    legal_random_action,
    validate_selection,
)
from agents.protocol import KNOWN_SELECT_TYPES  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


# --------------------------------------------------------------------------- #
# obs fixture builders — a well-formed observation dict the engine would emit
# --------------------------------------------------------------------------- #
def make_select(
    *,
    type_: int,
    context: int,
    n_options: int,
    min_count: int = 1,
    max_count: int = 1,
) -> dict:
    """A fully-populated ``select`` dict with ``n_options`` trivial options."""
    return {
        "type": type_,
        "context": context,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [{"type": 0, "index": i} for i in range(n_options)],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


def make_obs(select: dict | None) -> dict:
    return {"select": select, "logs": [], "current": None}


def parsed_select(select_dict: dict):
    """The typed ``SelectData`` for a raw select dict (as agents see it)."""
    return to_observation_class(make_obs(select_dict)).select


# A representative battery: every known type × a spread of contexts × count shapes,
# plus a few unknown (future-appended) type/context values.
def battery() -> list[dict]:
    fixtures: list[dict] = []
    contexts = [
        int(SelectContext.MAIN),
        int(SelectContext.SETUP_ACTIVE_POKEMON),
        int(SelectContext.DISCARD),
        int(SelectContext.IS_FIRST),
        int(SelectContext.ATTACK),
        int(SelectContext.DRAW_COUNT),
    ]
    for t in sorted(KNOWN_SELECT_TYPES):
        for c in contexts:
            fixtures.append(make_select(type_=t, context=c, n_options=3, min_count=1, max_count=1))
            fixtures.append(make_select(type_=t, context=c, n_options=5, min_count=0, max_count=3))
            fixtures.append(make_select(type_=t, context=c, n_options=1, min_count=1, max_count=1))
    # unknown type / unknown context (values the engine may append mid-competition)
    fixtures.append(make_select(type_=99, context=int(SelectContext.MAIN), n_options=4))
    fixtures.append(make_select(type_=int(SelectType.MAIN), context=99, n_options=4))
    fixtures.append(make_select(type_=123, context=456, n_options=2, min_count=1, max_count=2))
    return fixtures


# --------------------------------------------------------------------------- #
# validate_selection — the 選択数validator (range + duplicate + count)
# --------------------------------------------------------------------------- #
def test_validate_selection_accepts_legal():
    sel = parsed_select(make_select(type_=1, context=8, n_options=4, min_count=1, max_count=2))
    assert validate_selection([0], sel) == [0]
    assert validate_selection([3, 1], sel) == [3, 1]


def test_validate_selection_none_select():
    assert validate_selection([], None) == []
    assert validate_selection(None, None) == []
    with pytest.raises(InvalidSelectionError):
        validate_selection([0], None)


@pytest.mark.parametrize(
    "action",
    [
        [4],           # out of range (n=4 -> valid 0..3)
        [-1],          # negative
        [1, 1],        # duplicate
        [],            # below minCount=1
        [0, 1, 2],     # above maxCount=2
        "0",           # not a list
        [0, "1"],      # non-int element
        [True],        # bool is not a valid index
    ],
)
def test_validate_selection_rejects_illegal(action):
    sel = parsed_select(make_select(type_=1, context=8, n_options=4, min_count=1, max_count=2))
    with pytest.raises(InvalidSelectionError):
        validate_selection(action, sel)


# --------------------------------------------------------------------------- #
# legal_random_action — always legal, by construction
# --------------------------------------------------------------------------- #
def test_legal_random_action_always_legal():
    import random

    rng = random.Random(0)
    for f in battery():
        sel = parsed_select(f)
        for _ in range(20):
            action = legal_random_action(sel, rng)
            # must survive the validator every single time
            assert validate_selection(action, sel) == action


def test_legal_random_action_empty_and_none():
    import random

    rng = random.Random(0)
    assert legal_random_action(None, rng) == []
    sel = parsed_select(make_select(type_=0, context=0, n_options=0, min_count=0, max_count=0))
    assert legal_random_action(sel, rng) == []


# --------------------------------------------------------------------------- #
# The acceptance battery: illegal output 0 / exception 0
# --------------------------------------------------------------------------- #
class RaisingAgent(SafeAgent):
    name = "raising"

    def policy(self, obs, parsed, select):
        raise RuntimeError("policy blew up")


class GarbageAgent(SafeAgent):
    name = "garbage"

    def policy(self, obs, parsed, select):
        # Deliberately illegal: out-of-range + duplicate + wrong count.
        return [999, 999, 999]


class SlowAgent(SafeAgent):
    name = "slow"

    def policy(self, obs, parsed, select):
        time.sleep(0.02)
        return [0]


@pytest.mark.parametrize(
    "agent_factory",
    [
        lambda: SafeAgent(seed=1),
        lambda: RandomAgent(seed=1),
        lambda: RuleAgent(seed=1),
        lambda: RaisingAgent(seed=1),
        lambda: GarbageAgent(seed=1),
    ],
)
def test_battery_illegal_output_zero_exception_zero(agent_factory):
    agent = agent_factory()
    illegal = 0
    for f in battery():
        sel = parsed_select(f)
        action = agent.act(make_obs(f))  # must never raise
        try:
            validate_selection(action, sel)
        except InvalidSelectionError:
            illegal += 1
    assert illegal == 0


def test_random_agent_zero_unsupported():
    agent = RandomAgent(seed=7)
    for f in battery():
        if f["type"] in KNOWN_SELECT_TYPES and f["context"] in {int(c) for c in SelectContext}:
            agent.act(make_obs(f))
    # A real (always-applicable) policy: nothing on a known context is "unsupported".
    assert agent.unsupported_rate() == 0.0
    assert sum(s.handled for s in agent.stats.values()) > 0


def test_rule_agent_skeleton_fully_unsupported():
    agent = RuleAgent(seed=7)
    known = [f for f in battery()
             if f["type"] in KNOWN_SELECT_TYPES and f["context"] in {int(c) for c in SelectContext}]
    for f in known:
        agent.act(make_obs(f))
    # R1 skeleton has no tactics: every known-context decision is unsupported.
    assert agent.unsupported_rate() == 1.0
    assert sum(s.handled for s in agent.stats.values()) == 0
    # ...and it still emitted only legal actions (checked in the battery test above).


# --------------------------------------------------------------------------- #
# Fallback reasons are recorded per (type, context)
# --------------------------------------------------------------------------- #
def test_no_policy_fallback_recorded():
    agent = SafeAgent(seed=0)
    sel = make_select(type_=int(SelectType.MAIN), context=int(SelectContext.MAIN), n_options=3)
    agent.act(make_obs(sel))
    key = (int(SelectType.MAIN), int(SelectContext.MAIN))
    stat = agent.stats[key]
    assert stat.encounters == 1
    assert stat.fallbacks == 1
    assert stat.unsupported == 1
    assert stat.fallback_reasons == {FallbackReason.NO_POLICY.value: 1}
    assert agent.last_fallback == (key, FallbackReason.NO_POLICY)


def test_unknown_type_and_context_fallback():
    agent = SafeAgent(seed=0)
    agent.act(make_obs(make_select(type_=99, context=int(SelectContext.MAIN), n_options=3)))
    agent.act(make_obs(make_select(type_=int(SelectType.MAIN), context=99, n_options=3)))
    reasons = {r for s in agent.stats.values() for r in s.fallback_reasons}
    assert FallbackReason.UNKNOWN_TYPE.value in reasons
    assert FallbackReason.UNKNOWN_CONTEXT.value in reasons
    # Unknown type/context count as unsupported (no policy could exist).
    assert agent.unsupported_rate() == 1.0


def test_policy_exception_fallback_is_legal():
    agent = RaisingAgent(seed=0)
    sel = make_select(type_=int(SelectType.CARD), context=int(SelectContext.DISCARD), n_options=4)
    action = agent.act(make_obs(sel))
    validate_selection(action, parsed_select(sel))  # legal
    stat = agent.stats[(int(SelectType.CARD), int(SelectContext.DISCARD))]
    assert stat.fallback_reasons == {FallbackReason.POLICY_EXCEPTION.value: 1}
    # A policy exception is a runtime catch, NOT "unsupported".
    assert stat.unsupported == 0


def test_invalid_output_fallback_is_legal():
    agent = GarbageAgent(seed=0)
    sel = make_select(type_=int(SelectType.CARD), context=int(SelectContext.DISCARD),
                      n_options=4, min_count=1, max_count=1)
    action = agent.act(make_obs(sel))
    validate_selection(action, parsed_select(sel))  # legal despite garbage policy
    stat = agent.stats[(int(SelectType.CARD), int(SelectContext.DISCARD))]
    assert stat.fallback_reasons == {FallbackReason.INVALID_OUTPUT.value: 1}
    assert stat.unsupported == 0


def test_policy_timeout_fallback():
    agent = SlowAgent(seed=0, time_budget_s=0.001)
    sel = make_select(type_=int(SelectType.MAIN), context=int(SelectContext.MAIN), n_options=3)
    action = agent.act(make_obs(sel))
    validate_selection(action, parsed_select(sel))
    stat = agent.stats[(int(SelectType.MAIN), int(SelectContext.MAIN))]
    assert stat.fallback_reasons == {FallbackReason.POLICY_TIMEOUT.value: 1}


# --------------------------------------------------------------------------- #
# Trivial / malformed selections never crash and never count as encounters
# --------------------------------------------------------------------------- #
def test_none_selection_returns_empty():
    agent = SafeAgent(seed=0)
    assert agent.act(make_obs(None)) == []
    assert agent.trivial_counts["no_selection"] == 1
    assert agent.stats == {}


def test_empty_options_returns_empty():
    agent = SafeAgent(seed=0)
    sel = make_select(type_=int(SelectType.MAIN), context=int(SelectContext.MAIN),
                      n_options=0, min_count=0, max_count=0)
    assert agent.act(make_obs(sel)) == []
    assert agent.trivial_counts["empty_options"] == 1
    assert agent.stats == {}


def test_malformed_obs_never_crashes():
    agent = SafeAgent(seed=0)
    # select missing required SelectData fields -> parse raises -> guarded to [].
    assert agent.act({"select": {"type": 0}, "logs": [], "current": None}) == []
    assert agent.act({"not": "an obs"}) == []
    assert agent.trivial_counts["no_selection"] >= 2


# --------------------------------------------------------------------------- #
# RuleAgent extension seam: registering a handler makes a context supported
# --------------------------------------------------------------------------- #
def test_rule_agent_register_handler_drops_unsupported():
    agent = RuleAgent(seed=0)
    key = (int(SelectType.YES_NO), int(SelectContext.IS_FIRST))
    agent.register(key, lambda obs, parsed, select: [0])
    sel = make_select(type_=key[0], context=key[1], n_options=2, min_count=1, max_count=1)
    action = agent.act(make_obs(sel))
    assert action == [0]
    assert agent.stats[key].handled == 1
    assert agent.unsupported_rate(key) == 0.0


def test_stats_report_shape():
    agent = SafeAgent(seed=0)
    agent.act(make_obs(make_select(type_=0, context=0, n_options=3)))
    report = agent.stats_report()
    assert report["totals"]["encounters"] == 1
    assert report["totals"]["unsupported_rate"] == 1.0
    assert "0,0" in report["by_context"]
    assert report["by_context"]["0,0"]["type"] == "MAIN"
