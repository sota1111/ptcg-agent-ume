"""Fixed-dimension / unknown-safe contract of the observation featurizer (SOT-1688).

:mod:`agents.features` is pure Python and engine-free by design (PPO training and
CI must work without the gitignored cabt engine), so this file loads the module
**standalone** — straight from its file path, without executing the
``agents/__init__`` package import (which pulls in ``cg.api``). The tests pin:

* the fixed vector length (``FEATURE_DIM`` == ``featurize()`` == ``feature_names()``),
* the fallback behaviour on missing/None/malformed sub-objects,
* the unknown-bucket handling for enum values appended after this layout froze,
* the deciding-player perspective (me/opp blocks follow ``yourIndex``).
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_features():
    """Load ``agents/features.py`` without importing the ``agents`` package."""
    path = os.path.join(REPO, "agents", "features.py")
    spec = importlib.util.spec_from_file_location("_features_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


features = _load_features()


def _names():
    return features.feature_names()


def _slot(name: str) -> int:
    return _names().index(name)


def make_obs(**overrides) -> dict:
    """A well-formed observation dict in the shape ``cg.game`` emits."""
    mon = {
        "id": 1, "serial": 10, "hp": 120, "maxHp": 200, "appearThisTurn": False,
        "energies": [1, 1], "energyCards": [{"id": 2, "serial": 11, "playerIndex": 0}],
        "tools": [], "preEvolution": [],
    }
    player = {
        "active": [mon], "bench": [dict(mon, hp=60)], "benchMax": 5,
        "deckCount": 40, "discard": [], "prize": [None] * 6, "handCount": 5,
        "hand": None, "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    obs = {
        "select": {
            "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 14}, {"type": 13, "attackId": 7}],
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 3, "turnActionCount": 2, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": True, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [player, dict(player, deckCount=30, handCount=7)],
        },
    }
    obs.update(overrides)
    return obs


# --------------------------------------------------------------------------- #
# Fixed dimension / schema
# --------------------------------------------------------------------------- #
def test_dimension_is_fixed_and_consistent():
    names = _names()
    assert features.FEATURE_DIM == len(names)
    assert len(features.featurize(make_obs())) == features.FEATURE_DIM
    assert len(features.featurize({})) == features.FEATURE_DIM


def test_feature_names_are_unique():
    names = _names()
    assert len(names) == len(set(names))


def test_all_values_are_finite_floats():
    vec = features.featurize(make_obs())
    assert all(isinstance(x, float) and math.isfinite(x) for x in vec)


def test_featurize_is_deterministic():
    obs = make_obs()
    assert features.featurize(obs) == features.featurize(obs)


# --------------------------------------------------------------------------- #
# Fallbacks: missing / None / malformed input never raises, never changes shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("obs", [
    {},
    {"current": None, "select": None, "logs": []},
    {"current": {"players": "bogus", "turn": "NaNish"}, "select": {"option": "bad"}},
    {"current": {"yourIndex": 5, "players": []}, "select": {}},
    {"select": {"type": True, "context": None, "option": [None, 3, {"type": "x"}]}},
    make_obs(current=None),
])
def test_malformed_inputs_fall_back_to_fixed_shape(obs):
    vec = features.featurize(obs)
    assert len(vec) == features.FEATURE_DIM
    assert all(isinstance(x, float) and math.isfinite(x) for x in vec)


def test_empty_obs_zeroes_state_blocks():
    vec = features.featurize({})
    assert vec[_slot("turn")] == 0.0
    assert vec[_slot("me_active_exists")] == 0.0
    assert vec[_slot("opp_deck_count")] == 0.0
    # An absent select still one-hots deterministically into the unknown bucket.
    assert vec[_slot("select_type_unknown")] == 1.0
    assert vec[_slot("select_context_unknown")] == 1.0


# --------------------------------------------------------------------------- #
# Known values land in their slots
# --------------------------------------------------------------------------- #
def test_known_slots_reflect_observation():
    vec = features.featurize(make_obs())
    assert vec[_slot("turn")] == pytest.approx(3 / 50.0)
    assert vec[_slot("i_am_first")] == 1.0
    assert vec[_slot("opp_is_first")] == 0.0
    assert vec[_slot("supporter_played")] == 1.0
    assert vec[_slot("me_active_exists")] == 1.0
    assert vec[_slot("me_active_hp")] == pytest.approx(120 / 400.0)
    assert vec[_slot("me_active_hp_ratio")] == pytest.approx(120 / 200.0)
    assert vec[_slot("me_deck_count")] == pytest.approx(40 / 60.0)
    assert vec[_slot("opp_deck_count")] == pytest.approx(30 / 60.0)
    assert vec[_slot("me_prize_count")] == pytest.approx(1.0)
    assert vec[_slot("select_type_0")] == 1.0
    assert vec[_slot("select_context_0")] == 1.0
    assert vec[_slot("select_n_options")] == pytest.approx(2 / 50.0)
    # Option-type histogram: one END (14) + one ATTACK (13), each half.
    assert vec[_slot("option_type_frac_14")] == pytest.approx(0.5)
    assert vec[_slot("option_type_frac_13")] == pytest.approx(0.5)


def test_perspective_follows_your_index():
    obs = make_obs()
    obs["current"]["yourIndex"] = 1
    obs["current"]["firstPlayer"] = 0
    vec = features.featurize(obs)
    # "me" is now players[1] (deckCount=30) and the opponent moved first.
    assert vec[_slot("me_deck_count")] == pytest.approx(30 / 60.0)
    assert vec[_slot("opp_deck_count")] == pytest.approx(40 / 60.0)
    assert vec[_slot("i_am_first")] == 0.0
    assert vec[_slot("opp_is_first")] == 1.0


# --------------------------------------------------------------------------- #
# Unknown-enum safety (values appended mid-competition)
# --------------------------------------------------------------------------- #
def test_unknown_select_type_and_context_hit_unknown_bucket():
    obs = make_obs()
    obs["select"]["type"] = 99
    obs["select"]["context"] = 999
    obs["select"]["option"] = [{"type": 42}]
    vec = features.featurize(obs)
    assert vec[_slot("select_type_unknown")] == 1.0
    assert vec[_slot("select_context_unknown")] == 1.0
    assert vec[_slot("option_type_frac_unknown")] == pytest.approx(1.0)
    # No known slot may also fire.
    assert sum(vec[_slot(f"select_type_{i}")] for i in range(11)) == 0.0
