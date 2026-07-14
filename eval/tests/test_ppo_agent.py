"""PPO inference-agent acceptance tests (SOT-1689).

Two layers, matching the repo convention (cf. ``test_features`` vs
``test_safe_agent``):

* **engine-free** — :mod:`agents.policy_net` is pure Python/stdlib, so the
  artifact validation, the forward pass, and the legality-by-construction of
  :func:`~agents.policy_net.sample_action` are pinned with no ``cg/`` engine;
* **engine-backed** — :class:`agents.ppo_agent.PPOAgent` (a SafeAgent) is run
  over the crafted-selection battery: 違法出力0 with a real policy, with a
  missing artifact (legal-random fallback), and for the **committed**
  ``data/policy.json``.
"""
from __future__ import annotations

import json
import random

import pytest

from agents.features import FEATURE_DIM, FEATURE_VERSION
from agents.policy_net import (
    N_SLOTS,
    POLICY_SCHEMA,
    forward,
    load_policy,
    masked_log_softmax,
    sample_action,
    validate_policy,
)

HIDDEN = 4


def tiny_policy(n_slots: int = N_SLOTS, feature_version: int = FEATURE_VERSION) -> dict:
    """A small deterministic hand-built artifact (no numpy needed)."""
    return {
        "schema": POLICY_SCHEMA,
        "feature_version": feature_version,
        "feature_dim": FEATURE_DIM,
        "hidden": HIDDEN,
        "n_slots": n_slots,
        "w1": [[0.01 * ((i + j) % 5 - 2) for i in range(FEATURE_DIM)] for j in range(HIDDEN)],
        "b1": [0.1, -0.1, 0.0, 0.05],
        "w2": [[0.1 * ((k + j) % 3 - 1) for j in range(HIDDEN)] for k in range(n_slots)],
        "b2": [0.01 * (k % 7) for k in range(n_slots)],
        "vw": [0.2, -0.2, 0.1, 0.0],
        "vb": 0.05,
    }


# --------------------------------------------------------------------------- #
# validate_policy / load_policy (engine-free)
# --------------------------------------------------------------------------- #
def test_validate_policy_accepts_tiny_policy():
    assert validate_policy(tiny_policy()) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda p: p.update(schema="other"), "schema"),
    (lambda p: p.update(feature_dim=0), "feature_dim"),
    (lambda p: p.pop("w1"), "matrix 'w1'"),
    (lambda p: p["w1"][0].pop(), "matrix 'w1'"),
    (lambda p: p["b1"].append(0.0), "vector 'b1'"),
    (lambda p: p.update(vb=float("nan")), "vb"),
    (lambda p: p["w2"][0].__setitem__(0, "x"), "matrix 'w2'"),
])
def test_validate_policy_flags_violations(mutate, fragment):
    policy = tiny_policy()
    mutate(policy)
    errors = validate_policy(policy)
    assert errors and any(fragment in e for e in errors)


def test_load_policy_roundtrip_and_failure_modes(tmp_path):
    good = tmp_path / "policy.json"
    good.write_text(json.dumps(tiny_policy()), encoding="utf-8")
    assert load_policy(str(good)) is not None
    assert load_policy(str(tmp_path / "missing.json")) is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_policy(str(corrupt)) is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert load_policy(str(wrong)) is None


# --------------------------------------------------------------------------- #
# forward / masked softmax (engine-free)
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_determinism():
    policy = tiny_policy()
    features = [0.5] * FEATURE_DIM
    logits, value = forward(policy, features)
    assert len(logits) == N_SLOTS
    assert isinstance(value, float)
    assert (logits, value) == forward(policy, features)


def test_masked_log_softmax_covers_only_legal_slots():
    logits = [0.0, 1.0, 2.0, 3.0]
    import math

    logp = masked_log_softmax(logits, 3)
    assert len(logp) == 3  # slot 3 is masked out entirely
    assert math.isclose(sum(math.exp(x) for x in logp), 1.0, rel_tol=1e-9)
    assert masked_log_softmax(logits, 0) == []
    assert len(masked_log_softmax(logits, 99)) == 4  # capped at len(logits)


# --------------------------------------------------------------------------- #
# sample_action: legal by construction (engine-free)
# --------------------------------------------------------------------------- #
def test_sample_action_is_always_legal():
    policy = tiny_policy()
    rng = random.Random(7)
    features = [0.25] * FEATURE_DIM
    shapes = [
        (1, 1, 1), (2, 1, 1), (5, 1, 1), (5, 0, 3), (5, 2, 3), (10, 0, 0),
        (50, 1, 1), (80, 1, 1), (80, 70, 75), (3, 1, 3), (64, 1, 2), (65, 1, 1),
    ]
    for n, lo, hi in shapes:
        for _ in range(50):
            action = sample_action(policy, features, n, lo, hi, rng)
            assert len(set(action)) == len(action), (n, lo, hi, action)
            assert all(0 <= i < n for i in action), (n, lo, hi, action)
            expected_lo = max(0, lo)
            expected_hi = min(hi, n) if min(hi, n) >= expected_lo else expected_lo
            assert expected_lo <= len(action) <= max(expected_hi, expected_lo)


def test_sample_action_deterministic_mode_is_stable():
    policy = tiny_policy()
    features = [0.3] * FEATURE_DIM
    first = sample_action(policy, features, 8, 1, 1, random.Random(1), deterministic=True)
    second = sample_action(policy, features, 8, 1, 1, random.Random(2), deterministic=True)
    assert first == second  # rng must not matter when deterministic


def test_sample_action_prefers_high_logit_slots():
    """The argmax option must dominate sampling (distribution sanity, not exactness)."""
    policy = tiny_policy(n_slots=4)
    policy["w2"] = [[0.0] * HIDDEN for _ in range(4)]
    policy["b2"] = [5.0, 0.0, 0.0, 0.0]  # slot 0 ~ e^5 more likely
    rng = random.Random(11)
    picks = [sample_action(policy, [0.0] * FEATURE_DIM, 4, 1, 1, rng)[0] for _ in range(200)]
    assert picks.count(0) > 150


# --------------------------------------------------------------------------- #
# PPOAgent on the crafted-selection battery (engine-backed)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")


def _make_select(*, type_: int, context: int, n_options: int,
                 min_count: int = 1, max_count: int = 1) -> dict:
    return {
        "type": type_, "context": context,
        "minCount": min_count, "maxCount": max_count,
        "remainDamageCounter": 0, "remainEnergyCost": 0,
        "option": [{"type": 0, "index": i} for i in range(n_options)],
        "deck": None, "contextCard": None, "effect": None,
    }


def _battery() -> list[dict]:
    shapes = [(3, 1, 1), (5, 0, 3), (1, 1, 1), (12, 2, 4), (70, 1, 1)]
    selects = [
        _make_select(type_=t, context=c, n_options=n, min_count=lo, max_count=hi)
        for t in (0, 5, 9, 99)          # known types + an unknown one
        for c in (0, 41, 999)           # known contexts + an unknown one
        for n, lo, hi in shapes
    ]
    return selects


def _assert_all_acts_legal(agent, engine):
    from agents.protocol import validate_selection
    from cg.api import to_observation_class

    for select in _battery():
        obs = {"select": select, "logs": [], "current": None}
        action = agent.act(obs)
        parsed = to_observation_class(obs).select
        validate_selection(action, parsed)  # raises on any illegal output


def test_ppo_agent_never_emits_illegal_action(engine):
    from agents import PPOAgent

    agent = PPOAgent(seed=3, policy=tiny_policy())
    assert agent.policy_loaded
    _assert_all_acts_legal(agent, engine)


def test_ppo_agent_without_artifact_falls_back_legally(engine, tmp_path):
    from agents import PPOAgent

    agent = PPOAgent(seed=3, policy_path=str(tmp_path / "nope.json"))
    assert not agent.policy_loaded
    _assert_all_acts_legal(agent, engine)
    # every real decision was a no-policy fallback, never a crash
    assert agent.unsupported_rate() == 1.0


def test_ppo_agent_rejects_mismatched_feature_version(engine):
    from agents import PPOAgent

    agent = PPOAgent(seed=3, policy=tiny_policy(feature_version=FEATURE_VERSION + 1))
    assert not agent.policy_loaded
    _assert_all_acts_legal(agent, engine)


def test_committed_policy_artifact_is_usable(engine):
    """The repo's data/policy.json must load and drive legal decisions."""
    from agents import PPOAgent
    from agents.ppo_agent import DEFAULT_POLICY_PATH

    agent = PPOAgent(seed=5, policy_path=DEFAULT_POLICY_PATH)
    assert agent.policy_loaded, f"committed artifact unusable: {DEFAULT_POLICY_PATH}"
    _assert_all_acts_legal(agent, engine)


def test_ppo_agent_plays_real_matches_fault_free(engine):
    """A couple of live engine matches: fault 0 (the arena-scale run is the bench)."""
    from agents import PPOAgent, RandomAgent
    from eval.match import play_match
    from eval.selfplay import load_deck

    deck = load_deck("deck.csv")
    for game in range(2):
        result = play_match(
            deck, deck,
            [PPOAgent(seed=game, policy=tiny_policy()), RandomAgent(seed=100 + game)],
        )
        assert not result.is_fault, result
