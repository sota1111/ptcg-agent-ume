"""Critical-position determinized MCTS tests (SOT-1690).

Two layers, matching the repo convention:

* **engine-free** — the criticality judgement, the entropy helper, the stats
  accounting, and the ``sample_action`` logits short-circuit are pure Python
  (:mod:`agents.mcts` defers all ``cg`` imports to call time), so they are
  pinned with no engine. So is the **fail-closed eligibility gate**: a crafted
  observation with no ``search_begin_input`` must keep the plain PPO action —
  legal, no crash, counters recorded.
* **engine-backed** — a real match with an always-critical config pins the
  live contract: MCTS activates, every action stays legal (fault 0 on our
  side), and the measured per-decision search time respects the hard cap.
"""
from __future__ import annotations

import random

import pytest

from agents.mcts import (
    DeterminizedMCTS,
    MCTSConfig,
    MCTSStats,
    is_critical,
    policy_entropy,
)
from agents.policy_net import forward, masked_log_softmax, sample_action

from .conftest import requires_engine
from .test_ppo_agent import tiny_policy


# --------------------------------------------------------------------------- #
# criticality judgement (engine-free)
# --------------------------------------------------------------------------- #
def test_policy_entropy_uniform_and_peaked():
    import math

    uniform = masked_log_softmax([0.0, 0.0, 0.0, 0.0], 4)
    assert policy_entropy(uniform) == pytest.approx(math.log(4))
    peaked = masked_log_softmax([50.0, 0.0, 0.0, 0.0], 4)
    assert policy_entropy(peaked) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("entropy, value, expected", [
    (2.5, 0.9, True),    # uncertain policy alone qualifies
    (0.1, 0.01, True),   # coin-flip value alone qualifies
    (0.1, -0.03, True),  # ...and by absolute value
    (0.1, 0.9, False),   # confident policy, decided game
    (1.9, 0.9, True),    # threshold is inclusive
    (0.1, 0.06, True),
    (1.89, 0.061, False),
])
def test_is_critical_thresholds(entropy, value, expected):
    assert is_critical(entropy, value, MCTSConfig()) is expected


def test_stats_activation_rate_and_merge():
    a = MCTSStats(decisions=8, activations=2, elapsed_ms_total=30.0, elapsed_ms_max=20.0)
    b = MCTSStats(decisions=2, activations=3, elapsed_ms_total=10.0, elapsed_ms_max=9.0)
    a.merge(b)
    assert a.decisions == 10 and a.activations == 5
    assert a.activation_rate == pytest.approx(0.5)
    assert a.elapsed_ms_max == 20.0
    report = a.report()
    assert report["activation_rate"] == pytest.approx(0.5)
    assert report["search_ms_mean"] == pytest.approx(40.0 / 5)
    assert MCTSStats().activation_rate == 0.0  # no division by zero


def test_sample_action_reuses_precomputed_logits():
    policy = tiny_policy()
    features = [0.25] * policy["feature_dim"]
    logits, _ = forward(policy, features)
    for shape in [(5, 1, 1), (8, 1, 3), (70, 65, 68)]:
        n, lo, hi = shape
        recomputed = sample_action(policy, features, n, lo, hi, random.Random(3))
        reused = sample_action(
            policy, features, n, lo, hi, random.Random(3), logits=logits
        )
        assert recomputed == reused, shape


# --------------------------------------------------------------------------- #
# fail-closed gate on a crafted (searchless) observation (engine-free)
# --------------------------------------------------------------------------- #
def _crafted_obs(n_options: int = 4) -> dict:
    """A minimal MAIN selection with NO ``search_begin_input`` (search impossible)."""
    def player() -> dict:
        return {"active": [], "bench": [], "benchMax": 5, "deckCount": 40,
                "discard": [], "prize": [None] * 6, "handCount": 5, "hand": None,
                "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}

    return {
        "select": {
            "type": 0,
            "context": 0,
            "option": [{"type": 14, "index": i} for i in range(n_options)],
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 3,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [player(), player()],
        },
    }


def test_mcts_agent_fails_closed_without_search_input():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    from agents.ppo_agent import PPOAgent

    # Always-critical thresholds: only the missing search_begin_input stops it.
    cfg = MCTSConfig(entropy_threshold=0.0, value_threshold=1.0)
    agent = PPOAgent(seed=5, policy=tiny_policy(), mcts=True, mcts_config=cfg)
    assert agent.mcts_stats is not None and agent.name == "ppo+mcts"

    obs = _crafted_obs()
    for _ in range(10):
        action = agent.act(obs)
        assert len(action) == 1 and 0 <= action[0] < 4  # legal, single pick

    stats = agent.mcts_stats
    assert stats.decisions == 10
    assert stats.eligible == 0       # gate closed before any engine call
    assert stats.activations == 0
    assert stats.simulations == 0
    # And the SafeAgent skeleton saw only valid policy outputs (no fallbacks).
    assert all(s.fallbacks == 0 for s in agent.stats.values())


def test_plain_ppo_agent_has_no_mcts():
    pytest.importorskip("cg.api", reason="cabt engine (cg/) not installed")
    from agents.ppo_agent import PPOAgent

    agent = PPOAgent(seed=5, policy=tiny_policy())
    assert agent.mcts_stats is None
    assert agent.act(_crafted_obs()) is not None


# --------------------------------------------------------------------------- #
# live engine: activation, legality, time cap (engine-backed)
# --------------------------------------------------------------------------- #
@requires_engine
def test_mcts_agent_live_match_activates_legally_within_cap(deck):
    from agents.ppo_agent import PPOAgent
    from agents.random_agent import RandomAgent
    from eval.match import play_match

    time_limit = 0.2
    cfg = MCTSConfig(
        entropy_threshold=0.0,  # every eligible decision is critical:
        value_threshold=1.0,    # maximum exercise of the search path
        time_limit_s=time_limit,
        n_determinizations=2,
        rollout_depth=4,
    )
    agent = PPOAgent(seed=11, mcts=True, mcts_config=cfg)
    if not agent.policy_loaded:
        pytest.skip("data/policy.json not available")

    result = play_match(deck, deck, [agent, RandomAgent(seed=12)])
    assert result.faulted_player != 0  # 違法出力0 on the MCTS side

    stats = agent.mcts_stats
    assert stats.decisions > 0
    assert stats.activations > 0, "always-critical config must activate"
    assert stats.simulations > 0, "at least one determinization must simulate"
    assert 0.0 < stats.activation_rate <= 1.0
    # The per-decision cap holds with bounded overhead (one in-flight rollout
    # step may straddle the deadline; engine steps are milliseconds).
    assert stats.elapsed_ms_max <= time_limit * 1000.0 + 500.0


@requires_engine
def test_mcts_search_respects_tiny_time_cap(deck):
    from agents.ppo_agent import PPOAgent
    from agents.random_agent import RandomAgent
    from eval.match import play_match

    cfg = MCTSConfig(
        entropy_threshold=0.0, value_threshold=1.0, time_limit_s=0.01,
        n_determinizations=1, rollout_depth=2,
    )
    agent = PPOAgent(seed=21, mcts=True, mcts_config=cfg)
    if not agent.policy_loaded:
        pytest.skip("data/policy.json not available")

    result = play_match(deck, deck, [agent, RandomAgent(seed=22)])
    assert result.faulted_player != 0
    assert agent.mcts_stats.activations > 0
    assert agent.mcts_stats.elapsed_ms_max <= 10.0 + 300.0  # cap + bounded overhead
