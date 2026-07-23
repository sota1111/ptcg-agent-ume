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


# --------------------------------------------------------------------------- #
# SOT-1898 search-driven gate (all-decision + overspend guard, engine-free)
# --------------------------------------------------------------------------- #
class _FakeCurrent:
    yourIndex = 0


class _FakeParsed:
    """A searchable parsed observation (never reaches the engine — ``_search``
    is stubbed), enough to pass ``maybe_search``'s eligibility gate."""

    current = _FakeCurrent()
    search_begin_input = object()


class _FakeSelect:
    def __init__(self, n_options: int, min_count: int = 1, max_count: int = 1) -> None:
        self.option = list(range(n_options))
        self.minCount = min_count
        self.maxCount = max_count


def _confident_logits() -> list[float]:
    """A peaked, low-entropy distribution over 4 options — NOT critical."""
    return [10.0, 0.0, 0.0, 0.0]


def _make_mcts(cfg: MCTSConfig, stats: MCTSStats | None = None) -> DeterminizedMCTS:
    return DeterminizedMCTS(tiny_policy(), cfg, rng=random.Random(7),
                            stats=stats or MCTSStats())


def test_all_decision_activates_non_critical(monkeypatch):
    """all_decision searches an eligible position the critical gate would skip."""
    captured = {}

    def fake_search(parsed, logp, policy_pick, time_limit=None):
        captured["time_limit"] = time_limit
        # Make the searched best (option 1) beat the policy pick (option 0).
        return {0: 0.0, 1: 1.0}

    # value 0.9 + peaked logits => is_critical() is False.
    cfg = MCTSConfig(all_decision=True, deviate_margin=0.02, time_limit_s=1.5)
    mcts = _make_mcts(cfg)
    monkeypatch.setattr(mcts, "_search", fake_search)
    out = mcts.maybe_search({}, _FakeParsed(), _FakeSelect(4),
                            _confident_logits(), 0.9, [0])
    assert out == [1]                       # search overrode the policy pick
    assert mcts.stats.activations == 1      # activated despite non-critical
    assert mcts.stats.overrides == 1
    assert captured["time_limit"] == pytest.approx(1.5)  # no budget => full cap


def test_all_decision_off_keeps_critical_only(monkeypatch):
    """With all_decision off, a non-critical position is never searched."""
    calls = {"n": 0}

    def fake_search(*a, **k):
        calls["n"] += 1
        return {0: 0.0, 1: 1.0}

    cfg = MCTSConfig(all_decision=False)
    mcts = _make_mcts(cfg)
    monkeypatch.setattr(mcts, "_search", fake_search)
    out = mcts.maybe_search({}, _FakeParsed(), _FakeSelect(4),
                            _confident_logits(), 0.9, [0])
    assert out is None
    assert mcts.stats.eligible == 1 and mcts.stats.activations == 0
    assert calls["n"] == 0


def test_overspend_guard_reverts_to_critical(monkeypatch):
    """Once the match search budget is spent, all_decision falls back to the
    critical-only gate — a non-critical position stops activating."""
    def fake_search(*a, **k):
        return {0: 0.0, 1: 1.0}

    cfg = MCTSConfig(all_decision=True, match_search_budget_s=1.0, deviate_margin=0.02)
    # Pretend 1200 ms of search has already been spent (> 1.0 s budget).
    stats = MCTSStats(elapsed_ms_total=1200.0)
    mcts = _make_mcts(cfg, stats)
    monkeypatch.setattr(mcts, "_search", fake_search)
    out = mcts.maybe_search({}, _FakeParsed(), _FakeSelect(4),
                            _confident_logits(), 0.9, [0])
    assert out is None                      # guard tripped -> critical-only gate
    assert mcts.stats.activations == 0

    # A critical position (value ~0) still searches even past the budget.
    out2 = mcts.maybe_search({}, _FakeParsed(), _FakeSelect(4),
                             _confident_logits(), 0.0, [0])
    assert out2 == [1] and mcts.stats.activations == 1


def test_adaptive_time_limit_clamped_to_remaining_budget(monkeypatch):
    """The per-decision cap is clamped to the remaining match search budget."""
    captured = {}

    def fake_search(parsed, logp, policy_pick, time_limit=None):
        captured["time_limit"] = time_limit
        return {0: 1.0, 1: 0.0}

    cfg = MCTSConfig(all_decision=True, match_search_budget_s=2.0, time_limit_s=1.5)
    stats = MCTSStats(elapsed_ms_total=1000.0)  # 1.0 s spent of a 2.0 s budget
    mcts = _make_mcts(cfg, stats)
    monkeypatch.setattr(mcts, "_search", fake_search)
    mcts.maybe_search({}, _FakeParsed(), _FakeSelect(4),
                      _confident_logits(), 0.0, [0])
    # remaining = 2.0 - 1.0 = 1.0 s < time_limit_s (1.5) => clamped to 1.0.
    assert captured["time_limit"] == pytest.approx(1.0)


def test_rollout_temperature_default_and_config():
    assert MCTSConfig().rollout_temperature == 1.0
    assert MCTSConfig(rollout_temperature=0.35).rollout_temperature == 0.35
