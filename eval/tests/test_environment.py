"""Tests for the engine boundary and Agent Protocol (SOT-1623).

Covers the issue's acceptance / verification points:
* ``battle_finish()`` runs exactly once on both normal and exceptional exit.
* native resources are freed on exception (context-manager guarantee).
* the engine's global battle pointer does not leak past ``Environment`` (the
  single-active guard rejects a second live battle).
* legal moves come straight from the engine enumeration (``obs.select``).
* two different agent types are interchangeable under the same Protocol.
* the legacy ``run_match`` path still works.
"""
import random

import pytest


def _engine_available() -> bool:
    try:
        import cg.game  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _engine_available(),
    reason="cabt engine (cg/) not installed; run scripts/setup_engine.sh",
)

import cg.game as game  # noqa: E402
import eval.environment as environment  # noqa: E402
from eval.agents import (  # noqa: E402
    Agent,
    FirstOptionAgent,
    RandomAgent,
    SubmissionAgent,
)
from eval.environment import (  # noqa: E402
    EndReason,
    EngineError,
    Environment,
    IllegalActionError,
    MatchResult,
    validate_action,
)
from eval.match import play_match  # noqa: E402


@pytest.fixture
def count_finish(monkeypatch):
    """Wrap the engine's ``battle_finish`` with a call counter (still frees)."""
    calls = {"n": 0}
    real = game.battle_finish

    def counted():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(game, "battle_finish", counted)
    return calls


# --- pure unit: validate_action uses the engine's option list only -----------

class _FakeSelect:
    def __init__(self, n, minCount, maxCount):
        self.option = list(range(n))
        self.minCount = minCount
        self.maxCount = maxCount
        self.type = "FAKE"


def test_validate_action_accepts_wellformed():
    sel = _FakeSelect(5, 1, 2)
    assert validate_action([0], sel) == [0]
    assert validate_action([3, 4], sel) == [3, 4]


def test_validate_action_rejects_out_of_range_and_shape():
    sel = _FakeSelect(3, 1, 1)
    with pytest.raises(IllegalActionError):
        validate_action([5], sel)          # out of range
    with pytest.raises(IllegalActionError):
        validate_action([0, 1], sel)       # too many (> maxCount)
    with pytest.raises(IllegalActionError):
        validate_action([0, 0], _FakeSelect(3, 2, 2))  # duplicate
    with pytest.raises(IllegalActionError):
        validate_action("nope", sel)       # wrong type
    assert validate_action([], None) == []  # no selection pending


# --- finish() exactly once ---------------------------------------------------

def test_finish_once_on_normal_completion(deck, count_finish):
    result = play_match(deck, deck, [RandomAgent(seed=1), RandomAgent(seed=2)])
    assert isinstance(result, MatchResult)
    assert count_finish["n"] == 1


def test_finish_once_on_exception(deck, count_finish):
    env = Environment()
    with pytest.raises(RuntimeError, match="boom"):
        with env:
            env.start(deck, deck)
            raise RuntimeError("boom")
    assert count_finish["n"] == 1
    assert env._finished is True


def test_explicit_finish_then_context_exit_frees_once(deck, count_finish):
    with Environment() as env:
        env.start(deck, deck)
        env.finish()            # explicit
        # context-manager __exit__ will call finish() again -> must be a no-op
    assert count_finish["n"] == 1


# --- global-state confinement / single-active guard --------------------------

def test_second_environment_rejected_while_battle_live(deck):
    with Environment() as env1:
        env1.start(deck, deck)
        env2 = Environment()
        with pytest.raises(EngineError, match="one battle per process"):
            env2.start(deck, deck)
    # after env1 is finished a new battle can start again
    with Environment() as env3:
        env3.start(deck, deck)
        assert env3.observation.select is not None


def test_cannot_start_same_environment_twice(deck):
    with Environment() as env:
        env.start(deck, deck)
        with pytest.raises(EngineError, match="already started"):
            env.start(deck, deck)


def test_step_after_finish_raises(deck):
    env = Environment()
    with env:
        env.start(deck, deck)
    with pytest.raises(EngineError):
        env.step([0])


# --- legal moves from engine enumeration -------------------------------------

def test_select_is_the_legal_move_source(deck):
    with Environment() as env:
        obs = env.start(deck, deck)
        # The environment exposes exactly the engine's selection.
        assert env.select is obs.select
        assert env.select is not None
        assert len(env.select.option) >= 1


# --- Agent Protocol: two interchangeable implementations ---------------------

def test_random_and_first_agents_are_protocol_instances():
    assert isinstance(RandomAgent(seed=0), Agent)
    assert isinstance(FirstOptionAgent(), Agent)
    assert isinstance(SubmissionAgent(lambda o: []), Agent)


def test_two_agent_types_are_swappable(deck):
    # Same play_match Protocol, two different agent classes.
    r1 = play_match(deck, deck, [RandomAgent(seed=7), FirstOptionAgent()])
    r2 = play_match(deck, deck, [FirstOptionAgent(), RandomAgent(seed=7)])
    for r in (r1, r2):
        assert isinstance(r, MatchResult)
        assert r.winner in (0, 1, None)


def test_lifecycle_hooks_fire(deck):
    events = []

    class HookAgent(RandomAgent):
        def on_match_start(self, seat):
            events.append(("start", seat))

        def on_match_end(self, result):
            events.append(("end", result.reason))

    play_match(deck, deck, [HookAgent(seed=1), HookAgent(seed=2)])
    starts = [e for e in events if e[0] == "start"]
    ends = [e for e in events if e[0] == "end"]
    assert sorted(s[1] for s in starts) == [0, 1]
    assert len(ends) == 2


# --- structured faults: illegal / exception / timeout ------------------------

class _IllegalAgent(RandomAgent):
    def act(self, obs):
        return [10 ** 6]  # always out of range


class _RaisingAgent(RandomAgent):
    def act(self, obs):
        raise ValueError("agent blew up")


class _SlowAgent(RandomAgent):
    def act(self, obs):
        import time
        time.sleep(0.5)
        return super().act(obs)


def test_illegal_move_is_structured_loss(deck):
    result = play_match(deck, deck, [_IllegalAgent(), _IllegalAgent()])
    assert result.reason is EndReason.ILLEGAL_MOVE
    assert result.is_fault
    assert result.faulted_player in (0, 1)
    assert result.winner == 1 - result.faulted_player


def test_agent_exception_is_structured_loss(deck):
    result = play_match(deck, deck, [_RaisingAgent(), _RaisingAgent()])
    assert result.reason is EndReason.AGENT_EXCEPTION
    assert result.is_fault
    assert result.winner == 1 - result.faulted_player


def test_timeout_is_structured_loss(deck):
    result = play_match(
        deck, deck, [_SlowAgent(seed=1), _SlowAgent(seed=2)],
        per_move_timeout=0.05,
    )
    assert result.reason is EndReason.TIMEOUT
    assert result.is_fault
    assert result.winner == 1 - result.faulted_player


def test_fault_still_frees_native_resources(deck, count_finish):
    play_match(deck, deck, [_RaisingAgent(), _RaisingAgent()])
    assert count_finish["n"] == 1


# --- legacy run_match compat -------------------------------------------------

def test_run_match_compat_path(deck):
    import eval.run_match as run_match

    random.seed(42)
    winner, steps = run_match.run(deck, deck)
    assert winner in (-1, 0, 1)
    assert steps >= 1
    # legacy free-function agent still present and usable
    assert callable(run_match.random_agent)


def test_submission_agent_wraps_bare_callable(deck):
    calls = {"n": 0}

    def bare_agent(obs_dict):
        calls["n"] += 1
        ra = RandomAgent(seed=3)
        return ra.act(obs_dict)

    result = play_match(deck, deck, [SubmissionAgent(bare_agent), RandomAgent(seed=4)])
    assert isinstance(result, MatchResult)
    assert calls["n"] >= 1
