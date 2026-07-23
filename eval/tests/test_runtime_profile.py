import hashlib
import json
from pathlib import Path

from agents.runtime_profile import PROFILE_PATH, PROFILE_SCHEMA, load_runtime_profile


def test_promoted_profile_is_bound_to_runtime() -> None:
    profile = load_runtime_profile()
    assert profile.profile_id == "ume-high-variance-pressure-v1"
    assert profile.policy_temperature == 0.35
    assert profile.policy_temperature > 0.25
    assert profile.mcts.time_limit_s == 0.4
    assert profile.mcts.rollout_depth == 7
    assert profile.mcts.ucb_c == 1.25
    assert profile.mcts.ucb_c > 1.0
    assert profile.harness.top_alternatives == 4
    assert profile.per_move_timeout_s < profile.total_budget_s == 600
    assert profile.raw["evaluation"] == {
        "budget_hours": 8,
        "checkpoint_every_matchups": 1,
        "resume": True,
        "fixed_seed": 187500,
        "seat_swap": True,
    }

    payload = Path(PROFILE_PATH).read_bytes()
    assert profile.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert json.loads(payload)["schema"] == PROFILE_SCHEMA


def test_submission_defaults_to_promoted_core_strategy() -> None:
    source = Path("main.py").read_text(encoding="utf-8")
    assert 'load_runtime_profile()' in source
    assert 'PTCG_UME_MIGRATION_MODE", "core"' in source


def test_committed_champion_is_not_search_driven() -> None:
    """The committed default must stay the SOT-1690 critical-only champion."""
    profile = load_runtime_profile()
    assert profile.mcts.all_decision is False
    assert profile.mcts.match_search_budget_s == 0.0
    assert profile.mcts.rollout_temperature == 1.0


def test_search_driven_candidate_profile_parses() -> None:
    """SOT-1898 opt-in candidate profile loads with the search-driven knobs."""
    path = Path("agents/runtime_profile_search.json")
    profile = load_runtime_profile(str(path))
    assert profile.profile_id == "ume-search-driven-v1"
    assert profile.mcts.all_decision is True
    assert profile.mcts.match_search_budget_s == 300.0
    assert profile.mcts.rollout_temperature == 0.35
    # Adaptive per-decision cap still fits inside the per-move timeout.
    assert 0 < profile.mcts.time_limit_s <= profile.per_move_timeout_s
