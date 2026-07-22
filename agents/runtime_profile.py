"""Versioned runtime binding for Ume's hardened aggressive profile (SOT-1875)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from .harness import HarnessConfig
from .mcts import MCTSConfig

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "runtime_profile.json")
PROFILE_SCHEMA = "ume-runtime-profile/v1"


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    policy_temperature: float
    mcts: MCTSConfig
    harness: HarnessConfig
    total_budget_s: float
    per_move_timeout_s: float
    artifact_sha256: str
    raw: dict


def load_runtime_profile(path: str = PROFILE_PATH) -> RuntimeProfile:
    """Load and validate the committed profile; invalid values fail at import time."""
    with open(path, "rb") as fh:
        payload = fh.read()
    value = json.loads(payload)
    if value.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"unsupported runtime profile schema: {value.get('schema')!r}")

    temperature = float(value["policy_temperature"])
    search = value["search"]
    harness = value["harness"]
    runtime = value["runtime"]
    evaluation = value["evaluation"]
    if not 0 < temperature <= 1:
        raise ValueError("policy_temperature must be in (0, 1]")
    if not 0 < float(search["time_limit_s"]) <= float(runtime["per_move_timeout_s"]):
        raise ValueError("search time limit must fit within the per-move timeout")
    if not 0 < float(runtime["per_move_timeout_s"]) < float(runtime["total_budget_s"]):
        raise ValueError("runtime budgets are inconsistent")
    if runtime.get("illegal_action_fallback") != "highest-value-legal":
        raise ValueError("runtime profile must retain the legal-action fallback")
    if float(evaluation["budget_hours"]) > 8 or not evaluation.get("resume"):
        raise ValueError("evaluation must retain the 8-hour checkpoint/resume contract")
    if not evaluation.get("seat_swap") or int(evaluation["checkpoint_every_matchups"]) < 1:
        raise ValueError("evaluation must use seat swap and periodic checkpoints")

    return RuntimeProfile(
        profile_id=str(value["profile_id"]),
        policy_temperature=temperature,
        mcts=MCTSConfig(
            time_limit_s=float(search["time_limit_s"]),
            rollout_depth=int(search["rollout_depth"]),
            ucb_c=float(search["ucb_c"]),
            deviate_margin=float(search["deviate_margin"]),
            n_determinizations=int(search["n_determinizations"]),
            max_candidates=int(search["max_candidates"]),
        ),
        harness=HarnessConfig(
            top_alternatives=int(harness["top_alternatives"]),
            coverage_weight=float(harness["coverage_weight"]),
            act_bonus=float(harness["act_bonus"]),
        ),
        total_budget_s=float(runtime["total_budget_s"]),
        per_move_timeout_s=float(runtime["per_move_timeout_s"]),
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        raw=value,
    )
