"""Real-runtime promotion bench for the hardened SOT-1875 Ume profile.

Runs fixed agent seeds with seat reversal against the pre-promotion runtime and
the hard RuleAgent opponent.  The engine itself has no seed API, so the report
records that reproducibility limit together with every runtime/deck/policy hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.harness import HarnessAgent
from agents.mcts import MCTSConfig
from agents.rule_agent import RuleAgent
from agents.runtime_profile import PROFILE_PATH, load_runtime_profile
from eval.arena import _load_deck, run_arena
from eval.trace import RecordLevel, deck_hash

PRIOR_HARD_RUNTIME_WIN_RATE = 0.25

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(report) -> dict:
    return {
        "games": report.totals["n"],
        "candidate_wins": report.totals["a_wins"],
        "opponent_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "unfinished": report.totals["undecided"],
        "candidate_win_rate": report.win_rates["a_win_rate"],
        "wilson95": report.win_rates["a_win_rate_ci95"],
        "candidate_faults": report.safety["a_faults"],
        "opponent_faults": report.safety["b_faults"],
        "candidate_fault_categories": report.safety["a_fault_categories"],
        "candidate_latency": report.latency["candidate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="games per matchup; must be even")
    parser.add_argument("--seed", type=int, default=187500)
    parser.add_argument("--out", default="eval/runtime_promotion/sot-1875/report.json")
    args = parser.parse_args()
    if args.n <= 0 or args.n % 2:
        parser.error("--n must be a positive even number for exact seat reversal")

    os.chdir(ROOT)
    profile = load_runtime_profile()
    deck = _load_deck("deck.csv")
    policy = ROOT / "data" / "policy.json"

    def candidate(seed: int) -> HarnessAgent:
        cfg = MCTSConfig(**{**profile.mcts.__dict__, "deck_path": "deck.csv"})
        return HarnessAgent(seed=seed, policy_path=str(policy),
                            temperature=profile.policy_temperature, mcts=True,
                            mcts_config=cfg, harness_config=profile.harness)

    def legacy(seed: int) -> HarnessAgent:
        return HarnessAgent(seed=seed, policy_path=str(policy), temperature=0.25,
                            mcts=True,
                            mcts_config=MCTSConfig(time_limit_s=0.4, deck_path="deck.csv"))

    started = time.monotonic()
    common = dict(deck0=deck, n_matches=args.n, side_swap=True,
                  agent_seed=args.seed, per_move_timeout=profile.per_move_timeout_s,
                  record_traces=False, trace_level=RecordLevel.RESULT)
    ab = run_arena(candidate, legacy, label_a="candidate", label_b="legacy",
                   run_label=f"sot1875_ab_n{args.n}_s{args.seed}", **common)
    hard = run_arena(candidate, lambda seed: RuleAgent(seed=seed),
                     label_a="candidate", label_b="rule",
                     run_label=f"sot1875_rule_n{args.n}_s{args.seed + 100000}",
                     **{**common, "agent_seed": args.seed + 100000})
    elapsed = time.monotonic() - started

    ab_result, hard_result = result(ab), result(hard)
    safety = all(
        row["candidate_faults"] == 0 and row["unfinished"] == 0
        and row["candidate_latency"]["max_ms"] <= profile.per_move_timeout_s * 1000
        for row in (ab_result, hard_result)
    )
    legacy_gate = ab_result["wilson95"][0] > 0.5
    league_improvement = hard_result["candidate_win_rate"] > PRIOR_HARD_RUNTIME_WIN_RATE
    promotion = (legacy_gate or league_improvement) and safety
    report = {
        "schema": "ume-runtime-promotion/v1",
        "issue": "SOT-1875",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "profile_id": profile.profile_id,
        "profile_sha256": profile.artifact_sha256,
        "deck_sha256": deck_hash(deck),
        "policy_sha256": sha256(policy),
        "seed": args.seed,
        "seed_count": args.n // 2,
        "seat_swap": True,
        "engine_seed_support": False,
        "runtime_budget_s": profile.total_budget_s,
        "per_move_timeout_s": profile.per_move_timeout_s,
        "elapsed_s": elapsed,
        "matchups": {"legacy": ab_result, "rule": hard_result},
        "safety": {
            "faults": ab_result["candidate_faults"] + hard_result["candidate_faults"],
            "unfinished": ab_result["unfinished"] + hard_result["unfinished"],
            "illegal_actions": sum(
                row["candidate_fault_categories"].get("illegal_move", 0)
                for row in (ab_result, hard_result)
            ),
            "pass": safety,
        },
        "promotion_gate": {
            "criterion": (
                "legacy Wilson 95% lower bound > 0.5 OR hard-runtime league "
                "win rate improves over the prior 0.25; and zero safety faults"
            ),
            "legacy_wilson_pass": legacy_gate,
            "prior_hard_runtime_win_rate": PRIOR_HARD_RUNTIME_WIN_RATE,
            "hard_runtime_improvement": (
                hard_result["candidate_win_rate"] - PRIOR_HARD_RUNTIME_WIN_RATE
            ),
            "league_improvement_pass": league_improvement,
            "pass": promotion,
        },
        "high_variance": {
            "legacy_temperature": 0.25,
            "candidate_temperature": profile.policy_temperature,
            "temperature_ratio": profile.policy_temperature / 0.25,
            "exploration_constant": profile.mcts.ucb_c,
            "pass": profile.policy_temperature > 0.25 and profile.mcts.ucb_c > 1.0,
        },
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if promotion else 2


if __name__ == "__main__":
    raise SystemExit(main())
