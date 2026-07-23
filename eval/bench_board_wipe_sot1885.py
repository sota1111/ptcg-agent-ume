"""Small-N screen and conditional large-N board-survival confirmation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.harness import HarnessAgent, HarnessConfig
from eval.arena import run_arena
from eval.board_wipe import BoardWipeStats, BoardWipeTrackingAgent


def run_stage(n: int, seed: int, label: str) -> dict:
    candidate_stats, champion_stats = BoardWipeStats(), BoardWipeStats()

    def candidate(s):
        return BoardWipeTrackingAgent(
            HarnessAgent(
                seed=s,
                mcts=True,
                harness_config=HarnessConfig(board_survival_weight=2.0),
            ),
            candidate_stats,
        )

    def champion(s):
        return BoardWipeTrackingAgent(HarnessAgent(seed=s, mcts=True), champion_stats)

    deck = [int(line) for line in Path("deck.csv").read_text().splitlines()[:60]]
    started = time.perf_counter()
    arena = run_arena(
        candidate,
        champion,
        deck0=deck,
        n_matches=n,
        side_swap=True,
        agent_seed=seed,
        label_a="candidate",
        label_b="champion",
        run_label=f"sot1885_{label}",
        out_dir="eval/board_wipe_runs",
    )
    elapsed = time.perf_counter() - started
    ci = arena.win_rates["a_win_rate_ci95"]
    faults = arena.safety["a_faults"] + arena.safety["b_faults"]
    return {
        "stage": label,
        "arena": arena.to_dict(),
        "kpi": {
            "candidate": candidate_stats.report(),
            "champion": champion_stats.report(),
        },
        "elapsed_s": elapsed,
        "sims_per_sec": n / elapsed if elapsed else 0.0,
        "promotion_screen_pass": ci[0] > 0.5 and faults == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-n", type=int, default=40)
    parser.add_argument("--large-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1885)
    parser.add_argument("--out", default="eval/sot1885_board_wipe.json")
    args = parser.parse_args()
    small = run_stage(args.small_n, args.seed, "small_n")
    report = {
        "schema": "ume-board-wipe-ab/v1",
        "small_n": small,
        "large_n": None,
        "champion_updated": False,
        "decision": "retain_champion",
    }
    if small["promotion_screen_pass"]:
        large = run_stage(args.large_n, args.seed + 1, "large_n")
        report["large_n"] = large
        report["champion_updated"] = bool(large["promotion_screen_pass"])
        report["decision"] = (
            "promote_candidate" if report["champion_updated"] else "retain_champion"
        )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
