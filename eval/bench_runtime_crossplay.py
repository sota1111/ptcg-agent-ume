"""Seat-reversed real-submission cross-play against the heterogeneous agents."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.harness import HarnessAgent
from agents.mcts import MCTSConfig
from agents.runtime_profile import load_runtime_profile
from eval.arena import _load_deck, run_arena
from eval.trace import RecordLevel

OPPONENTS = ("sol", "debate", "fable", "zero")


class ExternalSubmissionAgent:
    """One isolated Python process per match avoids cross-repository imports."""

    def __init__(self, repo: Path, name: str) -> None:
        self.repo, self.name, self.process = repo, name, None

    def on_match_start(self, _player_index: int) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / "eval" / "submission_worker.py"), str(self.repo)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=self.repo,
        )

    def act(self, obs: dict) -> list[int]:
        assert self.process and self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(obs) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["action"]

    def on_match_end(self, _result) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="even games per opponent")
    parser.add_argument("--seed", type=int, default=187500)
    parser.add_argument("--out", default="eval/runtime_promotion/sot-1875/crossplay.json")
    args = parser.parse_args()
    if args.n <= 0 or args.n % 2:
        parser.error("--n must be a positive even number")

    profile, deck = load_runtime_profile(), _load_deck("deck.csv")

    def candidate(seed: int) -> HarnessAgent:
        cfg = MCTSConfig(**{**profile.mcts.__dict__, "deck_path": "deck.csv"})
        return HarnessAgent(seed=seed, policy_path="data/policy.json",
                            temperature=profile.policy_temperature, mcts=True,
                            mcts_config=cfg, harness_config=profile.harness)

    path = ROOT / args.out
    results = {}
    if profile.raw["evaluation"]["resume"] and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("seed") == args.seed:
            results = previous.get("results", {})
    safety = True
    deadline = time.monotonic() + float(profile.raw["evaluation"]["budget_hours"]) * 3600
    for index, name in enumerate(OPPONENTS):
        if name in results:
            row = results[name]
            safety &= (
                row["faults"] == 0 and row["opponent_faults"] == 0
                and row["unfinished"] == 0
            )
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("cross-play exceeded the configured 8-hour budget")
        repo = Path(f"/workspaces/ptcg-agent-{name}")
        report = run_arena(
            candidate, lambda _seed, r=repo, n=name: ExternalSubmissionAgent(r, n),
            deck0=deck, n_matches=args.n, side_swap=True,
            agent_seed=args.seed + index * 1000, per_move_timeout=profile.per_move_timeout_s,
            record_traces=False, trace_level=RecordLevel.RESULT,
            label_a="ume-hardened", label_b=name,
            run_label=f"sot1875_{name}_n{args.n}",
        )
        row = {
            "games": report.totals["n"], "wins": report.totals["a_wins"],
            "losses": report.totals["b_wins"], "draws": report.totals["draws"],
            "unfinished": report.totals["undecided"],
            "faults": report.safety["a_faults"],
            "opponent_faults": report.safety["b_faults"],
            "win_rate": report.win_rates["a_win_rate"],
        }
        results[name] = row
        safety &= (
            row["faults"] == 0 and row["opponent_faults"] == 0
            and row["unfinished"] == 0
        )
        checkpoint = {"schema": "ume-runtime-crossplay/v1", "issue": "SOT-1875",
                      "seed": args.seed, "seat_swap": True, "results": results,
                      "complete": False, "safety_pass": safety}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")

    output = {"schema": "ume-runtime-crossplay/v1", "issue": "SOT-1875",
              "seed": args.seed, "seat_swap": True, "results": results,
              "complete": len(results) == len(OPPONENTS), "safety_pass": safety}
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if safety else 2


if __name__ == "__main__":
    raise SystemExit(main())
