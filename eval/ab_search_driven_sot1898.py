"""SOT-1898 in-repo A/B: search-driven ume vs the committed champion.

Cheap, self-contained head-to-head on the real cabt engine (mirror deck,
seat-swapped) that confirms the acceptance-adjacent facts before the expensive
cross-agent league gate: fault 0 on the search-driven side, per-decision search
time inside the Kaggle budget, and a first strength signal vs the champion.

Run from the repo root with the venv python:
    venv/bin/python eval/ab_search_driven_sot1898.py --n 12 --seed 189800
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.harness import HarnessAgent
from agents.mcts import MCTSConfig
from agents.runtime_profile import load_runtime_profile
from eval.match import play_match

CHAMPION = ROOT / "agents" / "runtime_profile.json"
SEARCH = ROOT / "agents" / "runtime_profile_search.json"


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _make_agent(profile_path: Path, seed: int) -> HarnessAgent:
    prof = load_runtime_profile(str(profile_path))
    return HarnessAgent(
        seed=seed,
        policy_path=str(ROOT / "data" / "policy.json"),
        temperature=prof.policy_temperature,
        mcts=True,
        mcts_config=MCTSConfig(**{**prof.mcts.__dict__, "deck_path": str(ROOT / "deck.csv")}),
        harness_config=prof.harness,
    )


def read_deck() -> list[int]:
    rows = (ROOT / "deck.csv").read_text().split("\n")
    return [int(rows[i]) for i in range(60)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="games (seat-swapped pairs)")
    ap.add_argument("--seed", type=int, default=189800)
    ap.add_argument("--out", default="eval/runtime_promotion/sot-1898/ab_screen.json")
    args = ap.parse_args()

    deck = read_deck()
    search_wins = champ_wins = draws = 0
    search_faults = champ_faults = 0
    search_ms_max = 0.0
    search_act = search_decisions = 0

    for g in range(args.n):
        # Seat-swap each game so first-player advantage cancels.
        search_seat = g % 2
        sd = _make_agent(SEARCH, seed=args.seed + g)
        ch = _make_agent(CHAMPION, seed=args.seed + 10_000 + g)
        agents = [sd, ch] if search_seat == 0 else [ch, sd]
        result = play_match(deck, deck, agents)

        if result.is_fault:
            if result.faulted_player == search_seat:
                search_faults += 1
            else:
                champ_faults += 1
        if result.is_draw:
            draws += 1
        elif result.winner == search_seat:
            search_wins += 1
        elif result.winner is not None:
            champ_wins += 1

        st = sd.mcts_stats
        if st is not None:
            search_ms_max = max(search_ms_max, st.elapsed_ms_max)
            search_act += st.activations
            search_decisions += st.decisions
        print(f"  game {g:02d} seat={search_seat} winner={result.winner} "
              f"fault={result.faulted_player} steps={result.steps} "
              f"search_ms_max={search_ms_max:.0f}", flush=True)

    decided = search_wins + champ_wins
    wr = search_wins / decided if decided else 0.0
    lo, hi = _wilson(search_wins, decided)
    report = {
        "issue": "SOT-1898",
        "n_games": args.n,
        "seed": args.seed,
        "search_wins": search_wins,
        "champion_wins": champ_wins,
        "draws": draws,
        "decided": decided,
        "search_win_rate": wr,
        "search_win_rate_ci95": [lo, hi],
        "faults": {"search_driven": search_faults, "champion": champ_faults},
        "search_activation_rate": (search_act / search_decisions) if search_decisions else 0.0,
        "search_ms_max": search_ms_max,
        "per_move_budget_ms": 5000.0,
        "within_budget": search_ms_max <= 5000.0,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n=== SOT-1898 A/B (search-driven vs champion) ===")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
