"""25-deck mirror rotation safety bench for the submission agent (SOT-1695).

Plays the final :class:`agents.harness.HarnessAgent` configuration (the exact
``main.py`` setup: PPO policy + critical-position MCTS at the 0.4 s cap +
candidate harness) as a **mirror** on every ``*.csv`` deck under ``--deck-dir``
(default ``decks/initial``, the 25 tournament decks), ``--games-per-deck``
matches each, and verifies the legality gate holds on every archetype:

* **fault 0** on both sides (no illegal move / timeout / agent exception) —
  the features are card-ID independent, so the harness must stay safe on decks
  it never trained on;
* **invalid candidates never play**: the harness validates every candidate, so
  ``invalid_candidates`` in its stats is informational, faults are the gate.

Exit code 0 iff every deck completed its matches with zero faults::

    venv/bin/python eval/bench_deck_rotation.py --games-per-deck 2 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.harness import HarnessAgent, HarnessStats  # noqa: E402
from agents.mcts import MCTSConfig, MCTSStats          # noqa: E402
from agents.ppo_agent import DEFAULT_POLICY_PATH       # noqa: E402
from eval.match import play_match                      # noqa: E402
from eval.selfplay import load_deck_dir                # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="HarnessAgent mirror on every rotation deck: fault/illegal-move gate."
    )
    p.add_argument("--deck-dir", default="decks/initial", help="rotation deck directory")
    p.add_argument("--games-per-deck", type=int, default=2, help="mirror matches per deck")
    p.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="policy.json path")
    p.add_argument("--time-limit", type=float, default=0.4,
                   help="MCTS per-decision search cap (s) — keep = main.py's")
    p.add_argument("--per-move-timeout", type=float, default=5.0,
                   help="hard per-move timeout (s)")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--json", default=None, help="also write the raw JSON result here")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    probe = HarnessAgent(seed=0, policy_path=args.policy, mcts=False)
    if not probe.policy_loaded:
        print(f"POLICY INVALID: {args.policy} did not load — train it first (train/ppo.py)")
        return 2

    decks = load_deck_dir(args.deck_dir)
    cfg = MCTSConfig(time_limit_s=args.time_limit)
    harness_pooled, mcts_pooled = HarnessStats(), MCTSStats()
    per_deck: dict[str, dict] = {}
    total_faults = 0
    game_seed = args.seed

    for name, deck in decks:
        tally = {"games": 0, "faults": 0, "fault_reasons": []}
        for _ in range(args.games_per_deck):
            agents = [
                HarnessAgent(seed=game_seed + s, policy_path=args.policy,
                             mcts=True, mcts_config=cfg)
                for s in (0, 1)
            ]
            game_seed += 2
            result = play_match(deck, deck, agents,
                                per_move_timeout=args.per_move_timeout)
            tally["games"] += 1
            if result.is_fault:
                tally["faults"] += 1
                tally["fault_reasons"].append(result.reason.value)
            for a in agents:
                harness_pooled.merge(a.harness_stats)
                if a.mcts_stats is not None:
                    mcts_pooled.merge(a.mcts_stats)
        total_faults += tally["faults"]
        per_deck[name] = tally
        print(f"DECK {name}: games={tally['games']} faults={tally['faults']}")

    n_games = sum(t["games"] for t in per_deck.values())
    passed = total_faults == 0 and n_games == len(decks) * args.games_per_deck
    result = {
        "policy": args.policy,
        "deck_dir": args.deck_dir,
        "decks": len(decks),
        "games_per_deck": args.games_per_deck,
        "games": n_games,
        "faults": total_faults,
        "per_deck": per_deck,
        "harness_stats": harness_pooled.report(),
        "mcts_stats": mcts_pooled.report(),
        "passed": passed,
    }
    h = result["harness_stats"]
    print(
        f"ROTATION GATE {'PASS' if passed else 'FAIL'}: {len(decks)} decks x "
        f"{args.games_per_deck} mirror games, faults={total_faults}, "
        f"decisions={h['decisions']} invalid_candidates={h['invalid_candidates']} "
        f"decided_by={h['decided_by']}"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
