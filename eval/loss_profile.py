"""Profile HarnessAgent losses by cabt's terminal reason (SOT-1706).

The arena result schema intentionally keeps only transport-level end reasons;
the engine's granular reason (prize/deck/no-active) is preserved in each RESULT
trace.  This command joins those two durable artifacts and emits a compact JSON
profile suitable for before/after comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from agents.harness import HarnessAgent  # noqa: E402
from agents.mcts import MCTSConfig  # noqa: E402
from agents.rule_agent import RuleAgent  # noqa: E402
from eval.arena import _load_deck, run_arena  # noqa: E402
from eval.trace import RecordLevel, load_trace  # noqa: E402

REASONS = {1: "prize_out", 2: "deck_out", 3: "no_active", 4: "card_effect"}


def run(args: argparse.Namespace) -> dict:
    deck = _load_deck(args.deck)
    report = run_arena(
        lambda seed: HarnessAgent(
            seed=seed,
            temperature=args.temperature,
            mcts=True,
            mcts_config=MCTSConfig(time_limit_s=args.time_limit, deck_path=args.deck),
        ),
        lambda seed: RuleAgent(seed=seed),
        deck0=deck,
        n_matches=args.n,
        side_swap=True,
        agent_seed=args.seed,
        label_a="harness",
        label_b="rule",
        per_move_timeout=args.per_move_timeout,
        record_traces=True,
        trace_level=RecordLevel.RESULT,
        run_label=f"loss_profile_rule_n{args.n}_s{args.seed}",
    )
    counts: Counter[str] = Counter()
    for record in report.records:
        if not record.b_won:  # only HarnessAgent losses
            continue
        trace = load_trace(record.trace_path)
        code = (trace.result or {}).get("reason")
        counts[REASONS.get(code, f"unknown_{code}")] += 1
    return {
        "opponent": "rule",
        "n": report.totals["n"],
        "losses": report.totals["b_wins"],
        "loss_reasons": dict(sorted(counts.items())),
        "faults": report.safety["a_faults"],
        "run_dir": report.run_dir,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--time-limit", type=float, default=0.4)
    p.add_argument("--per-move-timeout", type=float, default=5.0)
    p.add_argument("--json", required=True)
    args = p.parse_args()
    result = run(args)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["faults"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
