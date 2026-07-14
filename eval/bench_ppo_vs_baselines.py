"""Evaluation benchmark: the PPO agent vs the Random / Rule baselines (SOT-1689).

Measures :class:`agents.ppo_agent.PPOAgent` (the ``data/policy.json`` artifact)
head-to-head against **both** baselines the issue's 受け入れ条件 name — the
random baseline and the RuleAgent champion — in N≥200 side-swapped
:func:`eval.arena.run_arena` runs, and reports the PPO win rate + **Wilson 95%
CI** per opponent.

Unlike :mod:`eval.bench_r4_vs_rule` this is a *measurement*, not a promotion
gate: SOT-1689 explicitly does not require beating the champion yet
(champion超えは必須としない). The pass criterion is safety + evidence only:

* the policy artifact loads (the committed ``policy.json`` is usable),
* **zero faults on the PPO side** in every matchup (違法出力0 / fault 0),
* both matchups ran the requested N with the CI recorded.

Usage::

    venv/bin/python eval/bench_ppo_vs_baselines.py [--n 200] [--seed 0]
        [--policy data/policy.json] [--json report.json]

Exit code 0 iff the safety criteria hold (whatever the win rates say). Run from
the repo root (needs the gitignored ``cg/`` engine + ``deck.csv``).
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

from agents.ppo_agent import DEFAULT_POLICY_PATH, PPOAgent  # noqa: E402
from agents.random_agent import RandomAgent                 # noqa: E402
from agents.rule_agent import RuleAgent                     # noqa: E402
from eval.arena import run_arena, _load_deck                # noqa: E402
from eval.trace import RecordLevel                          # noqa: E402

PPO_LABEL = "ppo"


def run_matchup(
    opponent_label: str,
    n: int,
    seed: int,
    policy_path: str,
    per_move_timeout: float,
    deck_path: str = "deck.csv",
) -> dict:
    """PPO (A) vs one baseline (B), paired/side-swapped; returns the summary dict."""
    deck = _load_deck(deck_path)
    baselines = {
        "random": lambda s: RandomAgent(seed=s),
        "rule": lambda s: RuleAgent(seed=s),
    }

    def make_ppo(s: int) -> PPOAgent:
        return PPOAgent(seed=s, policy_path=policy_path)

    report = run_arena(
        make_ppo,
        baselines[opponent_label],
        deck0=deck,
        n_matches=n,
        side_swap=True,
        agent_seed=seed,
        label_a=PPO_LABEL,
        label_b=opponent_label,
        per_move_timeout=per_move_timeout,
        record_traces=False,
        trace_level=RecordLevel.RESULT,
        run_label=f"bench_ppo_vs_{opponent_label}_n{n}",
    )
    wr = report.win_rates
    ci_low, ci_high = wr["a_win_rate_ci95"]
    return {
        "opponent": opponent_label,
        "n": report.totals["n"],
        "ppo_wins": report.totals["a_wins"],
        "opponent_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "undecided": report.totals["undecided"],
        "ppo_win_rate": wr["a_win_rate"],
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ppo_faults": report.safety["a_faults"],
        "opponent_faults": report.safety["b_faults"],
        "ppo_fault_categories": report.safety["a_fault_categories"],
        "ppo_latency": report.latency.get(PPO_LABEL, {}),
        "run_dir": report.run_dir,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PPOAgent vs Random/Rule baselines: win rate + Wilson CI + safety."
    )
    p.add_argument("--n", type=int, default=200, help="matches per matchup")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="policy.json path")
    p.add_argument("--per-move-timeout", type=float, default=5.0, help="hard per-move timeout (s)")
    p.add_argument("--json", default=None, help="also write the raw JSON result to this path")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    probe = PPOAgent(seed=0, policy_path=args.policy)
    if not probe.policy_loaded:
        print(f"POLICY INVALID: {args.policy} did not load — train it first (train/ppo.py)")
        return 2

    results = {}
    for opponent in ("random", "rule"):
        r = run_matchup(opponent, args.n, args.seed, args.policy, args.per_move_timeout)
        results[opponent] = r
        print(
            f"BENCH ppo vs {opponent}: n={r['n']} "
            f"W/D/L(ppo)={r['ppo_wins']}/{r['draws']}/{r['opponent_wins']} "
            f"undecided={r['undecided']} ppo_win_rate={r['ppo_win_rate']:.3f} "
            f"Wilson95=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] "
            f"faults(ppo/opp)={r['ppo_faults']}/{r['opponent_faults']}"
        )

    passed = all(r["n"] >= args.n and r["ppo_faults"] == 0 for r in results.values())
    out = {"policy": args.policy, "n_per_matchup": args.n, "seed": args.seed,
           "results": results, "passed": passed}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    print(f"SAFETY GATE {'PASS' if passed else 'FAIL'}: ppo faults "
          f"{[r['ppo_faults'] for r in results.values()]} over {args.n}x2 matches "
          f"(champion promotion is out of scope for SOT-1689)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
