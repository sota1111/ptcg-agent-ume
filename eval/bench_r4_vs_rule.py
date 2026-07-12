"""Evaluation benchmark: the R4 board-eval EvalAgent vs the R3 RuleAgent (SOT-1649).

R4 (:class:`agents.eval_agent.EvalAgent`) drives a one-ply lookahead with the unified
board evaluation :func:`agents.board_eval.score_state`. This benchmark quantifies it by
playing it **head to head against the R3 champion** (:class:`agents.rule_agent.RuleAgent`,
version 3) in an N≥200 side-swapped :func:`eval.arena.run_arena` and reporting exactly what
the issue's 受け入れ条件 ask for:

* **EvalAgent win rate + Wilson 95% CI** (R4 = agent A) over all matches (draws counted as
  non-wins, matching :func:`eval.arena.wilson_ci`). 受け入れ条件①: **CI lower bound
  strictly > 0.50** to promote R4 over the R3 champion; otherwise R3 stays champion and R4
  is reported honestly but not shipped as the champion.
* **Zero crashes / illegal moves / timeouts (違法出力0).** The arena scores an agent
  exception / illegal move / per-move-timeout as a *fault*; the gate requires
  ``a_faults == 0`` (受け入れ条件③). EvalAgent also self-reports its one-ply
  search bookkeeping (attempts / chosen / rule-fallbacks / leaked sessions), aggregated
  across every match, so a search-state leak can only surface, never hide.
* **Per-decision latency (mean / p99)** for the R4 side — the Kaggle time-budget watch
  value (受け入れ条件③: p99 が提出予算内).

Usage:
    venv/bin/python eval/bench_r4_vs_rule.py [--n 200] [--seed 0]
        [--search-budget 0.1] [--max-candidates 12] [--per-move-timeout 5.0]
        [--disable COMPONENT] [--json report.json]

``--disable`` zeroes one board-eval component (repeatable) so this same harness backs the
ablation runner. Exit code 0 iff the promotion gate passes (N≥200, zero faults/leaks, **and**
CI lower bound > 0.50). A valid run that does not clear 0.50 exits 1 — the expected, honest
"keep RuleAgent (R3) as champion" outcome, not an error. Run from the repo root (needs the
gitignored ``cg/`` engine + ``deck.csv``).
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

from agents.board_eval import COMPONENTS, DEFAULT_WEIGHTS, EvalWeights  # noqa: E402
from agents.eval_agent import EvalAgent          # noqa: E402
from agents.rule_agent import RuleAgent          # noqa: E402
from eval.arena import run_arena, _load_deck      # noqa: E402
from eval.trace import RecordLevel                # noqa: E402

R4_LABEL = "eval"
RULE_LABEL = "rule"
PROMOTE_THRESHOLD = 0.50


def run_bench(
    n: int,
    seed: int,
    search_budget_s: float,
    max_candidates: int,
    per_move_timeout: float,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    disabled: frozenset[str] = frozenset(),
    deck_path: str = "deck.csv",
    record_traces: bool = False,
) -> dict:
    """Run EvalAgent (A, R4) vs RuleAgent (B, R3) paired and return the aggregated result."""
    deck = _load_deck(deck_path)
    built: list[EvalAgent] = []

    def make_eval(s: int) -> EvalAgent:
        a = EvalAgent(
            seed=s, deck_path=deck_path,
            search_budget_s=search_budget_s, max_candidates=max_candidates,
            weights=weights, disabled=frozenset(disabled),
        )
        built.append(a)
        return a

    report = run_arena(
        make_eval,
        lambda s: RuleAgent(seed=s),
        deck0=deck,
        n_matches=n,
        side_swap=True,
        agent_seed=seed,
        label_a=R4_LABEL,
        label_b=RULE_LABEL,
        per_move_timeout=per_move_timeout,
        record_traces=record_traces,
        trace_level=RecordLevel.RESULT,
        run_label=f"bench_r4_vs_rule_n{n}",
    )

    wr = report.win_rates
    ci_low, ci_high = wr["a_win_rate_ci95"]
    # Aggregate the EvalAgent's own per-match one-ply search bookkeeping (leak accounting).
    agg = {"attempts": 0, "chosen": 0, "fallbacks": 0, "leaks": 0}
    for a in built:
        for k in agg:
            agg[k] += a.search_stats.get(k, 0)

    faults = report.safety["a_faults"] + report.safety["b_faults"]
    passed = (
        report.totals["n"] >= 200
        and faults == 0
        and agg["leaks"] == 0
        and ci_low > PROMOTE_THRESHOLD
    )
    return {
        "n": report.totals["n"],
        "eval_wins": report.totals["a_wins"],
        "rule_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "undecided": report.totals["undecided"],
        "eval_win_rate": wr["a_win_rate"],
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "a_faults": report.safety["a_faults"],
        "b_faults": report.safety["b_faults"],
        "a_fault_categories": report.safety["a_fault_categories"],
        "eval_latency": report.latency.get(R4_LABEL, {}),
        "search_stats": agg,
        "disabled": sorted(disabled),
        "run_dir": report.run_dir,
        "promote_threshold": PROMOTE_THRESHOLD,
        # Promotion gate: enough games, zero faults/leaks, AND CI lower bound > 0.50.
        "passed": passed,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EvalAgent(R4)-vs-RuleAgent(R3) promotion benchmark (win rate + Wilson CI + safety)."
    )
    p.add_argument("--n", type=int, default=200, help="matches (>=200 required by the gate)")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--search-budget", type=float, default=0.1, help="search per-decision time budget (s)")
    p.add_argument("--max-candidates", type=int, default=12, help="search per-decision candidate cap")
    p.add_argument("--per-move-timeout", type=float, default=5.0, help="hard per-move timeout (s)")
    p.add_argument("--disable", action="append", default=[], choices=COMPONENTS,
                   help="zero a board-eval component (repeatable; for ablation)")
    p.add_argument("--json", default=None, help="also write the raw JSON result to this path")
    return p.parse_args(argv)


def _fmt_lat(d: dict) -> str:
    if not d:
        return "n/a"
    return (f"n={d.get('n_decisions', 0)} mean={d.get('mean_ms', 0):.2f}ms "
            f"p99={d.get('p99_ms', 0):.2f}ms max={d.get('max_ms', 0):.2f}ms")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    disabled = frozenset(args.disable)
    r = run_bench(
        args.n, args.seed, args.search_budget, args.max_candidates, args.per_move_timeout,
        disabled=disabled,
    )
    dis = f" disabled={r['disabled']}" if r["disabled"] else ""
    print(
        f"BENCH eval(R4) vs rule(R3): n={r['n']}{dis} "
        f"W/D/L(eval)={r['eval_wins']}/{r['draws']}/{r['rule_wins']} undecided={r['undecided']} "
        f"eval_win_rate={r['eval_win_rate']:.3f} "
        f"Wilson95=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] "
        f"faults(eval/rule)={r['a_faults']}/{r['b_faults']}"
    )
    print(f"EVAL latency/decision: {_fmt_lat(r['eval_latency'])}")
    print(f"EVAL search stats (aggregate): {r['search_stats']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    if r["n"] < 200:
        print("GATE INVALID: need >= 200 games")
        return 2
    verdict = "PASS (promote R4)" if r["passed"] else "FAIL (keep RuleAgent R3 champion)"
    print(
        f"PROMOTION GATE {verdict}: N={r['n']}, faults={r['a_faults'] + r['b_faults']}, "
        f"leaks={r['search_stats']['leaks']}, CI_low={r['ci95_low']:.4f} vs threshold "
        f"{PROMOTE_THRESHOLD} (promote iff CI lower bound > {PROMOTE_THRESHOLD})"
    )
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
