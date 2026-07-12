"""Evaluation benchmark: the one-ply SearchAgent vs the RuleAgent (SOT-1650, R5).

The 案B search line (:class:`agents.search_agent.SearchAgent`) reuses the rule score
as a one-ply lookahead evaluation. This benchmark quantifies it by playing it **head
to head against the rule-based agent** (:class:`agents.rule_agent.RuleAgent`) in an
N≥200 side-swapped :func:`eval.arena.run_arena` and reporting what the parent Issue
(受け入れ条件) asks for:

* **SearchAgent win rate + Wilson 95% CI** (search = agent A) over all matches (draws
  counted as non-wins, matching :func:`eval.arena.wilson_ci`). The acceptance gate is
  *promotion-style*: the search line is promoted only when the CI **lower bound
  strictly exceeds 0.50**; otherwise the RuleAgent stays champion and the search is
  reported honestly but **not** shipped.
* **Zero crashes / illegal moves / timeouts (探索リーク・クラッシュ0).** The arena
  scores an agent exception / illegal move / per-move-timeout as a *fault*; the gate
  requires ``a_faults == 0``. The SearchAgent also self-reports its internal
  search-session bookkeeping (attempts / chosen / rule-fallbacks / leaked sessions),
  aggregated across every match, so a search-state leak can only surface, never hide.
* **Per-decision latency (mean / p99)** for the search side — the Kaggle time-budget
  watch value.

Usage:
    venv/bin/python eval/bench_search_vs_rule.py [--n 200] [--seed 0]
        [--search-budget 0.1] [--max-candidates 12] [--per-move-timeout 5.0]
        [--json report.json]

Exit code 0 iff the promotion gate passes (N≥200, zero faults, **and** CI lower
bound > 0.50). A valid run that does not clear 0.50 exits 1 — that is the expected,
honest "keep RuleAgent as champion" outcome, not an error. Run from the repo root
(needs the gitignored ``cg/`` engine + ``deck.csv``).
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

from agents.rule_agent import RuleAgent          # noqa: E402
from agents.search_agent import SearchAgent       # noqa: E402
from eval.arena import run_arena, _load_deck       # noqa: E402
from eval.trace import RecordLevel                 # noqa: E402

SEARCH_LABEL = "search"
RULE_LABEL = "rule"
PROMOTE_THRESHOLD = 0.50


def run_bench(
    n: int,
    seed: int,
    search_budget_s: float,
    max_candidates: int,
    per_move_timeout: float,
    deck_path: str = "deck.csv",
    record_traces: bool = False,
) -> dict:
    """Run SearchAgent (A) vs RuleAgent (B) paired and return the aggregated result."""
    deck = _load_deck(deck_path)
    built: list[SearchAgent] = []

    def make_search(s: int) -> SearchAgent:
        a = SearchAgent(
            seed=s, deck_path=deck_path,
            search_budget_s=search_budget_s, max_candidates=max_candidates,
        )
        built.append(a)
        return a

    report = run_arena(
        make_search,
        lambda s: RuleAgent(seed=s),
        deck0=deck,
        n_matches=n,
        side_swap=True,
        agent_seed=seed,
        label_a=SEARCH_LABEL,
        label_b=RULE_LABEL,
        per_move_timeout=per_move_timeout,
        record_traces=record_traces,
        trace_level=RecordLevel.RESULT,
        run_label=f"bench_search_vs_rule_n{n}",
    )

    wr = report.win_rates
    ci_low, ci_high = wr["a_win_rate_ci95"]
    # Aggregate the SearchAgent's own per-match search bookkeeping (leak accounting).
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
        "search_wins": report.totals["a_wins"],
        "rule_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "undecided": report.totals["undecided"],
        "search_win_rate": wr["a_win_rate"],
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "a_faults": report.safety["a_faults"],
        "b_faults": report.safety["b_faults"],
        "a_fault_categories": report.safety["a_fault_categories"],
        "search_latency": report.latency.get(SEARCH_LABEL, {}),
        "search_stats": agg,
        "run_dir": report.run_dir,
        "promote_threshold": PROMOTE_THRESHOLD,
        # Promotion gate: enough games, zero faults/leaks, AND CI lower bound > 0.50.
        "passed": passed,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SearchAgent-vs-RuleAgent promotion benchmark (win rate + Wilson CI + safety)."
    )
    p.add_argument("--n", type=int, default=200, help="matches (>=200 required by the gate)")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--search-budget", type=float, default=0.1, help="search per-decision time budget (s)")
    p.add_argument("--max-candidates", type=int, default=12, help="search per-decision candidate cap")
    p.add_argument("--per-move-timeout", type=float, default=5.0, help="hard per-move timeout (s)")
    p.add_argument("--json", default=None, help="also write the raw JSON result to this path")
    return p.parse_args(argv)


def _fmt_lat(d: dict) -> str:
    if not d:
        return "n/a"
    return (f"n={d.get('n_decisions', 0)} mean={d.get('mean_ms', 0):.2f}ms "
            f"p99={d.get('p99_ms', 0):.2f}ms max={d.get('max_ms', 0):.2f}ms")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    r = run_bench(
        args.n, args.seed, args.search_budget, args.max_candidates, args.per_move_timeout,
    )
    print(
        f"BENCH search vs rule: n={r['n']} "
        f"W/D/L(search)={r['search_wins']}/{r['draws']}/{r['rule_wins']} undecided={r['undecided']} "
        f"search_win_rate={r['search_win_rate']:.3f} "
        f"Wilson95=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] "
        f"faults(search/rule)={r['a_faults']}/{r['b_faults']}"
    )
    print(f"SEARCH latency/decision: {_fmt_lat(r['search_latency'])}")
    print(f"SEARCH stats (aggregate): {r['search_stats']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    if r["n"] < 200:
        print("GATE INVALID: need >= 200 games")
        return 2
    verdict = "PASS (promote search)" if r["passed"] else "FAIL (keep RuleAgent champion)"
    print(
        f"PROMOTION GATE {verdict}: N={r['n']}, faults={r['a_faults'] + r['b_faults']}, "
        f"leaks={r['search_stats']['leaks']}, CI_low={r['ci95_low']:.4f} vs threshold "
        f"{PROMOTE_THRESHOLD} (promote iff CI lower bound > {PROMOTE_THRESHOLD})"
    )
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
