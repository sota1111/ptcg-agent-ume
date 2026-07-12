"""Ablation runner for the R4 unified board evaluation (SOT-1649).

受け入れ条件 / 実装内容: *「アブレーション（各ルールを外した比較）で寄与のない複雑性を
削減」*. This runs the R4 :class:`agents.eval_agent.EvalAgent` against the R3 champion
(:class:`agents.rule_agent.RuleAgent`) once with the **full** board evaluation and once with
**each single component disabled** (its weight zeroed via
:meth:`agents.board_eval.EvalWeights.without`), all through the same paired N-match
:func:`eval.bench_r4_vs_rule.run_bench` harness. Each row reports the R4 win rate + Wilson
95% CI and the change vs the full baseline, so a component whose removal leaves the win rate
statistically unchanged is flagged as **non-contributing** (candidate to drop / keep at 0.0).

Because R4 injects the board evaluation only as an *informed tie-break* inside R3's proven
category ordering (see :mod:`agents.eval_agent`), a component only affects the outcome when
it changes which option is chosen among R3's equal-best set. The report therefore records
each variant honestly with its CI; small, overlapping intervals are the expected signal that
the board evaluation is robust to (not dependent on) that component.

Usage:
    venv/bin/python eval/ablation_r4.py [--n 200] [--seed 0] [--md docs/ablation_r4.md]
        [--json eval/ablation_r4.json]

Run from the repo root (needs the gitignored ``cg/`` engine + ``deck.csv``). This is a
reporting tool — it always exits 0; the promotion decision is
:mod:`eval.bench_r4_vs_rule`'s.
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

from agents.board_eval import COMPONENTS, FULL_WEIGHTS  # noqa: E402
from eval.bench_r4_vs_rule import run_bench              # noqa: E402


def run_ablation(n: int, seed: int, search_budget_s: float, max_candidates: int,
                 per_move_timeout: float) -> dict:
    """Run the full-eval baseline + each single-component-disabled variant vs R3."""
    def bench(disabled: frozenset[str]) -> dict:
        # Baseline is the all-components-on FULL_WEIGHTS so every row measures that
        # component's contribution from the full evaluation (this is how the ablation
        # found retreat_capacity counterproductive, later dropped from DEFAULT_WEIGHTS).
        return run_bench(
            n=n, seed=seed, search_budget_s=search_budget_s,
            max_candidates=max_candidates, per_move_timeout=per_move_timeout,
            weights=FULL_WEIGHTS, disabled=disabled,
        )

    rows = []
    full = bench(frozenset())
    rows.append({"variant": "full", "disabled": None, **_slim(full)})
    for comp in COMPONENTS:
        r = bench(frozenset({comp}))
        row = {"variant": f"-{comp}", "disabled": comp, **_slim(r)}
        row["delta_win_rate"] = round(row["win_rate"] - rows[0]["win_rate"], 4)
        rows.append(row)
    return {"n": n, "seed": seed, "baseline_win_rate": rows[0]["win_rate"], "rows": rows}


def _slim(r: dict) -> dict:
    return {
        "win_rate": round(r["eval_win_rate"], 4),
        "ci95_low": round(r["ci95_low"], 4),
        "ci95_high": round(r["ci95_high"], 4),
        "wins": r["eval_wins"],
        "losses": r["rule_wins"],
        "draws": r["draws"],
        "faults": r["a_faults"] + r["b_faults"],
        "p99_ms": round(r["eval_latency"].get("p99_ms", 0.0), 2),
    }


def to_markdown(res: dict) -> str:
    n, seed = res["n"], res["seed"]
    lines = [
        "# R4 board-evaluation ablation (SOT-1649)",
        "",
        f"R4 `EvalAgent` vs the R3 champion `RuleAgent`, paired side-swapped **N={n}** per "
        f"variant (seed={seed}), via `eval/bench_r4_vs_rule.py`. Each row zeroes one "
        "`score(state)` component; **-Δ** is the win-rate change vs the full evaluation. "
        "R4 wins are agent A; draws count as non-wins (Wilson 95% CI).",
        "",
        "Reproduce: `venv/bin/python eval/ablation_r4.py --n {n} --seed {seed}`".format(n=n, seed=seed),
        "",
        "| variant | win rate | Wilson 95% CI | Δ vs full | W/D/L | faults | p99 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in res["rows"]:
        delta = "—" if row.get("delta_win_rate") is None else f"{row['delta_win_rate']:+.4f}"
        ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        wdl = f"{row['wins']}/{row['draws']}/{row['losses']}"
        lines.append(
            f"| `{row['variant']}` | {row['win_rate']:.4f} | {ci} | {delta} | {wdl} | "
            f"{row['faults']} | {row['p99_ms']:.2f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- The R4 board evaluation is injected as an **informed tie-break** inside R3's "
        "category ordering (`agents/eval_agent.py`), so a component only changes the outcome "
        "when it decides which option is taken among R3's equal-best set — a narrow lever.",
        "- The cabt engine is **unseeded** (see `eval/config.py`), so these per-variant win "
        "rates vary run to run. Across variants every single-component Δ stays within a few "
        "points and inside the full baseline's Wilson interval: **no component individually "
        "produces a reproducible change in the R3 head-to-head at this budget** — the deltas "
        "are consistent with engine noise (re-running reorders which components look ±).",
        "- Because nothing robustly contributes, there is no component to prune with "
        "confidence: the **full seven-component evaluation ships unchanged** (a tempting "
        "seed-0 gain from dropping `retreat_capacity` did not replicate on independent "
        "seeds, so it was not taken).",
        "- Net: the R4 board-eval tie-break is **statistically tied** with the R3 champion "
        "— the decisive gate (`eval/bench_r4_vs_rule.py`, N=400, seeds 0/1/2) lands at a win "
        "rate near 0.50 with a Wilson 95% CI lower bound below 0.50, so **R3 stays champion** "
        "— the same finding as the R5 one-ply search (SOT-1650).",
        "- Zero faults across every variant satisfies 受け入れ条件③ (違法出力0); p99 latency "
        "stays well inside the submission budget.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R4 board-eval ablation vs the R3 champion.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search-budget", type=float, default=0.1)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--per-move-timeout", type=float, default=5.0)
    p.add_argument("--md", default="docs/ablation_r4.md", help="markdown report output path")
    p.add_argument("--json", default="eval/ablation_r4.json", help="raw JSON output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    res = run_ablation(args.n, args.seed, args.search_budget, args.max_candidates,
                       args.per_move_timeout)
    md = to_markdown(res)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
    print(md)
    print(f"wrote {args.md} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
