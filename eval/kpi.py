"""Ume battle KPI recording (SOT-1710).

Defines the KPI record schema (``ume-kpi-v1``, see ``docs/KPI.md``), computes
KPI records from measurements or existing bench results, and appends one line
per measurement to the committed history file ``eval/kpi_history.jsonl`` —
kept under ``eval/`` because ``/data/`` is double-gitignored.

Three ways to produce a record:

1. **Own measurement** (full KPI coverage: vs Rule + vs Random in one run,
   both via :func:`eval.bench_final_vs_rule.run_bench` — the exact submission
   configuration ``main.py`` uses)::

       venv/bin/python eval/kpi.py --measure --n-rule 48 --n-random 24 \
           --seed 20260718 --issue SOT-1710

2. **From existing bench JSON results** (``eval/bench_final_vs_rule.py``
   single-run or ``--aggregate`` output; the unmeasured opponent's KPI is
   null)::

       venv/bin/python eval/kpi.py --from-report rule_bench.json \
           --random-report random_bench.json --issue SOT-XXXX

3. **In-process hook** — ``eval/bench_final_vs_rule.py --kpi SOT-XXXX`` calls
   :func:`record_from_bench_result` + :func:`append_history` on its result.

History and comparison display: ``eval/kpi_report.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCHEMA = "ume-kpi-v1"
DEFAULT_HISTORY_PATH = os.path.join(REPO, "eval", "kpi_history.jsonl")

# Improvement direction per KPI: +1 higher is better, -1 lower is better,
# 0 must stay exactly zero (any nonzero value is a regression).
KPI_DIRECTIONS = {
    "winrate_vs_rule": 1,
    "winrate_vs_random": 1,
    "fault_total": 0,
    "decision_time_mean_ms": -1,
}


def history_path() -> str:
    """History file path; ``UME_KPI_HISTORY`` overrides (tests/verification)."""
    return os.environ.get("UME_KPI_HISTORY") or DEFAULT_HISTORY_PATH


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _winrate_kpi(result: dict | None) -> dict:
    if result is None:
        return {"value": None, "ci95": None, "wins": None, "losses": None,
                "draws": None, "undecided": None, "n": None,
                "note": "not measured in this record"}
    return {
        "value": round(result["final_win_rate"], 4),
        "ci95": [round(result["ci95_low"], 4), round(result["ci95_high"], 4)],
        "wins": result["final_wins"],
        "losses": result["opponent_wins"],
        "draws": result["draws"],
        "undecided": result["undecided"],
        "n": result["n"],
    }


def _latency(result: dict) -> tuple:
    """(mean_ms, max_ms, n_decisions) from a single-run or aggregate result."""
    lat = result.get("latency_final") or {}
    if lat.get("mean_ms") is not None:
        return (lat.get("mean_ms"), lat.get("max_ms"),
                lat.get("n_decisions"))
    chunk_means = result.get("latency_final_mean_ms_chunks") or []
    mean = (sum(chunk_means) / len(chunk_means)) if chunk_means else None
    return (mean, result.get("latency_final_max_ms"), None)


def build_record(rule_result: dict | None = None,
                 random_result: dict | None = None,
                 issue: str = None, source: str = "kpi-measure") -> dict:
    """One ``ume-kpi-v1`` record from bench_final_vs_rule result dict(s)."""
    if rule_result is None and random_result is None:
        raise ValueError("at least one of rule/random results is required")
    primary = rule_result or random_result
    faults = sum(r["final_faults"] for r in (rule_result, random_result) if r)
    fault_breakdown = {
        ("vs_" + r["opponent"]): {
            "faults": r["final_faults"],
            "categories": r.get("final_fault_categories") or {},
        }
        for r in (rule_result, random_result) if r
    }
    timing_src = rule_result or random_result
    mean_ms, max_ms, n_decisions = _latency(timing_src)
    rec = {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "issue": issue or "unknown",
        "source": source,
        "deck": "deck.csv",
        "n_rule": rule_result["n"] if rule_result else None,
        "n_random": random_result["n"] if random_result else None,
        "seed": primary.get("seed"),
        "temperature": primary.get("temperature"),
        "time_limit_s": primary.get("time_limit_s"),
        "kpis": {
            "winrate_vs_rule": _winrate_kpi(rule_result),
            "winrate_vs_random": _winrate_kpi(random_result),
            "fault_total": {"value": faults, "breakdown": fault_breakdown},
            "decision_time_mean_ms": {
                "value": round(mean_ms, 2) if mean_ms is not None else None,
                "max_ms": round(max_ms, 2) if max_ms is not None else None,
                "n_decisions": n_decisions,
                "timing_opponent": timing_src["opponent"],
                "per_move_timeout_ms":
                    timing_src.get("per_move_timeout_s", 0) * 1000 or None,
            },
        },
    }
    return rec


def record_from_bench_result(result: dict, issue: str = None) -> dict:
    """KPI record from ONE bench_final_vs_rule result (single-run or
    aggregate) — routes it to the vs-rule or vs-random slot by opponent."""
    opponent = result.get("opponent", "rule")
    return build_record(
        rule_result=result if opponent == "rule" else None,
        random_result=result if opponent == "random" else None,
        issue=issue, source="bench_final_vs_rule")


def append_history(record: dict, path: str = None) -> str:
    path = path or history_path()
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_history(path: str = None) -> list:
    path = path or history_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def shard_sizes(n: int, shards: int) -> list[int]:
    """Split *n* matches evenly, omitting empty shards."""
    if n < 0:
        raise ValueError("match count must be non-negative")
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if n == 0:
        return []
    count = min(n, shards)
    q, r = divmod(n, count)
    return [q + (i < r) for i in range(count)]


def _run_bench_shard(spec: tuple) -> dict:
    """Process-pool entry point; each shard owns an independent RNG range."""
    from eval.bench_final_vs_rule import run_bench
    opponent, n, seed, policy, time_limit, temperature, timeout = spec
    return run_bench(argparse.Namespace(
        opponent=opponent, n=n, seed=seed, policy=policy,
        time_limit=time_limit, temperature=temperature,
        per_move_timeout=timeout,
    ))


# ---------------------------------------------------------------- measurement

def run_measure(args) -> int:
    """Measure vs Rule (and vs Random) with the submission configuration and
    append one history record."""
    from eval.bench_final_vs_rule import aggregate_results
    from agents.ppo_agent import DEFAULT_POLICY_PATH

    def bench(opponent: str, n: int) -> dict:
        sizes = shard_sizes(n, args.shards)
        specs = [
            (opponent, size, args.seed + i * args.seed_stride,
             DEFAULT_POLICY_PATH, args.time_limit, 0.25, 5.0)
            for i, size in enumerate(sizes)
        ]
        print(f"[kpi] measuring vs {opponent} (n={n}, shards={len(specs)}, "
              f"seeds={[s[2] for s in specs]})...",
              flush=True)
        if len(specs) == 1:
            r = _run_bench_shard(specs[0])
        else:
            with ProcessPoolExecutor(max_workers=len(specs)) as pool:
                chunks = list(pool.map(_run_bench_shard, specs))
            r = aggregate_results(chunks)
        print(f"[kpi] vs {opponent}: win_rate={r['final_win_rate']:.3f} "
              f"Wilson95=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] "
              f"faults={r['final_faults']}", flush=True)
        return r

    rule_result = bench("rule", args.n_rule) if args.n_rule else None
    random_result = bench("random", args.n_random) if args.n_random else None
    rec = build_record(rule_result, random_result, issue=args.issue)
    print(json.dumps(rec, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec)}")
    return 0


def run_from_report(args) -> int:
    def load(path):
        if not path:
            return None
        with open(path) as f:
            return json.load(f)

    rule_result, random_result = None, None
    for path in (args.from_report, args.random_report):
        result = load(path)
        if result is None:
            continue
        if result.get("opponent", "rule") == "rule":
            rule_result = result
        else:
            random_result = result
    rec = build_record(rule_result, random_result, issue=args.issue,
                       source="from-report")
    print(json.dumps(rec, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure", action="store_true",
                   help="run vs-rule + vs-random benches and record KPIs")
    p.add_argument("--n-rule", type=int, default=48,
                   help="matches vs RuleAgent (0 skips)")
    p.add_argument("--n-random", type=int, default=24,
                   help="matches vs RandomAgent (0 skips)")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--shards", type=int, default=1,
                   help="parallel independent-seed shards per opponent")
    p.add_argument("--seed-stride", type=int, default=1000000,
                   help="distance between shard base seeds")
    p.add_argument("--time-limit", type=float, default=0.4,
                   help="MCTS search cap per decision (seconds)")
    p.add_argument("--from-report", default=None,
                   help="bench_final_vs_rule JSON result -> one record")
    p.add_argument("--random-report", default=None,
                   help="optional second JSON result (vs random)")
    p.add_argument("--issue", default=None, help="Linear issue id to record")
    p.add_argument("--no-append", action="store_true",
                   help="print the record without touching the history")
    args = p.parse_args(argv)
    if args.measure:
        return run_measure(args)
    if args.from_report or args.random_report:
        return run_from_report(args)
    raise SystemExit("one of --measure / --from-report required")


if __name__ == "__main__":
    sys.exit(main())
