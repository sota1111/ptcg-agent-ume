"""Promotion benchmark: the final submission agent vs the old champion (SOT-1691).

Plays the final agent — :class:`agents.harness.HarnessAgent`, the exact
PPO + critical-position-MCTS + candidate-harness configuration ``main.py``
submits (same 0.4 s search cap) — head to head against the reigning champion
(:class:`agents.rule_agent.RuleAgent`) in N side-swapped
:func:`eval.arena.run_arena` matches, and reports:

* the final agent's win rate + **Wilson 95% CI** — the champion promotion
  judgement: **promote only when the CI lower bound strictly exceeds 0.50**,
  otherwise the RuleAgent stays champion and that fact is reported honestly;
* safety: **zero faults / 違法出力0** on the final-agent side and per-decision
  thinking time (mean/max) within the hard per-move timeout;
* the harness' per-source decision tally (:class:`agents.harness.HarnessStats`)
  and the MCTS 発動率/override counters (:class:`agents.mcts.MCTSStats`),
  pooled across every agent instance the arena builds.

``--opponent random`` (SOT-1695) runs the same bench against the Random
baseline instead — the vs-Random no-regression check — where the 0.50 promotion
line is not printed (it is rule-specific).

A long N can be split into independent chunks (different ``--seed``) and merged
afterwards::

    venv/bin/python eval/bench_final_vs_rule.py --n 100 --seed 0   --json c0.json
    venv/bin/python eval/bench_final_vs_rule.py --n 100 --seed 100 --json c1.json
    venv/bin/python eval/bench_final_vs_rule.py --aggregate final.json c0.json c1.json

Exit code 0 iff the safety criteria hold (losing to the champion is an honest
"keep RuleAgent as champion" outcome, not an error). Run from the repo root
(needs the gitignored ``cg/`` engine + ``deck.csv`` + ``data/policy.json``).
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

from agents.harness import HarnessAgent, HarnessStats        # noqa: E402
from agents.mcts import MCTSConfig, MCTSStats                # noqa: E402
from agents.ppo_agent import DEFAULT_POLICY_PATH             # noqa: E402
from agents.random_agent import RandomAgent                  # noqa: E402
from agents.rule_agent import RuleAgent                      # noqa: E402
from eval.arena import run_arena, wilson_ci, _load_deck      # noqa: E402
from eval.trace import RecordLevel                           # noqa: E402

FINAL_LABEL = "harness"
OPPONENTS = {
    "rule": lambda s: RuleAgent(seed=s),
    "random": lambda s: RandomAgent(seed=s),
}


def run_bench(args: argparse.Namespace) -> dict:
    """One arena run of HarnessAgent (A) vs the chosen baseline (B); returns the result dict."""
    probe = HarnessAgent(seed=0, policy_path=args.policy, mcts=False)
    if not probe.policy_loaded:
        raise SystemExit(f"POLICY INVALID: {args.policy} did not load — train it first (train/ppo.py)")

    deck = _load_deck("deck.csv")
    cfg = MCTSConfig(time_limit_s=args.time_limit)
    built: list[HarnessAgent] = []

    def make_final(s: int) -> HarnessAgent:
        a = HarnessAgent(seed=s, policy_path=args.policy, mcts=True, mcts_config=cfg)
        built.append(a)
        return a

    report = run_arena(
        make_final,
        OPPONENTS[args.opponent],
        deck0=deck,
        n_matches=args.n,
        side_swap=True,
        agent_seed=args.seed,
        label_a=FINAL_LABEL,
        label_b=args.opponent,
        per_move_timeout=args.per_move_timeout,
        record_traces=False,
        trace_level=RecordLevel.RESULT,
        run_label=f"bench_final_vs_{args.opponent}_n{args.n}_s{args.seed}",
    )

    mcts_pooled, harness_pooled = MCTSStats(), HarnessStats()
    for a in built:
        if a.mcts_stats is not None:
            mcts_pooled.merge(a.mcts_stats)
        harness_pooled.merge(a.harness_stats)

    wr = report.win_rates
    ci_low, ci_high = wr["a_win_rate_ci95"]
    return {
        "policy": args.policy,
        "opponent": args.opponent,
        "n": report.totals["n"],
        "seed": args.seed,
        "time_limit_s": cfg.time_limit_s,
        "per_move_timeout_s": args.per_move_timeout,
        "final_wins": report.totals["a_wins"],
        "opponent_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "undecided": report.totals["undecided"],
        "final_win_rate": wr["a_win_rate"],
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "final_faults": report.safety["a_faults"],
        "opponent_faults": report.safety["b_faults"],
        "final_fault_categories": report.safety["a_fault_categories"],
        "latency_final": report.latency.get(FINAL_LABEL, {}),
        "latency_opponent": report.latency.get(args.opponent, {}),
        "harness_stats": harness_pooled.report(),
        "mcts_stats": mcts_pooled.report(),
        "run_dir": report.run_dir,
    }


def aggregate(paths: list[str]) -> dict:
    """Merge chunk results (independent seeds) into one promotion judgement."""
    chunks = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            chunks.append(json.load(fh))
    opponents = {c.get("opponent", "rule") for c in chunks}
    if len(opponents) != 1:
        raise SystemExit(f"cannot aggregate chunks of different opponents: {sorted(opponents)}")
    n = sum(c["n"] for c in chunks)
    wins = sum(c["final_wins"] for c in chunks)
    ci_low, ci_high = wilson_ci(wins, n)
    harness, mcts = HarnessStats(), MCTSStats()
    for c in chunks:
        h = HarnessStats(**{k: c["harness_stats"][k] for k in ("decisions", "candidates", "invalid_candidates")})
        h.decided_by = dict(c["harness_stats"]["decided_by"])
        harness.merge(h)
        m = c["mcts_stats"]
        other = MCTSStats(
            decisions=m["decisions"], eligible=m["eligible"], activations=m["activations"],
            searched=m["searched"], overrides=m["overrides"], failures=m["failures"],
            determinizations_ok=m["determinizations_ok"],
            determinizations_failed=m["determinizations_failed"],
            simulations=m["simulations"],
            elapsed_ms_total=m["search_ms_mean"] * m["activations"],
            elapsed_ms_max=m["search_ms_max"],
        )
        mcts.merge(other)
    return {
        "chunks": [{"seed": c["seed"], "n": c["n"], "final_wins": c["final_wins"]} for c in chunks],
        "opponent": opponents.pop(),
        "n": n,
        "final_wins": wins,
        "opponent_wins": sum(c.get("opponent_wins", c.get("rule_wins", 0)) for c in chunks),
        "draws": sum(c["draws"] for c in chunks),
        "undecided": sum(c["undecided"] for c in chunks),
        "final_win_rate": (wins / n) if n else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "final_faults": sum(c["final_faults"] for c in chunks),
        "opponent_faults": sum(c.get("opponent_faults", c.get("rule_faults", 0)) for c in chunks),
        "latency_final_max_ms": max((c["latency_final"].get("max_ms", 0.0) for c in chunks), default=0.0),
        "latency_final_mean_ms_chunks": [c["latency_final"].get("mean_ms", 0.0) for c in chunks],
        "per_move_timeout_s": max(c["per_move_timeout_s"] for c in chunks),
        "harness_stats": harness.report(),
        "mcts_stats": mcts.report(),
    }


def finish(result: dict, json_path: str | None, n_requested: int) -> int:
    """Print the verdict lines, apply the gates, optionally write JSON."""
    opponent = result.get("opponent", "rule")
    ci_low, ci_high = result["ci95_low"], result["ci95_high"]
    promote = ci_low > 0.5
    timeout_ms = result["per_move_timeout_s"] * 1000.0
    max_think = result.get("latency_final_max_ms", result.get("latency_final", {}).get("max_ms", 0.0))
    safety = (
        result["n"] >= n_requested
        and result["final_faults"] == 0
        and max_think <= timeout_ms
    )
    result["promote"] = promote
    result["passed"] = safety

    h, m = result["harness_stats"], result["mcts_stats"]
    print(
        f"BENCH final(harness) vs {opponent}: n={result['n']} "
        f"W/D/L(final)={result['final_wins']}/{result['draws']}/{result['opponent_wins']} "
        f"undecided={result['undecided']} win_rate={result['final_win_rate']:.3f} "
        f"Wilson95=[{ci_low:.4f}, {ci_high:.4f}] "
        f"faults(final/{opponent})={result['final_faults']}/{result['opponent_faults']}"
    )
    print(
        f"HARNESS decided_by={h['decided_by']} "
        f"candidates={h['candidates']} invalid={h['invalid_candidates']} "
        f"| MCTS 発動率={m['activation_rate']:.3f} overrides={m['overrides']} "
        f"failures={m['failures']} search_ms mean/max={m['search_ms_mean']:.1f}/{m['search_ms_max']:.1f}"
    )
    print(f"THINK ms (final side) max={max_think:.1f} (cap {timeout_ms:.0f})")
    if opponent == "rule":
        print(
            f"PROMOTION {'PASS (promote to champion)' if promote else 'FAIL (keep RuleAgent champion)'}: "
            f"CI lower bound {ci_low:.4f} {'>' if promote else '<='} 0.50"
        )
    print(f"SAFETY GATE {'PASS' if safety else 'FAIL'}: faults={result['final_faults']}, "
          f"n={result['n']}/{n_requested}, max think {max_think:.0f}ms <= {timeout_ms:.0f}ms")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {json_path}")
    return 0 if safety else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Final agent (PPO+MCTS+harness) vs RuleAgent champion: win rate + Wilson CI + promotion judgement."
    )
    p.add_argument("--n", type=int, default=400, help="number of matches")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--opponent", choices=sorted(OPPONENTS), default="rule",
                   help="baseline opponent (SOT-1695 adds random; default rule)")
    p.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="policy.json path")
    p.add_argument("--time-limit", type=float, default=0.4,
                   help="MCTS per-decision search cap (s) — keep = main.py's")
    p.add_argument("--per-move-timeout", type=float, default=5.0,
                   help="hard per-move timeout (s)")
    p.add_argument("--json", default=None, help="also write the raw JSON result here")
    p.add_argument("--aggregate", nargs="+", metavar=("OUT", "CHUNK"), default=None,
                   help="merge chunk JSONs: --aggregate out.json chunk1.json chunk2.json ...")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.aggregate:
        out, chunks = args.aggregate[0], args.aggregate[1:]
        if not chunks:
            p.error("--aggregate needs an output path followed by at least one chunk JSON")
        result = aggregate(chunks)
        return finish(result, out, n_requested=result["n"])

    result = run_bench(args)
    return finish(result, args.json, n_requested=args.n)


if __name__ == "__main__":
    raise SystemExit(main())
