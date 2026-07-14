"""Evaluation benchmark: PPO + critical-position MCTS vs plain PPO (SOT-1690).

Measures the determinized-MCTS reinforcement (:mod:`agents.mcts`) head-to-head
against the unreinforced PPO policy it wraps, in N side-swapped
:func:`eval.arena.run_arena` matches, and reports:

* the reinforced side's win rate + **Wilson 95% CI**,
* per-decision thinking time (mean/max) for both sides (the arena's own
  latency measurement — MCTS search time is inside it),
* the **MCTS発動率** — activated (critical) decisions over ALL decisions —
  plus override/failure/simulation counters (:class:`agents.mcts.MCTSStats`,
  pooled across every agent instance the arena builds).

Like :mod:`eval.bench_ppo_vs_baselines` this is a *measurement*, not a
promotion gate. The pass criterion is safety + evidence:

* **zero faults on the MCTS side** (違法出力0 / fault 0),
* the run completed the requested N with the CI recorded,
* MCTS actually activated (a reinforcement that never fires measures nothing),
* the MCTS side's max per-decision thinking time stayed within the hard
  per-move timeout (持ち時間 headroom).

Usage::

    venv/bin/python eval/bench_mcts_vs_ppo.py [--n 200] [--seed 0]
        [--time-limit 0.5] [--policy data/policy.json] [--json report.json]

Exit code 0 iff the safety criteria hold (whatever the win rate says). Run
from the repo root (needs the gitignored ``cg/`` engine + ``deck.csv``).
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

from agents.mcts import MCTSConfig, MCTSStats                # noqa: E402
from agents.ppo_agent import DEFAULT_POLICY_PATH, PPOAgent   # noqa: E402
from eval.arena import run_arena, _load_deck                 # noqa: E402
from eval.trace import RecordLevel                           # noqa: E402

MCTS_LABEL = "ppo+mcts"
PPO_LABEL = "ppo"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PPO+MCTS vs plain PPO: win rate + Wilson CI + 発動率 + safety."
    )
    p.add_argument("--n", type=int, default=200, help="number of matches")
    p.add_argument("--seed", type=int, default=0, help="base agent-RNG seed")
    p.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="policy.json path")
    p.add_argument("--time-limit", type=float, default=0.5,
                   help="MCTS per-decision search cap (s)")
    p.add_argument("--per-move-timeout", type=float, default=5.0,
                   help="hard per-move timeout (s)")
    p.add_argument("--json", default=None, help="also write the raw JSON result here")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    probe = PPOAgent(seed=0, policy_path=args.policy)
    if not probe.policy_loaded:
        print(f"POLICY INVALID: {args.policy} did not load — train it first (train/ppo.py)")
        return 2

    deck = _load_deck("deck.csv")
    cfg = MCTSConfig(time_limit_s=args.time_limit)
    mcts_agents: list[PPOAgent] = []

    def make_mcts(s: int) -> PPOAgent:
        agent = PPOAgent(seed=s, policy_path=args.policy, mcts=True, mcts_config=cfg)
        mcts_agents.append(agent)
        return agent

    def make_ppo(s: int) -> PPOAgent:
        return PPOAgent(seed=s, policy_path=args.policy)

    report = run_arena(
        make_mcts,
        make_ppo,
        deck0=deck,
        n_matches=args.n,
        side_swap=True,
        agent_seed=args.seed,
        label_a=MCTS_LABEL,
        label_b=PPO_LABEL,
        per_move_timeout=args.per_move_timeout,
        record_traces=False,
        trace_level=RecordLevel.RESULT,
        run_label=f"bench_mcts_vs_ppo_n{args.n}",
    )

    pooled = MCTSStats()
    for agent in mcts_agents:
        if agent.mcts_stats is not None:
            pooled.merge(agent.mcts_stats)
    mcts = pooled.report()

    wr = report.win_rates
    ci_low, ci_high = wr["a_win_rate_ci95"]
    latency_a = report.latency.get(MCTS_LABEL, {})
    latency_b = report.latency.get(PPO_LABEL, {})
    result = {
        "policy": args.policy,
        "n": report.totals["n"],
        "seed": args.seed,
        "mcts_config": {
            "time_limit_s": cfg.time_limit_s,
            "n_determinizations": cfg.n_determinizations,
            "max_candidates": cfg.max_candidates,
            "rollout_depth": cfg.rollout_depth,
            "ucb_c": cfg.ucb_c,
            "deviate_margin": cfg.deviate_margin,
            "entropy_threshold": cfg.entropy_threshold,
            "value_threshold": cfg.value_threshold,
            "min_options": cfg.min_options,
        },
        "mcts_wins": report.totals["a_wins"],
        "ppo_wins": report.totals["b_wins"],
        "draws": report.totals["draws"],
        "undecided": report.totals["undecided"],
        "mcts_win_rate": wr["a_win_rate"],
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mcts_faults": report.safety["a_faults"],
        "ppo_faults": report.safety["b_faults"],
        "mcts_fault_categories": report.safety["a_fault_categories"],
        "latency_mcts": latency_a,
        "latency_ppo": latency_b,
        "mcts_stats": mcts,
        "run_dir": report.run_dir,
    }

    print(
        f"BENCH ppo+mcts vs ppo: n={result['n']} "
        f"W/D/L(mcts)={result['mcts_wins']}/{result['draws']}/{result['ppo_wins']} "
        f"undecided={result['undecided']} mcts_win_rate={result['mcts_win_rate']:.3f} "
        f"Wilson95=[{ci_low:.4f}, {ci_high:.4f}] "
        f"faults(mcts/ppo)={result['mcts_faults']}/{result['ppo_faults']}"
    )
    print(
        f"MCTS 発動率={mcts['activation_rate']:.3f} "
        f"({mcts['activations']}/{mcts['decisions']} decisions, "
        f"eligible={mcts['eligible']}) searched={mcts['searched']} "
        f"overrides={mcts['overrides']} failures={mcts['failures']} "
        f"sims={mcts['simulations']} "
        f"det ok/fail={mcts['determinizations_ok']}/{mcts['determinizations_failed']} "
        f"search_ms mean/max={mcts['search_ms_mean']:.1f}/{mcts['search_ms_max']:.1f}"
    )
    print(
        f"THINK ms (mcts side) mean={latency_a.get('mean_ms', 0.0):.1f} "
        f"max={latency_a.get('max_ms', 0.0):.1f} | (ppo side) "
        f"mean={latency_b.get('mean_ms', 0.0):.1f} max={latency_b.get('max_ms', 0.0):.1f}"
    )

    passed = (
        result["n"] >= args.n
        and result["mcts_faults"] == 0
        and mcts["activations"] > 0
        and latency_a.get("max_ms", float("inf")) <= args.per_move_timeout * 1000.0
    )
    result["passed"] = passed
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"wrote JSON result -> {args.json}")
    print(
        f"SAFETY GATE {'PASS' if passed else 'FAIL'}: mcts faults={result['mcts_faults']}, "
        f"activations={mcts['activations']}, max think "
        f"{latency_a.get('max_ms', 0.0):.0f}ms <= {args.per_move_timeout * 1000:.0f}ms "
        f"(beating plain PPO is not required by SOT-1690)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
