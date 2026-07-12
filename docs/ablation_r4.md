# R4 board-evaluation ablation (SOT-1649)

R4 `EvalAgent` vs the R3 champion `RuleAgent`, paired side-swapped **N=200** per variant (seed=0), via `eval/bench_r4_vs_rule.py`. Each row zeroes one `score(state)` component; **-Δ** is the win-rate change vs the full evaluation. R4 wins are agent A; draws count as non-wins (Wilson 95% CI).

Reproduce: `venv/bin/python eval/ablation_r4.py --n 200 --seed 0`

| variant | win rate | Wilson 95% CI | Δ vs full | W/D/L | faults | p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `full` | 0.5100 | [0.4412, 0.5784] | — | 102/0/98 | 0 | 6.38 |
| `-prize` | 0.5400 | [0.4708, 0.6077] | +0.0300 | 108/0/92 | 0 | 6.17 |
| `-active_survival` | 0.4700 | [0.4020, 0.5391] | -0.0400 | 94/0/106 | 0 | 7.22 |
| `-ko_threat` | 0.4850 | [0.4167, 0.5539] | -0.0250 | 97/0/103 | 0 | 5.77 |
| `-bench_dev` | 0.5100 | [0.4412, 0.5784] | +0.0000 | 102/0/98 | 0 | 6.72 |
| `-hand_value` | 0.5200 | [0.4510, 0.5882] | +0.0100 | 104/0/96 | 0 | 7.18 |
| `-energy_tempo` | 0.4750 | [0.4069, 0.5440] | -0.0350 | 95/0/105 | 0 | 6.35 |
| `-retreat_capacity` | 0.5100 | [0.4412, 0.5784] | +0.0000 | 102/0/98 | 0 | 4.78 |

## Reading

- The R4 board evaluation is injected as an **informed tie-break** inside R3's category ordering (`agents/eval_agent.py`), so a component only changes the outcome when it decides which option is taken among R3's equal-best set — a narrow lever.
- The cabt engine is **unseeded** (see `eval/config.py`), so these per-variant win rates vary run to run. Across variants every single-component Δ stays within a few points and inside the full baseline's Wilson interval: **no component individually produces a reproducible change in the R3 head-to-head at this budget** — the deltas are consistent with engine noise (re-running reorders which components look ±).
- Because nothing robustly contributes, there is no component to prune with confidence: the **full seven-component evaluation ships unchanged** (a tempting seed-0 gain from dropping `retreat_capacity` did not replicate on independent seeds, so it was not taken).
- Net: the R4 board-eval tie-break is **statistically tied** with the R3 champion — the decisive gate (`eval/bench_r4_vs_rule.py`, N=400, seeds 0/1/2) lands at a win rate near 0.50 with a Wilson 95% CI lower bound below 0.50, so **R3 stays champion** — the same finding as the R5 one-ply search (SOT-1650).
- Zero faults across every variant satisfies 受け入れ条件③ (違法出力0); p99 latency stays well inside the submission budget.
