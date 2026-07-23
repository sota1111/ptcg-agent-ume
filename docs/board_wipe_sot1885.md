# SOT-1885 board-wipe KPI / board-survival candidate

## Result

The candidate was **not promoted**. The committed champion remains unchanged.
The required small-N screen did not clear the promotion threshold, so the
conditional large-N confirmation and Kaggle resubmission were correctly skipped.

## Candidate

`HarnessConfig.board_survival_weight` enables a lightweight, opt-in candidate.
When the opponent active's largest printed attack can reach the current active's
remaining HP, it evaluates only engine-offered legal actions and prefers:

1. a legal retreat when a bench replacement exists;
2. strengthening a bench replacement;
3. developing the board when no reserve exists.

Weakness is included in reachable-damage estimation. The default weight is zero,
so this experimental behavior is not part of the champion without promotion.

## KPI definition

`BoardWipeTrackingAgent` records the same metrics for both sides:

- `board_wipes`: losses where the last observed own board had at most one Pokémon;
- `board_wipe_rate_in_losses`: board-wipe losses / all losses;
- `risk_exposures`: matches that reached a one-Pokémon board at half HP or lower;
- `board_wipe_avoidance_rate`: exposed matches that did not end as a board wipe /
  exposed matches.

The final knockout happens before the loser receives another observation, so a
last-active loss is classified from the last one-Pokémon state rather than a
zero-HP terminal observation.

## Small-N screen

Command:

```bash
venv/bin/python -m eval.bench_board_wipe_sot1885 \
  --small-n 20 --large-n 200 --out eval/sot1885_board_wipe.json
```

| Metric | Candidate | Champion |
| --- | ---: | ---: |
| W-L | 9-11 | 11-9 |
| Win rate | 0.450 | 0.550 |
| Wilson 95% CI | [0.2582, 0.6579] | [0.3421, 0.7418] |
| Faults | 0 | 0 |
| board_wipe count | 11 | 9 |
| board_wipe rate in losses | 1.000 | 1.000 |
| risk exposures | 1 | 0 |
| avoidance rate | 0.000 | 1.000 |

- Throughput: 0.0859 sims/sec (20 games / 232.73 seconds).
- Screen gate: **FAIL** because candidate Wilson lower bound `0.2582 <= 0.5`.
- Large-N: not run, by design.
- Champion update: none.
- Kaggle resubmission: not applicable because the champion was not updated.

The machine-readable result is committed as `eval/sot1885_board_wipe.json`.
