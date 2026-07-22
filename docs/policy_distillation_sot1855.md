# SOT-1855 MCTS policy distillation

## Method

- Teacher: `ptcg-agent-fable` champion determinized MCTS, isolated through its
  JSON-line agent server to avoid sibling repositories' `agents`/`cg` package
  collisions.
- Collection: 8 side-alternating matches against ume's RuleAgent; 162
  non-trivial single-select teacher decisions; collection faults 0.
- Student update: initialise from committed `data/policy.json`, then run 12
  masked cross-entropy distillation passes (`lr=0.001`). The value head is not
  trained by teacher data.
- Gate: compare the committed baseline and distilled candidate with the same
  seed over 200 side-swapped games per opponent. Promote only when the
  aggregate Wilson 95% evidence establishes an improvement; otherwise retain
  the committed champion, preserving the SOT-1695 best-iteration rule.

## A/B result

| Policy | Opponent | W-L | Win rate | Wilson 95% CI | Faults |
| --- | --- | ---: | ---: | --- | ---: |
| committed baseline | Random | 148-52 | 0.740 | [0.6751, 0.7959] | 0 |
| distilled candidate | Random | 155-45 | 0.775 | [0.7123, 0.8274] | 0 |
| committed baseline | Rule | 41-159 | 0.205 | [0.1549, 0.2663] | 0 |
| distilled candidate | Rule | 42-158 | 0.210 | [0.1593, 0.2716] | 0 |

Decision: **not promoted**. Both point estimates increased, but both candidate
confidence intervals overlap the corresponding baseline intervals. The
committed `data/policy.json` therefore remains unchanged. The reusable
collector and supervised update are retained so a larger teacher corpus can be
evaluated without weakening the significance gate.

## Kaggle validation incident

Submissions `54904762` and `54904924` failed before their first action. Episode
logs show that `kaggle_environments` had preloaded its unrelated top-level
`agents` module; Python reused that cached module when `main.py` imported
`agents.harness`, despite the submission directory being first on `sys.path`.

The submission entry now evicts only foreign `agents` modules from
`sys.modules`, preserves modules already loaded from its own bundle, and has an
exec-loader regression test that preloads the conflicting package shape.
