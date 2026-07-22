# SOT-1875 hardened Ume runtime promotion

The committed `ume-high-variance-pressure-v1` profile is bound directly to the
Kaggle entry point and submission archive. The hardened setting retains more
exploration than legacy (temperature 0.35 vs 0.25; UCB 1.25 vs 1.0) while
restoring the 0.4-second search budget that performed reliably in production.

## Fixed-seed, seat-reversed runtime result

| Matchup | Result | Win rate | Wilson 95% | Safety |
| --- | ---: | ---: | ---: | ---: |
| hardened vs legacy | 10-10 | 50% | 29.9%–70.1% | 0 faults / unfinished / illegal |
| hardened vs RuleAgent | 8-12 | 40% | 21.9%–61.3% | 0 faults / unfinished / illegal |

The legacy Wilson lower-bound alternative was not met at this sample size. The
real hard-runtime league result improved from the previous integrated profile's
25% to 40% (+15 points), satisfying the issue's alternative improvement gate.
Maximum observed think time was 401.7 ms, below the 5-second per-move timeout;
the profile pins the Kaggle 600-second match budget. The engine does not expose
a seed API, so the artifact records fixed agent RNG seeds and exact seat reversal.

## Heterogeneous runtime cross-play

The exact repository submission entry points ran in isolated processes for two
seat-reversed games each: Sol 1-1, Debate 2-0, Fable 0-2, and Zero 2-0. Both
sides completed all eight games with zero faults, unfinished games, or illegal
actions. This small sample is a runtime compatibility/safety check, not a claim
of statistically significant superiority.

## Package verification

`scripts/build_submission.sh` produced a gzip-valid archive containing
`main.py`, the legal 60-card deck, policy, engine, and both runtime profile
files. The repository suite passed 402 tests; shared-core typecheck and all 13
shared-core tests passed.
