# SOT-1898 — Search-driven ume (all-decision MCTS + PPO prior)

**Goal.** Lift ume out of the bottom of the Kaggle field (444.9) by making it
*search-driven*: expand the SOT-1690 critical-position-only determinized MCTS to
**every** eligible decision, keep the PPO policy (`data/policy.json`) as the PUCT
prior and the rollout policy, and raise the per-decision search budget toward the
Kaggle match budget with an overspend guard. PPO-alone is known saturated for ume
(SOT-1855 distillation, SOT-1699 league, reward-shaping and board-eval ablation
all non-promoted; rule win-rate stuck at 0.52–0.56), while the Kaggle leaders
(fable/take/matsu) are all search-driven.

## What was built (opt-in; champion default unchanged)

All changes are gated behind a new, versioned candidate profile and are **off in
the committed champion** — `data/policy.json` is untouched.

- `MCTSConfig` gains three knobs (`agents/mcts.py`):
  - `all_decision` — search every eligible decision, not just critical ones.
  - `rollout_temperature` — softmax temperature of the PPO-guided rollout (was a
    fixed 1.0); tuned from the root `policy_temperature` (0.35).
  - `match_search_budget_s` — a match-level cumulative search-time budget.
- `DeterminizedMCTS.maybe_search` — in all-decision mode every eligible position
  is searched; once the match search budget is spent it **reverts to the
  critical-only gate** (現行harness挙動) and the per-decision cap is adaptively
  clamped to the remaining budget. With `all_decision=False` the behaviour is
  byte-for-byte the SOT-1690 champion.
- `agents/runtime_profile_search.json` — the `ume-search-driven-v1` candidate:
  `all_decision=true`, `time_limit_s=1.5` (vs champion 0.4), `rollout_temperature=0.35`,
  `deviate_margin=0.02` (vs 0.08 — a search-driven agent must be willing to act on
  the search), `match_search_budget_s=300`. Selectable at submission time via
  `PTCG_UME_PROFILE`, so the league gate can A/B it without touching the bundle.

## Evaluation

### In-repo A/B — search-driven vs the committed champion (self-play)

`eval/ab_search_driven_sot1898.py`, deck 01 mirror, seat-alternating, real cabt
engine.

| N  | search-driven wins | champion wins | search win-rate | CI95 | faults | search ms/max | activation |
| -- | ------------------ | ------------- | --------------- | ---- | ------ | ------------- | ---------- |
| 12 | 6 | 6 | **0.500** | [0.254, 0.746] | 0 / 0 | 1501 ms | 0.79 |

The search-driven agent is **statistically indistinguishable from the champion**
(exactly 6–6). Making ume search-driven — searching ~79% of decisions instead of
~25%, at nearly 4× the per-decision budget — produced **no strength gain**. Fault
0 held on both sides and the per-decision search time (max 1501 ms) stayed well
inside the 5 s per-move / 600 s match budget.

### League KPI gate (SOT-1896) — screen

Driver: `ptcg-agent-matsu/eval/battle_matsu_take_ume.py`, explicit-seat, deck 01
mirror, seat-alternating. ume (champion) vs the competitive pool {matsu, take} vs
ume (search-driven candidate, `PTCG_UME_PROFILE=agents/runtime_profile_search.json`).

Driver: `../ptcg-agent-matsu/eval/battle_matsu_take_ume.py`, explicit per-seat
(`--seat0 ume:01 --seat1 <opp>:01`), deck 01 mirror, seat-alternating, real cabt
engine, N=6 per pairing. `fable`/`sol` are excluded (their submission repos fail to
start in this environment, per SOT-1896), so the competitive pool is `{matsu, take}`
— the two agents directly above ume in the Kaggle field. The candidate is selected in
the ume server via `PTCG_UME_PROFILE=agents/runtime_profile_search.json`; the champion
run leaves it unset. Raw reports: `eval/runtime_promotion/sot-1898/league_screen/*.json`.

| ume variant | vs matsu | vs take | combined pool | faults |
| ----------- | -------- | ------- | ------------- | ------ |
| champion (critical-only, `all_decision=false`) | 2/6 = 0.333 | 0/6 = 0.000 | **2/12 = 0.167** | 0 |
| candidate (search-driven, `all_decision=true`)  | 1/6 = 0.167 | 1/6 = 0.167 | **2/12 = 0.167** | 0 |

The search-driven candidate does **not** lift ume's cross-agent league standing: the
combined win-rate against the competitive pool is **identical (2/12 = 0.167)** and, if
anything, the candidate *regresses* against matsu (0.333 → 0.167) while trading a
symmetric win against take — well inside the Wilson CIs (small-N screen). Fault 0 held
on both variants. Because the screen shows no positive signal (let alone the CI-clearing
lift required to justify the expensive large-N confirm), the gate stops at screen and
does not proceed to confirm — consistent with the SOT-1896 baseline that already places
ume at the bottom of the pool.

## Decision

**Non-promotion.** The all-decision search restructure does not convert extra
search into strength for ume: it ties the critical-only champion 6–6 in direct
self-play and does not lift ume's cross-agent league standing. This reproduces the
repeated SOT-1862/1864/1865 finding that for these saturated agents *more search
(quantity, depth, or now breadth) does not transfer to playing strength* — the
bottleneck is the value/policy quality, not the search budget.

Per the issue's contract, the champion is retained: the committed default profile
stays `ume-high-variance-pressure-v1` (critical-only, `all_decision=false`), so no
Kaggle re-submission is made (444.9 stands). The search-driven implementation is
kept as an **opt-in, off-by-default** capability (behind `PTCG_UME_PROFILE`) with
this negative result recorded, matching the sibling opt-in features SOT-1863/1864.

Acceptance:
- league KPI screen recorded (self-play A/B + cross-agent screen above) ✔
- non-promotion → champion retained + rationale docs (this file); no re-submission ✔
- fault 0 and per-decision time within budget (max 1501 ms ≤ 5000 ms) ✔
