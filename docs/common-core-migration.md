# 梅 compatibility adapter migration runbook

梅 keeps the existing PPO + MCTS `HarnessAgent` as the production default while exposing it through
the common `ptcg-deck-strategy/v1` / `ptcg-agent-adapter/v1` compatibility boundary. Both paths are
constructed from the same policy, search configuration, and per-process seed. The mode is read once at
startup, so a match never changes routing midway.

| Mode | `PTCG_UME_MIGRATION_MODE` | Authoritative result | Purpose |
| --- | --- | --- | --- |
| Legacy | unset or `legacy` | existing harness | default and rollback |
| Shadow | `shadow` | existing harness | compare candidate without gameplay risk |
| Core | `core` | versioned strategy adapter | staged cutover |

Shadow mode writes one JSON object per decision to stderr. `matched` reports exact option-index parity,
while `compatible` checks both results against the engine's action contract (selection count, unique
indices, and option range). Exact actions may differ because PPO sampling and time-bounded MCTS are
stochastic; treat mismatch rate as behavioral drift evidence, not an adapter failure. stdout remains
reserved for the battle protocol. Before cutover, run representative fixtures and real battles in
`shadow`, then require zero `compatible:false` events and zero agent/engine faults.

```bash
PTCG_UME_MIGRATION_MODE=shadow venv/bin/python -m pytest eval/tests/test_compatibility.py
PTCG_UME_MIGRATION_MODE=core venv/bin/python eval/arena.py --games 4 --workers 1
```

## Rollback

Set `PTCG_UME_MIGRATION_MODE=legacy` (or remove it) and restart the agent process. Verify startup by
requesting the initial observation and confirming the committed 60-card deck is returned, then run one
fixture match. No source revert, model change, or artifact rebuild is required. Unknown modes and
incompatible future strategy/adapter versions fail at startup instead of silently selecting a path.
