# ptcg-agent-ume

Agent & local evaluation environment for the **Pokémon TCG AI Battle Challenge** (Kaggle).

- Competition (Simulation): https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Competition (Strategy):   https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy
- **Competition info summary:** [`docs/competition.md`](docs/competition.md)

## ⚠️ License note
The battle engine (`cg/`, `libcg.so`) and card data (`data/`) are **competition-use-only and must not
be redistributed**. They are **gitignored** and never committed. Only our own code
(`main.py`, `deck.csv`, `eval/`, `scripts/`) lives in git.

## Layout
```
main.py             # submission entry: agent(obs_dict) -> list[int]  (tracked)
deck.csv            # our 60-card deck                                (tracked)
eval/environment.py # cabt engine boundary: Environment (global/native state confined here)
eval/agents/base.py # Agent Protocol act(obs)->list[int] + reference agents
eval/match.py       # play_match: agent-vs-agent loop, structured results/faults
eval/run_match.py   # local self-play CLI (backward-compatible wrapper)
eval/tests/         # pytest suite (skips when cg/ engine absent)
scripts/            # setup + build helpers                          (tracked)
cg/                 # cabt engine bindings (gitignored, license)
data/               # card CSVs (gitignored, license)
```

## Eval architecture (SOT-1623)
The cabt engine keeps the live battle in a **single process-global pointer** and talks over
`ctypes`. All of that is confined to `eval/environment.py`:

- **`Environment`** — a context manager owning one battle. `start/step/finish` are the typed API;
  `battle_finish()` is guaranteed to run (even on exception) and runs **exactly once**. A
  module-level single-active guard enforces **one battle per process** (run parallel matches in
  separate processes). `obs.select.option` is exposed as the **sole source of legal moves** — the
  rules are never re-implemented; `validate_action` only checks an action's shape against it.
- **Agent Protocol** (`eval/agents/base.py`) — `act(obs) -> list[int]`, matching the Kaggle
  submission entry point, plus optional `on_match_start` / `on_match_end` hooks. Reference agents:
  `RandomAgent`, `FirstOptionAgent`; `SubmissionAgent` adapts a bare `agent(obs_dict)` callable.
- **`play_match`** (`eval/match.py`) — drives a full match and returns a structured `MatchResult`.
  An illegal move, per-move timeout, or agent exception is reported as **that agent's loss**, not a
  crash.

Note: the engine's internal shuffle is unseeded, so match outcomes are non-deterministic even at a
fixed Python seed.

```bash
venv/bin/python -m pytest eval/tests/   # run the eval test suite (needs cg/)
```

## Setup
```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash scripts/setup_engine.sh          # copies cg/ + data/ from the Kaggle download
venv/bin/python eval/run_match.py     # run one local self-play match
```

## Build a submission
```bash
bash scripts/build_submission.sh      # -> submission.tar.gz (main.py + deck.csv + cg/)
```
