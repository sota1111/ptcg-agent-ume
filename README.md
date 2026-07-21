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

## Deck-optimization track (SOT-1651)
An **orthogonal** track that fixes the agent (the champion policy) and compares only the
**decks**, so deck iteration is never confounded with policy iteration. `eval/deck_eval.py`
holds one agent fixed on both seats and **swaps the decks** between seats every other match
(the dual of the Arena's agent-swap), giving an unbiased *paired* deck A/B: in a mirror the
win-rate CI brackets 0.5 even though the 先手 advantage is large. It also reports match-free
deck metrics — legality (60 cards / ≤4 copies / ≤1 ACE SPEC / ≥1 Basic Pokémon), energy ratio,
and 初動安定性 (hypergeometric P(≥1 Basic Pokémon in the opening hand)) — and `run_gauntlet`
for champion-vs-many matchup別勝率. Champion decks are version-managed under `decks/`
(`decks/registry.json` pins each version to a content hash).

```bash
venv/bin/python eval/deck_eval.py 200 decks/challenger_example.csv  # paired A/B, N=200
```

## Setup
```bash
git submodule update --init --recursive
bash scripts/check_core_compatibility.sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash scripts/setup_engine.sh          # copies cg/ + data/ from the Kaggle download
venv/bin/python eval/run_match.py     # run one local self-play match
```

## Shared core dependency

This repository consumes [`ptcg-agent-core`](vendor/ptcg-agent-core) as a
pinned Git submodule, using the same integration boundary as the other PTCG
agents. Core owns algorithm-independent contracts and the shared
[Kaggle submission guide](vendor/ptcg-agent-core/docs/kaggle-submission.md).
Ume continues to own its Python adapter, deck, PPO/MCTS policy, and evaluation
logic; those are deliberately outside the common-core contract.

The pinned commit keeps setup and submission builds reproducible. To update
core, review its schema versions and release notes, then run:

```bash
git -C vendor/ptcg-agent-core fetch origin main
git -C vendor/ptcg-agent-core checkout origin/main
bash scripts/check_core_compatibility.sh
venv/bin/python -m pytest eval/tests/
git add vendor/ptcg-agent-core
```

Commit the gitlink update together with the compatibility results. If the new
core is incompatible, restore the previous gitlink with
`git checkout -- vendor/ptcg-agent-core`, run
`git submodule update --init`, and re-run both checks. Do not use an unreviewed
moving branch for a submission build.

## Build a submission
```bash
bash scripts/build_submission.sh      # -> submission.tar.gz (main.py + deck.csv + cg/)
```

The builder follows the core-owned submission layout and checks that required
top-level files exist while development files, credentials, Git metadata, and
the core checkout remain outside the archive. For authentication, submission,
result checks, and troubleshooting, follow the
[shared core guide](vendor/ptcg-agent-core/docs/kaggle-submission.md).
