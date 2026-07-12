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
main.py            # submission entry: agent(obs_dict) -> list[int]  (tracked)
deck.csv           # our 60-card deck                                (tracked)
eval/run_match.py  # local self-play match runner                   (tracked)
scripts/           # setup + build helpers                          (tracked)
cg/                # cabt engine bindings (gitignored, license)
data/              # card CSVs (gitignored, license)
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
