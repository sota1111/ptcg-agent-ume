"""Local self-play match runner for the PTCG AI Battle eval environment.

Backward-compatible CLI wrapper over the typed engine boundary
(:mod:`eval.environment`) and the match loop (:mod:`eval.match`). The engine's
global/native state is confined to ``Environment``; this file only wires two
:class:`RandomAgent` s into ``play_match`` and prints the legacy result line.

Usage: python eval/run_match.py [deck0.csv] [deck1.csv]
Run from repo root (after scripts/setup_engine.sh has populated cg/).
"""
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)         # make `cg` and `eval` importable
os.chdir(REPO)                   # so libcg.so & deck.csv resolve

from cg.api import to_observation_class  # noqa: E402  (kept for compat imports)
from eval.agents import RandomAgent  # noqa: E402
from eval.match import play_match  # noqa: E402


def load_deck(path):
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def random_agent(obs_dict):
    """Legacy free-function agent kept for backward compatibility. Mirrors the
    original uniform-random selection over ``obs.select``."""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return None
    n = len(obs.select.option)
    k = max(obs.select.minCount, min(obs.select.maxCount, n))
    return random.sample(range(n), k) if n else []


def run(deck0, deck1, max_steps=100000):
    """Play one random self-play match. Returns ``(winner, steps)`` where winner
    is the player index (0/1) or -1 for a draw/unresolved — matching the original
    contract. Both agents draw from the global ``random`` stream, so seeding
    ``random`` before calling ``run`` keeps play reproducible as before."""
    agents = [RandomAgent(rng=random), RandomAgent(rng=random)]
    result = play_match(deck0, deck1, agents, max_steps=max_steps)
    winner = -1 if result.winner is None else result.winner
    return winner, result.steps


if __name__ == "__main__":
    random.seed(42)
    d0 = load_deck(sys.argv[1]) if len(sys.argv) > 1 else load_deck("deck.csv")
    d1 = load_deck(sys.argv[2]) if len(sys.argv) > 2 else d0
    result, steps = run(d0, d1)
    print(f"MATCH DONE: winner=player{result}  decisions={steps}")
