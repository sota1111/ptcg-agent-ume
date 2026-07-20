"""Kaggle submission entry — PPO policy + critical-position MCTS + decision harness.

SOT-1691 (the final stage of the SOT-1683 PPO track): every decision is made by
:class:`agents.harness.HarnessAgent` — the PPO policy (SOT-1689) plays each
selection, critical positions are re-evaluated by the determinized MCTS
(SOT-1690), and the candidate harness validates/scores/decides so the returned
action is legal unconditionally.

Runs on pure Python + the bundled files only (no pip deps): the policy weights
are ``data/policy.json``, the deck list is ``deck.csv``. Paths resolve relative
to this file first and then to the Kaggle agent bundle directory
(``/kaggle_simulations/agent/``), so the same entry point works from the repo
root, from an arbitrary cwd, and inside a Kaggle simulation.
"""

import os
import secrets
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"

# Make the bundled packages (agents/, cg/) importable wherever we were loaded from.
for _base in (_HERE, _KAGGLE_AGENT_DIR):
    if os.path.isdir(_base) and _base not in sys.path:
        sys.path.insert(0, _base)

from cg.api import Observation, to_observation_class  # noqa: E402

from agents.harness import HarnessAgent  # noqa: E402
from agents.compatibility import CompatibilityAdapter, LegacyDeckStrategy  # noqa: E402
from agents.mcts import MCTSConfig  # noqa: E402


def _resolve(relpath: str) -> str:
    """Absolute path of a bundled file: next to this file, else the Kaggle bundle."""
    for base in (_HERE, _KAGGLE_AGENT_DIR):
        path = os.path.join(base, relpath)
        if os.path.exists(path):
            return path
    return relpath


_DECK_PATH = _resolve("deck.csv")

# SOT-1701: calibrated from the enhanced battle evidence.  The learned PPO
# policy supplies most submitted decisions, so reducing tail-action sampling
# improves exploitation while retaining stochastic play.
PPO_TEMPERATURE = 0.25


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    with open(_DECK_PATH, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


# One agent instance for the whole match (keeps the RNG stream and the
# MCTS/harness measurement across decisions). time_limit_s=0.4 is the
# SOT-1690-measured per-decision search cap (max think ~0.4s per decision,
# well inside the competition持ち時間).
_AGENT_SEED = secrets.randbits(63)


def _new_harness_agent() -> HarnessAgent:
    return HarnessAgent(
        seed=_AGENT_SEED,
        policy_path=_resolve(os.path.join("data", "policy.json")),
        temperature=PPO_TEMPERATURE,
        mcts=True,
        mcts_config=MCTSConfig(time_limit_s=0.4, deck_path=_DECK_PATH),
    )


_legacy_agent = _new_harness_agent()
_candidate_agent = _new_harness_agent()
_agent = CompatibilityAdapter(
    legacy=_legacy_agent,
    candidate=LegacyDeckStrategy(_candidate_agent),
    mode=os.environ.get("PTCG_UME_MIGRATION_MODE", "legacy"),
)


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:  # noqa: E711 - keep the official template's contract check
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()

    return _agent.act(obs_dict)
