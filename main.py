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

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _EXEC_WITHOUT_FILE = False
except NameError:
    # Kaggle executes this file with exec() and does not define __file__.
    _HERE = os.path.abspath(os.getcwd())
    _EXEC_WITHOUT_FILE = True
_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"

# Make the bundled packages (agents/, cg/) importable wherever we were loaded
# from. Kaggle's extracted bundle is authoritative there; otherwise use the
# directory containing this source (or the exec() cwd fallback).
_BUNDLE_DIR = (
    _KAGGLE_AGENT_DIR
    if _EXEC_WITHOUT_FILE and os.path.isdir(_KAGGLE_AGENT_DIR)
    else _HERE
)
if not sys.path or sys.path[0] != _BUNDLE_DIR:
    sys.path.insert(0, _BUNDLE_DIR)

# kaggle_environments may preload an unrelated top-level package named
# ``agents`` (for example lux_ai_s3.agents) before exec() reaches this file.
# sys.path precedence cannot replace an entry already cached in sys.modules,
# so discard only foreign ``agents`` modules and let Python load our bundled
# package. Never evict an already-loaded module from this submission itself.
_bundle_agents = os.path.realpath(os.path.join(_BUNDLE_DIR, "agents"))
_loaded_agents = sys.modules.get("agents")
_loaded_agents_file = getattr(_loaded_agents, "__file__", "") if _loaded_agents else ""
if _loaded_agents is not None and not (
    _loaded_agents_file
    and os.path.realpath(_loaded_agents_file).startswith(_bundle_agents + os.sep)
):
    for _module_name in list(sys.modules):
        if _module_name == "agents" or _module_name.startswith("agents."):
            del sys.modules[_module_name]

from cg.api import Observation, to_observation_class  # noqa: E402

from agents.harness import HarnessAgent  # noqa: E402
from agents.compatibility import CompatibilityAdapter, LegacyDeckStrategy  # noqa: E402
from agents.mcts import MCTSConfig  # noqa: E402
from agents.runtime_profile import load_runtime_profile  # noqa: E402


def _resolve(relpath: str) -> str:
    """Absolute path of a bundled file: next to this file, else the Kaggle bundle."""
    for base in (_HERE, _KAGGLE_AGENT_DIR):
        path = os.path.join(base, relpath)
        if os.path.exists(path):
            return path
    return relpath


_DECK_PATH = _resolve("deck.csv")

# SOT-1875: the hardened high-variance profile is a bundled, versioned runtime
# artifact.  Loading one source of truth prevents the submission entry point
# and evaluation harness from silently drifting apart.
# SOT-1898: an alternate profile may be selected via PTCG_UME_PROFILE (a path,
# absolute or bundle-relative) so the league KPI gate can A/B the search-driven
# candidate against the committed champion without editing the deck bundle. The
# committed default remains the champion profile.
_PROFILE_OVERRIDE = os.environ.get("PTCG_UME_PROFILE")
RUNTIME_PROFILE = (
    load_runtime_profile(_resolve(_PROFILE_OVERRIDE))
    if _PROFILE_OVERRIDE
    else load_runtime_profile()
)
PPO_TEMPERATURE = RUNTIME_PROFILE.policy_temperature


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
# MCTS/harness measurement across decisions).  Search and candidate-generation
# knobs come from the hardened profile and remain below the 5 s move timeout
# and 600 s Kaggle match budget recorded in that artifact.
_AGENT_SEED = secrets.randbits(63)


def _new_harness_agent() -> HarnessAgent:
    return HarnessAgent(
        seed=_AGENT_SEED,
        policy_path=_resolve(os.path.join("data", "policy.json")),
        temperature=PPO_TEMPERATURE,
        mcts=True,
        mcts_config=MCTSConfig(**{
            **RUNTIME_PROFILE.mcts.__dict__,
            "deck_path": _DECK_PATH,
        }),
        harness_config=RUNTIME_PROFILE.harness,
    )


_legacy_agent = _new_harness_agent()
_candidate_agent = _new_harness_agent()
_agent = CompatibilityAdapter(
    legacy=_legacy_agent,
    candidate=LegacyDeckStrategy(_candidate_agent),
    mode=os.environ.get("PTCG_UME_MIGRATION_MODE", "core"),
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
