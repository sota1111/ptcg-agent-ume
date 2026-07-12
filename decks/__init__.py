"""Versioned champion decks for the deck-optimization track (SOT-1651, R6).

The **champion deck** is version-managed here, separately from the policy: each
champion version is a committed ``champion_<version>.csv`` (a 60-card id list) and an
entry in :data:`registry.json` recording its content hash, creation date and notes.
This keeps deck iteration (this track) auditable and disjoint from policy iteration
(the RuleAgent rounds) — a champion version pins an exact card multiset regardless of
file ordering (see :func:`decks.champion.load_champion`).
"""

from .champion import (
    CHAMPION_DIR,
    REGISTRY_PATH,
    champion_versions,
    current_version,
    load_champion,
)

__all__ = [
    "CHAMPION_DIR",
    "REGISTRY_PATH",
    "champion_versions",
    "current_version",
    "load_champion",
]
