"""Load a versioned champion deck (SOT-1651, R6).

``registry.json`` is the source of truth for champion versions: a ``current`` pointer
and a ``versions`` table mapping each version id to its ``champion_<version>.csv`` file
and metadata (content hash, creation date, notes). :func:`load_champion` resolves a
version (default: ``current``) into a :class:`~eval.deck_eval.DeckSpec` and, when the
registry records a ``deck_hash``, verifies the on-disk deck still hashes to it — so a
champion version is pinned to an exact, tamper-evident card multiset.
"""

from __future__ import annotations

import json
import os
from typing import Optional

CHAMPION_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(CHAMPION_DIR, "registry.json")


def _load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def current_version() -> str:
    """The registry's ``current`` champion version id."""
    return _load_registry()["current"]


def champion_versions() -> dict:
    """The full ``versions`` table (version id -> metadata) from the registry."""
    return _load_registry().get("versions", {})


def load_champion(version: Optional[str] = None):
    """Load the champion deck for ``version`` (default: the registry's ``current``).

    Returns a :class:`~eval.deck_eval.DeckSpec` named ``champion`` at that version.
    Raises ``KeyError`` for an unknown version, ``FileNotFoundError`` for a missing
    deck file, and ``ValueError`` if the recorded ``deck_hash`` no longer matches the
    file's contents (the champion has been edited out from under its version).
    """
    from eval.deck_eval import load_deck

    reg = _load_registry()
    version = version or reg["current"]
    versions = reg.get("versions", {})
    if version not in versions:
        raise KeyError(f"unknown champion version {version!r}; have {sorted(versions)}")
    entry = versions[version]
    path = os.path.join(CHAMPION_DIR, entry["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"champion deck file missing: {path}")
    deck = load_deck(path, name="champion", version=version)
    recorded = entry.get("deck_hash")
    if recorded and deck.deck_hash != recorded:
        raise ValueError(
            f"champion {version} hash mismatch: file={deck.deck_hash} "
            f"registry={recorded} (deck edited without a version bump?)"
        )
    return deck
