"""Hidden-information predictor for determinized search (SOT-1650 R5 → SOT-1691).

:class:`UniformDeckPredictor` samples the hidden zones (decks, prizes, the
opponent's hand/face-down Active) uniformly from an assumed 60-card deck list
minus the cards already visible, producing the six hidden-card lists
``cg.api.search_begin`` expects. Written for the R5 one-ply SearchAgent and now
owned by the determinized MCTS (:mod:`agents.mcts`) — the sole surviving search
line after SOT-1691 removed the superseded rule/search/eval agents.

Engine-free imports: the observation is only read via attributes, so this
module (and its tests) import without the gitignored ``cg/`` engine.
"""

from __future__ import annotations

import collections
import random
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # avoid importing the engine at module import time
    from cg.api import Observation

__all__ = ["UniformDeckPredictor"]


def _visible_card_ids(player, *, include_hand: bool) -> list[int]:
    """Card IDs currently visible on the board for one player.

    Pokémon in play (Active/Bench) with their attached energy/tool/pre-evolution cards,
    the discard pile, revealed prize cards, and — only where the hand is visible (our
    own side) — the hand. These are removed from the deck multiset to form the hidden
    pool. Fully guarded (an absent zone contributes nothing).
    """
    ids: list[int] = []

    def add_pokemon(pk) -> None:
        if pk is None:
            return
        pid = getattr(pk, "id", None)
        if pid is not None:
            ids.append(pid)
        for group in (getattr(pk, "energyCards", None), getattr(pk, "tools", None), getattr(pk, "preEvolution", None)):
            for card in group or []:
                cid = getattr(card, "id", None)
                if cid is not None:
                    ids.append(cid)

    for pk in getattr(player, "active", None) or []:
        add_pokemon(pk)
    for pk in getattr(player, "bench", None) or []:
        add_pokemon(pk)
    for card in getattr(player, "discard", None) or []:
        cid = getattr(card, "id", None)
        if cid is not None:
            ids.append(cid)
    for card in getattr(player, "prize", None) or []:
        cid = getattr(card, "id", None) if card is not None else None
        if cid is not None:  # revealed prize only
            ids.append(cid)
    if include_hand:
        for card in getattr(player, "hand", None) or []:
            cid = getattr(card, "id", None)
            if cid is not None:
                ids.append(cid)
    return ids


class UniformDeckPredictor:
    """Sample the hidden zones uniformly from a deck list minus visible cards.

    ``deck_ids`` is the assumed 60-card composition, reused for both players (no
    opponent-deck model yet). Seeded for reproducibility. A wrong guess just makes
    ``search_begin`` reject the candidate; the caller's fail-closed fallback still
    holds. Swap for a deck-tracking predictor later without touching the agent.
    """

    def __init__(self, deck_ids: Sequence[int], rng: random.Random) -> None:
        self.deck_ids = list(deck_ids)
        self.rng = rng

    def _pool(self, player, *, include_hand: bool) -> list[int]:
        remaining = collections.Counter(self.deck_ids)
        for cid in _visible_card_ids(player, include_hand=include_hand):
            remaining[cid] -= 1
        pool: list[int] = []
        for cid, n in remaining.items():
            if n > 0:
                pool.extend([cid] * n)
        self.rng.shuffle(pool)
        return pool

    def _take(self, pool: list[int], n: int) -> list[int]:
        """Pop ``n`` cards off ``pool``; top up from the deck list if short."""
        if n <= 0:
            return []
        out = pool[:n]
        del pool[:n]
        i = 0
        while len(out) < n and self.deck_ids:
            out.append(self.deck_ids[i % len(self.deck_ids)])
            i += 1
        return out

    def predict(self, obs: "Observation", your_index: int) -> tuple:
        """Return the six hidden-card lists in ``search_begin`` argument order."""
        state = obs.current
        players = state.players
        me = players[your_index]
        opp = players[1 - your_index]

        # Our side: the hand is visible, so subtract it; deck + prize are hidden.
        my_pool = self._pool(me, include_hand=True)
        your_deck = self._take(my_pool, me.deckCount)
        your_prize = self._take(my_pool, len(me.prize))

        # Opponent side: the hand is hidden too, so don't subtract it.
        opp_pool = self._pool(opp, include_hand=False)
        opponent_deck = self._take(opp_pool, opp.deckCount)
        opponent_prize = self._take(opp_pool, len(opp.prize))
        opponent_hand = self._take(opp_pool, opp.handCount)
        active = opp.active or []
        active_facedown = len(active) > 0 and active[0] is None
        opponent_active = self._take(opp_pool, 1) if active_facedown else []

        return (
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
        )
