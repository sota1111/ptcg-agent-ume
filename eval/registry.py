"""Card master registry + card-independence invariants for the eval env (SOT-1625).

Why this module exists
----------------------
The cabt engine is the **sole rule authority**: it owns the card list, the
attacks, and every legality/damage rule. Card pools grow *during* the
competition ("new attributes may be appended … during the competition", see
``cg/api.py``). If eval-core code branched on individual card IDs, every card
addition would force a core rewrite and silently rot.

``CardRegistry`` is the single, read-only access point for card/attack master
data in ``eval/``. It normalises the engine's flat ``AllCard()`` / ``AllAttack()``
lists into id-indexed lookups and caches them, so:

* **agent / trace / report** read card + attack data from one place
  (``eval.registry.get_registry()``) instead of re-scanning the engine lists;
* an **added card is retrieved dynamically** — the registry pulls from the
  engine, so a brand-new card id resolves with no core change (acceptance:
  「カード追加だけではコア実装変更が不要」);
* **duplicate / missing ids** have an explicit failure contract
  (:class:`DuplicateCardError`, :class:`CardNotFoundError`).

The registry only *reads* master data. It never re-implements a rule (damage,
legality, evolution) — those stay in the engine. The companion invariant test
``eval/tests/test_no_hardcoded_card_id.py`` enforces the boundary by rejecting
individual card/attack-id literals in eval-core modules.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator, Mapping, Optional

if TYPE_CHECKING:  # avoid importing the engine at module import time
    from cg.api import Attack, CardData

__all__ = [
    "CardRegistry",
    "RegistryError",
    "DuplicateCardError",
    "DuplicateAttackError",
    "CardNotFoundError",
    "AttackNotFoundError",
    "get_registry",
    "clear_registry_cache",
]


# --------------------------------------------------------------------------- #
# Failure contract
# --------------------------------------------------------------------------- #

class RegistryError(Exception):
    """Base class for all registry errors."""


class DuplicateCardError(RegistryError):
    """Two cards share a ``cardId`` — the master data is ambiguous."""

    def __init__(self, card_id: int):
        self.card_id = card_id
        super().__init__(f"duplicate cardId in master data: {card_id}")


class DuplicateAttackError(RegistryError):
    """Two attacks share an ``attackId`` — the master data is ambiguous."""

    def __init__(self, attack_id: int):
        self.attack_id = attack_id
        super().__init__(f"duplicate attackId in master data: {attack_id}")


class CardNotFoundError(RegistryError, KeyError):
    """No card with the requested ``cardId`` exists in the registry."""

    def __init__(self, card_id: int):
        self.card_id = card_id
        RegistryError.__init__(self, f"unknown cardId: {card_id}")


class AttackNotFoundError(RegistryError, KeyError):
    """No attack with the requested ``attackId`` exists in the registry."""

    def __init__(self, attack_id: int):
        self.attack_id = attack_id
        RegistryError.__init__(self, f"unknown attackId: {attack_id}")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class CardRegistry:
    """Read-only, id-indexed view over the engine's card + attack master data.

    Construct it once (usually via the cached :func:`get_registry`) and share it.
    The underlying dicts are exposed only through :class:`~types.MappingProxyType`
    so callers cannot mutate the master data.

    Two constructors:

    * :meth:`from_engine` — pull the live lists from ``cg.api`` (the source of
      truth). This is what makes a newly added card resolvable with no code change.
    * :meth:`from_records` — build from explicit ``CardData`` / ``Attack`` lists.
      Used by unit tests to exercise the duplicate/missing contracts without the
      engine installed.
    """

    def __init__(
        self,
        cards: Mapping[int, "CardData"],
        attacks: Mapping[int, "Attack"],
    ):
        # Wrap in read-only proxies so the master data can't be mutated in place.
        self._cards: Mapping[int, "CardData"] = MappingProxyType(dict(cards))
        self._attacks: Mapping[int, "Attack"] = MappingProxyType(dict(attacks))

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_records(
        cls,
        cards: "list[CardData]",
        attacks: "list[Attack]",
    ) -> "CardRegistry":
        """Index explicit card/attack records by id.

        Raises :class:`DuplicateCardError` / :class:`DuplicateAttackError` when an
        id appears twice — the master data must be unambiguous for a lookup to be
        well-defined.
        """
        card_index: dict[int, "CardData"] = {}
        for card in cards:
            cid = card.cardId
            if cid in card_index:
                raise DuplicateCardError(cid)
            card_index[cid] = card

        attack_index: dict[int, "Attack"] = {}
        for atk in attacks:
            aid = atk.attackId
            if aid in attack_index:
                raise DuplicateAttackError(aid)
            attack_index[aid] = atk

        return cls(card_index, attack_index)

    @classmethod
    def from_engine(cls) -> "CardRegistry":
        """Build the registry from the live engine master data (source of truth).

        Imports ``cg.api`` lazily so this module stays importable when the
        competition engine is absent (e.g. in CI).
        """
        from cg.api import all_attack, all_card_data

        return cls.from_records(all_card_data(), all_attack())

    # -- read-only mappings ------------------------------------------------ #
    @property
    def cards(self) -> Mapping[int, "CardData"]:
        """Read-only ``cardId -> CardData`` mapping."""
        return self._cards

    @property
    def attacks(self) -> Mapping[int, "Attack"]:
        """Read-only ``attackId -> Attack`` mapping."""
        return self._attacks

    # -- card lookups ------------------------------------------------------ #
    def card(self, card_id: int) -> "CardData":
        """Return the card for ``card_id`` or raise :class:`CardNotFoundError`."""
        try:
            return self._cards[card_id]
        except KeyError:
            raise CardNotFoundError(card_id) from None

    def get_card(self, card_id: int, default=None):
        """Return the card for ``card_id`` or ``default`` if it is unknown."""
        return self._cards.get(card_id, default)

    def has_card(self, card_id: int) -> bool:
        """Whether ``card_id`` exists in the master data."""
        return card_id in self._cards

    def card_name(self, card_id: int) -> str:
        """Convenience: the display name of ``card_id`` (raises if unknown)."""
        return self.card(card_id).name

    # -- attack lookups ---------------------------------------------------- #
    def attack(self, attack_id: int) -> "Attack":
        """Return the attack for ``attack_id`` or raise :class:`AttackNotFoundError`."""
        try:
            return self._attacks[attack_id]
        except KeyError:
            raise AttackNotFoundError(attack_id) from None

    def get_attack(self, attack_id: int, default=None):
        """Return the attack for ``attack_id`` or ``default`` if it is unknown."""
        return self._attacks.get(attack_id, default)

    def has_attack(self, attack_id: int) -> bool:
        """Whether ``attack_id`` exists in the master data."""
        return attack_id in self._attacks

    def attack_name(self, attack_id: int) -> str:
        """Convenience: the display name of ``attack_id`` (raises if unknown)."""
        return self.attack(attack_id).name

    def attacks_for(self, card_id: int) -> "list[Attack]":
        """Resolve a card's ``attacks`` id list to :class:`Attack` objects.

        Raises :class:`CardNotFoundError` if the card is unknown, and
        :class:`AttackNotFoundError` if the card references an attack id that is
        missing from the master data (an engine-data inconsistency).
        """
        card = self.card(card_id)
        return [self.attack(aid) for aid in card.attacks]

    # -- container protocol ------------------------------------------------ #
    def __len__(self) -> int:
        """Number of distinct cards."""
        return len(self._cards)

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._cards

    def __iter__(self) -> Iterator[int]:
        """Iterate over card ids."""
        return iter(self._cards)

    def __repr__(self) -> str:
        return f"CardRegistry(cards={len(self._cards)}, attacks={len(self._attacks)})"


# --------------------------------------------------------------------------- #
# Cached module-level accessor (the single reference point)
# --------------------------------------------------------------------------- #

_REGISTRY: Optional[CardRegistry] = None


def get_registry() -> CardRegistry:
    """Return the process-wide :class:`CardRegistry`, building it once and caching.

    This is the single entry point agent / trace / report code should use to read
    card + attack master data. The engine lists are large and immutable within a
    process, so they are loaded once and reused.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CardRegistry.from_engine()
    return _REGISTRY


def clear_registry_cache() -> None:
    """Drop the cached registry (next :func:`get_registry` rebuilds it).

    Mainly for tests; a running match never needs to invalidate the master data.
    """
    global _REGISTRY
    _REGISTRY = None
