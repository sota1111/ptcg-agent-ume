"""Unit tests for the CardRegistry (SOT-1625).

Most tests use lightweight stand-in records (``SimpleNamespace``) so they run
without the competition engine installed — they exercise the indexing, the
read-only guarantees, the duplicate/missing failure contract, and dynamic
retrieval of a freshly added card. A final engine-backed block (skipped when the
engine is absent) checks the live ``from_engine`` / ``get_registry`` path.
"""
from types import MappingProxyType, SimpleNamespace

import pytest

from eval.registry import (
    AttackNotFoundError,
    CardNotFoundError,
    CardRegistry,
    DuplicateAttackError,
    DuplicateCardError,
    RegistryError,
    clear_registry_cache,
    get_registry,
)


def _card(card_id, name="c", attacks=()):
    return SimpleNamespace(cardId=card_id, name=name, attacks=list(attacks))


def _attack(attack_id, name="a"):
    return SimpleNamespace(attackId=attack_id, name=name)


@pytest.fixture
def reg():
    cards = [_card(1, "Pikachu", attacks=[10, 11]), _card(2, "Bulbasaur", attacks=[12])]
    attacks = [_attack(10, "Thunder"), _attack(11, "Quick"), _attack(12, "Vine")]
    return CardRegistry.from_records(cards, attacks)


# --------------------------------------------------------------------------- #
# Indexing + lookups
# --------------------------------------------------------------------------- #

def test_from_records_indexes_by_id(reg):
    assert reg.card(1).name == "Pikachu"
    assert reg.card(2).name == "Bulbasaur"
    assert reg.attack(10).name == "Thunder"
    assert len(reg) == 2


def test_convenience_names(reg):
    assert reg.card_name(1) == "Pikachu"
    assert reg.attack_name(12) == "Vine"


def test_get_with_default(reg):
    assert reg.get_card(999) is None
    assert reg.get_card(999, "x") == "x"
    assert reg.get_attack(999) is None
    assert reg.has_card(1) is True
    assert reg.has_card(999) is False
    assert reg.has_attack(10) is True
    assert reg.has_attack(999) is False


def test_attacks_for_resolves_ids(reg):
    names = [a.name for a in reg.attacks_for(1)]
    assert names == ["Thunder", "Quick"]
    assert reg.attacks_for(2)[0].name == "Vine"


def test_container_protocol(reg):
    assert 1 in reg
    assert 999 not in reg
    assert sorted(iter(reg)) == [1, 2]
    assert "CardRegistry" in repr(reg)


# --------------------------------------------------------------------------- #
# Failure contract
# --------------------------------------------------------------------------- #

def test_duplicate_card_id_rejected():
    with pytest.raises(DuplicateCardError) as ei:
        CardRegistry.from_records([_card(1), _card(1)], [])
    assert ei.value.card_id == 1


def test_duplicate_attack_id_rejected():
    with pytest.raises(DuplicateAttackError) as ei:
        CardRegistry.from_records([], [_attack(7), _attack(7)])
    assert ei.value.attack_id == 7


def test_missing_card_raises(reg):
    with pytest.raises(CardNotFoundError) as ei:
        reg.card(999)
    assert ei.value.card_id == 999
    # KeyError subclass so existing `except KeyError` sites still catch it.
    assert isinstance(ei.value, KeyError)
    assert isinstance(ei.value, RegistryError)


def test_missing_attack_raises(reg):
    with pytest.raises(AttackNotFoundError):
        reg.attack(999)


def test_attacks_for_unknown_card_raises(reg):
    with pytest.raises(CardNotFoundError):
        reg.attacks_for(999)


def test_attacks_for_dangling_attack_raises():
    # Card references an attack id absent from the master data.
    reg = CardRegistry.from_records([_card(1, attacks=[10, 99])], [_attack(10)])
    with pytest.raises(AttackNotFoundError):
        reg.attacks_for(1)


# --------------------------------------------------------------------------- #
# Read-only guarantees
# --------------------------------------------------------------------------- #

def test_mappings_are_read_only(reg):
    assert isinstance(reg.cards, MappingProxyType)
    assert isinstance(reg.attacks, MappingProxyType)
    with pytest.raises(TypeError):
        reg.cards[3] = _card(3)  # type: ignore[index]
    with pytest.raises(TypeError):
        reg.attacks[3] = _attack(3)  # type: ignore[index]


def test_input_list_mutation_does_not_leak(reg):
    cards = [_card(1)]
    r = CardRegistry.from_records(cards, [])
    cards.append(_card(2))  # mutate the source list after construction
    assert 2 not in r  # the registry took its own snapshot


# --------------------------------------------------------------------------- #
# Dynamic retrieval — "adding a card needs no core change"
# --------------------------------------------------------------------------- #

def test_newly_added_card_is_retrievable_without_core_change():
    base = [_card(1, "Pikachu")]
    # Simulate the engine gaining a brand-new card id mid-competition.
    extended = base + [_card(50123, "Future-Mon", attacks=[777])]
    reg = CardRegistry.from_records(extended, [_attack(777, "Nova")])
    # No code enumerated id 50123 — it resolves purely because the engine lists it.
    assert reg.card(50123).name == "Future-Mon"
    assert reg.attacks_for(50123)[0].name == "Nova"


# --------------------------------------------------------------------------- #
# Cached accessor
# --------------------------------------------------------------------------- #

def test_get_registry_caches(monkeypatch):
    calls = {"n": 0}
    sentinel = CardRegistry.from_records([_card(1)], [])

    def fake_from_engine():
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(CardRegistry, "from_engine", staticmethod(fake_from_engine))
    clear_registry_cache()
    first = get_registry()
    second = get_registry()
    assert first is second is sentinel
    assert calls["n"] == 1  # built once, then cached
    clear_registry_cache()
    third = get_registry()
    assert third is sentinel
    assert calls["n"] == 2  # rebuilt after cache clear
    clear_registry_cache()


# --------------------------------------------------------------------------- #
# Engine-backed (skipped when the engine is absent)
# --------------------------------------------------------------------------- #

def _engine_available() -> bool:
    try:
        import cg.game  # noqa: F401
        return True
    except Exception:
        return False


requires_engine = pytest.mark.skipif(
    not _engine_available(),
    reason="cabt engine (cg/) not installed; run scripts/setup_engine.sh",
)


@requires_engine
def test_from_engine_builds_and_resolves_every_card():
    clear_registry_cache()
    reg = get_registry()
    assert len(reg) > 0
    assert len(reg.attacks) > 0
    # Every card the engine lists is retrievable and its attacks resolve — no
    # per-card code, no hard-coded id.
    for cid in reg:
        card = reg.card(cid)
        assert card.cardId == cid
        for atk in reg.attacks_for(cid):
            assert reg.has_attack(atk.attackId)
    # get_registry returns the same cached instance.
    assert get_registry() is reg
    clear_registry_cache()


@requires_engine
def test_from_engine_unknown_id_raises():
    clear_registry_cache()
    reg = get_registry()
    missing = max(reg.cards) + 10_000
    with pytest.raises(CardNotFoundError):
        reg.card(missing)
    clear_registry_cache()
