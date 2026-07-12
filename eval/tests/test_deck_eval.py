"""Tests for the deck-optimization track (SOT-1651, R6).

Split into pure-unit tests (no engine — the hypergeometric 初動安定性, deck hashing /
loading, and champion version registry logic) and engine-backed tests (static card
metrics + paired A/B through the real cabt engine). The engine (``cg/``) is
gitignored/absent in CI, so engine tests skip cleanly via ``requires_engine``.

Covers the R6 acceptance criteria:
* the agent is held fixed while only the deck varies — a paired deck A/B comparison;
* the champion deck is version-managed and its content hash is verified on load;
* legality / energy ratio / 初動安定性 are reported per deck;
* a mirror A/B (same deck both sides) is unbiased — win rate CI brackets 0.5 even
  though the 先手 advantage is large — which is exactly what the deck-swap gives over
  the seat-pinned Arena.
"""
from __future__ import annotations

import json
import os

import pytest

from eval.deck_eval import (
    DECK_SIZE,
    DeckSpec,
    _deck_hash,
    deck_static_metrics,
    load_deck,
    opening_no_mulligan_probability,
    run_deck_ab,
)

from .conftest import requires_engine


# --------------------------------------------------------------------------- #
# Pure-unit: 初動安定性 (opening / no-mulligan probability)
# --------------------------------------------------------------------------- #
def test_opening_probability_no_basics_is_zero():
    assert opening_no_mulligan_probability(0) == 0.0


def test_opening_probability_certain_when_no_bad_hand_possible():
    # With 54 Basic Pokémon in a 60-card deck, a 7-card hand cannot avoid them.
    assert opening_no_mulligan_probability(54) == 1.0


def test_opening_probability_known_value():
    # 1 - C(56,7)/C(60,7): a single Basic Pokémon in a 60-card deck opens ~11.4%.
    p = opening_no_mulligan_probability(4)
    assert 0.0 < p < 1.0
    assert round(opening_no_mulligan_probability(1), 4) == round(1 - (53 / 60), 4)


def test_opening_probability_monotonic_in_basics():
    ps = [opening_no_mulligan_probability(b) for b in range(1, 20)]
    assert ps == sorted(ps)  # more Basic Pokémon never lowers opening stability


# --------------------------------------------------------------------------- #
# Pure-unit: deck identity + loading
# --------------------------------------------------------------------------- #
def test_deck_hash_is_order_independent():
    assert _deck_hash([1, 2, 3, 3]) == _deck_hash([3, 1, 3, 2])
    assert _deck_hash([1, 2, 3]) != _deck_hash([1, 2, 4])


def test_load_deck_reads_ids_and_names(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("\n".join(["10"] * DECK_SIZE) + "\n")
    deck = load_deck(str(p), version="v9")
    assert deck.name == "d"
    assert deck.version == "v9"
    assert len(deck.cards) == DECK_SIZE
    assert deck.label() == "d@v9"


def test_deck_spec_label_hides_default_version():
    assert DeckSpec(name="x", cards=[1]).label() == "x"


# --------------------------------------------------------------------------- #
# Pure-unit: champion version registry
# --------------------------------------------------------------------------- #
def test_champion_registry_current_matches_file_hash():
    """The committed champion registry must point at a file whose contents hash to
    the recorded ``deck_hash`` — i.e. the champion version is pinned, not stale."""
    from decks import champion_versions, current_version
    from decks.champion import CHAMPION_DIR

    cur = current_version()
    entry = champion_versions()[cur]
    deck = load_deck(os.path.join(CHAMPION_DIR, entry["file"]))
    assert deck.deck_hash == entry["deck_hash"]


def test_load_champion_rejects_unknown_version():
    from decks.champion import load_champion

    with pytest.raises(KeyError):
        load_champion("does-not-exist")


# --------------------------------------------------------------------------- #
# Engine-backed: static metrics + paired A/B
# --------------------------------------------------------------------------- #
@requires_engine
def test_champion_static_metrics_are_legal():
    from decks import load_champion

    m = deck_static_metrics(load_champion())
    assert m["size"] == DECK_SIZE
    assert m["legal"] is True
    assert m["violations"] == []
    assert m["n_basic_pokemon"] >= 1
    # Energy ratio is the fraction of energy cards and matches the reported count.
    assert m["n_energy"] == sum(m["energy_by_type"].values())
    assert 0.0 <= m["energy_ratio"] <= 1.0
    # Opening stability equals the hypergeometric probability of its Basic count.
    assert m["opening_stability"] == pytest.approx(
        opening_no_mulligan_probability(m["n_basic_pokemon"])
    )


@requires_engine
def test_illegal_deck_is_flagged():
    from decks import load_champion

    champ = load_champion()
    # Five copies of some single card id -> exceeds the 4-copy limit (unless it is a
    # Basic Energy, which is exempt); pad to 60 with the champion's own cards.
    from collections import Counter

    from cg.api import CardType
    from eval.registry import get_registry

    reg = get_registry()
    non_energy = next(
        cid for cid in champ.cards
        if reg.get_card(cid) and reg.get_card(cid).cardType != CardType.BASIC_ENERGY
    )
    cards = [non_energy] * 5 + [c for c in champ.cards if c != non_energy][:55]
    bad = DeckSpec(name="bad", cards=cards)
    m = deck_static_metrics(bad)
    assert m["legal"] is False
    assert any("exceeds" in v for v in m["violations"])


@requires_engine
def test_paired_mirror_is_seat_unbiased_and_faultless():
    """The headline acceptance: fix the agent, run a paired N>=200 mirror A/B. Deck A
    plays each seat equally, so its win rate CI must bracket 0.5 even though the
    first-player win rate is far from it — and the fixed policy never faults."""
    from agents import RuleAgent
    from decks import load_champion

    champ = load_champion()
    mirror = DeckSpec(name="mirror", cards=list(champ.cards))
    out = run_deck_ab(
        champ, mirror,
        lambda s: RuleAgent(seed=s),
        n_matches=200,
        agent_label="RuleAgent",
        write_outputs=False,
    )
    assert out["totals"]["n"] == 200
    lo, hi = out["win_rates"]["a_win_rate_ci95"]
    assert lo <= 0.5 <= hi  # unbiased: no deck edge in a mirror
    # The fixed champion policy is legal on both seats -> zero faults.
    assert out["safety"]["a_faults"] == 0
    assert out["safety"]["b_faults"] == 0
    # Deck A occupied each seat an equal number of times (paired swap).
    assert out["config"]["fixed_agent"]["label"] == "RuleAgent"


@requires_engine
def test_run_deck_ab_writes_report(tmp_path):
    from agents import RuleAgent
    from decks import load_champion

    champ = load_champion()
    out = run_deck_ab(
        champ, DeckSpec(name="mirror", cards=list(champ.cards)),
        lambda s: RuleAgent(seed=s),
        n_matches=6,
        out_dir=str(tmp_path),
        run_label="unit",
    )
    assert os.path.exists(out["report_path"])
    with open(out["report_path"], encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["config"]["n_matches"] == 6
    assert "static_metrics" in saved
