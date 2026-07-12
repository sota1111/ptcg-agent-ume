"""Tests for the reproducible arena RunConfig / manifest (SOT-1626).

All pure — no engine. Covers agent-spec resolution, deck resolution + hashing,
preset match counts, the promotion-gate baseline selection, and manifest
round-tripping (受け入れ条件: 同一manifestから同一集計を再生成 starts with the
manifest itself being faithfully reproducible).
"""
from __future__ import annotations

import json

import pytest

from eval.agents import FirstOptionAgent, RandomAgent
from eval.config import (
    MANIFEST_SCHEMA,
    PRESETS,
    AgentSpec,
    DeckSpec,
    RunConfig,
    build_agent,
)


# --------------------------------------------------------------------------- #
# AgentSpec
# --------------------------------------------------------------------------- #
def test_agentspec_builds_known_kinds():
    assert isinstance(AgentSpec(kind="random").build(7), RandomAgent)
    assert isinstance(AgentSpec(kind="first").build(), FirstOptionAgent)


def test_agentspec_seed_pins_rng_stream():
    # An explicit params seed builds a private, reproducible RNG stream regardless
    # of the arena's per-match seed; two builds with the same seed match bit-for-bit.
    a = AgentSpec(kind="random", params={"seed": 123}).build(999)
    b = AgentSpec(kind="random", params={"seed": 123}).build(0)
    assert isinstance(a, RandomAgent) and isinstance(b, RandomAgent)
    assert a._rng.random() == b._rng.random()
    # build_agent is a thin wrapper around AgentSpec.build.
    assert isinstance(build_agent(AgentSpec(kind="random")), RandomAgent)


def test_agentspec_label_defaults_to_kind():
    assert AgentSpec(kind="random").label == "random"
    assert AgentSpec(kind="random", name="candidate").label == "candidate"


def test_agentspec_unknown_kind_raises():
    with pytest.raises(KeyError):
        AgentSpec(kind="does-not-exist").build()


def test_agentspec_round_trip():
    spec = AgentSpec(kind="import", name="best", params={"target": "m:fn"})
    assert AgentSpec.from_dict(spec.to_dict()) == spec


# --------------------------------------------------------------------------- #
# DeckSpec
# --------------------------------------------------------------------------- #
def test_deckspec_inline_cards_resolve_and_hash_stable():
    spec = DeckSpec(cards=tuple(range(60)))
    assert spec.resolve() == list(range(60))
    # order-sensitive, stable hash; a reorder changes it.
    assert spec.hash() == DeckSpec(cards=tuple(range(60))).hash()
    assert spec.hash() != DeckSpec(cards=tuple(range(59, -1, -1))).hash()


def test_deckspec_path_resolve(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("\n".join(str(i) for i in range(60)) + "\n")
    assert DeckSpec(path=str(p)).resolve() == list(range(60))


def test_deckspec_requires_cards_or_path():
    with pytest.raises(ValueError):
        DeckSpec().resolve()


def test_deckspec_round_trip():
    spec = DeckSpec(cards=(1, 2, 3))
    assert DeckSpec.from_dict(spec.to_dict()) == spec


# --------------------------------------------------------------------------- #
# RunConfig presets + gate baseline
# --------------------------------------------------------------------------- #
def _cfg(**kw) -> RunConfig:
    base = dict(
        candidate=AgentSpec(kind="random", name="cand"),
        baselines=[AgentSpec(kind="random", name="rnd"),
                   AgentSpec(kind="first", name="best")],
        deck0=DeckSpec(cards=tuple(range(60))),
    )
    base.update(kw)
    return RunConfig(**base)


def test_preset_match_counts():
    assert PRESETS == {"smoke": 20, "iteration": 200, "promotion": 1000}
    for name, n in PRESETS.items():
        cfg = RunConfig.preset_run(
            name,
            candidate=AgentSpec(kind="random"),
            baselines=[AgentSpec(kind="first")],
        )
        assert cfg.n_matches == n
        assert cfg.preset == name


def test_preset_run_rejects_unknown():
    with pytest.raises(ValueError):
        RunConfig.preset_run("mega", candidate=AgentSpec(kind="random"), baselines=[])


def test_gate_baseline_defaults_to_last():
    cfg = _cfg()
    assert cfg.gate_baseline is cfg.baselines[-1]
    assert cfg.gate_baseline.name == "best"


def test_gate_baseline_index_selects():
    cfg = _cfg(gate_baseline_index=0)
    assert cfg.gate_baseline.name == "rnd"


def test_gate_baseline_none_when_no_baselines():
    cfg = _cfg(baselines=[])
    assert cfg.gate_baseline is None


def test_deck1_defaults_to_deck0_mirror():
    cfg = _cfg(deck1=None)
    assert cfg.deck1 == cfg.deck0


def test_with_matches_copies_count():
    cfg = _cfg(n_matches=20)
    assert cfg.with_matches(1000).n_matches == 1000
    assert cfg.n_matches == 20  # original untouched


# --------------------------------------------------------------------------- #
# Manifest round-trip (reproducibility)
# --------------------------------------------------------------------------- #
def test_manifest_round_trip_preserves_fields():
    cfg = _cfg(n_matches=200, side_swap=True, agent_seed=3, time_limit_s=42.0,
               gate_baseline_index=1)
    m = cfg.to_manifest()
    assert m["manifest_schema"] == MANIFEST_SCHEMA
    back = RunConfig.from_manifest(m)
    assert back.candidate == cfg.candidate
    assert back.baselines == cfg.baselines
    assert back.n_matches == cfg.n_matches
    assert back.agent_seed == cfg.agent_seed
    assert back.time_limit_s == cfg.time_limit_s
    assert back.gate_baseline_index == cfg.gate_baseline_index
    assert back.gate_baseline.name == "best"


def test_manifest_is_json_serialisable_and_stable(tmp_path):
    cfg = _cfg()
    path = cfg.write_manifest(str(tmp_path / "manifest.json"))
    loaded = RunConfig.load_manifest(path)
    # Re-serialising the loaded config yields the same manifest (idempotent).
    assert loaded.to_manifest() == json.loads(open(path).read())
