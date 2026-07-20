"""Regression tests for the staged common-core compatibility adapter."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import pytest

from agents.compatibility import (
    ADAPTER_API_VERSION,
    DECK_STRATEGY_API_VERSION,
    CompatibilityAdapter,
    LegacyDeckStrategy,
    stderr_shadow_sink,
)


class FixedAgent:
    def __init__(self, result: list[int]) -> None:
        self.result = result
        self.calls = 0

    def act(self, _observation: dict) -> list[int]:
        self.calls += 1
        return list(self.result)


@dataclass
class FixedStrategy:
    result: list[int]
    api_version: str = DECK_STRATEGY_API_VERSION
    implementation_version: str = "fixture/v1"
    compatible_adapter_apis: tuple[str, ...] = (ADAPTER_API_VERSION,)

    def decide(self, _observation: dict) -> list[int]:
        return list(self.result)


def test_legacy_is_default_and_candidate_is_not_executed():
    legacy, candidate = FixedAgent([1]), FixedAgent([0])
    adapter = CompatibilityAdapter(legacy, LegacyDeckStrategy(candidate))
    assert adapter.act({}) == [1]
    assert legacy.calls == 1 and candidate.calls == 0


def test_shadow_compares_both_paths_and_keeps_legacy_authoritative():
    comparisons = []
    adapter = CompatibilityAdapter(
        FixedAgent([1]), FixedStrategy([0]), mode="shadow", shadow_sink=comparisons.append
    )
    observation = {"select": {"option": [{}, {}], "minCount": 1, "maxCount": 1}}
    assert adapter.act(observation) == [1]
    assert comparisons[0].matched is False
    assert comparisons[0].compatible is True
    assert comparisons[0].legacy == (1,) and comparisons[0].candidate == (0,)


def test_core_switch_and_rollback_to_legacy():
    legacy, candidate = FixedAgent([1]), FixedStrategy([0])
    assert CompatibilityAdapter(legacy, candidate, mode="core").act({}) == [0]
    assert CompatibilityAdapter(legacy, candidate, mode="legacy").act({}) == [1]


@pytest.mark.parametrize(
    ("strategy", "message"),
    [
        (FixedStrategy([], api_version="ptcg-deck-strategy/v2"), "incompatible strategy API"),
        (FixedStrategy([], compatible_adapter_apis=()), "is incompatible"),
    ],
)
def test_incompatible_contract_versions_fail_closed(strategy, message):
    with pytest.raises(ValueError, match=message):
        CompatibilityAdapter(FixedAgent([]), strategy)


def test_shadow_log_is_machine_readable():
    output = io.StringIO()
    adapter = CompatibilityAdapter(
        FixedAgent([1]),
        FixedStrategy([1]),
        mode="shadow",
        shadow_sink=lambda comparison: stderr_shadow_sink(comparison, output),
    )
    adapter.act({"select": {"option": [{}, {}], "minCount": 1, "maxCount": 1}})
    assert json.loads(output.getvalue()) == {
        "event": "ptcg_ume_shadow_comparison",
        "sequence": 0,
        "matched": True,
        "compatible": True,
        "legacy": [1],
        "candidate": [1],
    }


def test_shadow_marks_invalid_candidate_as_incompatible():
    comparisons = []
    adapter = CompatibilityAdapter(
        FixedAgent([0]), FixedStrategy([2]), mode="shadow", shadow_sink=comparisons.append
    )
    observation = {"select": {"option": [{}], "minCount": 1, "maxCount": 1}}
    assert adapter.act(observation) == [0]
    assert comparisons[0].matched is False
    assert comparisons[0].compatible is False


@pytest.mark.parametrize("mode", ["legacy", "shadow", "core"])
def test_submission_starts_in_every_mode(mode):
    env = {**os.environ, "PTCG_UME_MIGRATION_MODE": mode}
    completed = subprocess.run(
        [sys.executable, "-c", "import main; print(type(main._agent).__name__)"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "CompatibilityAdapter"
    assert completed.stderr == ""
