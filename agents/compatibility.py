"""Compatibility boundary for the staged common-core migration.

The shared contract versions the deck strategy and adapter APIs independently.
This Python submission mirrors the narrow decision boundary: ``legacy`` stays
the default, ``shadow`` compares both paths while returning legacy output, and
``core`` makes the versioned candidate authoritative.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Protocol, TextIO

DECK_STRATEGY_API_VERSION = "ptcg-deck-strategy/v1"
ADAPTER_API_VERSION = "ptcg-agent-adapter/v1"
MIGRATION_MODES = frozenset({"legacy", "shadow", "core"})


class Agent(Protocol):
    def act(self, observation: dict) -> list[int]: ...


class DeckStrategy(Protocol):
    api_version: str
    implementation_version: str
    compatible_adapter_apis: tuple[str, ...]

    def decide(self, observation: dict) -> list[int]: ...


@dataclass
class LegacyDeckStrategy:
    """Expose an existing 梅 agent through the versioned strategy boundary."""

    agent: Agent
    implementation_version: str = "ume-ppo-mcts-harness/v1"
    api_version: str = DECK_STRATEGY_API_VERSION
    compatible_adapter_apis: tuple[str, ...] = (ADAPTER_API_VERSION,)

    def decide(self, observation: dict) -> list[int]:
        return self.agent.act(observation)


@dataclass(frozen=True)
class ShadowComparison:
    sequence: int
    matched: bool
    compatible: bool
    legacy: tuple[int, ...]
    candidate: tuple[int, ...]


@dataclass
class CompatibilityAdapter:
    legacy: Agent
    candidate: DeckStrategy
    mode: str = "legacy"
    shadow_sink: Callable[[ShadowComparison], None] | None = None
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in MIGRATION_MODES:
            raise ValueError(
                f"unknown migration mode {self.mode!r}; expected one of {sorted(MIGRATION_MODES)}"
            )
        if self.candidate.api_version != DECK_STRATEGY_API_VERSION:
            raise ValueError(
                f"incompatible strategy API {self.candidate.api_version!r}; "
                f"expected {DECK_STRATEGY_API_VERSION!r}"
            )
        if ADAPTER_API_VERSION not in self.candidate.compatible_adapter_apis:
            raise ValueError(
                f"strategy {self.candidate.implementation_version!r} is incompatible "
                f"with {ADAPTER_API_VERSION!r}"
            )

    def act(self, observation: dict) -> list[int]:
        if self.mode == "legacy":
            return self.legacy.act(observation)
        if self.mode == "core":
            return self.candidate.decide(observation)

        legacy_result = self.legacy.act(observation)
        candidate_result = self.candidate.decide(observation)
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = int(select.get("minCount", 0))
        maximum = int(select.get("maxCount", len(options)))

        def contract_valid(result: list[int]) -> bool:
            return (
                minimum <= len(result) <= maximum
                and len(set(result)) == len(result)
                and all(isinstance(index, int) and 0 <= index < len(options) for index in result)
            )

        comparison = ShadowComparison(
            sequence=self._sequence,
            matched=legacy_result == candidate_result,
            compatible=contract_valid(legacy_result) and contract_valid(candidate_result),
            legacy=tuple(legacy_result),
            candidate=tuple(candidate_result),
        )
        self._sequence += 1
        (self.shadow_sink or stderr_shadow_sink)(comparison)
        return legacy_result


def stderr_shadow_sink(comparison: ShadowComparison, stream: TextIO = sys.stderr) -> None:
    """Emit machine-readable parity evidence without contaminating stdout."""
    stream.write(
        json.dumps(
            {
                "event": "ptcg_ume_shadow_comparison",
                "sequence": comparison.sequence,
                "matched": comparison.matched,
                "compatible": comparison.compatible,
                "legacy": list(comparison.legacy),
                "candidate": list(comparison.candidate),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()
