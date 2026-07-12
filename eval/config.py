"""Reproducible run configuration for the Arena (SOT-1626).

:class:`RunConfig` aggregates *everything* needed to reproduce an arena run — the
candidate agent, the frozen baselines it is measured against, the decks, the match
count, side-swap, the per-agent seed, the resolved engine binary hash, and the
output location — into a single JSON-serialisable **manifest**. Re-running from the
same manifest reproduces the same *configuration* and the same aggregation
procedure; note that the cabt engine takes **no seed** (see :mod:`eval.trace`), so
individual match *outcomes* are not bit-reproducible. Faithful reproducibility of
the statistics is instead guaranteed by recording every match to a results file
that :mod:`eval.report` re-aggregates deterministically.

Agents are described declaratively as :class:`AgentSpec` (a ``kind`` resolved
through a small registry + JSON ``params``), so a manifest fully describes which
agents played without pickling live objects. The built-in kinds are ``random`` and
``first`` (a deterministic lowest-index baseline that stands in for the rule-based
heuristic until SOT-1631 lands); register more with :func:`register_agent`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from eval.agents import Agent, FirstOptionAgent, RandomAgent
from eval.trace import deck_hash, engine_hash, git_sha

__all__ = [
    "AgentSpec",
    "DeckSpec",
    "RunConfig",
    "register_agent",
    "build_agent",
    "MANIFEST_SCHEMA",
]

# Bump when the manifest shape changes so a reader can flag an incompatible manifest.
MANIFEST_SCHEMA = "1.0.0"


# -- agent registry ---------------------------------------------------------------
# A factory maps (params, seed) -> Agent. ``seed`` is the per-match seed derived by
# the arena; a factory ignores it when its agent is deterministic.
AgentFactory = Callable[[dict, Optional[int]], Agent]

_AGENT_REGISTRY: dict[str, AgentFactory] = {}


def register_agent(kind: str, factory: AgentFactory) -> None:
    """Register an agent ``kind`` so it can be named from a manifest."""
    _AGENT_REGISTRY[kind] = factory


def _random_factory(params: dict, seed: Optional[int]) -> Agent:
    # An explicit params["seed"] pins the stream; otherwise the arena's per-match
    # seed is used so a run is reproducible from its manifest + match index.
    s = params.get("seed", seed)
    return RandomAgent(seed=s)


def _first_factory(params: dict, seed: Optional[int]) -> Agent:
    return FirstOptionAgent()


def _import_factory(params: dict, seed: Optional[int]) -> Agent:
    """Build a :class:`SubmissionAgent` from a ``module:attr`` target.

    Lets a manifest reference a champion / candidate shipped as a plain
    ``agent(obs) -> list[int]`` callable (e.g. a saved "直前best") without pickling.
    """
    from importlib import import_module

    from eval.agents import SubmissionAgent

    target = params["target"]
    mod_name, _, attr = target.partition(":")
    fn = getattr(import_module(mod_name), attr)
    return SubmissionAgent(fn, name=params.get("name", attr))


register_agent("random", _random_factory)
register_agent("first", _first_factory)
register_agent("import", _import_factory)


@dataclass(frozen=True)
class AgentSpec:
    """Declarative description of an agent: a registry ``kind`` + JSON ``params``.

    ``name`` is a display label (defaults to ``kind``). :meth:`build` resolves it to
    a live :class:`~eval.agents.Agent`; ``seed`` is the arena's per-match seed and is
    honoured only by seedable kinds.
    """

    kind: str
    name: Optional[str] = None
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.kind

    def build(self, seed: Optional[int] = None) -> Agent:
        try:
            factory = _AGENT_REGISTRY[self.kind]
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(
                f"unknown agent kind {self.kind!r}; registered: "
                f"{sorted(_AGENT_REGISTRY)}"
            ) from exc
        return factory(dict(self.params), seed)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSpec":
        return cls(kind=d["kind"], name=d.get("name"), params=d.get("params") or {})


def build_agent(spec: AgentSpec, seed: Optional[int] = None) -> Agent:
    """Convenience wrapper around :meth:`AgentSpec.build`."""
    return spec.build(seed)


@dataclass(frozen=True)
class DeckSpec:
    """A deck given either inline (``cards``) or by ``path`` to a 60-line CSV.

    :meth:`resolve` returns the concrete ``list[int]``; :meth:`hash` is its stable,
    order-sensitive sha256 (shared with the trace's ``deck_hash``).
    """

    path: Optional[str] = None
    cards: Optional[tuple[int, ...]] = None

    def resolve(self, repo_dir: Optional[str] = None) -> list[int]:
        if self.cards is not None:
            return list(self.cards)
        if self.path is None:
            raise ValueError("DeckSpec needs either `cards` or `path`")
        path = self.path
        if repo_dir and not os.path.isabs(path):
            path = os.path.join(repo_dir, path)
        with open(path) as f:
            return [int(x) for x in f.read().split("\n") if x.strip()][:60]

    def hash(self, repo_dir: Optional[str] = None) -> str:
        return deck_hash(self.resolve(repo_dir))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "cards": list(self.cards) if self.cards is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeckSpec":
        cards = d.get("cards")
        return cls(path=d.get("path"),
                   cards=tuple(cards) if cards is not None else None)


# Standard preset match counts (issue: smoke 20 / iteration ≥200 / promotion ≥1000).
PRESETS = {"smoke": 20, "iteration": 200, "promotion": 1000}


@dataclass
class RunConfig:
    """Everything required to run and reproduce an arena evaluation.

    ``candidate`` is measured against each frozen ``baselines`` entry over
    ``n_matches`` matches per matchup, with seats swapped every other match when
    ``side_swap`` is set. ``agent_seed`` seeds the per-match agent construction so
    the run is reproducible from this manifest (modulo the engine's unseeded
    shuffles). ``time_limit_s`` is the per-matchup wall-clock budget used by the
    promotion gate; ``engine`` / ``git_sha`` are stamped at construction for
    provenance.
    """

    candidate: AgentSpec
    baselines: list[AgentSpec]
    deck0: DeckSpec = field(default_factory=lambda: DeckSpec(path="deck.csv"))
    deck1: Optional[DeckSpec] = None  # defaults to deck0 (mirror match)
    n_matches: int = PRESETS["smoke"]
    side_swap: bool = True
    agent_seed: int = 0
    max_steps: int = 100_000
    per_move_timeout: Optional[float] = None
    time_limit_s: Optional[float] = None
    # Which baseline is the "直前best" the promotion gate judges the candidate
    # against (an index into ``baselines``; the default ``-1`` = the last one, the
    # convention that the frozen previous-best is appended last).
    gate_baseline_index: int = -1
    out_dir: str = "eval/arena_runs"
    label: Optional[str] = None
    preset: Optional[str] = None
    engine: dict = field(default_factory=engine_hash)
    git_sha: Optional[str] = field(default_factory=git_sha)

    def __post_init__(self) -> None:
        if self.deck1 is None:
            self.deck1 = self.deck0

    @property
    def gate_baseline(self) -> Optional[AgentSpec]:
        """The frozen baseline the promotion gate compares the candidate against."""
        if not self.baselines:
            return None
        return self.baselines[self.gate_baseline_index]

    # -- presets -----------------------------------------------------------------
    @classmethod
    def preset_run(
        cls,
        name: str,
        candidate: AgentSpec,
        baselines: list[AgentSpec],
        **kwargs: Any,
    ) -> "RunConfig":
        """Build a config for a named preset (``smoke`` / ``iteration`` / ``promotion``)."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        kwargs.setdefault("n_matches", PRESETS[name])
        kwargs.setdefault("label", name)
        return cls(candidate=candidate, baselines=baselines, preset=name, **kwargs)

    def with_matches(self, n: int) -> "RunConfig":
        """A copy with a different match count (handy for scaling a preset up)."""
        return replace(self, n_matches=n)

    # -- manifest (serialisation) ------------------------------------------------
    def to_manifest(self) -> dict:
        return {
            "manifest_schema": MANIFEST_SCHEMA,
            "label": self.label,
            "preset": self.preset,
            "candidate": self.candidate.to_dict(),
            "baselines": [b.to_dict() for b in self.baselines],
            "deck0": self.deck0.to_dict(),
            "deck1": (self.deck1 or self.deck0).to_dict(),
            "n_matches": self.n_matches,
            "side_swap": self.side_swap,
            "agent_seed": self.agent_seed,
            "max_steps": self.max_steps,
            "per_move_timeout": self.per_move_timeout,
            "time_limit_s": self.time_limit_s,
            "gate_baseline_index": self.gate_baseline_index,
            "out_dir": self.out_dir,
            "engine": self.engine,
            "git_sha": self.git_sha,
        }

    @classmethod
    def from_manifest(cls, m: dict) -> "RunConfig":
        return cls(
            candidate=AgentSpec.from_dict(m["candidate"]),
            baselines=[AgentSpec.from_dict(b) for b in m.get("baselines", [])],
            deck0=DeckSpec.from_dict(m["deck0"]),
            deck1=DeckSpec.from_dict(m.get("deck1", m["deck0"])),
            n_matches=m.get("n_matches", PRESETS["smoke"]),
            side_swap=m.get("side_swap", True),
            agent_seed=m.get("agent_seed", 0),
            max_steps=m.get("max_steps", 100_000),
            per_move_timeout=m.get("per_move_timeout"),
            time_limit_s=m.get("time_limit_s"),
            gate_baseline_index=m.get("gate_baseline_index", -1),
            out_dir=m.get("out_dir", "eval/arena_runs"),
            label=m.get("label"),
            preset=m.get("preset"),
            engine=m.get("engine") or {},
            git_sha=m.get("git_sha"),
        )

    def write_manifest(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_manifest(), f, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_manifest(cls, path: str) -> "RunConfig":
        with open(path) as f:
            return cls.from_manifest(json.load(f))
