"""Agent Protocol and reference agents for the eval environment.

The protocol mirrors the Kaggle submission entry point exactly: an agent is a
callable ``act(obs) -> list[int]`` that, given the current observation, returns a
list of option indices into ``obs.select.option`` (the engine-supplied legal
moves). The same object plugs into local self-play (:mod:`eval.match`) and, via
:class:`SubmissionAgent`, wraps a bare submission ``agent(obs_dict)`` function so
that the *same* code can be evaluated locally and submitted.

Two lifecycle hooks — :meth:`Agent.on_match_start` and
:meth:`Agent.on_match_end` — let a stateful agent set up / tear down per match.
They default to no-ops so a stateless agent only implements ``act``.

Observation shape
-----------------
``act`` receives the raw ``obs`` dict — byte-for-byte what a Kaggle submission's
``agent(obs_dict)`` gets — so agents written here are submission-ready as-is. Use
``self.parse(obs)`` (or ``cg.api.to_observation_class``) for typed access to
``obs.select`` and the board state. During local self-play the environment always
supplies the decks, so ``obs.select`` is never ``None`` mid-match; the ``None``
case only occurs in the official harness's initial deck selection.
"""

from __future__ import annotations

import random
from typing import Optional, Protocol, runtime_checkable

from cg.api import Observation, to_observation_class

__all__ = [
    "Agent",
    "BaseAgent",
    "RandomAgent",
    "FirstOptionAgent",
    "SubmissionAgent",
]


@runtime_checkable
class Agent(Protocol):
    """Structural Agent Protocol: anything with a compatible ``act`` is an Agent.

    ``act(obs) -> list[int]`` returns option indices into ``obs["select"]``.
    Lifecycle hooks are optional (see :class:`BaseAgent` for no-op defaults).
    """

    def act(self, obs: dict) -> list[int]:
        ...


class BaseAgent:
    """Convenience base with no-op lifecycle hooks and a typed-parse helper.

    Subclasses implement :meth:`act`. Reference agents below subclass this.
    """

    #: Human-readable name, handy for reports / logs.
    name: str = "base"

    def on_match_start(self, player_index: int) -> None:
        """Called once before a match, with this agent's seat (0 or 1)."""

    def on_match_end(self, result) -> None:  # result: eval.environment.MatchResult
        """Called once after a match with the structured result."""

    def parse(self, obs: dict) -> Observation:
        """Typed view of the raw observation dict."""
        return to_observation_class(obs)

    def act(self, obs: dict) -> list[int]:  # pragma: no cover - abstract
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Selects a uniformly random legal action.

    Picks ``clamp(minCount, maxCount, len(option))`` distinct option indices at
    random. Pass ``rng`` to share a stream (e.g. the global ``random`` module for
    reproducible self-play) or ``seed`` for a private stream; default uses the
    global ``random`` module.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None, rng=None) -> None:
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random  # share the global stream

    def act(self, obs: dict) -> list[int]:
        select = self.parse(obs).select
        if select is None:
            return []
        n = len(select.option)
        if n == 0:
            return []
        k = max(select.minCount, min(select.maxCount, n))
        return self._rng.sample(range(n), k)


class FirstOptionAgent(BaseAgent):
    """Deterministic baseline: always picks the lowest-index legal option(s).

    Useful as a stable opponent and to demonstrate that two different agent
    implementations are interchangeable under the same Protocol.
    """

    name = "first"

    def act(self, obs: dict) -> list[int]:
        select = self.parse(obs).select
        if select is None:
            return []
        n = len(select.option)
        if n == 0:
            return []
        k = max(select.minCount, min(select.maxCount, n))
        return list(range(k))


class SubmissionAgent(BaseAgent):
    """Adapts a bare submission-style ``agent(obs_dict) -> list[int]`` callable.

    Lets the exact function shipped in ``main.py`` be evaluated locally without
    modification: ``SubmissionAgent(main.agent)``.
    """

    def __init__(self, fn, name: str = "submission") -> None:
        self._fn = fn
        self.name = name

    def act(self, obs: dict) -> list[int]:
        return self._fn(obs)
