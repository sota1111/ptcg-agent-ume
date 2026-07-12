"""Agents for the PTCG AI Battle eval environment.

The Agent Protocol (``act(obs) -> list[int]``) and reference agents live in
:mod:`eval.agents.base`.
"""

from .base import (
    Agent,
    BaseAgent,
    FirstOptionAgent,
    RandomAgent,
    SubmissionAgent,
)

__all__ = [
    "Agent",
    "BaseAgent",
    "FirstOptionAgent",
    "RandomAgent",
    "SubmissionAgent",
]
