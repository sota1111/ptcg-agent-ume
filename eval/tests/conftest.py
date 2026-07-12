"""Shared pytest fixtures for eval tests.

The cabt engine (``cg/``, ``libcg.so``) is competition-licensed and gitignored,
so it is absent in CI. These tests import it opportunistically and skip cleanly
when it is not installed.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope="session", autouse=True)
def _chdir_repo():
    """Run from repo root so ``libcg.so`` / ``deck.csv`` / ``data`` resolve."""
    prev = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(prev)


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


@pytest.fixture
def deck() -> list[int]:
    with open(os.path.join(REPO, "deck.csv")) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


@pytest.fixture(autouse=True)
def _reset_active_guard():
    """Ensure the module-level single-active guard never leaks across tests."""
    yield
    try:
        import eval.environment as environment
    except Exception:
        return
    if environment._ACTIVE is not None:
        # A test left a battle live; free it defensively so later tests are clean.
        try:
            environment._ACTIVE.finish()
        except Exception:
            environment._ACTIVE = None
