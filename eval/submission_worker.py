"""JSON-lines bridge that runs an exact repository submission in isolation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "src"))
# The shared devcontainer has a different extracted submission mounted at the
# Kaggle path. Force repository-local imports for cross-play isolation.
_isdir = os.path.isdir
os.path.isdir = lambda path: False if str(path).startswith("/kaggle_simulations/agent") else _isdir(path)
from main import agent  # type: ignore  # noqa: E402
os.path.isdir = _isdir

for line in sys.stdin:
    try:
        action = agent(json.loads(line))
        print(json.dumps({"action": action}), flush=True)
    except Exception as exc:  # pragma: no cover - exercised through parent process
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
