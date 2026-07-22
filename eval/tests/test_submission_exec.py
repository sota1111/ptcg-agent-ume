"""Regression tests for Kaggle's exec()-based submission loader."""
from __future__ import annotations

import os
import subprocess
import sys


def test_exec_replaces_preloaded_foreign_agents_package():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = r'''
import os, sys, types
real_isdir = os.path.isdir
os.path.isdir = lambda path: False if path == "/kaggle_simulations/agent" else real_isdir(path)
foreign = types.ModuleType("agents")
foreign.__file__ = "/usr/local/lib/python3.11/site-packages/kaggle_environments/envs/lux/agents.py"
sys.modules["agents"] = foreign
sys.modules["agents.harness"] = types.ModuleType("agents.harness")
code = open("main.py", encoding="utf-8").read()
env = {}
exec(compile(code, "/kaggle_simulations/agent/main.py", "exec"), env)
assert callable(env["agent"])
assert len(env["agent"]({"select": None, "logs": [], "current": None})) == 60
assert os.path.realpath(sys.modules["agents"].__file__).startswith(os.getcwd())
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo, text=True,
        capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
