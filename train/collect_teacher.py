"""Collect policy targets from a strong agent in a bounded subprocess."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agents.features import FEATURE_VERSION, featurize  # noqa: E402
from agents.policy_net import N_SLOTS  # noqa: E402
from agents.rule_agent import RuleAgent  # noqa: E402
from eval.arena import _load_deck  # noqa: E402
from eval.match import play_match  # noqa: E402


class TeacherError(RuntimeError):
    """The isolated teacher failed to start or answer."""


class JsonLineTeacher:
    name = "search-teacher"
    version = "1"

    def __init__(self, repo: str, python: str, out, teacher: str) -> None:
        server = os.path.join(repo, "eval", "agent_server.py")
        # Local multi-repo benches may have a stale /kaggle_simulations/agent
        # left by another agent. Hide only that directory while importing the
        # teacher; the real Kaggle bundle path is irrelevant to this collector.
        bootstrap = (
            "import os,runpy; real=os.path.isdir; "
            "os.path.isdir=lambda p: False if p=='/kaggle_simulations/agent' else real(p); "
            f"runpy.run_path({server!r}, run_name='__main__')"
        )
        self._proc = subprocess.Popen(
            [python, "-c", bootstrap], cwd=repo, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        ready = self._proc.stderr.readline().strip()
        if ready != "READY":
            raise TeacherError(f"teacher failed to start: {ready}")
        self._out = out
        self._teacher = teacher
        self._game = -1
        self._seat = -1
        self._decision = 0
        self.records = 0

    def on_match_start(self, seat: int) -> None:
        self._game += 1
        self._seat = seat
        self._decision = 0

    def on_match_end(self, _result) -> None:
        pass

    def act(self, obs: dict) -> list[int]:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(obs, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise TeacherError("teacher exited without an action")
        reply = json.loads(line)
        if isinstance(reply, dict) and "__error__" in reply:
            raise TeacherError(reply["__error__"])
        select = obs.get("select") or {}
        options = select.get("option") or []
        if (len(options) >= 2 and isinstance(reply, list) and len(reply) == 1
                and isinstance(reply[0], int) and 0 <= reply[0] < min(len(options), N_SLOTS)):
            record = {
                "schema": "ume-policy-distill-v1", "feature_version": FEATURE_VERSION,
                "teacher": self._teacher, "game": self._game, "player": self._seat,
                "decision": self._decision, "features": featurize(obs),
                "n_options": len(options), "teacher_action_index": reply[0],
            }
            self._out.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.records += 1
        self._decision += 1
        return reply

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Collect strong-search policy targets")
    p.add_argument("--teacher-repo", required=True)
    p.add_argument("--teacher-python", default=sys.executable)
    p.add_argument("--teacher-name", default="fable-champion-mcts")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.games <= 0:
        p.error("--games must be positive")
    if os.path.exists(args.out):
        p.error(f"refusing to overwrite existing output: {args.out}")

    deck = _load_deck(args.deck)
    faults = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        teacher = JsonLineTeacher(args.teacher_repo, args.teacher_python, out, args.teacher_name)
        try:
            for game in range(args.games):
                rule = RuleAgent(seed=args.seed + game)
                agents = (teacher, rule) if game % 2 == 0 else (rule, teacher)
                result = play_match(deck, deck, agents, per_move_timeout=5.0)
                faults += int(result.is_fault)
        finally:
            teacher.close()
    print(json.dumps({"games": args.games, "records": teacher.records, "faults": faults,
                      "teacher": args.teacher_name, "out": args.out}))
    return 1 if faults else 0


if __name__ == "__main__":
    raise SystemExit(main())
