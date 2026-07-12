"""Baseline matchup suite + promotion gate for the Arena (SOT-1626).

:mod:`eval.arena` plays *two* injected agents head-to-head; :mod:`eval.report`
turns candidate-centric match records into statistics and a promotion decision;
:mod:`eval.config` freezes the whole thing into a reproducible manifest. This
module is the orchestrator that ties them together into the automated
**promotion pipeline** the issue asks for:

    RunConfig(candidate, [frozen baselines…])  →  run_suite()  →  SuiteResult

For every frozen baseline it runs a paired, side-swapped :func:`~eval.arena.run_arena`
matchup with the candidate always seated as agent A, flattens each match into a
**candidate-centric** row (``candidate_won`` / ``candidate_faulted`` / …), and lets
:func:`eval.report.summarize_matchup` aggregate it (win/draw/loss, Wilson 95% CI,
手数, 意思決定時間, 例外率, 先後別席数). The candidate-vs-*直前best* summary is then
fed to :func:`eval.report.promotion_gate`, so "baseline昇格判定が自動化される" is a
single call: CI lower bound > 0.5, zero candidate faults, and the matchup inside its
wall-clock budget → promote.

Reproducibility. The engine is unseeded, so match *outcomes* are not bit-reproducible;
instead every match is written to ``results.jsonl`` (one candidate-centric row per
line, tagged with its ``matchup``), and the statistics are a pure function of those
rows — :func:`eval.report.aggregate_run` re-derives byte-identical summaries (incl. the
95% CI) from the saved file without replaying a single match. ``manifest.json`` records
the exact :class:`~eval.config.RunConfig` that produced them.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

# When run as a script (``python eval/suite.py``) the repo root is not on sys.path.
if __package__ in (None, ""):  # pragma: no cover - only when executed as a script
    import sys

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

from eval.arena import MatchRecord, run_arena
from eval.config import AgentSpec, RunConfig
from eval.report import (
    MatchupSummary,
    PromotionVerdict,
    aggregate_run,
    format_summary,
    promotion_gate,
    summarize_matchup,
)

__all__ = [
    "candidate_row",
    "SuiteResult",
    "run_suite",
]


def _matchup_label(candidate: str, opponent: str) -> str:
    """Stable per-matchup key used to group rows and name outputs."""
    return f"{candidate}_vs_{opponent}"


def candidate_row(record: MatchRecord, matchup: str) -> dict:
    """Flatten an arena :class:`~eval.arena.MatchRecord` into a candidate-centric row.

    The suite always seats the candidate as agent **A**, so agent-A fields become the
    candidate's and agent-B's the opponent's. The result is exactly the schema
    :func:`eval.report.summarize_matchup` reads, so a run's ``results.jsonl`` is
    re-aggregatable on its own.
    """
    faulted = bool(record.faulted_label) and record.faulted_label == record.label_a
    return {
        "matchup": matchup,
        "candidate": record.label_a,
        "opponent": record.label_b,
        "match_index": record.match_index,
        "candidate_seat": record.seat_of_a,
        "first_player": record.first_player,
        "candidate_won": record.a_won,
        "draw": record.draw,
        "undecided": record.undecided,
        "candidate_faulted": faulted,
        "fault_category": record.fault_category if faulted else None,
        "reason": record.reason,
        "steps": record.steps,
        "candidate_decisions": record.a_decisions,
        "candidate_decision_ms": record.a_decision_ms,
        "seed_candidate": record.seed_a,
        "seed_opponent": record.seed_b,
    }


@dataclass
class SuiteResult:
    """Outcome of a full baseline suite: per-matchup stats + the promotion verdict."""

    manifest: dict
    summaries: dict[str, MatchupSummary]  # matchup label -> summary
    gate: PromotionVerdict
    gate_matchup: Optional[str]
    elapsed_s: dict[str, float] = field(default_factory=dict)  # per matchup wall-clock
    rows: list[dict] = field(default_factory=list)
    suite_dir: Optional[str] = None
    results_path: Optional[str] = None
    summary_path: Optional[str] = None
    gate_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest,
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "gate": self.gate.to_dict(),
            "gate_matchup": self.gate_matchup,
            "elapsed_s": self.elapsed_s,
            "suite_dir": self.suite_dir,
            "results_path": self.results_path,
            "summary_path": self.summary_path,
        }

    def summary_lines(self) -> list[str]:
        lines = [format_summary(s) for s in self.summaries.values()]
        verdict = "PROMOTE" if self.gate.promote else "HOLD"
        gate = f"gate[{self.gate_matchup}]: {verdict}"
        if self.gate.reasons:
            gate += " — " + "; ".join(self.gate.reasons)
        lines.append(gate)
        return lines


def run_suite(
    config: RunConfig,
    *,
    repo_dir: Optional[str] = None,
    write_outputs: bool = True,
    record_traces: bool = False,
    run_label: Optional[str] = None,
) -> SuiteResult:
    """Run the candidate against every frozen baseline and apply the promotion gate.

    Each baseline is a paired, side-swapped matchup of ``config.n_matches`` matches
    (candidate seated as A). Rows are aggregated per matchup by
    :func:`eval.report.summarize_matchup`, and the candidate-vs-``config.gate_baseline``
    summary is judged by :func:`eval.report.promotion_gate` using ``config.time_limit_s``
    and the measured wall-clock of that matchup.

    With ``write_outputs`` (default) a run directory is written holding ``manifest.json``,
    ``results.jsonl`` (all candidate rows, tagged by ``matchup``), ``summary.json`` and
    ``gate.json`` — from which the whole result is regenerable via
    :func:`eval.report.aggregate_run`.
    """
    if not config.baselines:
        raise ValueError("RunConfig needs at least one baseline to run a suite")

    deck0 = config.deck0.resolve(repo_dir)
    deck1 = (config.deck1 or config.deck0).resolve(repo_dir)
    candidate_label = config.candidate.label
    gate_spec = config.gate_baseline

    now = datetime.datetime.now(datetime.timezone.utc)
    run_label = run_label or (
        config.label
        and f"{config.label}_{now.strftime('%Y%m%dT%H%M%S')}"
        or f"suite_{candidate_label}_{now.strftime('%Y%m%dT%H%M%S')}"
    )
    suite_dir = os.path.join(config.out_dir, run_label)

    all_rows: list[dict] = []
    summaries: dict[str, MatchupSummary] = {}
    elapsed: dict[str, float] = {}
    gate_matchup: Optional[str] = None

    for i, baseline in enumerate(config.baselines):
        matchup = _matchup_label(candidate_label, baseline.label)
        if matchup in summaries:  # two baselines share a label — disambiguate
            matchup = f"{matchup}#{i}"

        t0 = time.perf_counter()
        report = run_arena(
            lambda s, spec=config.candidate: spec.build(s),
            lambda s, spec=baseline: spec.build(s),
            deck0=deck0,
            deck1=deck1,
            n_matches=config.n_matches,
            side_swap=config.side_swap,
            agent_seed=config.agent_seed,
            label_a=candidate_label,
            label_b=baseline.label,
            max_steps=config.max_steps,
            per_move_timeout=config.per_move_timeout,
            write_outputs=False,
            record_traces=False,
        )
        elapsed[matchup] = time.perf_counter() - t0

        rows = [candidate_row(r, matchup) for r in report.records]
        all_rows.extend(rows)
        summaries[matchup] = summarize_matchup(rows)
        if gate_spec is not None and baseline is gate_spec and gate_matchup is None:
            gate_matchup = matchup

    # Promotion gate on the candidate-vs-直前best matchup.
    if gate_matchup is not None:
        gate = promotion_gate(
            summaries[gate_matchup],
            time_limit_s=config.time_limit_s,
            elapsed_s=elapsed.get(gate_matchup),
        )
    else:  # no baselines matched the gate index — nothing to promote against
        gate = promotion_gate(
            MatchupSummary(
                candidate=candidate_label, opponent="?", n=0, wins=0, losses=0,
                draws=0, win_rate=0.0, draw_rate=0.0, loss_rate=0.0, ci_low=0.0,
                ci_high=1.0, decisive_win_rate=None, mean_steps=0.0,
                mean_decision_ms=0.0, exceptions=0, exception_rate=0.0,
            ),
        )

    result = SuiteResult(
        manifest=config.to_manifest(),
        summaries=summaries,
        gate=gate,
        gate_matchup=gate_matchup,
        elapsed_s=elapsed,
        rows=all_rows,
    )

    if write_outputs:
        os.makedirs(suite_dir, exist_ok=True)
        result.suite_dir = suite_dir
        result.results_path = os.path.join(suite_dir, "results.jsonl")
        result.summary_path = os.path.join(suite_dir, "summary.json")
        result.gate_path = os.path.join(suite_dir, "gate.json")
        config.write_manifest(os.path.join(suite_dir, "manifest.json"))
        with open(result.results_path, "w", encoding="utf-8") as fh:
            for row in all_rows:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
        with open(result.summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: v.to_dict() for k, v in summaries.items()},
                fh, indent=2, sort_keys=True,
            )
        with open(result.gate_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "gate_matchup": gate_matchup,
                    "elapsed_s": elapsed,
                    "verdict": gate.to_dict(),
                },
                fh, indent=2, sort_keys=True,
            )

    return result


# --------------------------------------------------------------------------- #
# CLI: run a preset suite (candidate vs a frozen random + first baseline pair)
# --------------------------------------------------------------------------- #
def _main(argv: list[str]) -> int:
    """``python eval/suite.py [preset] [candidate_kind] [baseline_kind…]``.

    Defaults to a ``smoke`` suite of ``random`` (candidate) vs a frozen ``first``
    baseline. Kinds name an agent registered in :mod:`eval.config` (``random`` /
    ``first`` / ``import``). The last baseline is the promotion-gate target.
    Example: ``python eval/suite.py promotion random first`` runs the candidate over
    1,000 side-swapped matches and prints the automated promotion verdict.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)

    preset = argv[1] if len(argv) > 1 else "smoke"
    candidate_kind = argv[2] if len(argv) > 2 else "random"
    baseline_kinds = argv[3:] if len(argv) > 3 else ["first"]

    config = RunConfig.preset_run(
        preset,
        candidate=AgentSpec(kind=candidate_kind, name=f"cand:{candidate_kind}"),
        baselines=[AgentSpec(kind=k, name=f"base:{k}") for k in baseline_kinds],
        deck0=RunConfig.__dataclass_fields__["deck0"].default_factory(),
    )
    result = run_suite(config)
    for line in result.summary_lines():
        print(line)
    if result.suite_dir:
        print(f"artifacts: {result.suite_dir}")
    return 0 if result.gate.promote else 0  # gate result is informational, not an error


if __name__ == "__main__":
    import sys

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    raise SystemExit(_main(sys.argv))
