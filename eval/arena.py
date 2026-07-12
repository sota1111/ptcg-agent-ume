"""Paired-match evaluation Arena for the PTCG AI Battle eval environment (SOT-1645).

The Arena is the reproducible harness that lets **any two agents** be injected and
played head-to-head so a new agent can be measured against a rule-based (or any
other) baseline in the *same* environment. It sits directly on top of the engine
boundary and the match loop already established by the eval stack:

* :class:`eval.environment.Environment` — the one-process/one-match engine boundary
  that always runs ``battle_finish()`` (native cleanup) exactly once per match;
* :func:`eval.match.play_match` — the single-match loop that isolates every agent
  fault (illegal move / per-move timeout / exception) as *that agent's loss* rather
  than crashing the batch;
* :mod:`eval.trace` — the versioned JSONL trace (schema/engine/git stamp, agents,
  decks, one record per decision, terminal result).

What this module adds
---------------------
* **Arbitrary agent injection.** ``run_arena(agent_a, agent_b, ...)`` accepts either
  a live :class:`~eval.agents.Agent` instance *or* a factory (``f(seed)`` / ``f()``)
  for each side, so a fresh, per-match-seeded agent is built for every match. A
  RuleAgent vs a learned agent is exactly ``run_arena(rule_factory, learned_factory)``.
* **Paired matches with side-swap.** Turn-order (先後) is a real advantage, so matches
  are played in pairs with the seats swapped, keeping each agent on each side an
  equal number of times (``side_swap=True``, the default).
* **Machine-readable outputs.** For every match a versioned JSONL trace is written
  (stamped with each agent's identity **and its per-match seed**, both decks and
  their hashes, and the engine/git version); every match also appends a compact
  :class:`MatchRecord` to ``results.jsonl``; and the whole run is summarised into a
  ``report.json`` — win rate + **Wilson 95% CI**, draw/undecided rates, per-agent
  **decision-latency p50/p95/p99**, **safety** (illegal/timeout/exception) rates, and
  **先後別 (by-seat) win rates**.

Reproducibility. The cabt engine takes **no seed** (see :mod:`eval.trace`), so match
*outcomes* are not bit-reproducible; the recorded traces + ``results.jsonl`` are the
faithful record, and :func:`aggregate` is a pure function of the records, so the
report (incl. the 95% CI) is always regenerable from a run's ``results.jsonl``.

One process = one match. The engine keeps a single process-global battle pointer, so
matches here run **sequentially** in-process (each fully finished before the next
starts). Scale out by running Arenas in **separate processes** — never two live
battles in one process.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Any, Callable, Optional, Sequence, Union

# When run as a script (``python eval/arena.py``) the repo root is not on sys.path,
# so make ``eval`` / ``cg`` importable before the package imports below.
if __package__ in (None, ""):  # pragma: no cover - only when executed as a script
    import sys

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

from eval.agents import Agent, FirstOptionAgent, RandomAgent
from eval.environment import EndReason
from eval.match import play_match
from eval.trace import RecordLevel, TraceWriter, deck_hash, engine_hash, git_sha

__all__ = [
    "wilson_ci",
    "percentile",
    "AgentProvider",
    "MatchRecord",
    "ArenaReport",
    "run_arena",
    "aggregate",
    "Z_95",
]

# 95% two-sided normal quantile (shared with the eval report/promotion gate).
Z_95 = 1.959963984540054

# Fault categories, in the reason vocabulary of :class:`eval.environment.EndReason`.
_FAULT_REASONS = {
    EndReason.ILLEGAL_MOVE.value,
    EndReason.TIMEOUT.value,
    EndReason.AGENT_EXCEPTION.value,
}


# --------------------------------------------------------------------------- #
# Statistics helpers (pure — the report is a deterministic function of records)
# --------------------------------------------------------------------------- #
def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(low, high)`` clamped to ``[0, 1]``. With no evidence (``n == 0``) the
    maximally-uncertain ``(0.0, 1.0)`` is returned. Robust for small ``n`` and
    near-0/1 rates, which is why it — not the normal approximation — is the interval
    the promotion gate reads. Known value: ``wilson_ci(50, 100) ≈ (0.4038, 0.5962)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default ``linear`` method).

    ``p`` is in ``[0, 100]``. Returns ``0.0`` for an empty input. ``values`` need not
    be pre-sorted (a sorted copy is taken).
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return float(xs[lo] * (hi - k) + xs[hi] * (k - lo))


def _latency_stats(values: Sequence[float]) -> dict:
    """p50/p95/p99 (+ mean/max/n) for a list of per-decision latencies (ms)."""
    return {
        "n_decisions": len(values),
        "mean_ms": (sum(values) / len(values)) if values else 0.0,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values) if values else 0.0,
    }


# --------------------------------------------------------------------------- #
# Agent injection: a live Agent, or a factory f(seed) / f()
# --------------------------------------------------------------------------- #
# Anything that yields an Agent: an Agent instance (reused as-is) or a callable that
# builds one, optionally taking the arena's per-match seed.
AgentProvider = Union[Agent, Callable[..., Agent], Callable[[int], Agent]]


def _resolve_factory(provider: AgentProvider) -> Callable[[int], Agent]:
    """Normalise a provider into ``make(seed) -> Agent``.

    * an :class:`~eval.agents.Agent` *instance* (has ``act`` but is not callable) is
      reused for every match — its ``seed`` is fixed at construction, so pass a
      factory instead if you want per-match reseeding;
    * a **factory** (a function or an Agent *class*) is called per match; if it
      accepts an argument it receives the arena's per-match ``seed``, otherwise it is
      called with none.
    """
    if callable(provider):  # function or class (an Agent instance is NOT callable)
        try:
            takes_arg = len(signature(provider).parameters) >= 1
        except (TypeError, ValueError):
            takes_arg = True
        if takes_arg:
            def make(seed: int, _f=provider) -> Agent:
                try:
                    return _f(seed)
                except TypeError:
                    return _f()
            return make
        return lambda seed, _f=provider: _f()
    # A pre-built Agent instance: reuse it (seed already fixed at construction).
    return lambda seed, _a=provider: _a


def _agent_label(agent: Any, fallback: str) -> str:
    return str(getattr(agent, "name", None) or fallback)


def _agent_version(agent: Any) -> str:
    return str(getattr(agent, "version", "0"))


class _Instrumented:
    """Wraps an agent to (a) time every ``act`` and (b) capture the first player.

    Per-decision wall-clock latencies are appended to ``latencies`` (the arena reads
    them for the p50/p95/p99 stats); the engine-decided first player (a seat index)
    is recorded into the shared per-match ``ctx`` the first time it is observed.
    Lifecycle hooks are forwarded so stateful agents still work. The wrapper is
    transparent: play_match calls ``act`` exactly as it would on the bare agent.
    """

    def __init__(self, inner: Agent, latencies: list[float], ctx: dict) -> None:
        self._inner = inner
        self._latencies = latencies
        self._ctx = ctx

    def act(self, obs: dict) -> list[int]:
        if self._ctx.get("first_player") is None:
            fp = (obs.get("current") or {}).get("firstPlayer", -1)
            if fp not in (-1, None):
                self._ctx["first_player"] = fp
        t0 = time.perf_counter()
        try:
            return self._inner.act(obs)
        finally:
            self._latencies.append((time.perf_counter() - t0) * 1000.0)

    def on_match_start(self, player_index: int) -> None:
        hook = getattr(self._inner, "on_match_start", None)
        if callable(hook):
            hook(player_index)

    def on_match_end(self, result) -> None:
        hook = getattr(self._inner, "on_match_end", None)
        if callable(hook):
            hook(result)


# --------------------------------------------------------------------------- #
# Records + report
# --------------------------------------------------------------------------- #
@dataclass
class MatchRecord:
    """Compact, machine-readable outcome of one arena match (one ``results.jsonl`` line).

    Neutral in the two agents A/B; ``seat_of_a`` records which engine seat (0/1) agent
    A played this match (so side-swap is auditable), and ``first_player`` is the seat
    the engine moved first. Faults are attributed to the faulting agent (``*_won`` are
    both ``False`` for that agent) and categorised in ``fault_category``.
    """

    match_index: int
    pair_index: int
    seat_of_a: int
    first_player: Optional[int]
    label_a: str
    label_b: str
    seed_a: int
    seed_b: int
    winner_seat: Optional[int]
    winner_label: Optional[str]
    a_won: bool
    b_won: bool
    draw: bool
    undecided: bool
    reason: str
    faulted_seat: Optional[int]
    faulted_label: Optional[str]
    fault_category: Optional[str]
    steps: int
    a_decisions: int
    b_decisions: int
    a_decision_ms: float
    b_decision_ms: float
    trace_path: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArenaReport:
    """Aggregated result of an arena run — the content of ``report.json``."""

    config: dict
    totals: dict
    win_rates: dict
    seat_winrate: dict
    latency: dict
    safety: dict
    reason_counts: dict
    run_dir: Optional[str] = None
    results_path: Optional[str] = None
    report_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        t = self.totals
        wr = self.win_rates
        a = self.config.get("label_a", "A")
        b = self.config.get("label_b", "B")
        ci = wr.get("a_win_rate_ci95", (0.0, 1.0))
        return (
            f"{a} vs {b}: n={t['n']} "
            f"W/D/L(A)={t['a_wins']}/{t['draws']}/{t['b_wins']} "
            f"A_win_rate={wr['a_win_rate']:.3f} CI95=[{ci[0]:.3f},{ci[1]:.3f}] "
            f"undecided={t['undecided']} faults(A/B)={self.safety['a_faults']}/"
            f"{self.safety['b_faults']} "
            f"first_player_win_rate={self.seat_winrate['first_player_win_rate']}"
        )


# --------------------------------------------------------------------------- #
# Aggregation (pure function of the records + collected latencies)
# --------------------------------------------------------------------------- #
def aggregate(
    records: Sequence[MatchRecord],
    latencies_a: Sequence[float],
    latencies_b: Sequence[float],
    config: dict,
) -> ArenaReport:
    """Summarise a completed run's records into an :class:`ArenaReport`.

    Deterministic: the same records + latencies always yield the same report, so a
    run's statistics (win rate, Wilson CI, seat win rates, safety rates) can be
    regenerated from its ``results.jsonl`` without re-running any match.
    """
    n = len(records)
    a_wins = sum(1 for r in records if r.a_won)
    b_wins = sum(1 for r in records if r.b_won)
    draws = sum(1 for r in records if r.draw)
    undecided = sum(1 for r in records if r.undecided)

    a_ci = wilson_ci(a_wins, n)
    b_ci = wilson_ci(b_wins, n)

    # 先後別勝率: win rate of whichever agent moved first, over decisive matches with a
    # known first player, plus the per-agent by-seat split. Agent A sat in
    # ``seat_of_a``; agent B sat in the other seat — so A moved first when
    # ``seat_of_a == first_player`` and B moved first exactly when it did not.
    decisive_fp = [r for r in records
                   if r.first_player is not None and not r.draw and not r.undecided]

    def _seat_split(is_a: bool) -> dict:
        out = {}
        for role, first in (("as_first", True), ("as_second", False)):
            sub = [
                r for r in decisive_fp
                if (((r.seat_of_a == r.first_player) if is_a
                     else (r.seat_of_a != r.first_player)) == first)
            ]
            wins = sum(1 for r in sub if (r.a_won if is_a else r.b_won))
            m = len(sub)
            out[role] = {
                "n": m,
                "wins": wins,
                "win_rate": (wins / m) if m else None,
            }
        return out

    first_wins = sum(1 for r in decisive_fp if r.winner_seat == r.first_player)
    n_fp = len(decisive_fp)
    seat_winrate = {
        "n_decisive_known_first_player": n_fp,
        "first_player_win_rate": (first_wins / n_fp) if n_fp else None,
        "second_player_win_rate": ((n_fp - first_wins) / n_fp) if n_fp else None,
        "A": _seat_split(True),
        "B": _seat_split(False),
    }

    # Safety: faults attributed to each agent, broken down by category.
    def _fault_categories(label: str) -> dict:
        cats: dict[str, int] = {}
        for r in records:
            if r.faulted_label == label and r.fault_category:
                cats[r.fault_category] = cats.get(r.fault_category, 0) + 1
        return cats

    a_faults = sum(1 for r in records if r.faulted_label == config.get("label_a"))
    b_faults = sum(1 for r in records if r.faulted_label == config.get("label_b"))
    safety = {
        "a_faults": a_faults,
        "b_faults": b_faults,
        "a_fault_rate": (a_faults / n) if n else 0.0,
        "b_fault_rate": (b_faults / n) if n else 0.0,
        "a_fault_categories": _fault_categories(config.get("label_a")),
        "b_fault_categories": _fault_categories(config.get("label_b")),
        "undecided": undecided,
        "undecided_rate": (undecided / n) if n else 0.0,
    }

    reason_counts: dict[str, int] = {}
    for r in records:
        reason_counts[r.reason] = reason_counts.get(r.reason, 0) + 1

    totals = {
        "n": n,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "undecided": undecided,
        "mean_steps": (sum(r.steps for r in records) / n) if n else 0.0,
    }
    win_rates = {
        "a_win_rate": (a_wins / n) if n else 0.0,
        "a_win_rate_ci95": [a_ci[0], a_ci[1]],
        "b_win_rate": (b_wins / n) if n else 0.0,
        "b_win_rate_ci95": [b_ci[0], b_ci[1]],
        "draw_rate": (draws / n) if n else 0.0,
    }
    latency = {
        config.get("label_a", "A"): _latency_stats(latencies_a),
        config.get("label_b", "B"): _latency_stats(latencies_b),
    }
    return ArenaReport(
        config=config,
        totals=totals,
        win_rates=win_rates,
        seat_winrate=seat_winrate,
        latency=latency,
        safety=safety,
        reason_counts=reason_counts,
    )


# --------------------------------------------------------------------------- #
# The arena runner
# --------------------------------------------------------------------------- #
def run_arena(
    agent_a: AgentProvider,
    agent_b: AgentProvider,
    *,
    deck0: list[int],
    deck1: Optional[list[int]] = None,
    n_matches: int = 200,
    side_swap: bool = True,
    agent_seed: int = 0,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    max_steps: int = 100_000,
    per_move_timeout: Optional[float] = None,
    out_dir: str = "eval/arena_runs",
    run_label: Optional[str] = None,
    record_traces: bool = True,
    trace_level: RecordLevel = RecordLevel.LOGS,
    write_outputs: bool = True,
) -> ArenaReport:
    """Play ``n_matches`` between two injectable agents and aggregate the result.

    Args:
        agent_a, agent_b: the two sides — each an :class:`~eval.agents.Agent` instance
            or a factory (``f(seed)`` / ``f()``). A fresh agent is built per match, so
            a factory keyed on the per-match ``seed`` gives a reproducible run.
        deck0, deck1: the two 60-card decks (``deck1`` defaults to ``deck0``: a mirror).
        n_matches: number of matches to play (sequentially, one battle at a time).
        side_swap: alternate seats every other match so each agent plays first/second
            an equal number of times (paired matches).
        agent_seed: base seed; match ``i`` uses seeds ``agent_seed+2i`` / ``+2i+1``.
        label_a, label_b: display labels (default: the agents' ``name`` attribute).
        max_steps / per_move_timeout: per-match safety caps forwarded to
            :func:`eval.match.play_match`.
        out_dir / run_label: where run artifacts go — a per-run subdirectory holding
            ``manifest.json``, ``results.jsonl`` and ``report.json`` (+ ``traces/`` when
            ``record_traces``).
        record_traces / trace_level: whether to write a versioned JSONL trace per match
            and at what verbosity.
        write_outputs: set ``False`` to run purely in-memory (writes nothing to disk).

    Returns:
        The aggregated :class:`ArenaReport` (also written to ``report.json`` unless
        ``write_outputs`` is off).
    """
    if n_matches <= 0:
        raise ValueError("n_matches must be positive")
    if deck1 is None:
        deck1 = deck0

    fa = _resolve_factory(agent_a)
    fb = _resolve_factory(agent_b)

    # Build one sample of each side to discover default labels/versions.
    sample_a = fa(agent_seed)
    sample_b = fb(agent_seed + 1)
    label_a = label_a or _agent_label(sample_a, "A")
    label_b = label_b or _agent_label(sample_b, "B")
    if label_a == label_b:  # keep the two sides distinguishable in the outputs
        label_a, label_b = f"{label_a}#A", f"{label_b}#B"
    version_a = _agent_version(sample_a)
    version_b = _agent_version(sample_b)
    type_a = type(sample_a).__name__
    type_b = type(sample_b).__name__

    now = datetime.datetime.now(datetime.timezone.utc)
    run_label = run_label or f"{label_a}_vs_{label_b}_{now.strftime('%Y%m%dT%H%M%S')}"
    run_dir = os.path.join(out_dir, run_label)
    traces_dir = os.path.join(run_dir, "traces")
    results_path = os.path.join(run_dir, "results.jsonl")
    report_path = os.path.join(run_dir, "report.json")
    manifest_path = os.path.join(run_dir, "manifest.json")

    config = {
        "run_label": run_label,
        "created_at": now.isoformat(),
        "label_a": label_a,
        "label_b": label_b,
        "agent_a": {"label": label_a, "type": type_a, "version": version_a},
        "agent_b": {"label": label_b, "type": type_b, "version": version_b},
        "n_matches": n_matches,
        "side_swap": side_swap,
        "agent_seed": agent_seed,
        "max_steps": max_steps,
        "per_move_timeout": per_move_timeout,
        "deck0_hash": deck_hash(deck0),
        "deck1_hash": deck_hash(deck1),
        "engine": engine_hash(),
        "git_sha": git_sha(),
    }

    results_fh = None
    if write_outputs:
        os.makedirs(run_dir, exist_ok=True)
        if record_traces:
            os.makedirs(traces_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
        results_fh = open(results_path, "w", encoding="utf-8")

    records: list[MatchRecord] = []
    latencies_a: list[float] = []
    latencies_b: list[float] = []

    try:
        for i in range(n_matches):
            seat_of_a = 0 if (not side_swap or i % 2 == 0) else 1
            seed_a = agent_seed + 2 * i
            seed_b = agent_seed + 2 * i + 1
            a = fa(seed_a)
            b = fb(seed_b)
            ctx: dict = {"first_player": None}
            pre_a, pre_b = len(latencies_a), len(latencies_b)
            wa = _Instrumented(a, latencies_a, ctx)
            wb = _Instrumented(b, latencies_b, ctx)

            if seat_of_a == 0:
                seat_agents = [wa, wb]
                seeds_by_seat = [seed_a, seed_b]
                labels_by_seat = [label_a, label_b]
                types_by_seat = [type_a, type_b]
                versions_by_seat = [version_a, version_b]
            else:
                seat_agents = [wb, wa]
                seeds_by_seat = [seed_b, seed_a]
                labels_by_seat = [label_b, label_a]
                types_by_seat = [type_b, type_a]
                versions_by_seat = [version_b, version_a]

            agent_meta = [
                {
                    "index": s,
                    "label": labels_by_seat[s],
                    "name": labels_by_seat[s],
                    "type": types_by_seat[s],
                    "version": versions_by_seat[s],
                    "seed": seeds_by_seat[s],
                }
                for s in (0, 1)
            ]

            trace_path = None
            if write_outputs and record_traces:
                trace_path = os.path.join(traces_dir, f"match_{i:04d}.jsonl")

            if trace_path is not None:
                with TraceWriter(trace_path, trace_level) as writer:
                    result = play_match(
                        deck0, deck1, seat_agents,
                        max_steps=max_steps, per_move_timeout=per_move_timeout,
                        writer=writer, agent_meta=agent_meta,
                        trace_id=f"{run_label}-m{i:04d}",
                    )
            else:
                result = play_match(
                    deck0, deck1, seat_agents,
                    max_steps=max_steps, per_move_timeout=per_move_timeout,
                )

            record = _build_record(
                match_index=i,
                seat_of_a=seat_of_a,
                first_player=ctx.get("first_player"),
                labels_by_seat=labels_by_seat,
                label_a=label_a,
                label_b=label_b,
                seed_a=seed_a,
                seed_b=seed_b,
                result=result,
                a_decisions=len(latencies_a) - pre_a,
                b_decisions=len(latencies_b) - pre_b,
                a_decision_ms=sum(latencies_a[pre_a:]),
                b_decision_ms=sum(latencies_b[pre_b:]),
                trace_path=trace_path,
            )
            records.append(record)
            if results_fh is not None:
                results_fh.write(json.dumps(record.to_dict(), ensure_ascii=False))
                results_fh.write("\n")
                results_fh.flush()
    finally:
        if results_fh is not None:
            results_fh.close()

    report = aggregate(records, latencies_a, latencies_b, config)
    report.run_dir = run_dir if write_outputs else None
    report.results_path = results_path if write_outputs else None
    report.report_path = report_path if write_outputs else None
    if write_outputs:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
    return report


def _build_record(
    *,
    match_index: int,
    seat_of_a: int,
    first_player: Optional[int],
    labels_by_seat: list[str],
    label_a: str,
    label_b: str,
    seed_a: int,
    seed_b: int,
    result,
    a_decisions: int,
    b_decisions: int,
    a_decision_ms: float,
    b_decision_ms: float,
    trace_path: Optional[str],
) -> MatchRecord:
    """Translate a :class:`~eval.environment.MatchResult` into a :class:`MatchRecord`."""
    reason = result.reason.value
    winner_seat = result.winner
    winner_label = (
        labels_by_seat[winner_seat] if winner_seat in (0, 1) else None
    )
    draw = result.is_draw
    undecided = winner_seat is None and not draw  # MAX_STEPS / truncated
    faulted_seat = result.faulted_player
    faulted_label = (
        labels_by_seat[faulted_seat] if faulted_seat in (0, 1) else None
    )
    fault_category = reason if reason in _FAULT_REASONS else None
    return MatchRecord(
        match_index=match_index,
        pair_index=match_index // 2,
        seat_of_a=seat_of_a,
        first_player=first_player,
        label_a=label_a,
        label_b=label_b,
        seed_a=seed_a,
        seed_b=seed_b,
        winner_seat=winner_seat,
        winner_label=winner_label,
        a_won=winner_label == label_a,
        b_won=winner_label == label_b,
        draw=draw,
        undecided=undecided,
        reason=reason,
        faulted_seat=faulted_seat,
        faulted_label=faulted_label,
        fault_category=fault_category,
        steps=result.steps,
        a_decisions=a_decisions,
        b_decisions=b_decisions,
        a_decision_ms=a_decision_ms,
        b_decision_ms=b_decision_ms,
        trace_path=trace_path,
    )


# --------------------------------------------------------------------------- #
# CLI: demonstrate arbitrary-agent injection + a paired Random-vs-Random run
# --------------------------------------------------------------------------- #
def _load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n") if x.strip()][:60]


_BUILTIN_AGENTS = {
    "random": lambda s: RandomAgent(seed=s),
    "first": lambda s: FirstOptionAgent(),
}


def _main(argv: list[str]) -> int:
    """``python eval/arena.py [n_matches] [agent_b] [agent_a]``.

    Defaults to a paired Random-vs-Random run of 200 matches. ``agent_b`` /
    ``agent_a`` name a built-in (``random`` | ``first``) — e.g.
    ``python eval/arena.py 200 first`` injects two *different* agents (Random vs the
    deterministic FirstOption baseline) to demonstrate arbitrary A/B injection.
    """
    import random as _random

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)

    n = int(argv[1]) if len(argv) > 1 else 200
    b_kind = argv[2] if len(argv) > 2 else "random"
    a_kind = argv[3] if len(argv) > 3 else "random"
    fa = _BUILTIN_AGENTS.get(a_kind, _BUILTIN_AGENTS["random"])
    fb = _BUILTIN_AGENTS.get(b_kind, _BUILTIN_AGENTS["random"])

    _random.seed(0)
    deck = _load_deck("deck.csv")
    report = run_arena(
        fa, fb,
        deck0=deck,
        n_matches=n,
        side_swap=True,
        label_a=a_kind,
        label_b=b_kind,
    )
    print(report.summary_line())
    if report.run_dir:
        print(f"artifacts: {report.run_dir}")
    return 0


if __name__ == "__main__":
    import sys

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    raise SystemExit(_main(sys.argv))
