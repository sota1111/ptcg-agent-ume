"""Statistics and the promotion gate for arena runs (SOT-1626).

Pure, engine-free aggregation over the per-match records produced by
:mod:`eval.arena`. Every number here is a deterministic function of the recorded
results, so re-running :func:`summarize_matchup` (or :func:`aggregate_run`) on the
same ``results.jsonl`` yields byte-identical statistics — that is how "結果から
95% CI を再生成可能" is guaranteed even though the engine itself is unseeded.

Metrics per matchup (candidate's perspective):

* win / draw / loss rate over ``n`` matches (draws count as non-wins);
* **Wilson 95% CI** on the win rate (:func:`wilson_ci`) — the interval used by the
  promotion gate, robust for small ``n`` and near-0/1 rates;
* mean steps (手数) and mean candidate decision time (意思決定時間);
* exception rate — the fraction of matches the candidate lost by its own fault
  (illegal move / timeout / crash), which must be 0 to promote.

The promotion gate (:func:`promotion_gate`) turns the candidate-vs-直前best summary
into a pass/fail decision: CI lower bound > 0.5, zero candidate exceptions, and the
run inside its wall-clock budget.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

__all__ = [
    "wilson_ci",
    "MatchupSummary",
    "summarize_matchup",
    "aggregate_run",
    "PromotionVerdict",
    "promotion_gate",
    "format_summary",
]

# 95% two-sided normal quantile.
Z_95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(low, high)`` clamped to ``[0, 1]``. For ``n == 0`` there is no
    evidence, so the maximally-uncertain ``(0.0, 1.0)`` is returned. This is the
    interval the promotion gate reads (its ``low`` is the CI lower bound).

    Known value: ``wilson_ci(50, 100)`` ≈ ``(0.4038, 0.5962)``.
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


@dataclass
class MatchupSummary:
    """Aggregated candidate-vs-opponent statistics over ``n`` matches."""

    candidate: str
    opponent: str
    n: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    # Wilson 95% CI on the win rate (draws as non-wins).
    ci_low: float
    ci_high: float
    # Decisive win rate = wins / (wins + losses); None when every match drew.
    decisive_win_rate: Optional[float]
    mean_steps: float
    mean_decision_ms: float
    exceptions: int
    exception_rate: float
    seat_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _rec(r, name: str, default=None):
    """Read a field from either a MatchRecord dataclass or a plain dict."""
    if isinstance(r, dict):
        return r.get(name, default)
    return getattr(r, name, default)


def summarize_matchup(records: Sequence) -> MatchupSummary:
    """Aggregate one matchup's :class:`~eval.arena.MatchRecord` list (or dicts)."""
    n = len(records)
    if n == 0:
        return MatchupSummary(
            candidate="?", opponent="?", n=0, wins=0, losses=0, draws=0,
            win_rate=0.0, draw_rate=0.0, loss_rate=0.0, ci_low=0.0, ci_high=1.0,
            decisive_win_rate=None, mean_steps=0.0, mean_decision_ms=0.0,
            exceptions=0, exception_rate=0.0, seat_counts={},
        )

    wins = draws = losses = exceptions = 0
    total_steps = 0
    total_decision_ms = 0.0
    total_decisions = 0
    seat_counts: dict[int, int] = {}
    candidate = _rec(records[0], "candidate", "?")
    opponent = _rec(records[0], "opponent", "?")

    for r in records:
        if _rec(r, "draw"):
            draws += 1
        elif _rec(r, "candidate_won"):
            wins += 1
        else:
            losses += 1
        if _rec(r, "candidate_faulted"):
            exceptions += 1
        total_steps += int(_rec(r, "steps", 0) or 0)
        total_decision_ms += float(_rec(r, "candidate_decision_ms", 0.0) or 0.0)
        total_decisions += int(_rec(r, "candidate_decisions", 0) or 0)
        seat = _rec(r, "candidate_seat")
        if seat is not None:
            seat_counts[seat] = seat_counts.get(seat, 0) + 1

    decisive = wins + losses
    ci_low, ci_high = wilson_ci(wins, n)
    return MatchupSummary(
        candidate=candidate,
        opponent=opponent,
        n=n,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / n,
        draw_rate=draws / n,
        loss_rate=losses / n,
        ci_low=ci_low,
        ci_high=ci_high,
        decisive_win_rate=(wins / decisive) if decisive else None,
        mean_steps=total_steps / n,
        mean_decision_ms=(total_decision_ms / total_decisions) if total_decisions else 0.0,
        exceptions=exceptions,
        exception_rate=exceptions / n,
        seat_counts={int(k): v for k, v in sorted(seat_counts.items())},
    )


def aggregate_run(records: Iterable) -> dict[str, MatchupSummary]:
    """Group records by their ``matchup`` label and summarise each group."""
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(_rec(r, "matchup", "?"), []).append(r)
    return {label: summarize_matchup(rs) for label, rs in groups.items()}


@dataclass
class PromotionVerdict:
    """Whether a candidate may be promoted over the previous best."""

    promote: bool
    reasons: list[str]
    ci_low: float
    exceptions: int
    elapsed_s: Optional[float]
    time_limit_s: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def promotion_gate(
    summary: MatchupSummary,
    *,
    min_ci_low: float = 0.5,
    max_exceptions: int = 0,
    time_limit_s: Optional[float] = None,
    elapsed_s: Optional[float] = None,
) -> PromotionVerdict:
    """Decide promotion from the candidate-vs-直前best summary.

    Promote only when *all* hold: the win-rate Wilson CI lower bound strictly
    exceeds ``min_ci_low`` (default 0.5 — the candidate is statistically better than
    the previous best), the candidate committed no more than ``max_exceptions``
    faults (default 0), and the run finished within ``time_limit_s`` (when given).
    Each failing condition is recorded in ``reasons``.
    """
    reasons: list[str] = []
    ci_ok = summary.ci_low > min_ci_low
    if not ci_ok:
        reasons.append(
            f"CI lower bound {summary.ci_low:.4f} ≤ {min_ci_low:.2f} "
            f"(win_rate={summary.win_rate:.4f}, n={summary.n})"
        )
    exc_ok = summary.exceptions <= max_exceptions
    if not exc_ok:
        reasons.append(
            f"candidate exceptions {summary.exceptions} > {max_exceptions} "
            f"(exception_rate={summary.exception_rate:.4f})"
        )
    time_ok = True
    if time_limit_s is not None and elapsed_s is not None:
        time_ok = elapsed_s <= time_limit_s
        if not time_ok:
            reasons.append(
                f"elapsed {elapsed_s:.2f}s exceeds limit {time_limit_s:.2f}s"
            )
    return PromotionVerdict(
        promote=ci_ok and exc_ok and time_ok,
        reasons=reasons,
        ci_low=summary.ci_low,
        exceptions=summary.exceptions,
        elapsed_s=elapsed_s,
        time_limit_s=time_limit_s,
    )


def format_summary(summary: MatchupSummary) -> str:
    """One-line human-readable rendering of a matchup summary."""
    return (
        f"{summary.candidate} vs {summary.opponent}: "
        f"n={summary.n} W/D/L={summary.wins}/{summary.draws}/{summary.losses} "
        f"win_rate={summary.win_rate:.3f} "
        f"CI95=[{summary.ci_low:.3f},{summary.ci_high:.3f}] "
        f"exc={summary.exceptions} ({summary.exception_rate:.3f}) "
        f"steps~{summary.mean_steps:.1f} dec~{summary.mean_decision_ms:.2f}ms "
        f"seats={summary.seat_counts}"
    )
