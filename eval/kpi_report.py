"""Ume KPI history report (SOT-1710).

Prints the KPI history (``eval/kpi_history.jsonl``, see ``docs/KPI.md``) as a
time-ordered table plus a comparison of the two most recent measurements:
delta and 改善/悪化/横ばい per KPI (``fault_total`` is an OK/NG gate instead
of a trend). Adoption decisions should use CI non-overlap, which is reported
for the win-rate KPIs.

    venv/bin/python eval/kpi_report.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from eval.kpi import KPI_DIRECTIONS, load_history  # noqa: E402

# Below these absolute deltas a change counts as 横ばい, not a trend.
FLAT_EPS = {
    "winrate_vs_rule": 0.005,
    "winrate_vs_random": 0.005,
    "decision_time_mean_ms": 5.0,
}


def _kpi_value(record: dict, name: str):
    return ((record.get("kpis") or {}).get(name) or {}).get("value")


def _kpi_ci(record: dict, name: str):
    return ((record.get("kpis") or {}).get(name) or {}).get("ci95")


def _fmt(value, ci=None) -> str:
    if value is None:
        return "-"
    s = f"{value:.4f}" if isinstance(value, float) else str(value)
    if ci:
        s += f" [{ci[0]:.3f},{ci[1]:.3f}]"
    return s


def compare_last_two(history: list) -> list:
    """Verdict dicts for the last two records; [] when fewer than 2.

    Each dict: {kpi, prev, cur, delta, verdict} with verdict in
    改善 / 悪化 / 横ばい / OK / NG / n/a (+ CI overlap note for win rates).
    """
    if len(history) < 2:
        return []
    prev, cur = history[-2], history[-1]
    out = []
    for name, direction in KPI_DIRECTIONS.items():
        pv, cv = _kpi_value(prev, name), _kpi_value(cur, name)
        row = {"kpi": name, "prev": pv, "cur": cv, "delta": None,
               "verdict": "n/a"}
        if direction == 0:
            if cv is not None:
                row["verdict"] = "OK" if cv == 0 else "NG"
                row["delta"] = (cv - pv) if pv is not None else None
        elif pv is not None and cv is not None:
            delta = cv - pv
            row["delta"] = delta
            eps = FLAT_EPS.get(name, 0.0)
            if abs(delta) <= eps:
                row["verdict"] = "横ばい"
            elif delta * direction > 0:
                row["verdict"] = "改善"
            else:
                row["verdict"] = "悪化"
            pci, cci = _kpi_ci(prev, name), _kpi_ci(cur, name)
            if pci and cci:
                disjoint = cci[0] > pci[1] or cci[1] < pci[0]
                row["ci_disjoint"] = disjoint
        out.append(row)
    return out


def print_report(history: list) -> None:
    if not history:
        print(f"no KPI history at {os.environ.get('UME_KPI_HISTORY') or 'eval/kpi_history.jsonl'}")
        return
    print(f"KPI history ({len(history)} record(s)) — schema ume-kpi-v1\n")
    header = (f"{'ts':20} {'git_sha':8} {'issue':10} {'n_r':>4} {'n_x':>4} "
              f"{'vs_rule':24} {'vs_random':24} {'faults':>6} {'time_ms':>8}")
    print(header)
    print("-" * len(header))
    for r in history:
        print(f"{r.get('ts', '-'):20} {r.get('git_sha', '-'):8} "
              f"{str(r.get('issue', '-')):10} "
              f"{str(r.get('n_rule') or '-'):>4} "
              f"{str(r.get('n_random') or '-'):>4} "
              f"{_fmt(_kpi_value(r, 'winrate_vs_rule'), _kpi_ci(r, 'winrate_vs_rule')):24} "
              f"{_fmt(_kpi_value(r, 'winrate_vs_random'), _kpi_ci(r, 'winrate_vs_random')):24} "
              f"{_fmt(_kpi_value(r, 'fault_total')):>6} "
              f"{_fmt(_kpi_value(r, 'decision_time_mean_ms')):>8}")
    rows = compare_last_two(history)
    if not rows:
        print("\n(1 record only — no comparison yet)")
        return
    print(f"\ncomparison: {history[-2].get('ts')} ({history[-2].get('issue')})"
          f" -> {history[-1].get('ts')} ({history[-1].get('issue')})")
    for row in rows:
        delta = (f"{row['delta']:+.4f}" if isinstance(row["delta"], float)
                 else f"{row['delta']:+d}" if row["delta"] is not None
                 else "-")
        note = ""
        if "ci_disjoint" in row:
            note = ("  (CI非重複=有意)" if row["ci_disjoint"]
                    else "  (CI重複=有意差なし)")
        print(f"  {row['kpi']:24} {_fmt(row['prev']):>10} -> "
              f"{_fmt(row['cur']):>10}  Δ={delta:>9}  {row['verdict']}{note}")


def main(argv=None) -> int:
    print_report(load_history())
    return 0


if __name__ == "__main__":
    sys.exit(main())
