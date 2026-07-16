"""Self-play data generation pipeline for PPO training (SOT-1688).

Runs N self-play matches — the current champion (RuleAgent) mirror, or Random
混在 pairings — and writes one JSONL record per *decision* in a form PPO training
(SOT-1689) consumes directly: observation features, the chosen action (option
indices), the legal-move count, the deciding player, and the final outcome as a
±1/0 reward from that player's perspective.

Reuse, not reimplementation
---------------------------
The match loop is :func:`eval.match.play_match` — fault isolation, the engine
boundary and native cleanup all stay in one place. This module only *wraps* each
agent in a :class:`_RecordingAgent` that captures ``(obs, action)`` pairs as the
match runs (the same pattern as :class:`eval.arena._Instrumented`), then stamps
the terminal result into the captured records afterwards.

Reproducibility. The cabt engine takes **no seed** (see :mod:`eval.trace`), so a
run is not bit-replayable; instead every record is self-contained — features,
action and outcome are *in the record* — which is exactly what PPO needs. Agent
RNGs are still seeded per match (``base_seed + 2*game [+1]``) so agent behaviour
is deterministic given the engine's stream.

Record schema (one JSON object per line; see :data:`RECORD_FIELDS`)::

    {"schema": "ume-selfplay-v1", "feature_version": 1, "game": 0, "decision": 3,
     "player": 0, "agent": "rule", "features": [...], "action": [2],
     "action_index": 2, "n_options": 5, "min_count": 1, "max_count": 1,
     "select_type": 0, "select_context": 0, "reward": 1.0, "result": "win",
     "winner": 0, "reason": "normal", "steps": 74}

CLI::

    python -m eval.selfplay --games 100 --agents rule,rule \
        --out eval/selfplay_runs/selfplay.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Optional

# When run as a script (``python eval/selfplay.py``) the repo root is not on
# sys.path, so make ``eval`` / ``agents`` / ``cg`` importable first.
if __package__ in (None, ""):  # pragma: no cover - only when executed as a script
    import sys

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

from agents.features import FEATURE_DIM, FEATURE_VERSION, featurize

__all__ = [
    "SCHEMA",
    "RECORD_FIELDS",
    "AGENT_KINDS",
    "validate_record",
    "run_selfplay",
    "load_deck",
    "load_deck_dir",
]

#: Schema tag stamped into every record (bump on layout change).
SCHEMA = "ume-selfplay-v1"

#: Required fields of a record and their (duck) types — the machine-checkable
#: contract shared by :func:`validate_record`, the tests and downstream training.
RECORD_FIELDS = {
    "schema": str,
    "feature_version": int,
    "game": int,
    "decision": int,
    "player": int,
    "agent": str,
    "features": list,
    "action": list,
    "action_index": int,   # first chosen option index; -1 for an empty action
    "n_options": int,
    "min_count": int,
    "max_count": int,
    "select_type": int,
    "select_context": int,
    "reward": float,
    "result": str,         # win | loss | draw | undecided (deciding player's view)
    "winner": (int, type(None)),
    "reason": str,
    "steps": int,
}

# The self-play pairings this CLI offers: the current champion (rule), the
# random baseline, and the PPO policy (SOT-1689), built per match with the
# run's per-match seed.
AGENT_KINDS = ("rule", "random", "ppo")


def _agent_factory(kind: str) -> Callable[[int], object]:
    """``kind -> f(seed) -> Agent``. Engine-importing, so resolved lazily here."""
    if kind == "rule":
        from agents import RuleAgent

        return lambda seed: RuleAgent(seed=seed)
    if kind == "random":
        from agents import RandomAgent

        return lambda seed: RandomAgent(seed=seed)
    if kind == "ppo":
        from agents import PPOAgent

        return lambda seed: PPOAgent(seed=seed)
    raise ValueError(f"unknown agent kind {kind!r} (choose from {AGENT_KINDS})")


def _resolve_agent_spec(spec) -> tuple[str, Callable[[int], object]]:
    """Normalise one side of the ``agents`` pairing to ``(label, factory)``.

    A spec is either a kind string from :data:`AGENT_KINDS` or a
    ``(label, factory)`` pair — the latter lets the PPO training loop
    (:mod:`train.ppo`) inject agents carrying in-memory weights while reusing
    this pipeline unchanged for its per-iteration self-play.
    """
    if isinstance(spec, str):
        return spec, _agent_factory(spec)
    if (
        isinstance(spec, tuple)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and callable(spec[1])
    ):
        return spec
    raise ValueError(
        f"agent spec must be one of {AGENT_KINDS} or a (label, factory) pair, got {spec!r}"
    )


def validate_record(record: dict) -> list[str]:
    """Return the list of schema violations in ``record`` (empty = valid).

    Checks the required fields/types (:data:`RECORD_FIELDS`), the fixed feature
    dimension, and the action/option consistency PPO relies on (indices in
    ``[0, n_options)``, no duplicates, count within ``[min_count, max_count]``).
    """
    errors: list[str] = []
    for field, expected in RECORD_FIELDS.items():
        if field not in record:
            errors.append(f"missing field {field!r}")
            continue
        value = record[field]
        if field == "reward":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif isinstance(value, bool) and expected is int:
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            errors.append(f"field {field!r} has wrong type {type(value).__name__}")

    features = record.get("features")
    if isinstance(features, list):
        if len(features) != FEATURE_DIM:
            errors.append(f"features length {len(features)} != FEATURE_DIM {FEATURE_DIM}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in features):
            errors.append("features contains a non-numeric value")

    action = record.get("action")
    n = record.get("n_options")
    if isinstance(action, list) and isinstance(n, int):
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in action):
            errors.append("action contains a non-int")
        elif any(i < 0 or i >= n for i in action):
            errors.append(f"action {action} outside [0, {n})")
        elif len(set(action)) != len(action):
            errors.append(f"action {action} has duplicates")
        lo, hi = record.get("min_count"), record.get("max_count")
        if isinstance(lo, int) and isinstance(hi, int) and not (lo <= len(action) <= hi):
            errors.append(f"action length {len(action)} outside [{lo}, {hi}]")

    if record.get("result") not in ("win", "loss", "draw", "undecided"):
        errors.append(f"result {record.get('result')!r} invalid")
    if record.get("reward") not in (1.0, -1.0, 0.0, 1, -1, 0):
        errors.append(f"reward {record.get('reward')!r} not in {{1, -1, 0}}")
    if record.get("player") not in (0, 1):
        errors.append(f"player {record.get('player')!r} not a seat")
    return errors


#: cabt RESULT reason codes (engine reason=<code> in MatchResult.detail):
#: 1 no prizes (prize-out), 2 empty deck (deck-out), 3 no active, 4 card effect.
_DECKOUT_REASON = 2
_NOACTIVE_REASON = 3


def _prize_counts(obs: dict) -> tuple[int, int]:
    """``(own_remaining, opp_remaining)`` prize cards from the deciding player's view.

    Mirrors :func:`agents.features.featurize`'s perspective (``current.yourIndex``
    is "me"). Missing/malformed state falls back to the full 6 prizes so a record
    is always self-contained. Used only for the SOT-1699 reward shaping.
    """
    current = obs.get("current") if isinstance(obs.get("current"), dict) else {}
    me = current.get("yourIndex")
    me = me if me in (0, 1) else 0
    players = current.get("players") if isinstance(current.get("players"), list) else []

    def remaining(idx: int) -> int:
        player = players[idx] if 0 <= idx < len(players) else None
        prize = player.get("prize") if isinstance(player, dict) else None
        return len(prize) if isinstance(prize, list) else 6

    return remaining(me), remaining(1 - me)


def _engine_reason_code(result) -> Optional[int]:
    """Parse the ``engine reason=<code>`` note :class:`MatchResult` carries in
    ``detail`` (see :meth:`eval.environment.Environment._result_reason_code`).

    Returns the int code (1..4) or ``None`` when absent (e.g. fault results the
    match runner produces, whose detail is not the engine RESULT note).
    """
    detail = getattr(result, "detail", None)
    if isinstance(detail, str) and "engine reason=" in detail:
        token = detail.rsplit("engine reason=", 1)[1].strip()
        if token.isdigit():
            return int(token)
    return None


class _RecordingAgent:
    """Wraps an agent to capture one pending record per real decision.

    A *real decision* is a pending selection with a non-empty option list — the
    states PPO trains on. Trivial no-ops (no selection / no options, where ``[]``
    is the only legal action) are not decisions and are skipped. The terminal
    result is unknown mid-match, so records are buffered in ``pending`` and
    finalised by :func:`_finalize` after :func:`~eval.match.play_match` returns.
    Lifecycle hooks are forwarded so stateful agents keep working.
    """

    def __init__(self, inner, seat: int, label: str) -> None:
        self._inner = inner
        self._seat = seat
        self._label = label
        self.pending: list[dict] = []

    def act(self, obs: dict) -> list[int]:
        action = self._inner.act(obs)
        select = obs.get("select") if isinstance(obs.get("select"), dict) else {}
        options = select.get("option") if isinstance(select.get("option"), list) else []
        if options:
            act_list = [int(i) for i in action] if isinstance(action, list) else []
            own_prizes, opp_prizes = _prize_counts(obs)
            self.pending.append({
                "schema": SCHEMA,
                "feature_version": FEATURE_VERSION,
                "player": self._seat,
                "agent": self._label,
                "features": featurize(obs),
                "action": act_list,
                "action_index": act_list[0] if act_list else -1,
                "n_options": len(options),
                "min_count": int(select.get("minCount") or 0),
                "max_count": int(select.get("maxCount") or 0),
                "select_type": int(select.get("type") or 0),
                "select_context": int(select.get("context") or 0),
                # Reward-shaping signal (SOT-1699): prizes *remaining* at this
                # decision, from the deciding player's perspective. Extra fields —
                # the ume-selfplay-v1 required-field contract is unchanged.
                "own_prizes": own_prizes,
                "opp_prizes": opp_prizes,
            })
        return action

    def on_match_start(self, player_index: int) -> None:
        hook = getattr(self._inner, "on_match_start", None)
        if callable(hook):
            hook(player_index)

    def on_match_end(self, result) -> None:
        hook = getattr(self._inner, "on_match_end", None)
        if callable(hook):
            hook(result)


def _finalize(recorder: _RecordingAgent, game: int, result) -> list[dict]:
    """Stamp the terminal outcome into a seat's buffered records.

    Reward is from the deciding player's perspective: +1 win, -1 loss, 0 for a
    draw or an undecided (step-capped) match — the PPO terminal return.
    """
    seat = recorder._seat
    if result.winner == seat:
        outcome, reward = "win", 1.0
    elif result.winner == (1 - seat):
        outcome, reward = "loss", -1.0
    elif result.is_draw:
        outcome, reward = "draw", 0.0
    else:
        outcome, reward = "undecided", 0.0

    reason_code = _engine_reason_code(result)
    records = []
    for t, rec in enumerate(recorder.pending):
        rec = dict(rec)
        rec.update({
            "game": game,
            "decision": t,
            "reward": reward,
            "result": outcome,
            "winner": result.winner,
            "reason": result.reason.value,
            "steps": result.steps,
            # Granular engine end-reason (SOT-1699 loss shaping); extra field.
            "end_reason_code": reason_code,
        })
        records.append(rec)
    return records


def load_deck(path: str) -> list[int]:
    """Read a 60-card deck csv (one card id per line, ``deck.csv`` format)."""
    with open(path, encoding="utf-8") as fh:
        return [int(x) for x in fh.read().split("\n") if x.strip()][:60]


def load_deck_dir(deck_dir: str) -> list[tuple[str, list[int]]]:
    """Load every ``*.csv`` deck in a directory as sorted ``(name, deck)`` pairs.

    The rotation pool for multi-deck self-play (SOT-1695) — e.g. the 25
    tournament decks under ``decks/initial/``. Sorted by filename so the
    game→deck assignment is stable across runs.
    """
    names = sorted(f for f in os.listdir(deck_dir) if f.endswith(".csv"))
    if not names:
        raise ValueError(f"no *.csv decks found in {deck_dir!r}")
    return [(name, load_deck(os.path.join(deck_dir, name))) for name in names]


def run_selfplay(
    games: int,
    out_path: str,
    *,
    agents: tuple = ("rule", "rule"),  # each side: an AGENT_KINDS str or (label, factory)
    deck0: Optional[list[int]] = None,
    deck1: Optional[list[int]] = None,
    decks: Optional[list[tuple[str, list[int]]]] = None,
    base_seed: int = 0,
    max_steps: int = 100_000,
    validate: bool = True,
    record_labels: Optional[set] = None,
) -> dict:
    """Play ``games`` self-play matches and append decision records to ``out_path``.

    Matches run sequentially (one live battle per process — see
    :mod:`eval.environment`). Seat order alternates every game so both agent
    kinds see both seats in a mixed pairing. Returns a summary dict::

        {"games": N, "decisions": int, "faults": int, "invalid_records": int,
         "wins": {"<label>": int}, "draws": int, "undecided": int,
         "feature_dim": FEATURE_DIM, "out_path": ...}

    Deck rotation (SOT-1695): pass ``decks`` — ``(name, deck)`` pairs, e.g. from
    :func:`load_deck_dir` — and game ``g`` is a **mirror** of ``decks[g % len]``
    (both seats play the same deck, so records stay opponent-symmetric while the
    corpus covers every archetype). Mutually exclusive with ``deck0``/``deck1``.
    Rotation adds a ``deck`` name field to each record (an *extra* field — the
    ``ume-selfplay-v1`` required-field contract is unchanged) and a ``per_deck``
    ``{name: {"games": n, "faults": n}}`` tally to the summary.

    A fault (illegal move / timeout / agent exception — see
    :class:`eval.environment.EndReason`) is counted, never raised; the SafeAgent
    families used here are expected to keep this at 0.

    League play (SOT-1699): ``record_labels`` restricts *which seats' records are
    written* to the label set (e.g. ``{"ppo"}`` keeps only the learner's
    on-policy records when the opponent is a past-policy snapshot). ``None`` (the
    default) writes both seats, unchanged. Game outcomes/faults are still counted
    for every game regardless of the filter.
    """
    from eval.match import play_match

    if games <= 0:
        raise ValueError("games must be positive")
    if decks is not None and (deck0 is not None or deck1 is not None):
        raise ValueError("decks (rotation) and deck0/deck1 are mutually exclusive")
    if decks is not None and not decks:
        raise ValueError("decks rotation pool must be non-empty")
    kind_a, factory_a = _resolve_agent_spec(agents[0])
    kind_b, factory_b = _resolve_agent_spec(agents[1])
    if decks is None:
        if deck0 is None:
            deck0 = load_deck("deck.csv")
        if deck1 is None:
            deck1 = deck0

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    decisions = 0
    faults = 0
    invalid = 0
    draws = 0
    undecided = 0
    wins: dict[str, int] = {kind_a: 0, kind_b: 0}
    per_deck: dict[str, dict[str, int]] = {}

    with open(out_path, "a", encoding="utf-8") as fh:
        for game in range(games):
            if decks is not None:
                deck_name, rotated = decks[game % len(decks)]
                deck0 = deck1 = rotated
            else:
                deck_name = None
            # Alternate seats each game (paired seating, as in eval.arena).
            swap = game % 2 == 1
            seat_kinds = (kind_b, kind_a) if swap else (kind_a, kind_b)
            seat_factories = (factory_b, factory_a) if swap else (factory_a, factory_b)
            recorders = [
                _RecordingAgent(seat_factories[s](base_seed + 2 * game + s), s, seat_kinds[s])
                for s in (0, 1)
            ]

            result = play_match(deck0, deck1, recorders, max_steps=max_steps)

            if result.is_fault:
                faults += 1
            if result.is_draw:
                draws += 1
            elif result.winner in (0, 1):
                wins[seat_kinds[result.winner]] = wins.get(seat_kinds[result.winner], 0) + 1
            else:
                undecided += 1
            if deck_name is not None:
                tally = per_deck.setdefault(deck_name, {"games": 0, "faults": 0})
                tally["games"] += 1
                tally["faults"] += int(result.is_fault)

            for recorder in recorders:
                if record_labels is not None and recorder._label not in record_labels:
                    continue
                for record in _finalize(recorder, game, result):
                    if deck_name is not None:
                        record["deck"] = deck_name
                    if validate:
                        errors = validate_record(record)
                        if errors:
                            invalid += 1
                            record["schema_errors"] = errors
                    fh.write(json.dumps(record, ensure_ascii=False))
                    fh.write("\n")
                    decisions += 1
            fh.flush()

    summary = {
        "games": games,
        "decisions": decisions,
        "faults": faults,
        "invalid_records": invalid,
        "wins": wins,
        "draws": draws,
        "undecided": undecided,
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "agents": [kind_a, kind_b],
        "out_path": out_path,
    }
    if decks is not None:
        summary["decks"] = [name for name, _ in decks]
        summary["per_deck"] = per_deck
    return summary


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PPO-ready self-play decision records (JSONL).",
    )
    parser.add_argument("--games", type=int, default=100,
                        help="number of matches to play (default: 100)")
    parser.add_argument("--out", default="eval/selfplay_runs/selfplay.jsonl",
                        help="output JSONL path (appended)")
    parser.add_argument("--agents", default="rule,rule",
                        help="comma-separated pairing, e.g. rule,rule or rule,random")
    parser.add_argument("--deck0", default="deck.csv", help="player 0 deck csv")
    parser.add_argument("--deck1", default=None,
                        help="player 1 deck csv (default: same as --deck0)")
    parser.add_argument("--deck-dir", default=None,
                        help="rotate mirror decks over every *.csv in this directory "
                             "(e.g. decks/initial); overrides --deck0/--deck1")
    parser.add_argument("--seed", type=int, default=0, help="base agent seed")
    parser.add_argument("--max-steps", type=int, default=100_000,
                        help="per-match selection-step safety cap")
    args = parser.parse_args(argv)

    kinds = tuple(k.strip() for k in args.agents.split(","))
    if len(kinds) != 2 or not all(k in AGENT_KINDS for k in kinds):
        parser.error(f"--agents must be two of {AGENT_KINDS}, got {args.agents!r}")

    summary = run_selfplay(
        args.games,
        args.out,
        agents=kinds,  # type: ignore[arg-type]
        deck0=None if args.deck_dir else load_deck(args.deck0),
        deck1=load_deck(args.deck1) if args.deck1 and not args.deck_dir else None,
        decks=load_deck_dir(args.deck_dir) if args.deck_dir else None,
        base_seed=args.seed,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["faults"] == 0 and summary["invalid_records"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
