"""Deck-optimization track: fix the agent, compare the decks (SOT-1651, R6).

This is the **orthogonal deck track** — kept deliberately separate from the
policy-improvement track so that "did the *deck* get better?" is never confounded
with "did the *policy* get better?". The one lever it varies is the deck; the agent
(the champion policy) is held fixed on **both** seats, so any win-rate gap between two
decks is attributable to the decks alone.

Why not just use :func:`eval.arena.run_arena`?
---------------------------------------------
The Arena pins each deck to a *seat* and swaps the two **agents** every other match
(``side_swap``). That is exactly right for an *agent* A/B test, but wrong for a *deck*
A/B test: with the agent fixed, swapping identical agents does nothing, the decks stay
glued to their seats, and whichever deck sits in seat 0 soaks up the entire 先手
(first-player) advantage. This module does the dual thing — it holds the agent fixed
and **swaps the decks** between the seats every other match — so each deck plays first
and second an equal number of times (a genuine *paired* deck comparison). The heavy
lifting (single-match loop, fault isolation, Wilson-CI aggregation, 先後別 win rates)
is reused from :mod:`eval.match` / :mod:`eval.arena` unchanged.

What it produces
----------------
* **Static deck metrics** (no matches needed, engine card data only):
  legality (60 cards, ≤4 copies per non-basic-energy card, ≤1 ACE SPEC, ≥1 Basic
  Pokémon, all ids known), energy ratio + per-type breakdown, and 初動安定性 — the
  hypergeometric probability of opening with ≥1 Basic Pokémon (i.e. not having to
  mulligan) in the opening ``HAND_SIZE`` cards.
* **Paired A/B win rate** with the agent fixed: champion vs challenger, N≥200,
  Wilson 95% CI, draw/undecided/fault rates, and 先後別 (by-seat) win rates — all via
  the reused :func:`eval.arena.aggregate`.
* **Matchup別勝率**: :func:`run_gauntlet` runs the champion against several challengers
  and reports each matchup's paired win rate.

One process = one match (the engine's process-global battle pointer), so matches run
sequentially in-process; scale out across processes, never two live battles in one.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Allow ``python eval/deck_eval.py`` to import the package when the repo root is not
# yet on sys.path (mirrors eval/arena.py).
if __package__ in (None, ""):  # pragma: no cover - only when executed as a script
    import sys

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

from eval.arena import MatchRecord, aggregate
from eval.match import play_match

__all__ = [
    "HAND_SIZE",
    "DECK_SIZE",
    "MAX_COPIES",
    "DeckSpec",
    "load_deck",
    "opening_no_mulligan_probability",
    "deck_static_metrics",
    "AgentFactory",
    "run_deck_ab",
    "run_gauntlet",
]

# Pokémon TCG constants. A legal constructed deck is exactly 60 cards with at most 4
# copies of any one card (Basic Energy exempt) and at most 1 ACE SPEC; the opening
# hand is 7 cards and a hand with no Basic Pokémon forces a mulligan.
DECK_SIZE = 60
HAND_SIZE = 7
MAX_COPIES = 4

# An agent factory ``f(seed) -> agent`` (or ``f()``). The deck track builds a fresh
# agent per seat per match so a stateful champion policy still works.
AgentFactory = Callable[..., object]


# --------------------------------------------------------------------------- #
# Deck loading + identity
# --------------------------------------------------------------------------- #
def _deck_hash(cards: Sequence[int]) -> str:
    """Order-independent content hash of a deck (a multiset of card ids).

    Sorting first makes the hash a function of the *card multiset*, not the file
    order, so two files listing the same 60 cards in a different order are the same
    champion version. Kept local (a short sha1) to avoid importing the trace module's
    engine-stamped hasher for a pure-data operation.
    """
    import hashlib

    payload = ",".join(str(c) for c in sorted(cards))
    return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class DeckSpec:
    """A named, hashable deck: the card-id list plus its identity for reports.

    ``name``/``version`` label the deck in the A/B report; ``deck_hash`` is the
    order-independent content hash so a champion version is pinned to its exact card
    multiset regardless of file ordering.
    """

    name: str
    cards: list[int]
    version: str = "0"
    path: Optional[str] = None

    @property
    def deck_hash(self) -> str:
        return _deck_hash(self.cards)

    def label(self) -> str:
        return f"{self.name}@{self.version}" if self.version not in ("", "0") else self.name


def load_deck(path: str, name: Optional[str] = None, version: str = "0") -> DeckSpec:
    """Load a ``deck.csv``-style file (one card id per line) into a :class:`DeckSpec`.

    Blank lines are ignored and at most :data:`DECK_SIZE` ids are read, matching
    :func:`eval.arena._load_deck` / the submission loader. ``name`` defaults to the
    file's basename.
    """
    with open(path) as f:
        cards = [int(x) for x in f.read().split("\n") if x.strip()][:DECK_SIZE]
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    return DeckSpec(name=name, cards=cards, version=version, path=path)


# --------------------------------------------------------------------------- #
# Static deck metrics (legality / energy ratio / 初動安定性)
# --------------------------------------------------------------------------- #
def opening_no_mulligan_probability(
    n_basic_pokemon: int, deck_size: int = DECK_SIZE, hand_size: int = HAND_SIZE
) -> float:
    """P(≥1 Basic Pokémon in the opening hand) — the deck's 初動安定性.

    A player who opens with no Basic Pokémon must mulligan, so this hypergeometric
    probability is a deck-only, match-free measure of opening reliability::

        P(≥1) = 1 - C(deck - basics, hand) / C(deck, hand)

    Returns ``0.0`` if there are no Basic Pokémon and ``1.0`` once a no-Basic hand is
    impossible (``deck - basics < hand``). Deterministic — no engine/match needed.
    """
    if n_basic_pokemon <= 0:
        return 0.0
    non_basic = deck_size - n_basic_pokemon
    if non_basic < hand_size:
        return 1.0
    p_none = math.comb(non_basic, hand_size) / math.comb(deck_size, hand_size)
    return 1.0 - p_none


def _enum_name(enum_cls, value) -> str:
    """Human-readable name for an int/enum value, unknown-safe.

    The engine deserialises ``cardType`` / ``energyType`` as plain ints; resolve them
    through the enum for readable report keys, but fall back to ``str(value)`` for a
    value the enum does not know yet (the enums may gain members mid-competition).
    """
    try:
        return enum_cls(int(value)).name
    except (ValueError, TypeError):
        return str(value)


def deck_static_metrics(deck: DeckSpec, registry=None) -> dict:
    """Match-free metrics for a single deck, read from the engine's card master data.

    Computes composition (counts by card type), **energy ratio** (energy cards / 60,
    plus a per-energy-type breakdown), **初動安定性** (see
    :func:`opening_no_mulligan_probability`), and a **legality** verdict with itemised
    violations:

    * deck size must be exactly :data:`DECK_SIZE`;
    * ≤ :data:`MAX_COPIES` copies of any single card, **except Basic Energy** (which is
      unlimited);
    * ≤ 1 ACE SPEC card in total;
    * ≥ 1 Basic Pokémon (else every opening hand mulligans);
    * every card id must exist in the master data.

    ``registry`` defaults to the shared :func:`eval.registry.get_registry`. The card
    types/flags come straight from :class:`~cg.api.CardData`, so a newly added card is
    classified with no code change here.
    """
    from cg.api import CardType, EnergyType

    if registry is None:
        from eval.registry import get_registry

        registry = get_registry()

    cards = deck.cards
    counts: dict[int, int] = {}
    for cid in cards:
        counts[cid] = counts.get(cid, 0) + 1

    type_counts: dict[str, int] = {}
    energy_by_type: dict[str, int] = {}
    n_energy = 0
    n_basic_pokemon = 0
    n_ace_spec = 0
    unknown_ids: list[int] = []
    over_copies: list[dict] = []

    for cid, copies in counts.items():
        card = registry.get_card(cid)
        if card is None:
            unknown_ids.append(cid)
            type_counts["UNKNOWN"] = type_counts.get("UNKNOWN", 0) + copies
            continue

        ctype = card.cardType
        tname = _enum_name(CardType, ctype)
        type_counts[tname] = type_counts.get(tname, 0) + copies

        is_basic_energy = ctype == CardType.BASIC_ENERGY
        if ctype in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_energy += copies
            ename = _enum_name(EnergyType, card.energyType)
            energy_by_type[ename] = energy_by_type.get(ename, 0) + copies
        if ctype == CardType.POKEMON and getattr(card, "basic", False):
            n_basic_pokemon += copies
        if getattr(card, "aceSpec", False):
            n_ace_spec += copies

        # Copy-limit: Basic Energy is exempt from the 4-copy rule.
        if not is_basic_energy and copies > MAX_COPIES:
            over_copies.append(
                {"card_id": cid, "name": card.name, "copies": copies}
            )

    violations: list[str] = []
    if len(cards) != DECK_SIZE:
        violations.append(f"deck size {len(cards)} != {DECK_SIZE}")
    if unknown_ids:
        violations.append(f"unknown card ids: {sorted(unknown_ids)}")
    for oc in over_copies:
        violations.append(
            f"{oc['copies']}x '{oc['name']}' (id={oc['card_id']}) exceeds {MAX_COPIES}"
        )
    if n_ace_spec > 1:
        violations.append(f"{n_ace_spec} ACE SPEC cards (max 1)")
    if n_basic_pokemon < 1:
        violations.append("no Basic Pokémon (every opening hand mulligans)")

    return {
        "name": deck.name,
        "version": deck.version,
        "deck_hash": deck.deck_hash,
        "size": len(cards),
        "distinct_cards": len(counts),
        "composition": dict(sorted(type_counts.items())),
        "n_basic_pokemon": n_basic_pokemon,
        "n_energy": n_energy,
        "energy_ratio": (n_energy / len(cards)) if cards else 0.0,
        "energy_by_type": dict(sorted(energy_by_type.items())),
        "n_ace_spec": n_ace_spec,
        "opening_stability": opening_no_mulligan_probability(n_basic_pokemon),
        "legal": not violations,
        "violations": violations,
    }


# --------------------------------------------------------------------------- #
# Paired A/B: fix the agent, swap the decks between seats
# --------------------------------------------------------------------------- #
class _FirstPlayerProbe:
    """Wraps a fixed agent to capture the engine-decided first player, transparently.

    The deck track needs the first-player seat for 先後別 win rates but does not need
    per-agent latency (both seats run the *same* fixed policy). This thin wrapper only
    records the first player it observes into the shared per-match ``ctx`` and forwards
    every call — so ``play_match`` behaves exactly as on the bare agent.
    """

    def __init__(self, inner, ctx: dict) -> None:
        self._inner = inner
        self._ctx = ctx

    def act(self, obs: dict) -> list[int]:
        if self._ctx.get("first_player") is None:
            fp = (obs.get("current") or {}).get("firstPlayer", -1)
            if fp not in (-1, None):
                self._ctx["first_player"] = fp
        return self._inner.act(obs)

    def on_match_start(self, player_index: int) -> None:
        hook = getattr(self._inner, "on_match_start", None)
        if callable(hook):
            hook(player_index)

    def on_match_end(self, result) -> None:
        hook = getattr(self._inner, "on_match_end", None)
        if callable(hook):
            hook(result)


_FAULT_REASONS = {"illegal_move", "timeout", "agent_exception"}


def _record_for_deck_match(
    *,
    match_index: int,
    deck_a_seat: int,
    first_player: Optional[int],
    label_a: str,
    label_b: str,
    seed0: int,
    seed1: int,
    result,
) -> MatchRecord:
    """Build an arena :class:`MatchRecord` for one deck-A/B match.

    ``deck_a_seat`` is the seat deck A occupied this match (the deck analog of the
    arena's ``seat_of_a``), so the reused :func:`eval.arena.aggregate` computes deck
    A's win rate, Wilson CI and 先後別 splits directly — treating the *decks* as the
    two contestants. Faults are attributed to the deck whose seat faulted.
    """
    labels_by_seat = [label_a, label_b] if deck_a_seat == 0 else [label_b, label_a]
    reason = result.reason.value
    winner_seat = result.winner
    winner_label = labels_by_seat[winner_seat] if winner_seat in (0, 1) else None
    draw = result.is_draw
    undecided = winner_seat is None and not draw
    faulted_seat = result.faulted_player
    faulted_label = labels_by_seat[faulted_seat] if faulted_seat in (0, 1) else None
    fault_category = reason if reason in _FAULT_REASONS else None
    return MatchRecord(
        match_index=match_index,
        pair_index=match_index // 2,
        seat_of_a=deck_a_seat,
        first_player=first_player,
        label_a=label_a,
        label_b=label_b,
        seed_a=seed0,
        seed_b=seed1,
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
        a_decisions=0,
        b_decisions=0,
        a_decision_ms=0.0,
        b_decision_ms=0.0,
        trace_path=None,
    )


def run_deck_ab(
    deck_a: DeckSpec,
    deck_b: DeckSpec,
    agent_factory: AgentFactory,
    *,
    n_matches: int = 200,
    agent_seed: int = 0,
    agent_label: Optional[str] = None,
    max_steps: int = 100_000,
    per_move_timeout: Optional[float] = None,
    out_dir: Optional[str] = "eval/deck_runs",
    run_label: Optional[str] = None,
    write_outputs: bool = True,
) -> dict:
    """Paired A/B comparison of two decks with **one fixed agent** on both seats.

    Deck A and deck B each play seat 0 and seat 1 an equal number of times (deck A
    sits in seat ``match_index % 2``), so the 先手 advantage is split evenly — a true
    paired deck comparison. The agent built by ``agent_factory`` is used on *both*
    seats every match, so the only thing that differs between the sides is the deck.

    Returns a JSON-serialisable dict: the reused arena report (win rate + Wilson CI +
    先後別 + safety, keyed on the two *decks*), each deck's static metrics, and the
    fixed-agent identity (so the report is self-documenting about what was held fixed).
    Written to ``<out_dir>/<run_label>/report.json`` unless ``write_outputs`` is off.
    """
    if n_matches <= 0:
        raise ValueError("n_matches must be positive")

    # Discover the fixed agent's identity for the report (build one sample).
    sample = _build_agent(agent_factory, agent_seed)
    fixed_label = agent_label or str(getattr(sample, "name", type(sample).__name__))
    fixed_version = str(getattr(sample, "version", "0"))

    label_a = deck_a.label()
    label_b = deck_b.label()
    if label_a == label_b:  # keep the two decks distinguishable in the report
        label_a, label_b = f"{label_a}#A", f"{label_b}#B"

    records: list[MatchRecord] = []
    for i in range(n_matches):
        deck_a_seat = i % 2  # paired seat-swap of the DECKS
        decks_by_seat = (
            [deck_a.cards, deck_b.cards] if deck_a_seat == 0
            else [deck_b.cards, deck_a.cards]
        )
        seed0 = agent_seed + 2 * i
        seed1 = agent_seed + 2 * i + 1
        ctx: dict = {"first_player": None}
        seat_agents = [
            _FirstPlayerProbe(_build_agent(agent_factory, seed0), ctx),
            _FirstPlayerProbe(_build_agent(agent_factory, seed1), ctx),
        ]
        result = play_match(
            decks_by_seat[0], decks_by_seat[1], seat_agents,
            max_steps=max_steps, per_move_timeout=per_move_timeout,
        )
        records.append(
            _record_for_deck_match(
                match_index=i,
                deck_a_seat=deck_a_seat,
                first_player=ctx.get("first_player"),
                label_a=label_a,
                label_b=label_b,
                seed0=seed0,
                seed1=seed1,
                result=result,
            )
        )

    config = {
        "run_label": run_label or f"{label_a}_vs_{label_b}",
        "fixed_agent": {"label": fixed_label, "version": fixed_version},
        "n_matches": n_matches,
        "agent_seed": agent_seed,
        "label_a": label_a,
        "label_b": label_b,
        "deck_a": {"name": deck_a.name, "version": deck_a.version, "hash": deck_a.deck_hash},
        "deck_b": {"name": deck_b.name, "version": deck_b.version, "hash": deck_b.deck_hash},
    }
    report = aggregate(records, [], [], config)

    out = {
        "config": config,
        "totals": report.totals,
        "win_rates": report.win_rates,
        "seat_winrate": report.seat_winrate,
        "safety": report.safety,
        "reason_counts": report.reason_counts,
        "static_metrics": {
            "deck_a": _safe_static(deck_a),
            "deck_b": _safe_static(deck_b),
        },
    }

    if write_outputs and out_dir:
        now = datetime.datetime.now(datetime.timezone.utc)
        rl = run_label or f"{config['run_label']}_{now.strftime('%Y%m%dT%H%M%S')}"
        run_dir = os.path.join(out_dir, rl)
        os.makedirs(run_dir, exist_ok=True)
        report_path = os.path.join(run_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True, ensure_ascii=False)
        out["report_path"] = report_path

    return out


def run_gauntlet(
    champion: DeckSpec,
    challengers: Sequence[DeckSpec],
    agent_factory: AgentFactory,
    *,
    n_matches: int = 200,
    agent_seed: int = 0,
    **kwargs,
) -> dict:
    """Run the champion against several challengers — 主要matchup別勝率.

    Each matchup is an independent paired :func:`run_deck_ab` (champion = deck A), so
    every challenger plays the champion first and second equally. Returns the champion
    static metrics plus one entry per matchup with the challenger's win rate against
    the champion and its Wilson CI.
    """
    matchups = []
    fixed_agent = None
    for challenger in challengers:
        ab = run_deck_ab(
            champion, challenger, agent_factory,
            n_matches=n_matches, agent_seed=agent_seed, **kwargs,
        )
        fixed_agent = ab["config"]["fixed_agent"]
        wr = ab["win_rates"]
        matchups.append(
            {
                "challenger": challenger.label(),
                "champion_win_rate": wr["a_win_rate"],
                "champion_win_rate_ci95": wr["a_win_rate_ci95"],
                "challenger_win_rate": wr["b_win_rate"],
                "draw_rate": wr["draw_rate"],
                "n": ab["totals"]["n"],
                "faults": {
                    "champion": ab["safety"]["a_faults"],
                    "challenger": ab["safety"]["b_faults"],
                },
                "challenger_static": ab["static_metrics"]["deck_b"],
            }
        )
    return {
        "champion": champion.label(),
        "champion_static": _safe_static(champion),
        "fixed_agent": fixed_agent,
        "matchups": matchups,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_agent(factory: AgentFactory, seed: int):
    """Call ``factory(seed)`` if it accepts an arg, else ``factory()``."""
    try:
        return factory(seed)
    except TypeError:
        return factory()


def _safe_static(deck: DeckSpec) -> dict:
    """Static metrics, degrading to an error note if the engine card data is absent."""
    try:
        return deck_static_metrics(deck)
    except Exception as e:  # pragma: no cover - only without the engine
        return {"name": deck.name, "error": f"static metrics unavailable: {e}"}


# --------------------------------------------------------------------------- #
# CLI: champion vs a challenger (defaults to a mirror of the champion)
# --------------------------------------------------------------------------- #
def _main(argv: list[str]) -> int:
    """``python eval/deck_eval.py [n_matches] [challenger.csv]``.

    Fixes the champion policy (:class:`agents.RuleAgent`) and runs a paired A/B of the
    versioned champion deck against ``challenger.csv`` (default: a mirror of the
    champion). Prints each deck's static metrics and the paired win rate + CI.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)

    from agents import RuleAgent
    from decks.champion import load_champion

    n = int(argv[1]) if len(argv) > 1 else 200
    champion = load_champion()
    if len(argv) > 2:
        challenger = load_deck(argv[2], name=os.path.splitext(os.path.basename(argv[2]))[0])
    else:
        challenger = DeckSpec(name="mirror", cards=list(champion.cards), version=champion.version)

    result = run_deck_ab(
        champion, challenger,
        lambda s: RuleAgent(seed=s),
        n_matches=n,
        agent_label="RuleAgent",
        write_outputs=True,
    )
    sa = result["static_metrics"]["deck_a"]
    sb = result["static_metrics"]["deck_b"]
    wr = result["win_rates"]
    t = result["totals"]
    ci = wr["a_win_rate_ci95"]
    print(f"fixed agent: {result['config']['fixed_agent']['label']}")
    for tag, s in (("champion", sa), ("challenger", sb)):
        print(
            f"  {tag} {s['name']}@{s['version']}: legal={s['legal']} "
            f"energy_ratio={s['energy_ratio']:.2f} "
            f"opening_stability={s['opening_stability']:.3f} "
            f"basic_pokemon={s['n_basic_pokemon']}"
            + ("" if s["legal"] else f" violations={s['violations']}")
        )
    print(
        f"paired A/B: n={t['n']} champion W/D/L={t['a_wins']}/{t['draws']}/{t['b_wins']} "
        f"champion_win_rate={wr['a_win_rate']:.3f} CI95=[{ci[0]:.3f},{ci[1]:.3f}] "
        f"first_player_win_rate={result['seat_winrate']['first_player_win_rate']}"
    )
    if result.get("report_path"):
        print(f"artifacts: {result['report_path']}")
    return 0


if __name__ == "__main__":
    import sys

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    raise SystemExit(_main(sys.argv))
