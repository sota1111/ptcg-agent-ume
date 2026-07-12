"""Pure per-context tactics for setup / forced-selection turns (SOT-1648, R3).

R2 (:mod:`agents.rule_scoring`) grew the MAIN-turn policy. Everything *other* than a
MAIN selection — the initial active/bench placement, the go-first coin, mulligan,
promoting a new active after a knock-out, attaching / discarding energy, searching the
deck, setting prizes, special-condition choices, … — is a **forced or setup selection**
the engine asks for between (or around) MAIN turns. This module is the decision logic
for those, kept in the same 案B split as ``rule_scoring``:

* It is a **pure function** of the parsed :class:`~cg.api.Observation`, the pending
  :class:`~cg.api.SelectData`, and a static :class:`~agents.rule_scoring.CardIndex`.
  It never calls the engine, never touches global state, and never reads/writes I/O, so
  it is exhaustively unit-testable against hand-built selections with no live battle.
* Legality is **not** re-implemented: the engine already enumerates only legal moves in
  ``select.option``; a tactic only *ranks / counts* them and the
  :class:`~agents.protocol.SafeAgent` still validates the chosen action. An unknown
  context, or a tactic that declines, still falls back to a guaranteed-legal random
  action (違法出力0), so the agent never crashes and never emits an illegal move.

Coverage model
--------------
:data:`CONTEXT_POLICIES` maps **every** known :class:`~cg.api.SelectContext` (except
``MAIN``, owned by R2) to a tactic, so any *encountered* non-MAIN context is handled by a
dedicated policy rather than falling through to legal-random (受け入れ条件①: 遭遇context
のhandler網羅率100%). The tactics fall into a few reasoned families:

* **place / promote the strongest** — pick the best attacker for the active spot and
  develop the bench (initial active/bench, promote-after-KO, forced switch);
* **acquire resources** — a deck→hand search or a draw-count takes the *maximum* offered
  (more cards / more draw = more future options);
* **spend the minimum** — a discard / detach / return-to-deck gives up the *fewest*
  allowed and the *least valuable* of them, preserving resources for future attacks
  (受け入れ条件③: 複数選択が組合せ・将来資源を考慮);
* **beneficial effects** — heal / evolve / recover-condition / apply-condition take the
  offered upside up to the cap;
* **initiative / forced answers** — go first, reveal on mulligan;
* **neutral minimum** — for blind or genuinely indifferent choices (prize set-up, look,
  put-to-deck) satisfy the requirement with the fewest, lowest options.

Multi-selection (``maxCount > 1``) is handled as a *combination*, not option-by-option:
:func:`select_beneficial_combo` takes the top-valued **set** up to the cap (and stops
early once the remaining options add no benefit), and :func:`select_cost_combo` gives up
the minimum-size, least-valuable subset — both reason about the whole set and about what
is worth keeping for later, not just each option in isolation.
"""

from __future__ import annotations

from typing import Callable, Optional

from cg.api import (
    AreaType,
    Observation,
    Option,
    OptionType,
    SelectContext,
    SelectData,
)

from .rule_scoring import CardIndex

__all__ = [
    "ContextPolicy",
    "CONTEXT_POLICIES",
    "select_for_context",
    "best_attack_damage",
    "card_pokemon_value",
    "board_pokemon_value",
    "select_beneficial_combo",
    "select_cost_combo",
    "select_neutral_min",
    "select_max_count",
]

# A per-context tactic: ``policy(parsed, select, cards, rng) -> option indices | None``.
# Returning ``None`` defers to the SafeAgent legal-random fallback, exactly like an
# absent policy. Every entry in ``CONTEXT_POLICIES`` returns a legal selection for a
# non-empty option list, so an encountered context is never left unhandled.
ContextPolicy = Callable[[Observation, SelectData, CardIndex, object], Optional[list[int]]]


# --------------------------------------------------------------------------- #
# card / pokemon value — the "how strong is this?" signal the tactics rank on
# --------------------------------------------------------------------------- #
def best_attack_damage(card, cards: CardIndex) -> float:
    """The highest static attack damage printed on ``card`` (0 if none / unknown).

    Many attacks scale their damage (static ``damage`` == 0); those contribute 0, which
    only affects *ordering among options*, never legality. Weakness/effects are ignored —
    this is a coarse "can it hit hard?" signal for placement, not a damage calculator.
    """
    if card is None:
        return 0.0
    best = 0.0
    for aid in (getattr(card, "attacks", None) or []):
        atk = cards.attack(aid)
        if atk is not None and atk.damage:
            best = max(best, float(atk.damage))
    return best


def card_pokemon_value(card, cards: CardIndex) -> float:
    """Value a *hand* Pokémon card for the active/bench spot (static data only).

    Attack potential dominates (we want a threat in the active spot), HP is a secondary
    durability term, and a heavy retreat cost is a small penalty (harder to pivot out).
    """
    if card is None:
        return 0.0
    dmg = best_attack_damage(card, cards)
    hp = float(getattr(card, "hp", 0) or 0)
    retreat = float(getattr(card, "retreatCost", 0) or 0)
    return dmg * 2.0 + hp * 0.1 - retreat


def board_pokemon_value(pk, cards: CardIndex) -> float:
    """Value an *in-play* Pokémon for promotion: current HP weighs more than for a
    fresh hand card (we must not promote a nearly-fainted mon when a healthy one is
    available), on top of its attack potential."""
    if pk is None:
        return float("-inf")
    card = cards.card(getattr(pk, "id", None))
    dmg = best_attack_damage(card, cards)
    hp = float(getattr(pk, "hp", 0) or 0)
    return dmg * 2.0 + hp * 0.5


# --------------------------------------------------------------------------- #
# board / option resolution (best-effort; always degrades to a legal choice)
# --------------------------------------------------------------------------- #
def _me(parsed: Observation):
    state = getattr(parsed, "current", None)
    if state is None or getattr(state, "players", None) is None:
        return None
    yi = state.yourIndex
    return state.players[yi] if 0 <= yi < len(state.players) else None


def _hand_card(me, opt: Option, cards: CardIndex):
    """The static :class:`CardData` for the hand card an option references, or ``None``."""
    if me is None:
        return None
    hand = getattr(me, "hand", None) or []
    idx = int(getattr(opt, "index", 0) or 0)
    if 0 <= idx < len(hand):
        return cards.card(getattr(hand[idx], "id", None))
    return None


def _board_pokemon(me, opt: Option):
    """The in-play Pokémon an option references (active or a bench slot), or ``None``."""
    if me is None:
        return None
    area = int(getattr(opt, "area", 0) or 0)
    idx = int(getattr(opt, "index", 0) or 0)
    if area == int(AreaType.ACTIVE):
        active = getattr(me, "active", None) or []
        return active[0] if active else None
    if area == int(AreaType.BENCH):
        bench = getattr(me, "bench", None) or []
        if 0 <= idx < len(bench):
            return bench[idx]
        return bench[0] if bench else None
    return None


def _bounds(select: SelectData) -> tuple[int, int, int]:
    """``(n, lo, hi)``: option count and the clamped legal selection-size window.

    ``lo``/``hi`` are ``minCount``/``maxCount`` clamped into ``[0, n]`` with ``lo <= hi``,
    so any count taken from ``[lo, hi]`` satisfies the engine's count rule by construction.
    """
    n = len(select.option)
    lo = max(0, min(int(select.minCount), n))
    hi = min(int(select.maxCount), n)
    if hi < lo:
        hi = lo
    return n, lo, hi


# --------------------------------------------------------------------------- #
# generic, combination- & resource-aware selectors (受け入れ条件③)
# --------------------------------------------------------------------------- #
def select_beneficial_combo(
    scored: list[tuple[int, float]], lo: int, hi: int, rng
) -> list[int]:
    """Pick the highest-value **set** of ``lo..hi`` options (a combination, not one-by-one).

    Takes the best options in value order up to ``hi``, but stops as soon as the count has
    reached the required minimum ``lo`` and the next option adds no positive value — so a
    multi-select acquires *only* what is worth keeping and never pads the hand with junk
    just because more slots are offered (future-resource aware). Exact ties at the cutoff
    are broken by ``rng`` for a stable, non-degenerate pick.
    """
    order = _value_order(scored, rng, ascending=False)
    chosen: list[int] = []
    for idx, val in order:
        if len(chosen) >= hi:
            break
        if len(chosen) >= lo and val <= 0.0:
            break
        chosen.append(idx)
    _pad_to(chosen, order, lo)
    return sorted(chosen)


def select_cost_combo(
    scored: list[tuple[int, float]], lo: int, rng
) -> list[int]:
    """Give up the **minimum-size, least-valuable** subset (a resource-preserving cost).

    A discard/detach/return spends resources, so the tactic parts with exactly the fewest
    the engine allows (``lo``) and, among the options, the *lowest*-valued ones — keeping
    the strongest cards/energy for future turns. Reasoning over the whole set (which subset
    to keep) rather than each option alone is the 組合せ/将来資源 requirement for costs.
    """
    order = _value_order(scored, rng, ascending=True)
    return sorted(idx for idx, _ in order[:lo])


def select_neutral_min(select: SelectData, rng) -> list[int]:
    """Satisfy a blind/indifferent selection with the fewest, lowest-index options.

    For choices with no value signal to the agent (prize set-up is face-down, a forced
    "look", a shuffle-back), there is nothing to optimise, so take the minimum legal count
    deterministically. Still a real decision — a legal, minimal, reproducible one.
    """
    _n, lo, _hi = _bounds(select)
    return list(range(lo))


def select_max_count(select: SelectData, rng) -> list[int]:
    """For a numeric COUNT selection, take the option with the largest ``number``.

    Draw-count / heal-count / counter-count: a larger number means more cards, more
    healing, or more damage placed — always the resource-maximising choice here. Ties
    (two options with the same number) break stably via ``rng``.
    """
    opts = select.option
    if not opts:
        return []
    vals = [(i, float(getattr(o, "number", 0) or 0)) for i, o in enumerate(opts)]
    best = max(v for _, v in vals)
    top = [i for i, v in vals if v == best]
    return [top[0] if len(top) == 1 else rng.choice(top)]


def _pick_option_of_type(select: SelectData, wanted: int) -> list[int]:
    """Index of the first option whose ``type`` equals ``wanted`` (else the first option)."""
    for i, o in enumerate(select.option):
        if int(getattr(o, "type", -1)) == wanted:
            return [i]
    return [0] if select.option else []


def _value_order(scored, rng, *, ascending: bool):
    """``scored`` sorted by value (stable on index), with tie groups shuffled by ``rng``.

    Deterministic given ``rng``: options are grouped by equal value and each tie group is
    ``rng``-shuffled, so genuine ties are broken stably/uniformly while distinct values
    keep their order (受け入れ条件: 同点のみ安定した乱択, mirrored from R2).
    """
    base = sorted(scored, key=lambda t: (t[1], t[0]), reverse=not ascending)
    out: list[tuple[int, float]] = []
    i = 0
    while i < len(base):
        j = i + 1
        while j < len(base) and base[j][1] == base[i][1]:
            j += 1
        group = base[i:j]
        if len(group) > 1:
            group = [group[k] for k in rng.sample(range(len(group)), len(group))]
        out.extend(group)
        i = j
    return out


def _pad_to(chosen: list[int], order: list[tuple[int, float]], lo: int) -> None:
    """Top ``chosen`` up to ``lo`` items from ``order`` (defensive; ``lo`` is always <= n)."""
    if len(chosen) >= lo:
        return
    for idx, _val in order:
        if idx not in chosen:
            chosen.append(idx)
            if len(chosen) >= lo:
                return


# --------------------------------------------------------------------------- #
# per-context tactics
# --------------------------------------------------------------------------- #
def _score_hand_options(parsed, select, cards) -> list[tuple[int, float]]:
    me = _me(parsed)
    return [
        (i, card_pokemon_value(_hand_card(me, o, cards), cards))
        for i, o in enumerate(select.option)
    ]


def _score_board_options(parsed, select, cards) -> list[tuple[int, float]]:
    me = _me(parsed)
    return [
        (i, board_pokemon_value(_board_pokemon(me, o), cards))
        for i, o in enumerate(select.option)
    ]


def place_active_from_hand(parsed, select, cards, rng) -> Optional[list[int]]:
    """Initial active: put the strongest hand Pokémon into the active spot."""
    _n, lo, hi = _bounds(select)
    if hi == 0:
        return list(range(lo))
    return select_beneficial_combo(
        [(i, v if v > 0 else 0.001) for i, v in _score_hand_options(parsed, select, cards)],
        max(lo, 1) if hi >= 1 else lo,
        1,
        rng,
    )


def develop_bench_from_hand(parsed, select, cards, rng) -> Optional[list[int]]:
    """Bench development: fill the offered bench slots with the strongest basics.

    Benching is board-building (a knocked-out active can be replaced — avoiding a
    no-Active loss — and it widens attack options), so take up to the cap, best first.
    """
    _n, lo, hi = _bounds(select)
    scored = [(i, v if v > 0 else 0.001) for i, v in _score_hand_options(parsed, select, cards)]
    return select_beneficial_combo(scored, lo, hi, rng)


def promote_best_from_board(parsed, select, cards, rng) -> Optional[list[int]]:
    """Promote / switch in the strongest in-play Pokémon (best attacker, most HP).

    Used after a knock-out (TO_ACTIVE) and for a forced switch of our own active: the new
    active should be the bench Pokémon most able to attack and survive.
    """
    _n, lo, hi = _bounds(select)
    scored = _score_board_options(parsed, select, cards)
    take = max(lo, 1) if hi >= 1 else lo
    return select_beneficial_combo(scored, take, max(take, 1), rng)


def attach_to_attacker(parsed, select, cards, rng) -> Optional[list[int]]:
    """Attach energy to the active attacker when offered; else to the best target.

    Energy on the active advances this turn's attack, so an option targeting the active
    area is preferred; otherwise the highest-value in-play Pokémon (falling back to the
    first option) receives it. Attaching is beneficial, so at least one is taken when the
    engine allows it.
    """
    _n, lo, hi = _bounds(select)
    if hi == 0:
        return list(range(lo))
    for i, o in enumerate(select.option):
        if int(getattr(o, "inPlayArea", 0) or 0) == int(AreaType.ACTIVE) or int(
            getattr(o, "area", 0) or 0
        ) == int(AreaType.ACTIVE):
            return [i]
    scored = _score_board_options(parsed, select, cards)
    return select_beneficial_combo(
        [(i, v if v != float("-inf") else 0.0) for i, v in scored],
        max(lo, 1),
        1,
        rng,
    )


def take_max_resources(parsed, select, cards, rng) -> Optional[list[int]]:
    """Acquire the maximum offered (deck→hand search, put-to-hand): more cards = options.

    The engine hides the identity of a searched card from the agent, so there is no
    per-card signal; the resource-maximising choice is simply to take as many as the cap
    allows — reasoned about as a set (take the whole allowed combination).
    """
    _n, lo, hi = _bounds(select)
    scored = [(i, 1.0) for i in range(len(select.option))]
    return select_beneficial_combo(scored, hi, hi, rng) if hi > 0 else list(range(lo))


def take_beneficial(parsed, select, cards, rng) -> Optional[list[int]]:
    """A beneficial effect (heal / evolve / recover- or apply-condition): take the upside.

    These help us (or hurt the opponent), so accept the offered options up to the cap,
    best-valued first where a board signal exists, at least the required minimum.
    """
    _n, lo, hi = _bounds(select)
    scored = _score_board_options(parsed, select, cards)
    if all(v == float("-inf") for _, v in scored):
        scored = [(i, 1.0) for i in range(len(select.option))]
    else:
        scored = [(i, v if v != float("-inf") else 0.0) for i, v in scored]
    return select_beneficial_combo(scored, max(lo, min(1, hi)), hi, rng)


def spend_minimum(parsed, select, cards, rng) -> Optional[list[int]]:
    """A cost (discard / detach / return-to-deck): give up the fewest, least valuable."""
    _n, lo, _hi = _bounds(select)
    scored = _score_board_options(parsed, select, cards)
    if all(v == float("-inf") for _, v in scored):
        scored = [(i, 0.0) for i in range(len(select.option))]
    else:
        scored = [(i, v if v != float("-inf") else 0.0) for i, v in scored]
    return select_cost_combo(scored, lo, rng)


def go_first(parsed, select, cards, rng) -> Optional[list[int]]:
    """Take the initiative: choose to go first (the YES option)."""
    return _pick_option_of_type(select, int(OptionType.YES))


def answer_reveal(parsed, select, cards, rng) -> Optional[list[int]]:
    """A forced disclosure (mulligan reveal): answer YES when offered, else the only move."""
    return _pick_option_of_type(select, int(OptionType.YES))


def neutral_min(parsed, select, cards, rng) -> Optional[list[int]]:
    """Blind / indifferent selection: satisfy the minimum count deterministically."""
    return select_neutral_min(select, rng)


def max_count(parsed, select, cards, rng) -> Optional[list[int]]:
    """Numeric COUNT selection: take the largest number offered (draw/heal/counters)."""
    return select_max_count(select, rng)


# --------------------------------------------------------------------------- #
# context → tactic table — every known non-MAIN context has a dedicated policy
# --------------------------------------------------------------------------- #
CONTEXT_POLICIES: dict[int, ContextPolicy] = {
    # placement / promotion of our own Pokémon
    int(SelectContext.SETUP_ACTIVE_POKEMON): place_active_from_hand,
    int(SelectContext.SETUP_BENCH_POKEMON): develop_bench_from_hand,
    int(SelectContext.SWITCH): promote_best_from_board,
    int(SelectContext.TO_ACTIVE): promote_best_from_board,
    int(SelectContext.TO_BENCH): develop_bench_from_hand,
    int(SelectContext.TO_FIELD): take_beneficial,
    # acquire resources
    int(SelectContext.TO_HAND): take_max_resources,
    int(SelectContext.TO_HAND_ENERGY): take_max_resources,
    int(SelectContext.DRAW_COUNT): max_count,
    int(SelectContext.REMOVE_DAMAGE_COUNTER_COUNT): max_count,
    int(SelectContext.DAMAGE_COUNTER_COUNT): max_count,
    # spend the minimum (costs)
    int(SelectContext.DISCARD): spend_minimum,
    int(SelectContext.DISCARD_ENERGY): spend_minimum,
    int(SelectContext.DISCARD_ENERGY_CARD): spend_minimum,
    int(SelectContext.DISCARD_TOOL_CARD): spend_minimum,
    int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD): spend_minimum,
    int(SelectContext.DETACH_FROM): spend_minimum,
    # attach energy
    int(SelectContext.ATTACH_TO): attach_to_attacker,
    int(SelectContext.ATTACH_FROM): neutral_min,
    # beneficial effects
    int(SelectContext.HEAL): take_beneficial,
    int(SelectContext.REMOVE_DAMAGE_COUNTER): take_beneficial,
    int(SelectContext.EVOLVES_FROM): take_beneficial,
    int(SelectContext.EVOLVES_TO): take_beneficial,
    int(SelectContext.EVOLVE): take_beneficial,
    int(SelectContext.AFFECT_SPECIAL_CONDITION): take_beneficial,
    int(SelectContext.RECOVER_SPECIAL_CONDITION): take_beneficial,
    # offensive damage placement — put as much on the opponent as offered
    int(SelectContext.DAMAGE): take_max_resources,
    int(SelectContext.DAMAGE_COUNTER): take_max_resources,
    int(SelectContext.DAMAGE_COUNTER_ANY): take_max_resources,
    # initiative / forced answers
    int(SelectContext.IS_FIRST): go_first,
    int(SelectContext.MULLIGAN): answer_reveal,
    # blind / indifferent — minimal legal decision
    int(SelectContext.TO_DECK): neutral_min,
    int(SelectContext.TO_DECK_BOTTOM): neutral_min,
    int(SelectContext.TO_DECK_ENERGY): neutral_min,
    int(SelectContext.TO_PRIZE): neutral_min,
    int(SelectContext.NOT_MOVE): neutral_min,
    int(SelectContext.DEVOLVE): neutral_min,
    int(SelectContext.LOOK): neutral_min,
    int(SelectContext.EFFECT_TARGET): take_beneficial,
    int(SelectContext.SWITCH_ENERGY_CARD): neutral_min,
    int(SelectContext.SWITCH_ENERGY): neutral_min,
    int(SelectContext.SKILL_ORDER): neutral_min,
    int(SelectContext.ATTACK): neutral_min,
    int(SelectContext.DISABLE_ATTACK): neutral_min,
    int(SelectContext.ACTIVATE): neutral_min,
    int(SelectContext.FIRST_EFFECT): neutral_min,
    int(SelectContext.MORE_DEVOLVE): neutral_min,
    int(SelectContext.COIN_HEAD): neutral_min,
}


def select_for_context(
    parsed: Observation, select: SelectData, cards: CardIndex, rng
) -> Optional[list[int]]:
    """Run the tactic registered for ``select.context`` (``None`` if none / it declines).

    ``None`` means *no policy for this context* — the caller (:class:`RuleAgent`) then
    lets the :class:`~agents.protocol.SafeAgent` fall back to a legal-random action, so an
    unknown or unmapped context is still handled safely (違法出力0).
    """
    if select is None or not select.option:
        return None
    fn = CONTEXT_POLICIES.get(int(select.context))
    if fn is None:
        return None
    return fn(parsed, select, cards, rng)
