"""Pure MAIN-turn scoring for the rule-based agent (SOT-1647, R2).

This module is the **decision logic** of :class:`agents.rule_agent.RuleAgent`, kept
deliberately separate from all I/O and the engine (SOT-1631 案B / 受け入れ条件③):

* It never calls the cabt engine, never touches global battle state, and never
  reads/writes anything. It is a **pure function** of a parsed
  :class:`~cg.api.Observation`, the pending :class:`~cg.api.SelectData`, and a static
  :class:`CardIndex` (card/attack reference data passed in). Same inputs → same
  ``list[ScoredOption]`` every time, so it is exhaustively unit-testable against
  hand-built fixtures with no live battle.
* Legality is **not** re-implemented here: the engine already enumerates only legal
  moves in ``select.option`` (an offered ATTACK therefore already has its energy
  cost met, an offered EVOLVE is already a legal evolution, …). Scoring only *ranks*
  the legal options; the :class:`~agents.protocol.SafeAgent` still validates the
  chosen action before it reaches the engine.

The policy — a *minimal winning* MAIN-turn policy (最小勝利方策)
--------------------------------------------------------------
A MAIN selection asks which single sub-action to take this turn (PLAY / ATTACH /
EVOLVE / ABILITY / DISCARD / RETREAT / ATTACK / END); after each non-terminating
sub-action the engine re-enters MAIN, and an ATTACK (or END) ends the turn. The
policy therefore spends the turn on value-building setup first and ends it on the
best attack:

  KO攻撃 > エネルギー付与(active) > 進化 > 展開(play) > エネルギー付与(bench)
        > 通常攻撃 > 交代(きぜつ回避) > ターン終了

Rationale for the ordering:

* **勝ちにつながる攻撃 (KO).** A knock-out is taken immediately — further setup is
  wasted once the turn ends, and a KO converts directly toward the prize-based win.
* **Setup before a non-KO attack.** Attaching energy / evolving / developing the
  bench all keep the turn alive and strengthen future turns, so they outrank a
  non-lethal attack, which is played *last* (it ends the turn). Every setup action
  strictly consumes a resource (an energy attachment, a hand card, …), so the MAIN
  loop always terminates — there is no risk of cycling.
* **展開 (develop).** Playing basics/items/supporters both grows the board (so a
  knocked-out active can be replaced — avoiding a no-Active loss) and draws cards.
* **交代 (きぜつ回避).** A guarded retreat: only when the active is badly hurt and a
  bench Pokémon can take over, and always below "just attack" (a KO/attack this turn
  beats retreating). Retreat sets the once-per-turn retreated flag, so it too cannot
  loop.
* **ABILITY / DISCARD are not used proactively.** They are scored below END so the
  policy never triggers them for their own sake — abilities in particular can be
  repeatable and would risk a non-terminating MAIN loop. The always-legal fallback
  still covers them if they are ever the only move.

This module is intentionally engine-*type* aware but engine-*call* free: it imports
the ``cg.api`` enums/dataclasses only for typed access to the observation the engine
already produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cg.api import (
    Attack,
    CardData,
    Observation,
    Option,
    OptionType,
    SelectData,
    AreaType,
)

__all__ = [
    "OptionCategory",
    "ScoredOption",
    "CardIndex",
    "estimate_attack",
    "score_main_options",
    "pick_best_option",
]


class OptionCategory(str, Enum):
    """The tactical bucket a MAIN option falls into (for scoring + audit/logging)."""

    WINNING_ATTACK = "winning_attack"   # 勝ちにつながる攻撃 (estimated KO)
    ENERGY_ACTIVE = "energy_active"     # エネルギー付与 (to the active attacker)
    EVOLVE = "evolve"                   # 進化
    DEVELOP = "develop"                 # 展開 (play a card from hand)
    ENERGY_BENCH = "energy_bench"       # エネルギー付与 (to a bench Pokémon)
    ATTACK = "attack"                   # 通常攻撃 (non-lethal; taken last)
    SWITCH = "switch"                   # きぜつ回避/交代 (guarded retreat)
    END = "end"                         # ターン終了
    AVOID = "avoid"                     # ability/discard/unknown — never proactive


# Deterministic base scores. The gaps encode the priority ordering; the attack
# refinements (below) stay inside their band so the ordering is never violated:
#   * a KO attack adds its damage on top of WINNING_ATTACK (so, among KOs, the
#     hardest hit wins) — always the top of the table;
#   * a non-KO attack adds a damage bonus capped strictly below the ENERGY_BENCH
#     gap, so setup (attach/evolve/develop) is always preferred over a non-lethal
#     attack, which in turn always beats a retreat or ending the turn.
_S_WINNING_ATTACK = 1000.0
_S_ENERGY_ACTIVE = 500.0
_S_EVOLVE = 450.0
_S_DEVELOP = 400.0
_S_ENERGY_BENCH = 350.0
_S_ATTACK = 200.0
_S_SWITCH = 150.0
_S_END = 0.0
_S_AVOID = -1.0

# A non-KO attack's damage bonus is capped here so _S_ATTACK + bonus < _S_ENERGY_BENCH.
_ATTACK_DMG_BONUS_CAP = 149.0

# Retreat only when the active has lost this fraction of its max HP (and a bench
# Pokémon can take over). Conservative so retreat's tempo cost rarely backfires.
_RETREAT_HP_FRACTION = 1.0 / 3.0


@dataclass(frozen=True)
class ScoredOption:
    """One scored MAIN option: its index, priority score, category, and a reason."""

    index: int
    score: float
    category: OptionCategory
    reason: str


class CardIndex:
    """Static card/attack reference data, indexed by id — the scorer's only "world".

    Built once from the engine (:meth:`from_engine`) and passed in, so the scoring
    functions stay pure (no engine calls). Tests construct one directly from a few
    hand-made :class:`~cg.api.CardData` / :class:`~cg.api.Attack` objects.
    """

    def __init__(self, cards: dict[int, CardData], attacks: dict[int, Attack]) -> None:
        self._cards = cards
        self._attacks = attacks

    @classmethod
    def from_engine(cls, cards: list[CardData], attacks: list[Attack]) -> "CardIndex":
        return cls(
            {c.cardId: c for c in cards},
            {a.attackId: a for a in attacks},
        )

    def card(self, card_id: Optional[int]) -> Optional[CardData]:
        return self._cards.get(card_id) if card_id is not None else None

    def attack(self, attack_id: Optional[int]) -> Optional[Attack]:
        return self._attacks.get(attack_id) if attack_id is not None else None


def estimate_attack(
    attack: Optional[Attack],
    my_active_card: Optional[CardData],
    opp_active_hp: Optional[int],
    opp_active_card: Optional[CardData],
) -> tuple[float, bool]:
    """Estimate an attack's damage and whether it knocks out the opponent's active.

    A deliberately simple heuristic (the engine, not this, computes real damage):
    the attack's base ``damage``, **doubled** when the defender is weak to the
    attacker's type (our active's ``energyType`` matches the defender's ``weakness``).
    Resistance and text effects are ignored. A KO is estimated when the (nonzero)
    damage meets or exceeds the defender's *current* HP.

    Many real attacks scale their damage (base ``damage`` == 0 in the static data);
    those score as 0 here, which only affects *ordering among attacks* — the policy
    still prefers any attack over ending the turn.
    """
    if attack is None:
        return (0.0, False)
    dmg = float(attack.damage or 0)
    if (
        dmg > 0
        and my_active_card is not None
        and opp_active_card is not None
        and opp_active_card.weakness is not None
        and my_active_card.energyType == opp_active_card.weakness
    ):
        dmg *= 2.0
    is_ko = dmg > 0 and opp_active_hp is not None and dmg >= opp_active_hp
    return (dmg, is_ko)


def _active_pokemon(player):
    """The active Pokémon dataclass for a player state, or ``None``."""
    active = getattr(player, "active", None) or []
    return active[0] if active and active[0] is not None else None


def _score_option(
    opt: Option,
    parsed: Observation,
    cards: CardIndex,
) -> tuple[float, OptionCategory, str]:
    """Score a single MAIN option → ``(score, category, reason)`` (pure)."""
    otype = int(opt.type)
    state = parsed.current

    if otype == int(OptionType.END):
        return (_S_END, OptionCategory.END, "end turn")

    if otype == int(OptionType.ATTACK):
        atk = cards.attack(opt.attackId)
        my = _me(state)
        opp = _opp(state)
        my_active = _active_pokemon(my) if my else None
        opp_active = _active_pokemon(opp) if opp else None
        my_card = cards.card(my_active.id) if my_active else None
        opp_card = cards.card(opp_active.id) if opp_active else None
        opp_hp = opp_active.hp if opp_active else None
        dmg, is_ko = estimate_attack(atk, my_card, opp_hp, opp_card)
        name = atk.name if atk else f"attack#{opt.attackId}"
        if is_ko:
            return (
                _S_WINNING_ATTACK + dmg,
                OptionCategory.WINNING_ATTACK,
                f"KO with {name} (~{dmg:g} dmg)",
            )
        bonus = min(dmg, _ATTACK_DMG_BONUS_CAP)
        return (_S_ATTACK + bonus, OptionCategory.ATTACK, f"attack {name} (~{dmg:g} dmg)")

    if otype == int(OptionType.ATTACH):
        to_active = opt.inPlayArea == int(AreaType.ACTIVE)
        if to_active:
            return (_S_ENERGY_ACTIVE, OptionCategory.ENERGY_ACTIVE, "attach energy to active")
        return (_S_ENERGY_BENCH, OptionCategory.ENERGY_BENCH, "attach energy to bench")

    if otype == int(OptionType.EVOLVE):
        return (_S_EVOLVE, OptionCategory.EVOLVE, "evolve a Pokémon")

    if otype == int(OptionType.PLAY):
        return (_S_DEVELOP, OptionCategory.DEVELOP, "play a card (develop/draw)")

    if otype == int(OptionType.RETREAT):
        my = _me(state)
        my_active = _active_pokemon(my) if my else None
        bench = (getattr(my, "bench", None) or []) if my else []
        if my_active is not None and bench and _is_hurt(my_active):
            return (_S_SWITCH, OptionCategory.SWITCH, "retreat hurt active (きぜつ回避)")
        return (_S_AVOID, OptionCategory.AVOID, "retreat not beneficial")

    # ABILITY / DISCARD / anything else: never used proactively (loop-safe).
    return (_S_AVOID, OptionCategory.AVOID, f"non-proactive option type {otype}")


def _me(state):
    if state is None or getattr(state, "players", None) is None:
        return None
    yi = state.yourIndex
    return state.players[yi] if 0 <= yi < len(state.players) else None


def _opp(state):
    if state is None or getattr(state, "players", None) is None:
        return None
    yi = state.yourIndex
    oi = 1 - yi
    return state.players[oi] if 0 <= oi < len(state.players) else None


def _is_hurt(pokemon) -> bool:
    """True if the Pokémon has lost at least ``_RETREAT_HP_FRACTION`` of its max HP."""
    max_hp = getattr(pokemon, "maxHp", 0) or 0
    hp = getattr(pokemon, "hp", 0) or 0
    if max_hp <= 0:
        return False
    return hp <= max_hp * (1.0 - _RETREAT_HP_FRACTION)


def score_main_options(
    parsed: Observation,
    select: SelectData,
    cards: CardIndex,
) -> list[ScoredOption]:
    """Score every option of a MAIN selection, in option order (pure).

    Returns one :class:`ScoredOption` per ``select.option`` entry (same order/length).
    Requires a live board (``parsed.current`` present); with no state there is nothing
    to reason about, so an empty list is returned and the caller should defer to the
    safety fallback.
    """
    if parsed is None or parsed.current is None or select is None:
        return []
    scored: list[ScoredOption] = []
    for i, opt in enumerate(select.option):
        score, category, reason = _score_option(opt, parsed, cards)
        scored.append(ScoredOption(index=i, score=score, category=category, reason=reason))
    return scored


def pick_best_option(scored: list[ScoredOption], rng) -> Optional[int]:
    """The index of the highest-scoring option; ties broken by ``rng`` (stable).

    Deterministic given ``rng``'s state: the top score is unique in almost all real
    positions, and when several options tie exactly (e.g. two interchangeable energy
    attachments) ``rng.choice`` picks stably among them (受け入れ条件: 同点のみ安定した
    乱択). Returns ``None`` for an empty input.
    """
    if not scored:
        return None
    best = max(s.score for s in scored)
    top = [s.index for s in scored if s.score == best]
    if len(top) == 1:
        return top[0]
    return rng.choice(top)
