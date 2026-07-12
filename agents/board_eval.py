"""Unified board evaluation ``score(state)`` for the rule-based agent line (SOT-1649, R4).

R2/R3 grew the RuleAgent as a stack of *local* tactics: a MAIN-turn category ordering
(:mod:`agents.rule_scoring`) and per-context setup policies (:mod:`agents.setup_scoring`).
Those local rules each answer "which option here?" in isolation from a hand-ranked table.
R4 adds the missing *global* signal: a single, weighted **board evaluation** —
``score(state)`` — that says how good a whole board position is for a player, so the agent
can compare the board positions its moves actually lead to rather than trusting a static
category ranking. The R4 agent (:class:`agents.eval_agent.EvalAgent`) plugs this in as the
evaluator of the leak-safe one-ply lookahead built in SOT-1650, so every MAIN option is
played on a search copy and the *resulting* board is scored by :func:`score_state`.

Design (same SOT-1631 案B split as the rest of the policy)
--------------------------------------------------------
* **Pure.** :func:`score_state` is a pure function of a parsed :class:`~cg.api.State`,
  the evaluating player's seat index, a static :class:`~agents.rule_scoring.CardIndex`,
  and a set of :class:`EvalWeights`. It never calls the engine, never touches global
  state, never reads/writes I/O — same inputs → same :class:`BoardEval`, so it is
  exhaustively unit-testable on hand-built states with no live battle.
* **Weighted + configurable (受け入れ条件: 重みは設定値化).** Every component is a term
  ``weight * feature(me, opp)``; the weights live in :class:`EvalWeights` so they can be
  tuned / ablated without touching the logic. :data:`DEFAULT_WEIGHTS` is the shipped R4
  set (tuned by the ablation, ``docs/ablation_r4.md``); :data:`FULL_WEIGHTS` keeps every
  component non-zero as the ablation baseline.
* **Traceable (判断ごとに score 内訳と reason code).** :class:`BoardEval` carries the
  per-component ``components`` breakdown and a compact ``reasons`` list of reason codes,
  so every evaluation is auditable (the R4 agent records the chosen one on ``last_eval``).
* **Ablatable (寄与のない複雑性を削減).** ``disabled`` zeroes named components so an
  ablation can measure each one's contribution. The R4 ablation (``docs/ablation_r4.md``)
  found ``prize`` dominant and no *other* single component whose removal robustly changed
  the R3 head-to-head, so the full seven-component set is kept; a component whose weight is
  set to ``0.0`` is dropped from the live decision while staying configurable and measured.

The seven components (受け入れ条件 / 実装内容)
--------------------------------------------
Each is a *differential* (my side minus the opponent's), so the score is symmetric and
zero on a mirrored board:

* ``prize`` — 勝敗 / サイド差: prizes remaining is the win clock (0 left = win), so the
  differential of prizes-remaining is the dominant term.
* ``active_survival`` — active 生存: whether each side still has an Active (losing it can
  lose the game) plus its remaining HP fraction.
* ``ko_threat`` — 次ターン KO 確率: whether my Active can already KO the opponent's Active
  with an affordable attack next turn (and, subtracted, the reverse threat on me).
* ``bench_dev`` — bench 育成: bench Pokémon developed (board width / KO insurance).
* ``hand_value`` — 手札価値: cards in hand (future options).
* ``energy_tempo`` — energy tempo: total energy in play (progress toward attacks).
* ``retreat_capacity`` — retreat 余力: whether the Active can afford to retreat (a healthy
  pivot out of a bad matchup), for me minus the opponent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from cg.api import State

from .rule_scoring import CardIndex

__all__ = [
    "EvalWeights",
    "DEFAULT_WEIGHTS",
    "FULL_WEIGHTS",
    "COMPONENTS",
    "BoardEval",
    "score_state",
    "WIN_SCORE",
    "LOSS_SCORE",
]

# The board-eval component names, in report/ablation order. Kept as data so the ablation
# runner and tests iterate the same source of truth as :class:`EvalWeights`.
COMPONENTS = (
    "prize",
    "active_survival",
    "ko_threat",
    "bench_dev",
    "hand_value",
    "energy_tempo",
    "retreat_capacity",
)

# Sentinels for a decided position. Large finite values (not ±inf) so score *deltas*
# used by the caller stay finite and orderable even next to a terminal board.
WIN_SCORE = 1_000_000.0
LOSS_SCORE = -1_000_000.0


@dataclass(frozen=True)
class EvalWeights:
    """Per-component weights for :func:`score_state` (受け入れ条件: 重みは設定値化).

    A component's contribution is ``weight * differential_feature``. Set a weight to
    ``0.0`` to drop that component from the live evaluation while keeping it configurable
    and measurable — the R4 ablation ships the non-contributing components that way.
    """

    prize: float = 1000.0
    active_survival: float = 100.0
    ko_threat: float = 60.0
    bench_dev: float = 20.0
    hand_value: float = 5.0
    energy_tempo: float = 30.0
    retreat_capacity: float = 10.0

    def without(self, component: str) -> "EvalWeights":
        """A copy with ``component``'s weight zeroed (for ablation)."""
        if component not in COMPONENTS:
            raise KeyError(f"unknown component {component!r}")
        return replace(self, **{component: 0.0})


#: The shipped R4 weights. The ablation (``docs/ablation_r4.md``) found no single component
#: whose removal *robustly* changed the R3 head-to-head — the one apparent gain (dropping
#: ``retreat_capacity``) did not replicate on independent seeds — so the full seven-component
#: evaluation is kept, with ``prize`` the dominant term.
DEFAULT_WEIGHTS = EvalWeights()

#: Alias of :data:`DEFAULT_WEIGHTS` (every component non-zero) — the named ablation baseline
#: each ``-component`` variant is measured against, kept so ``docs/ablation_r4.md`` reproduces.
FULL_WEIGHTS = EvalWeights()


@dataclass(frozen=True)
class BoardEval:
    """The result of one :func:`score_state` call: total, breakdown, and reason codes."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# feature helpers (pure; tolerant of partially-built / hand-made states)
# --------------------------------------------------------------------------- #
def _active(player) -> Optional[object]:
    """The Active Pokémon of a player state, or ``None`` (empty / face-down slot)."""
    active = getattr(player, "active", None) or []
    return active[0] if active and active[0] is not None else None


def _hp_fraction(pokemon) -> float:
    """Remaining HP as a fraction of max (0.0 when absent / unknown)."""
    if pokemon is None:
        return 0.0
    max_hp = getattr(pokemon, "maxHp", 0) or 0
    if max_hp <= 0:
        return 0.0
    return (getattr(pokemon, "hp", 0) or 0) / max_hp


def _energy_count(pokemon) -> int:
    return len(getattr(pokemon, "energies", None) or []) if pokemon is not None else 0


def _total_energy(player) -> int:
    """Total energy attached across a player's Active + bench."""
    total = _energy_count(_active(player))
    for pk in (getattr(player, "bench", None) or []):
        if pk is not None:
            total += _energy_count(pk)
    return total


def _can_ko(attacker_player, defender_player, cards: CardIndex) -> int:
    """1 if ``attacker`` can KO ``defender``'s Active with an affordable attack now.

    An attack is *affordable* when its energy cost (number of required energies) is met
    by the attacker's currently-attached energy; a KO is when the printed damage meets or
    exceeds the defender's current HP. Coarse on purpose (scaling attacks with static 0
    damage never register here — the engine's forward search, not this, resolves those);
    it is a cheap "is a lethal already on the board?" flag, not a damage calculator.
    """
    atk = _active(attacker_player)
    dfn = _active(defender_player)
    if atk is None or dfn is None:
        return 0
    attached = _energy_count(atk)
    card = cards.card(getattr(atk, "id", None))
    if card is None:
        return 0
    def_hp = getattr(dfn, "hp", 0) or 0
    for aid in (getattr(card, "attacks", None) or []):
        attack = cards.attack(aid)
        if attack is None or not attack.damage:
            continue
        if len(attack.energies or []) <= attached and float(attack.damage) >= def_hp > 0:
            return 1
    return 0


def _can_retreat(player, cards: CardIndex) -> int:
    """1 if the player's Active can afford to retreat and has a bench to retreat to."""
    active = _active(player)
    if active is None or not (getattr(player, "bench", None) or []):
        return 0
    card = cards.card(getattr(active, "id", None))
    retreat_cost = (getattr(card, "retreatCost", 0) or 0) if card is not None else 0
    return 1 if _energy_count(active) >= retreat_cost else 0


def _prizes_left(player) -> int:
    return len(getattr(player, "prize", None) or [])


# --------------------------------------------------------------------------- #
# the unified evaluation
# --------------------------------------------------------------------------- #
def score_state(
    state: State,
    my_index: int,
    cards: CardIndex,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    disabled: frozenset[str] = frozenset(),
) -> BoardEval:
    """Evaluate ``state`` from ``my_index``'s seat → a weighted :class:`BoardEval`.

    Each component is ``weight * (my_feature - opp_feature)`` (a mirrored board scores 0).
    A decided position short-circuits to :data:`WIN_SCORE` / :data:`LOSS_SCORE` on the
    prize clock. ``disabled`` names components to zero out (ablation) — they are dropped
    from ``total`` but still reported (value 0.0) so a trace shows what was excluded.

    Returns a :class:`BoardEval` with the ``total``, the per-component ``components``
    breakdown, and a compact ``reasons`` list (the non-zero components, largest |value|
    first) for audit/logging.
    """
    if state is None or getattr(state, "players", None) is None:
        return BoardEval(total=0.0)
    players = state.players
    if not (0 <= my_index < len(players)):
        return BoardEval(total=0.0)
    me = players[my_index]
    opp = players[1 - my_index]

    my_prizes = _prizes_left(me)
    opp_prizes = _prizes_left(opp)
    # Terminal on the win clock: no prizes left to take = that player has won.
    if my_prizes == 0:
        return BoardEval(total=WIN_SCORE, components={"prize": WIN_SCORE}, reasons=["win"])
    if opp_prizes == 0:
        return BoardEval(total=LOSS_SCORE, components={"prize": LOSS_SCORE}, reasons=["loss"])

    my_active = _active(me)
    opp_active = _active(opp)

    # Raw differential features (my side minus opponent), before weighting.
    features = {
        "prize": float(opp_prizes - my_prizes),
        "active_survival": (
            2.0 * ((1 if my_active is not None else 0) - (1 if opp_active is not None else 0))
            + (_hp_fraction(my_active) - _hp_fraction(opp_active))
        ),
        "ko_threat": float(_can_ko(me, opp, cards) - _can_ko(opp, me, cards)),
        "bench_dev": float(len(getattr(me, "bench", None) or []) - len(getattr(opp, "bench", None) or [])),
        "hand_value": float((getattr(me, "handCount", 0) or 0) - (getattr(opp, "handCount", 0) or 0)),
        "energy_tempo": float(_total_energy(me) - _total_energy(opp)),
        "retreat_capacity": float(_can_retreat(me, cards) - _can_retreat(opp, cards)),
    }

    components: dict[str, float] = {}
    total = 0.0
    for name in COMPONENTS:
        if name in disabled:
            components[name] = 0.0
            continue
        value = getattr(weights, name) * features[name]
        components[name] = value
        total += value

    reasons = [
        f"{name}{value:+.0f}"
        for name, value in sorted(components.items(), key=lambda kv: -abs(kv[1]))
        if value
    ]
    return BoardEval(total=total, components=components, reasons=reasons)
