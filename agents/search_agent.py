"""SearchAgent — one-ply lookahead agent reusing the rule score (SOT-1650, R5).

The B-line experiment of the parent plan (SOT-1631 案B/2): instead of choosing a
MAIN sub-action purely from the static option score, this agent uses the engine's
official one-ply lookahead API (``search_begin`` / ``search_step`` / ``search_end`` /
``search_release`` — see :mod:`cg.api`) to actually *play* each candidate MAIN option
on a search copy of the position, evaluate the resulting board from our perspective,
and take the best. Everything else is inherited unchanged from
:class:`~agents.rule_agent.RuleAgent`.

Design contract
---------------
* **Separate module, RuleAgent unchanged (受け入れ条件③).** :class:`SearchAgent`
  *subclasses* :class:`RuleAgent`; it only overrides the MAIN-turn tactic. Every
  non-MAIN selection (setup / forced contexts) and the whole safety skeleton come
  straight from :class:`RuleAgent` / :class:`~agents.protocol.SafeAgent`, so the rule
  agent's code is never touched and stays independently the champion baseline.
* **One-ply search on MAIN, else the rule policy.** For a single-select MAIN decision
  every legal option is a candidate; each is stepped on a fresh search copy and the
  resulting :class:`~cg.api.Observation` is scored by :func:`evaluate_state` (a pure
  perspective value). The best-scoring option is chosen. A MAIN selection that is
  multi-select, degenerate, or that the search cannot handle defers to the inherited
  rule policy (:meth:`RuleAgent._main_policy`).
* **Fail-closed to the rule policy (失敗時は即RuleAgentへ, 探索リーク0).** Any failure
  whatsoever — a rejected hidden-info prediction, an unknown/newly-appended enum, a
  raising evaluator, a broken engine copy, or the per-decision time budget expiring
  before a single candidate is scored — makes :meth:`_main_policy` return the rule
  policy's move instead. Every search session is torn down (``search_release`` +
  ``search_end``) in a ``finally`` so no engine search state is ever leaked, and the
  :class:`SafeAgent` skeleton still validates the final action, so the agent can never
  crash the match or emit an illegal move.

The evaluation reuses only **merged** pieces: prize/HP differential from the resulting
observation, plus an offensive term computed with the R2 attack estimate
(:func:`agents.rule_scoring.estimate_attack`, weakness ×2). It deliberately does *not*
depend on the R4 unified ``score(state)`` (a separate, not-yet-merged round), so this
module is self-contained and mergeable on its own; the evaluator is pluggable and can
be swapped for the R4 board evaluation later without touching the search machinery.

``search_begin`` needs a full prediction of every hidden zone (both decks, prizes, the
opponent's hand / face-down Active) that is unknown in live play. A best-effort
:class:`UniformDeckPredictor` samples it from our own 60-card deck list; a wrong guess
just makes ``search_begin`` reject the candidate, and the rule-policy fallback holds.
"""

from __future__ import annotations

import collections
import random
import time
from functools import lru_cache
from typing import Callable, Optional, Sequence

from cg.api import (
    Observation,
    OptionType,
    SelectContext,
    SelectType,
)

from .rule_agent import RuleAgent
from .rule_scoring import CardIndex, estimate_attack

__all__ = [
    "Evaluator",
    "evaluate_state",
    "UniformDeckPredictor",
    "SearchAgent",
]

# The MAIN turn selection SearchAgent overrides with a one-ply lookahead.
_MAIN_KEY = (int(SelectType.MAIN), int(SelectContext.MAIN))

# An evaluator scores a resulting :class:`~cg.api.Observation` from the acting player's
# perspective (higher = better for ``your_index``). Pluggable: the default is
# :func:`evaluate_state`; any callable with this signature can replace it.
Evaluator = Callable[[Observation, int, CardIndex], float]

# evaluate_state weights: a terminal win dominates everything, then prize progress
# (fewer of our own prize cards remaining ⇒ we have taken more prizes ⇒ winning), then
# a damage/KO offensive term, then board HP as the finest tie-break. The offensive band
# sits below one prize so the search never trades a prize for chip damage.
_WIN_SCORE = 1_000_000.0
_PRIZE_WEIGHT = 1_000.0
_KO_BONUS = 500.0
_DMG_WEIGHT = 2.0
_HP_WEIGHT = 1.0


@lru_cache(maxsize=1)
def _card_index() -> CardIndex:
    """Static card/attack reference data, loaded from the engine once per process.

    Mirrors :func:`agents.rule_agent._card_index` but kept local so this module stays
    self-contained; the ``lru_cache`` means the single engine read happens once.
    """
    from cg.api import all_attack, all_card_data

    return CardIndex.from_engine(all_card_data(), all_attack())


def _board_hp(player) -> int:
    """Total current HP of a player's in-play Pokémon (Active + Bench)."""
    total = 0
    for group in ((getattr(player, "active", None) or []), (getattr(player, "bench", None) or [])):
        for pk in group:
            if pk is not None:
                total += getattr(pk, "hp", 0) or 0
    return total


def _active(player):
    active = getattr(player, "active", None) or []
    return active[0] if active and active[0] is not None else None


def _best_attack(attacker, defender, cards: CardIndex) -> tuple[float, bool]:
    """Best single-attack (damage, can_ko) ``attacker``'s Active threatens on ``defender``.

    Reuses the merged R2 estimate (:func:`agents.rule_scoring.estimate_attack`,
    weakness ×2). Any missing piece — no Active on either side, an unknown card id, a
    Pokémon with no attacks — yields ``(0.0, False)`` rather than raising.
    """
    if attacker is None or defender is None:
        return 0.0, False
    atk_card = cards.card(getattr(attacker, "id", None))
    def_card = cards.card(getattr(defender, "id", None))
    if atk_card is None or not atk_card.attacks:
        return 0.0, False
    best_dmg = 0.0
    can_ko = False
    def_hp = getattr(defender, "hp", None)
    for aid in atk_card.attacks:
        dmg, is_ko = estimate_attack(cards.attack(aid), atk_card, def_hp, def_card)
        if dmg > best_dmg:
            best_dmg = dmg
        if is_ko:
            can_ko = True
    return best_dmg, can_ko


def evaluate_state(observation: Observation, your_index: int, cards: CardIndex) -> float:
    """Perspective score for a resulting position (higher = better for ``your_index``).

    A terminal win/loss/draw is decisive. Otherwise: the prize-card differential (how
    much closer we are to taking all our prizes than the opponent) dominates, then an
    offensive damage/KO differential computed with weakness (×2), then board HP as a
    tie-break. Guarded throughout — a malformed observation, unknown enum, or missing
    field degrades to the neutral (0) contribution and never raises, so the agent's
    rule-policy fallback is never needed merely to evaluate.
    """
    current = getattr(observation, "current", None)
    if current is None:
        return 0.0
    result = getattr(current, "result", -1)
    if result in (0, 1):
        return _WIN_SCORE if result == your_index else -_WIN_SCORE
    if result == 2:  # draw
        return 0.0

    try:
        players = current.players
        me = players[your_index]
        opp = players[1 - your_index]
    except (IndexError, TypeError, AttributeError):
        return 0.0

    # Prize progress: fewer of our prizes remaining, or more of theirs, both mean we lead.
    prize_term = (len(opp.prize) - len(me.prize)) * _PRIZE_WEIGHT

    offense = 0.0
    try:
        my_active, opp_active = _active(me), _active(opp)
        if my_active is not None or opp_active is not None:
            my_dmg, my_ko = _best_attack(my_active, opp_active, cards)
            opp_dmg, opp_ko = _best_attack(opp_active, my_active, cards)
            offense = _DMG_WEIGHT * (my_dmg - opp_dmg)
            if my_ko:
                offense += _KO_BONUS
            if opp_ko:
                offense -= _KO_BONUS
    except Exception:  # noqa: BLE001 - evaluation must never raise
        offense = 0.0

    hp_term = (_board_hp(me) - _board_hp(opp)) * _HP_WEIGHT
    return prize_term + offense + hp_term


def _visible_card_ids(player, *, include_hand: bool) -> list[int]:
    """Card IDs currently visible on the board for one player.

    Pokémon in play (Active/Bench) with their attached energy/tool/pre-evolution cards,
    the discard pile, revealed prize cards, and — only where the hand is visible (our
    own side) — the hand. These are removed from the deck multiset to form the hidden
    pool. Fully guarded (an absent zone contributes nothing).
    """
    ids: list[int] = []

    def add_pokemon(pk) -> None:
        if pk is None:
            return
        pid = getattr(pk, "id", None)
        if pid is not None:
            ids.append(pid)
        for group in (getattr(pk, "energyCards", None), getattr(pk, "tools", None), getattr(pk, "preEvolution", None)):
            for card in group or []:
                cid = getattr(card, "id", None)
                if cid is not None:
                    ids.append(cid)

    for pk in getattr(player, "active", None) or []:
        add_pokemon(pk)
    for pk in getattr(player, "bench", None) or []:
        add_pokemon(pk)
    for card in getattr(player, "discard", None) or []:
        cid = getattr(card, "id", None)
        if cid is not None:
            ids.append(cid)
    for card in getattr(player, "prize", None) or []:
        cid = getattr(card, "id", None) if card is not None else None
        if cid is not None:  # revealed prize only
            ids.append(cid)
    if include_hand:
        for card in getattr(player, "hand", None) or []:
            cid = getattr(card, "id", None)
            if cid is not None:
                ids.append(cid)
    return ids


class UniformDeckPredictor:
    """Sample the hidden zones uniformly from a deck list minus visible cards.

    ``deck_ids`` is the assumed 60-card composition, reused for both players (R5 has no
    opponent-deck model yet). Seeded for reproducibility. A wrong guess just makes
    ``search_begin`` reject the candidate; the rule-policy fallback still holds. Swap
    for a deck-tracking predictor later without touching the agent.
    """

    def __init__(self, deck_ids: Sequence[int], rng: random.Random) -> None:
        self.deck_ids = list(deck_ids)
        self.rng = rng

    def _pool(self, player, *, include_hand: bool) -> list[int]:
        remaining = collections.Counter(self.deck_ids)
        for cid in _visible_card_ids(player, include_hand=include_hand):
            remaining[cid] -= 1
        pool: list[int] = []
        for cid, n in remaining.items():
            if n > 0:
                pool.extend([cid] * n)
        self.rng.shuffle(pool)
        return pool

    def _take(self, pool: list[int], n: int) -> list[int]:
        """Pop ``n`` cards off ``pool``; top up from the deck list if short."""
        if n <= 0:
            return []
        out = pool[:n]
        del pool[:n]
        i = 0
        while len(out) < n and self.deck_ids:
            out.append(self.deck_ids[i % len(self.deck_ids)])
            i += 1
        return out

    def predict(self, obs: Observation, your_index: int) -> tuple:
        """Return the six hidden-card lists in ``search_begin`` argument order."""
        state = obs.current
        players = state.players
        me = players[your_index]
        opp = players[1 - your_index]

        # Our side: the hand is visible, so subtract it; deck + prize are hidden.
        my_pool = self._pool(me, include_hand=True)
        your_deck = self._take(my_pool, me.deckCount)
        your_prize = self._take(my_pool, len(me.prize))

        # Opponent side: the hand is hidden too, so don't subtract it.
        opp_pool = self._pool(opp, include_hand=False)
        opponent_deck = self._take(opp_pool, opp.deckCount)
        opponent_prize = self._take(opp_pool, len(opp.prize))
        opponent_hand = self._take(opp_pool, opp.handCount)
        active = opp.active or []
        active_facedown = len(active) > 0 and active[0] is None
        opponent_active = self._take(opp_pool, 1) if active_facedown else []

        return (
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
        )


class SearchAgent(RuleAgent):
    """One-ply search agent: lookahead on MAIN, the rule policy everywhere else.

    Subclass of :class:`RuleAgent` — only the MAIN-turn tactic is overridden with a
    one-ply search; all other contexts and the safety skeleton are inherited unchanged.
    On any search failure (rejected prediction, unknown enum, raising evaluator, budget
    expiry) the MAIN tactic falls back to the inherited rule policy, so the agent is
    never worse-behaved than :class:`RuleAgent` and never leaks a search session.
    """

    name = "search"
    version = "1"

    def __init__(
        self,
        *args,
        deck_path: str = "deck.csv",
        search_budget_s: float = 0.1,
        max_candidates: int = 12,
        evaluate: Optional[Evaluator] = None,
        manual_coin: bool = False,
        **kwargs,
    ) -> None:
        """Args (beyond :class:`SafeAgent`'s ``seed`` / ``rng`` / ``time_budget_s``):
        deck_path: deck CSV used as the hidden-info composition prior for both players.
        search_budget_s: soft wall-clock budget per MAIN decision; candidates are
            searched in order until spent, then the best scored so far is returned.
        max_candidates: cap on candidate options searched per decision (bounds cost).
        evaluate: position evaluator (defaults to :func:`evaluate_state`).
        manual_coin: forwarded to ``search_begin`` (fix coin flips during lookahead).
        """
        super().__init__(*args, **kwargs)
        self.deck_path = deck_path
        self.search_budget_s = search_budget_s
        self.max_candidates = max_candidates
        self.evaluate = evaluate or evaluate_state
        self.manual_coin = manual_coin
        self._deck_ids: Optional[list[int]] = None
        # Per-agent safety counters (leak/crash accounting for the acceptance report).
        self.search_stats = {"attempts": 0, "chosen": 0, "fallbacks": 0, "leaks": 0}

    # -- MAIN tactic: one-ply search, else the inherited rule policy ---------
    def _main_policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        """Override the R2 MAIN tactic with a one-ply search; fall back to the rule.

        Attempts the lookahead only for a single-select MAIN with a live board. Returns
        the best searched option, or — on any failure — the inherited rule policy's
        move (:meth:`RuleAgent._main_policy`), so the safety contract is preserved.
        """
        best = None
        try:
            best = self._search_main(parsed, select)
        except Exception:  # noqa: BLE001 - the search must never crash the agent
            best = None
        if best is not None:
            self.search_stats["chosen"] += 1
            return [best]
        self.search_stats["fallbacks"] += 1
        # Fail-closed: use the merged rule MAIN policy (KO > energy > evolve > …).
        return super()._main_policy(obs, parsed, select)

    def _search_main(self, parsed: Observation, select) -> Optional[int]:
        """One-ply search over MAIN options → best option index, or ``None`` to defer.

        Only handles a single-select decision (``minCount<=1<=maxCount``) on a live
        board. Predicts hidden info once (so every candidate sees the same sampled
        world), steps each candidate on a fresh search copy, and scores the result.
        Returns ``None`` (defer to the rule policy) when no candidate can be scored.
        """
        if parsed is None or parsed.current is None or select is None:
            return None
        if not (select.minCount <= 1 <= select.maxCount):
            return None  # multi/zero-select MAIN → rule policy
        options = select.option or []
        if not options:
            return None
        # Only search "real" sub-actions; never spend the whole turn searching END.
        candidates = [
            i for i, opt in enumerate(options) if int(opt.type) != int(OptionType.END)
        ][: self.max_candidates]
        if not candidates:
            return None

        your_index = parsed.current.yourIndex
        cards = _card_index()
        predictor = UniformDeckPredictor(self._deck(), self._rng)
        hidden = predictor.predict(parsed, your_index)

        self.search_stats["attempts"] += 1
        deadline = time.perf_counter() + max(0.0, self.search_budget_s)
        best_idx: Optional[int] = None
        best_score = float("-inf")
        scored_any = False
        for idx in candidates:
            if scored_any and time.perf_counter() >= deadline:
                break  # budget spent and we already have something to return
            score = self._score_candidate(parsed, hidden, idx, your_index, cards)
            if score is None:
                continue
            scored_any = True
            if score > best_score:
                best_score, best_idx = score, idx
        return best_idx

    def _score_candidate(
        self, parsed: Observation, hidden: tuple, idx: int, your_index: int, cards: CardIndex
    ) -> Optional[float]:
        """Step option ``idx`` on a fresh search copy and score the resulting position.

        The search session is ALWAYS torn down (``search_release`` + ``search_end`` in
        a ``finally``) even if reconstruction, the step, or the evaluator raises — so no
        engine search state ever leaks. Returns ``None`` on any failure (candidate
        skipped). A teardown failure is recorded as a leak for the safety report.
        """
        from cg.api import search_begin, search_end, search_release, search_step

        search_id: Optional[int] = None
        started = False
        try:
            root = search_begin(parsed, *hidden, manual_coin=self.manual_coin)
            started = True
            search_id = root.searchId
            state = search_step(search_id, [idx])
            return float(self.evaluate(state.observation, your_index, cards))
        except Exception:  # noqa: BLE001 - a rejected/failed candidate is just skipped
            return None
        finally:
            if started:
                try:
                    if search_id is not None:
                        search_release(search_id)
                    search_end()
                except Exception:  # noqa: BLE001
                    self.search_stats["leaks"] += 1

    def _deck(self) -> list[int]:
        """Load and cache the deck-id list used as the hidden-info prior."""
        if self._deck_ids is None:
            try:
                with open(self.deck_path) as f:
                    self._deck_ids = [int(x) for x in f.read().split() if x.strip()][:60]
            except Exception:  # noqa: BLE001 - missing deck ⇒ empty prior, search defers
                self._deck_ids = []
        return list(self._deck_ids)
