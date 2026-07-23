"""Pure, lightweight next-turn board-wipe risk scoring."""

from __future__ import annotations

from dataclasses import dataclass

from cg.api import AreaType, OptionType


@dataclass(frozen=True)
class SurvivalAssessment:
    threatened: bool
    active_hp: float
    reachable_damage: float
    bench_count: int
    switch_available: bool


def _active(player):
    active = getattr(player, "active", None) or []
    return active[0] if active and active[0] is not None else None


def assess_board_survival(parsed, select, cards) -> SurvivalAssessment:
    """Estimate whether the active can be knocked out on the opponent's turn."""
    state = getattr(parsed, "current", None)
    players = getattr(state, "players", None) if state is not None else None
    if not players:
        return SurvivalAssessment(False, 0.0, 0.0, 0, False)
    yi = int(state.yourIndex)
    me, opp = players[yi], players[1 - yi]
    mine, theirs = _active(me), _active(opp)
    hp = float(getattr(mine, "hp", 0) or 0)
    bench = [p for p in (getattr(me, "bench", None) or []) if p is not None]

    reachable = 0.0
    opp_card = cards.card(getattr(theirs, "id", None)) if theirs else None
    my_card = cards.card(getattr(mine, "id", None)) if mine else None
    for attack_id in getattr(opp_card, "attacks", None) or []:
        attack = cards.attack(attack_id)
        damage = float(getattr(attack, "damage", 0) or 0)
        if (
            damage > 0
            and opp_card is not None
            and my_card is not None
            and getattr(my_card, "weakness", None) == getattr(opp_card, "energyType", None)
        ):
            damage *= 2.0
        reachable = max(reachable, damage)

    switch_available = any(
        int(getattr(option, "type", -1)) == int(OptionType.RETREAT)
        for option in (getattr(select, "option", None) or [])
    )
    return SurvivalAssessment(
        threatened=bool(hp > 0 and reachable >= hp),
        active_hp=hp,
        reachable_damage=reachable,
        bench_count=len(bench),
        switch_available=switch_available,
    )


def survival_option_score(parsed, select, option_index: int, cards) -> float:
    """Return a bonus for an offered option that reduces wipe exposure."""
    assessment = assess_board_survival(parsed, select, cards)
    if not assessment.threatened:
        return 0.0
    options = getattr(select, "option", None) or []
    if not 0 <= option_index < len(options):
        return 0.0
    option = options[option_index]
    option_type = int(getattr(option, "type", -1))
    if option_type == int(OptionType.RETREAT) and assessment.bench_count:
        return 2.0
    if (
        option_type == int(OptionType.ATTACH)
        and int(getattr(option, "inPlayArea", -1)) == int(AreaType.BENCH)
        and assessment.bench_count
    ):
        return 0.75
    if option_type == int(OptionType.PLAY) and assessment.bench_count == 0:
        return 0.5
    return 0.0
