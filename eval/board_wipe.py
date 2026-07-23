"""Board-wipe KPI collection for candidate/champion arena runs (SOT-1885)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoardWipeStats:
    losses: int = 0
    board_wipes: int = 0
    risk_exposures: int = 0
    avoided: int = 0

    def report(self) -> dict:
        return {
            "losses": self.losses,
            "board_wipes": self.board_wipes,
            "board_wipe_rate_in_losses": (
                self.board_wipes / self.losses if self.losses else 0.0
            ),
            "risk_exposures": self.risk_exposures,
            "board_wipe_avoidance_rate": (
                self.avoided / self.risk_exposures if self.risk_exposures else 1.0
            ),
        }


class BoardWipeTrackingAgent:
    """Transparent wrapper classifying losses from the last observed own board."""

    def __init__(self, agent, stats: BoardWipeStats):
        self.agent = agent
        self.stats = stats
        self._seat = None
        self._last_board_size = 0
        self._last_active_hp = 0
        self._exposed = False
        self.name = getattr(agent, "name", type(agent).__name__)
        self.version = getattr(agent, "version", "0")

    def act(self, obs):
        current = obs.get("current") or {}
        players = current.get("players") or []
        yi = current.get("yourIndex", 0)
        if 0 <= yi < len(players):
            player = players[yi] or {}
            active = [p for p in (player.get("active") or []) if p]
            bench = [p for p in (player.get("bench") or []) if p]
            self._last_board_size = len(active) + len(bench)
            self._last_active_hp = int(active[0].get("hp", 0)) if active else 0
            if self._last_board_size == 1 and active:
                max_hp = int(active[0].get("maxHp", 0) or 0)
                if (
                    max_hp > 0
                    and self._last_active_hp <= max_hp / 2
                    and not self._exposed
                ):
                    self._exposed = True
                    self.stats.risk_exposures += 1
        return self.agent.act(obs)

    def on_match_start(self, seat):
        self._seat = seat
        self._last_board_size = 0
        self._last_active_hp = 0
        self._exposed = False
        hook = getattr(self.agent, "on_match_start", None)
        if callable(hook):
            hook(seat)

    def on_match_end(self, result):
        lost = (
            self._seat is not None
            and result.winner is not None
            and result.winner != self._seat
        )
        if lost:
            self.stats.losses += 1
            # The engine terminates before the loser receives another observation,
            # so the last-active KO is represented by "one Pokémon left" rather
            # than an observed zero-HP state.
            wiped = self._last_board_size <= 1
            if wiped:
                self.stats.board_wipes += 1
            elif self._exposed:
                self.stats.avoided += 1
        hook = getattr(self.agent, "on_match_end", None)
        if callable(hook):
            hook(result)
