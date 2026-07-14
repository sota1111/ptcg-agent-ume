"""Observation featurization for learned agents (SOT-1688, PPO第1段).

Turns the engine's raw observation **dict** into a fixed-length ``list[float]``
feature vector — the state input for PPO training (SOT-1689) and for the
self-play data pipeline (:mod:`eval.selfplay`).

Design contract
---------------
* **Pure Python, engine-free.** This module never imports ``cg`` — it reads the
  raw dict shape a Kaggle submission's ``agent(obs_dict)`` receives. The enum
  cardinalities below are frozen copies of the ``cg.api`` docs, so the module
  (and its tests) work with no engine installed (e.g. CI).
* **Fixed dimension.** ``featurize`` always returns exactly :data:`FEATURE_DIM`
  floats, whatever the input: missing keys, ``None`` sub-objects, or malformed
  values fall back to ``0.0`` for their slots. PPO tensors never change shape.
* **Unknown-safe.** The enums "may be appended during the competition"
  (``cg/api.py``), so every one-hot block carries a trailing *unknown* bucket:
  a value outside the known range sets that bucket instead of raising.
* **Card-identity free.** No feature branches on an individual card/attack id
  (the SOT-1625 invariant); only counts, HP and engine-provided flags are used.
  Card embeddings are a later, separate concern.
* **Perspective.** The vector is always from the viewpoint of the player the
  engine is asking to select (``current.yourIndex``): "me" is that player,
  "opp" the other. A record is therefore self-describing for either seat.

Normalisation uses fixed constants (deck size, prize count, HP cap, ...) so the
mapping is a pure function of the observation — no running statistics.
"""

from __future__ import annotations

import math

__all__ = ["FEATURE_VERSION", "FEATURE_DIM", "featurize", "feature_names"]

# Bump when the vector layout changes — stamped into every self-play record so
# training data from different layouts is never silently mixed.
FEATURE_VERSION = 1

# Frozen enum cardinalities (from cg/api.py at FEATURE_VERSION=1). Values beyond
# these ranges are engine additions newer than this layout → the unknown bucket.
_N_SELECT_TYPES = 11      # SelectType MAIN..SPECIAL_CONDITION (0..10)
_N_SELECT_CONTEXTS = 49   # SelectContext MAIN..RECOVER_SPECIAL_CONDITION (0..48)
_N_OPTION_TYPES = 17      # OptionType NUMBER..SPECIAL_CONDITION (0..16)

# Fixed normalisation constants (game invariants, not tuned parameters).
_DECK_SIZE = 60.0
_PRIZES = 6.0
_HP_CAP = 400.0           # generous max-HP cap (Mega ex era)
_BENCH_CAP = 8.0
_ENERGY_CAP = 10.0
_TURN_CAP = 50.0
_ACTIONS_CAP = 30.0
_COUNT_CAP = 10.0         # select min/max count
_OPTIONS_CAP = 50.0       # legal-option count


def _num(value, scale: float = 1.0, cap: float = 1.0) -> float:
    """``value / scale`` clamped to ``[0, cap]``; 0.0 for missing/malformed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(cap, float(value) / scale))


def _flag(value) -> float:
    """1.0 for a truthy engine flag, else 0.0."""
    return 1.0 if value is True else 0.0


def _one_hot(value, n: int) -> list[float]:
    """Length ``n + 1`` one-hot; the trailing slot is the *unknown* bucket."""
    vec = [0.0] * (n + 1)
    if isinstance(value, bool) or not isinstance(value, int):
        vec[n] = 1.0
    elif 0 <= value < n:
        vec[value] = 1.0
    else:
        vec[n] = 1.0
    return vec


def _pokemon_features(mon) -> list[float]:
    """Per-Pokémon block (8 slots); all zeros for ``None``/facedown/malformed."""
    if not isinstance(mon, dict):
        return [0.0] * 8
    hp = _num(mon.get("hp"), _HP_CAP)
    max_hp = _num(mon.get("maxHp"), _HP_CAP)
    ratio = (hp / max_hp) if max_hp > 0 else 0.0
    energies = mon.get("energies") if isinstance(mon.get("energies"), list) else []
    energy_cards = mon.get("energyCards") if isinstance(mon.get("energyCards"), list) else []
    tools = mon.get("tools") if isinstance(mon.get("tools"), list) else []
    pre = mon.get("preEvolution") if isinstance(mon.get("preEvolution"), list) else []
    return [
        1.0,                                   # exists
        hp,
        max_hp,
        min(1.0, ratio),
        _num(len(energies), _ENERGY_CAP),
        _num(len(energy_cards), _ENERGY_CAP),
        _num(len(tools), 2.0),
        _num(len(pre), 3.0),                   # evolution depth proxy
    ]


_POKEMON_SLOTS = [
    "exists", "hp", "max_hp", "hp_ratio",
    "energies", "energy_cards", "tools", "pre_evolution",
]


def _player_features(player) -> list[float]:
    """Per-player block (22 slots); all zeros when the state is absent."""
    if not isinstance(player, dict):
        return [0.0] * 22

    active_list = player.get("active") if isinstance(player.get("active"), list) else []
    active = active_list[0] if active_list else None
    bench = [m for m in (player.get("bench") or []) if isinstance(m, dict)] \
        if isinstance(player.get("bench"), list) else []

    bench_hp = sum(_num(m.get("hp"), _HP_CAP) for m in bench)
    bench_max_hp = sum(_num(m.get("maxHp"), _HP_CAP) for m in bench)
    bench_energy = sum(
        len(m.get("energies")) if isinstance(m.get("energies"), list) else 0
        for m in bench
    )
    prize = player.get("prize") if isinstance(player.get("prize"), list) else []
    discard = player.get("discard") if isinstance(player.get("discard"), list) else []

    return _pokemon_features(active) + [
        _num(len(bench), _BENCH_CAP),
        _num(player.get("benchMax"), _BENCH_CAP),
        _num(bench_hp, _BENCH_CAP),            # sum of per-mon [0,1] HP shares
        min(1.0, (bench_hp / bench_max_hp) if bench_max_hp > 0 else 0.0),
        _num(bench_energy, _ENERGY_CAP * 2),
        _num(player.get("deckCount"), _DECK_SIZE),
        _num(player.get("handCount"), _DECK_SIZE),
        _num(len(discard), _DECK_SIZE),
        _num(len(prize), _PRIZES),
        _flag(player.get("poisoned")),
        _flag(player.get("burned")),
        _flag(player.get("asleep")),
        _flag(player.get("paralyzed")),
        _flag(player.get("confused")),
    ]


_PLAYER_SLOTS = [f"active_{s}" for s in _POKEMON_SLOTS] + [
    "bench_count", "bench_max", "bench_hp", "bench_hp_ratio", "bench_energy",
    "deck_count", "hand_count", "discard_count", "prize_count",
    "poisoned", "burned", "asleep", "paralyzed", "confused",
]


def _select_features(select) -> list[float]:
    """Selection block: type/context one-hots, option-type histogram, counts."""
    if not isinstance(select, dict):
        select = {}
    options = select.get("option") if isinstance(select.get("option"), list) else []

    histogram = [0.0] * (_N_OPTION_TYPES + 1)
    for opt in options:
        ot = opt.get("type") if isinstance(opt, dict) else None
        if isinstance(ot, int) and not isinstance(ot, bool) and 0 <= ot < _N_OPTION_TYPES:
            histogram[ot] += 1.0
        else:
            histogram[_N_OPTION_TYPES] += 1.0
    if options:
        histogram = [c / len(options) for c in histogram]

    return (
        _one_hot(select.get("type"), _N_SELECT_TYPES)
        + _one_hot(select.get("context"), _N_SELECT_CONTEXTS)
        + histogram
        + [
            _num(select.get("minCount"), _COUNT_CAP),
            _num(select.get("maxCount"), _COUNT_CAP),
            _num(len(options), _OPTIONS_CAP),
            _num(select.get("remainDamageCounter"), _COUNT_CAP),
            _num(select.get("remainEnergyCost"), _ENERGY_CAP),
        ]
    )


def featurize(obs: dict) -> list[float]:
    """Map a raw observation dict to exactly :data:`FEATURE_DIM` floats.

    Never raises on missing/None/malformed sub-objects — the affected slots are
    simply 0.0 — so any observation the engine (or a test) produces yields a
    valid, fixed-shape vector.
    """
    if not isinstance(obs, dict):
        obs = {}
    current = obs.get("current") if isinstance(obs.get("current"), dict) else {}
    me = current.get("yourIndex")
    me = me if me in (0, 1) else 0
    players = current.get("players") if isinstance(current.get("players"), list) else []

    def player(idx: int):
        return players[idx] if 0 <= idx < len(players) else None

    first = current.get("firstPlayer")
    stadium = current.get("stadium") if isinstance(current.get("stadium"), list) else []

    global_block = [
        _num(current.get("turn"), _TURN_CAP),
        _num(current.get("turnActionCount"), _ACTIONS_CAP),
        1.0 if first == me else 0.0,
        1.0 if first in (0, 1) and first != me else 0.0,
        _flag(current.get("supporterPlayed")),
        _flag(current.get("stadiumPlayed")),
        _flag(current.get("energyAttached")),
        _flag(current.get("retreated")),
        1.0 if stadium else 0.0,
    ]

    return (
        global_block
        + _player_features(player(me))
        + _player_features(player(1 - me))
        + _select_features(obs.get("select"))
    )


def feature_names() -> list[str]:
    """Stable, human-readable name per slot (``len == FEATURE_DIM``)."""
    names = [
        "turn", "turn_action_count", "i_am_first", "opp_is_first",
        "supporter_played", "stadium_played", "energy_attached", "retreated",
        "stadium_in_play",
    ]
    for side in ("me", "opp"):
        names += [f"{side}_{s}" for s in _PLAYER_SLOTS]
    names += [f"select_type_{i}" for i in range(_N_SELECT_TYPES)]
    names += ["select_type_unknown"]
    names += [f"select_context_{i}" for i in range(_N_SELECT_CONTEXTS)]
    names += ["select_context_unknown"]
    names += [f"option_type_frac_{i}" for i in range(_N_OPTION_TYPES)]
    names += ["option_type_frac_unknown"]
    names += [
        "select_min_count", "select_max_count", "select_n_options",
        "remain_damage_counter", "remain_energy_cost",
    ]
    return names


#: The fixed vector length. Derived from the layout so it can never drift from
#: ``feature_names()`` / ``featurize()``.
FEATURE_DIM = len(feature_names())
