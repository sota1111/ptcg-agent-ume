"""Pure-Python policy network core for the PPO agent (SOT-1689).

The **runtime half** of the PPO learning loop: :mod:`train.ppo` learns a small
policy+value MLP with numpy and serialises the weights to ``data/policy.json``;
this module loads those weights and evaluates the policy with nothing but the
standard library, so the inference agent (:mod:`agents.ppo_agent`) carries **no
runtime pip dependency** — the submission constraint every ``agents/`` module
shares.

Design contract
---------------
* **Engine-free and dependency-free.** Like :mod:`agents.features`, this module
  never imports ``cg`` or numpy, so it (and its tests) run anywhere.
* **One math definition, two implementations.** The forward pass here must stay
  in lockstep with the numpy forward in :mod:`train.ppo` (same layout: tanh MLP,
  masked softmax over option slots). ``POLICY_SCHEMA`` is bumped on any layout
  change so a stale ``policy.json`` is rejected, not silently misread.
* **Never raises on bad artifacts.** :func:`load_policy` returns ``None`` for a
  missing/corrupt/mismatched file and :func:`validate_policy` reports what is
  wrong; the agent then simply defers to its SafeAgent legal-random fallback.
* **Legal by construction.** :func:`sample_action` only ever returns distinct
  option indices in ``[0, n_options)`` with a count clamped to the engine's
  ``[minCount, maxCount]`` — the same guarantee as
  :func:`agents.protocol.legal_random_action`.

Action model. The policy head scores a fixed number of *option slots*
(:data:`N_SLOTS`); a pending selection's legal options map onto the first
``min(n_options, N_SLOTS)`` slots and the rest are masked out. Sampling uses the
Gumbel top-k trick, so the *first* chosen index is an exact softmax sample —
which is precisely the quantity PPO trains on (the ``action_index`` of a
self-play record). Multi-count selections take the top-``k`` perturbed slots.
"""

from __future__ import annotations

import json
import math
import random
from typing import Optional

__all__ = [
    "POLICY_SCHEMA",
    "N_SLOTS",
    "validate_policy",
    "load_policy",
    "forward",
    "masked_log_softmax",
    "sample_action",
]

#: Schema tag of the ``data/policy.json`` artifact (bump on layout change).
POLICY_SCHEMA = "ume-policy-v1"

#: Number of option slots the policy head scores. Options beyond this (rare —
#: the featurizer's own option cap is 50) are unscored and only ever used as
#: random padding when a selection *requires* more picks than there are slots.
N_SLOTS = 64

# The weight matrices a policy artifact must carry: name -> (rows of, cols of),
# where a string refers to another artifact field ("hidden"/"feature_dim"/...).
_MATRIX_FIELDS = {"w1": ("hidden", "feature_dim"), "w2": ("n_slots", "hidden")}
_VECTOR_FIELDS = {"b1": "hidden", "b2": "n_slots", "vw": "hidden"}


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_policy(policy) -> list[str]:
    """Return the list of layout violations in a policy dict (empty = valid)."""
    if not isinstance(policy, dict):
        return ["policy is not a dict"]
    errors: list[str] = []
    if policy.get("schema") != POLICY_SCHEMA:
        errors.append(f"schema {policy.get('schema')!r} != {POLICY_SCHEMA!r}")
    for field in ("feature_version", "feature_dim", "hidden", "n_slots"):
        v = policy.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            errors.append(f"field {field!r} must be a positive int, got {v!r}")
    if errors:
        return errors  # sizes unusable — shape checks below would misfire

    def dim(name: str) -> int:
        return int(policy[name])

    for name, (rows_of, cols_of) in _MATRIX_FIELDS.items():
        m = policy.get(name)
        rows, cols = dim(rows_of), dim(cols_of)
        if (
            not isinstance(m, list)
            or len(m) != rows
            or not all(
                isinstance(r, list) and len(r) == cols and all(_is_num(x) for x in r)
                for r in m
            )
        ):
            errors.append(f"matrix {name!r} is not a finite {rows}x{cols} float matrix")
    for name, len_of in _VECTOR_FIELDS.items():
        v = policy.get(name)
        if not isinstance(v, list) or len(v) != dim(len_of) or not all(_is_num(x) for x in v):
            errors.append(f"vector {name!r} is not a finite float vector of length {dim(len_of)}")
    if not _is_num(policy.get("vb")):
        errors.append("field 'vb' must be a finite number")
    return errors


def load_policy(path: str) -> Optional[dict]:
    """Load and validate a policy artifact; ``None`` on any problem (never raises)."""
    try:
        with open(path, encoding="utf-8") as fh:
            policy = json.load(fh)
    except Exception:  # noqa: BLE001 - missing/corrupt file must never crash an agent
        return None
    return policy if not validate_policy(policy) else None


def forward(policy: dict, features: list[float]) -> tuple[list[float], float]:
    """Evaluate the MLP: ``features -> (slot logits, state value)``.

    ``h = tanh(w1 @ x + b1)``; ``logits = w2 @ h + b2``; ``value = vw . h + vb``.
    Must match the numpy forward in :mod:`train.ppo` exactly.
    """
    w1, b1, w2, b2 = policy["w1"], policy["b1"], policy["w2"], policy["b2"]
    h = [
        math.tanh(sum(wr[i] * features[i] for i in range(len(wr))) + b1[j])
        for j, wr in enumerate(w1)
    ]
    logits = [
        sum(wr[j] * h[j] for j in range(len(h))) + b2[k]
        for k, wr in enumerate(w2)
    ]
    value = sum(policy["vw"][j] * h[j] for j in range(len(h))) + policy["vb"]
    return logits, value


def masked_log_softmax(logits: list[float], n_options: int) -> list[float]:
    """Log-probabilities over the first ``min(n_options, len(logits))`` slots.

    Returns a list the length of the *scored* slots (masked slots are simply
    absent). Empty for ``n_options <= 0``.
    """
    n = min(max(0, n_options), len(logits))
    if n == 0:
        return []
    scored = logits[:n]
    top = max(scored)
    exps = [math.exp(z - top) for z in scored]
    log_total = math.log(sum(exps)) + top
    return [z - log_total for z in scored]


def _pick_count(n_options: int, min_count: int, max_count: int) -> int:
    """How many indices to select: prefer acting (k >= 1) when allowed.

    Clamped to the engine contract ``[minCount, min(maxCount, n)]`` (mirroring
    :func:`agents.protocol.legal_random_action`), so the result always passes
    the selection validator.
    """
    lo = max(0, int(min_count))
    hi = min(int(max_count), n_options)
    if hi < lo:  # defensive: engine guarantees lo <= hi
        hi = lo
    k = max(lo, min(1, hi))
    return min(k, n_options)


def sample_action(
    policy: dict,
    features: list[float],
    n_options: int,
    min_count: int,
    max_count: int,
    rng: random.Random,
    *,
    deterministic: bool = False,
    temperature: float = 1.0,
    logits: Optional[list[float]] = None,
) -> list[int]:
    """A guaranteed-legal action sampled from the policy distribution.

    Scores the first ``min(n_options, N_SLOTS)`` option indices and takes the
    top-``k`` by Gumbel-perturbed logit (``deterministic=True`` drops the
    noise), so the first index is an exact softmax sample of the trained
    distribution. If the selection requires more picks than there are scored
    slots, the remainder is padded with random *unscored* legal indices.
    ``logits`` short-circuits the forward pass when the caller already ran it
    for this exact ``features`` vector (e.g. for a criticality judgement).
    """
    k = _pick_count(n_options, min_count, max_count)
    if k == 0:
        return []
    if logits is None:
        logits, _ = forward(policy, features)
    scored = min(n_options, len(logits))
    temp = temperature if temperature > 0 else 1.0

    def perturbed(i: int) -> float:
        z = logits[i] / temp
        if deterministic:
            return z
        u = rng.random()
        # Gumbel(0,1) noise; clamp u away from {0,1} for a finite double log.
        return z - math.log(-math.log(min(max(u, 1e-12), 1.0 - 1e-12)))

    ranked = sorted(range(scored), key=perturbed, reverse=True)
    action = ranked[: min(k, scored)]
    if len(action) < k:  # selection demands more picks than scored slots
        action += rng.sample(range(scored, n_options), k - len(action))
    return action
