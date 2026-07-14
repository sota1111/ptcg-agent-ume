"""PPOAgent — inference agent for the PPO-trained policy (SOT-1689).

A :class:`~agents.protocol.SafeAgent` whose policy is the small MLP learned by
:mod:`train.ppo` from self-play decision records (SOT-1688). Inference is **pure
Python + the JSON weight artifact** (``data/policy.json``): the heavy learning
dependencies (numpy) exist only on the training side, never at submission
runtime.

Decision path per selection:

1. featurize the raw observation (:func:`agents.features.featurize` — the same
   vector the policy was trained on, checked via ``feature_version``);
2. score the legal option indices with the policy head and sample from the
   masked softmax (:func:`agents.policy_net.sample_action`), which is legal by
   construction (distinct indices in range, count within min/max);
3. anything off the happy path — missing/corrupt/mismatched ``policy.json``,
   an empty option list, a feature-layout mismatch — makes :meth:`policy`
   *defer* (return ``None``), so the SafeAgent skeleton falls back to a legal
   random action. 違法出力0 therefore holds no matter what state the artifact
   is in, and the skeleton still revalidates every proposed action anyway.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from cg.api import Observation

from .features import FEATURE_DIM, FEATURE_VERSION, featurize
from .policy_net import load_policy, sample_action, validate_policy
from .protocol import SafeAgent

__all__ = ["PPOAgent", "DEFAULT_POLICY_PATH"]

#: The committed training artifact, resolved relative to the repo root so the
#: agent works whatever the caller's cwd is.
DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "policy.json"
)


class PPOAgent(SafeAgent):
    """Plays the PPO-trained policy; legal-random on anything it cannot score."""

    name = "ppo"
    version = "1"

    def __init__(
        self,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        *,
        policy: Optional[dict] = None,
        policy_path: Optional[str] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        time_budget_s: Optional[float] = None,
    ) -> None:
        """Args:
        policy: an in-memory policy dict (used by the training loop between
            updates). Takes precedence over ``policy_path``; rejected (agent
            defers everywhere) if it fails :func:`~agents.policy_net.validate_policy`.
        policy_path: JSON artifact to load (default :data:`DEFAULT_POLICY_PATH`).
        deterministic: argmax/top-k instead of softmax sampling.
        temperature: softmax temperature for sampling (ignored when deterministic).
        """
        super().__init__(seed=seed, rng=rng, time_budget_s=time_budget_s)
        if policy is not None:
            self._policy = policy if not validate_policy(policy) else None
        else:
            self._policy = load_policy(policy_path or DEFAULT_POLICY_PATH)
        if self._policy is not None and (
            self._policy["feature_version"] != FEATURE_VERSION
            or self._policy["feature_dim"] != FEATURE_DIM
        ):
            self._policy = None  # trained on another feature layout — don't misread it
        self._deterministic = deterministic
        self._temperature = temperature

    @property
    def policy_loaded(self) -> bool:
        """Whether a usable policy artifact is in place (else: legal-random)."""
        return self._policy is not None

    def policy(self, obs: dict, parsed: Observation, select) -> Optional[list[int]]:
        if self._policy is None or not select.option:
            return None
        return sample_action(
            self._policy,
            featurize(obs),
            len(select.option),
            int(select.minCount),
            int(select.maxCount),
            self._rng,
            deterministic=self._deterministic,
            temperature=self._temperature,
        )
