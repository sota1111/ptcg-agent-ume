"""PPO training-side acceptance tests (SOT-1689).

Pin the trainer's contract without the engine: numpy-only (importorskip), using
synthetic ``ume-selfplay-v1`` records. Covered:

* the numpy forward is the same math as the pure-Python inference forward
  (:mod:`agents.policy_net`) — the lockstep the artifact depends on;
* GAE on a hand-computed trajectory;
* record filtering keeps only PPO-trainable decisions and tallies the drops;
* an offline training run (``--iters 0 --bootstrap-data``) **generates a valid
  ``policy.json`` from self-play data**, changes the weights, and can be
  **re-trained** from its own artifact via ``--init-from`` (再学習可能).
"""
from __future__ import annotations

import json
import math
import random

import pytest

np = pytest.importorskip("numpy", reason="training-only dependency (requirements.txt)")

from agents.features import FEATURE_DIM, FEATURE_VERSION  # noqa: E402
from agents.policy_net import forward, load_policy, validate_policy  # noqa: E402
from train.ppo import (  # noqa: E402
    _shaped_rewards,
    build_batch,
    compute_gae,
    forward_np,
    init_params,
    load_records,
    main,
    masked_log_softmax_np,
    params_to_policy,
    policy_to_params,
)


def _record(game: int, decision: int, *, player: int = 0, reward: float = 1.0,
            action_index: int = 0, n_options: int = 3, seed: int = 0) -> dict:
    rng = random.Random(seed * 1000 + game * 100 + decision)
    return {
        "schema": "ume-selfplay-v1", "feature_version": FEATURE_VERSION,
        "game": game, "decision": decision, "player": player, "agent": "ppo",
        "features": [rng.random() for _ in range(FEATURE_DIM)],
        "action": [action_index], "action_index": action_index,
        "n_options": n_options, "min_count": 1, "max_count": 1,
        "select_type": 0, "select_context": 0, "reward": reward,
        "result": "win" if reward > 0 else "loss", "winner": 0,
        "reason": "normal", "steps": 40,
    }


def _write_records(path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_numpy_forward_matches_pure_python_forward():
    params = init_params(hidden=8, seed=42)
    policy = params_to_policy(params)
    features = [random.Random(1).random() for _ in range(FEATURE_DIM)]
    logits_py, value_py = forward(policy, features)
    _, logits_np, values_np = forward_np(params, np.asarray([features]))
    assert np.allclose(logits_np[0], logits_py, atol=1e-9)
    assert math.isclose(float(values_np[0]), value_py, abs_tol=1e-9)


def test_masked_log_softmax_np_masks_illegal_slots():
    logits = np.zeros((2, 5))
    logp = masked_log_softmax_np(logits, np.asarray([2, 5]))
    assert np.allclose(np.exp(logp[0][:2]), [0.5, 0.5])
    assert logp[0][2] < -1e8  # masked
    assert np.allclose(np.exp(logp[1]).sum(), 1.0)


def test_compute_gae_hand_case():
    # Two steps, terminal reward 1, gamma=lam=1: adv_t = G_t - V_t.
    adv, ret = compute_gae(np.asarray([0.0, 1.0]), np.asarray([0.2, 0.5]), 1.0, 1.0)
    assert np.allclose(ret, [1.0, 1.0])
    assert np.allclose(adv, [0.8, 0.5])


def test_load_records_filters_and_tallies(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        _record(0, 0),
        _record(0, 1, n_options=1),                 # trivial: no real choice
        _record(0, 2, action_index=-1),             # empty action
        _record(0, 3, action_index=200),            # beyond the slot head
        {**_record(0, 4), "schema": "other"},       # wrong schema
    ]
    _write_records(path, records)
    kept, tally = load_records([str(path)])
    assert len(kept) == 1
    assert tally["read"] == 5
    assert tally["skipped"] == {
        "trivial_choice": 1, "empty_action": 1, "beyond_slots": 1, "schema_mismatch": 1,
    }


def test_build_batch_groups_trajectories_and_normalises(tmp_path):
    path = tmp_path / "records.jsonl"
    records = (
        [_record(0, d, player=0, reward=1.0, seed=1) for d in range(4)]
        + [_record(0, d, player=1, reward=-1.0, seed=2) for d in range(3)]
        + [_record(1, d, player=0, reward=-1.0, seed=3) for d in range(2)]
    )
    _write_records(path, records)
    kept, _ = load_records([str(path)])
    params = init_params(hidden=8, seed=0)
    batch = build_batch(kept, params, gamma=0.99, lam=0.95)
    assert batch["n_trajectories"] == 3
    assert len(batch["x"]) == 9
    assert abs(float(batch["advantages"].mean())) < 1e-9  # normalised
    assert np.isfinite(batch["old_logp"]).all()


def test_shaping_off_is_terminal_only():
    # No shaping args -> reward is 0 everywhere but the terminal ±1 (SOT-1689).
    recs = [_record(0, d, reward=-1.0) for d in range(3)]
    r = _shaped_rewards(recs, [0, 1, 2], gamma=0.99, prize_shaping=0.0, loss_shaping=0.0)
    assert np.allclose(r, [0.0, 0.0, -1.0])


def test_prize_shaping_is_potential_based_and_telescopes():
    # A prize-taking win: own prizes 6->5->4, opp 6->6->5, coef 0.6, gamma 1.
    recs = [
        {**_record(0, 0, reward=1.0), "own_prizes": 6, "opp_prizes": 6},
        {**_record(0, 1, reward=1.0), "own_prizes": 5, "opp_prizes": 6},
        {**_record(0, 2, reward=1.0), "own_prizes": 4, "opp_prizes": 5},
    ]
    r = _shaped_rewards(recs, [0, 1, 2], gamma=1.0, prize_shaping=0.6, loss_shaping=0.0)
    # phi = 0.6*(opp-own)/6 = [0, 0.1, 0.1]; F_t = phi_{t+1}-phi_t (terminal phi=0).
    assert np.allclose(r, [0.1, 0.0, 0.9])
    # Potential-based with phi_0=0 and gamma=1 preserves the total return.
    assert math.isclose(float(r.sum()), 1.0, abs_tol=1e-9)


def test_prize_shaping_ignored_without_prize_fields():
    recs = [_record(0, d, reward=1.0) for d in range(3)]  # no own/opp_prizes
    r = _shaped_rewards(recs, [0, 1, 2], gamma=0.99, prize_shaping=0.5, loss_shaping=0.0)
    assert np.allclose(r, [0.0, 0.0, 1.0])


def test_loss_shaping_only_penalizes_deckout_or_no_active_losses():
    def traj(reward, code):
        recs = [{**_record(0, d, reward=reward), "end_reason_code": code} for d in range(2)]
        return _shaped_rewards(recs, [0, 1], gamma=0.99, prize_shaping=0.0, loss_shaping=0.3)

    assert np.allclose(traj(-1.0, 2), [0.0, -1.3])   # deck-out loss penalised
    assert np.allclose(traj(-1.0, 3), [0.0, -1.3])   # no-active loss penalised
    assert np.allclose(traj(-1.0, 1), [0.0, -1.0])   # prize-out loss: no penalty
    assert np.allclose(traj(1.0, 2), [0.0, 1.0])     # a *win* is never penalised


def test_build_batch_accepts_shaping_and_stays_finite(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {**_record(0, d, player=0, reward=1.0, seed=1, action_index=d % 3, n_options=4),
         "own_prizes": 6 - d, "opp_prizes": 6, "end_reason_code": 1}
        for d in range(4)
    ]
    _write_records(path, records)
    kept, _ = load_records([str(path)])
    params = init_params(hidden=8, seed=0)
    batch = build_batch(kept, params, gamma=0.99, lam=0.95,
                        prize_shaping=0.1, loss_shaping=0.1)
    assert np.isfinite(batch["advantages"]).all()
    assert np.isfinite(batch["returns"]).all()


def test_offline_training_produces_valid_reloadable_artifact(tmp_path):
    """--iters 0 --bootstrap-data: policy.json from self-play data, 再学習可能."""
    data = tmp_path / "selfplay.jsonl"
    records = []
    for game in range(6):
        reward = 1.0 if game % 2 == 0 else -1.0
        records += [
            _record(game, d, player=0, reward=reward, seed=game,
                    action_index=d % 3, n_options=4)
            for d in range(5)
        ]
    _write_records(data, records)
    out = tmp_path / "policy.json"

    rc = main([
        "--iters", "0", "--bootstrap-data", str(data), "--bootstrap-passes", "2",
        "--out", str(out), "--hidden", "8", "--seed", "7",
        "--data-dir", str(tmp_path / "runs"),
    ])
    assert rc == 0
    policy = load_policy(str(out))
    assert policy is not None and validate_policy(policy) == []

    # The update actually moved the weights away from the init.
    init = params_to_policy(init_params(hidden=8, seed=7))
    assert not np.allclose(policy["w2"], init["w2"])

    # 再学習: resume from the artifact and write it again.
    out2 = tmp_path / "policy2.json"
    rc = main([
        "--iters", "0", "--bootstrap-data", str(data),
        "--init-from", str(out), "--out", str(out2), "--seed", "8",
        "--data-dir", str(tmp_path / "runs2"),
    ])
    assert rc == 0
    resumed = load_policy(str(out2))
    assert resumed is not None
    # and the resumed run continued from out, not from a fresh init
    params_out = policy_to_params(policy)
    params_resumed = policy_to_params(resumed)
    assert not np.allclose(params_resumed["w2"], params_out["w2"])
