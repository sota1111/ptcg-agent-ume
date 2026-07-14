# PPO方策学習ベンチマーク (SOT-1689)

PPO学習 (`train/ppo.py`) で生成した `data/policy.json` を純Python推論エージェント
(`agents/ppo_agent.py`) で指し、ベースライン2種と N=200 ずつ側交替で対戦した記録。
再現: `venv/bin/python eval/bench_ppo_vs_baselines.py --n 200 --seed 0`。

## 学習レシピ

```
venv/bin/python train/ppo.py --iters 6 --games-per-iter 32 \
    --bootstrap-data eval/selfplay_runs/acceptance_sot1688.jsonl \
    --out data/policy.json
```

- ネットワーク: tanh MLP (138特徴 → hidden 64 → 64スロット方策 + 価値ヘッド)
- アルゴリズム: clipped surrogate (clip 0.2) + GAE(γ=0.99, λ=0.95) + entropy 0.01, Adam lr 3e-4
- ブートストラップ: SOT-1688 acceptance データ (3,164 学習可能決定 / 120 trajectories) ×3 passes
- Self-play反復: 6 iterations × 32 mirror games (毎iteration新しい重みで生成、faults 0 / invalid 0)
- 最終losses: policy −0.035, value 0.129, entropy 1.371, clip_frac 0.022 (`data/policy.json` の meta 参照)

## 結果 (N=200/対戦, side-swap, seed=0)

| 対戦相手 | W/D/L (ppo) | 勝率 | Wilson 95% CI | faults (ppo/opp) | ppo latency mean/max |
| --- | --- | --- | --- | --- | --- |
| Random | 129/0/71 | **0.645** | [0.577, 0.708] | 0 / 0 | 0.56ms / 2.0ms |
| RuleAgent (champion) | 36/0/164 | 0.180 | [0.133, 0.239] | 0 / 0 | 0.63ms / 2.9ms |

- **vs Random: CI下限 0.577 > 0.5** — ランダムより有意に強く、方策は学習出来ている。
- **vs RuleAgent: 0.180** — champion には未達。SOT-1689 は「champion超えは必須としない」
  (計測・記録が受け入れ条件)。強化は後続の SOT-1690 (重要局面MCTS補強) の範囲。
- 違法出力 0・fault 0 (両対戦、計400試合・7,614決定)。SafeAgent骨格 + 構成的合法性
  (`agents/policy_net.sample_action`) により artifact が壊れていても legal-random に落ちる。

## 受け入れ条件との対応

- [x] PPO学習が self-play データから policy.json を生成し再学習可能
  (`--init-from` での再開・オフライン再学習を `eval/tests/test_ppo_train.py` で固定)
- [x] `agents/ppo_agent.py` が純Python+JSON重みで動作、違法出力0・fault 0 (N=200×2)
- [x] vs Random / vs RuleAgent の勝率+CI を本レポートに記録
