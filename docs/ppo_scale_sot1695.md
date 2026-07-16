# 25デッキ・マルチデッキ自己対戦 PPO スケールアップ (SOT-1695)

SOT-1689 の PPO 方策（単一デッキ・60試合・6反復）を、25大会デッキ（`decks/initial/`）の
ミラーローテーション self-play + RuleAgent スパーリング混合で1桁スケールアップし、
反復ごとの評価ログ・昇格ゲート（最良反復スナップショット採用）・停滞早期終了付きで
学習し直した記録。提出構成（`main.py` = HarnessAgent: PPO方策 + 重要局面MCTS 0.4s +
候補手ハーネス）での N=400 ベンチで採用判定した。

## 学習レシピ（採用 artifact の再現コマンド）

```
venv/bin/python train/ppo.py --iters 20 --games-per-iter 60 \
    --deck-dir decks/initial --sparring-frac 0.5 \
    --eval-games 150 --patience 10 --seed 5695 \
    --out data/policy.json
```

- ネットワーク/アルゴリズムは SOT-1689 と同一: tanh MLP (138特徴 → hidden 64)、
  clipped surrogate (clip 0.2) + GAE(γ=0.99, λ=0.95) + entropy 0.01, Adam lr 3e-4（既定を維持）
- self-play: 毎ゲーム `decks/initial/` の25デッキを順繰りミラー、`--sparring-frac 0.5` で
  半分を RuleAgent スパーリングに割当（データ多様性）
- 各反復後に vs RuleAgent / vs Random を n=150 で評価ログ（`policy.json` meta の
  `eval_history`）。**最良反復のスナップショットを採用する昇格ゲート**と patience=10 の
  停滞早期終了付き
- 学習規模: 20反復 × 60試合 = 1,200 ローテーション試合 / `records_trained` 136,453 決定
  （SOT-1689 は 60試合・3,747決定）
- 実際の学習経過: **best_iter=3**（vs rule 0.253）以降 PPO は単調劣化（iter 13 で 0.087 まで
  崩壊）→ patience で iter 13 早期終了、iter 3 スナップショットを artifact 化。昇格ゲートが
  無ければ劣化重みを掴んでいた
- 学習メタ（デッキ数・sparring_frac・eval_history・best_iter・停止理由 等）は
  `data/policy.json` の meta に焼き込み済み（SOT-1679 の教訓）

## ハイパラ探索（データミックスの A/B）

コア・ハイパラは既定のまま、データミックスだけ変えた4候補を同一シード帯
（vs RuleAgent, HarnessAgent 構成, n=120, seed=91000）でプローブ比較:

| 候補 | iters × games/iter | sparring_frac | probe 勝率 (n=120) | 備考 |
| --- | --- | --- | --- | --- |
| A | 20 × 30 | 0.25 | 0.233 | best_iter=0（改善せず） |
| B | 20 × 30 | 0.25 | 0.45 (n=20) | best_iter=−1 → 破棄 |
| C | 15 × 60 | 0.8 | 0.283 | スパーリング過多 |
| **C2（採用）** | **20 × 60** | **0.5** | **0.367** | best_iter=3 |
| D | 12 × 50 | 0.5 | — | best_iter=−1（初期方策未満）→ 破棄 |

## 結果（採用 policy.json / HarnessAgent 構成）

vs RuleAgent は seed 0/100/200/300 の 100試合チャンク×4、vs Random は seed 3000–3300 の
100試合チャンク×4 を `eval/bench_final_vs_rule.py --aggregate` で再集計（チャンク分割 +
再集計は SOT-1691 の教訓）。

| 対戦相手 | N | W/L | 勝率 | Wilson 95% CI | 旧値 | faults |
| --- | --- | --- | --- | --- | --- | --- |
| RuleAgent | 400 | 142/258 | **0.355** | **[0.310, 0.403]** | 0.20 [0.133, 0.289] (raw PPO) / 0.223 [0.185, 0.266] (harness+旧policy, SOT-1691) | 0 / 0 |
| Random | 400 | 359/41 | **0.898** | [0.864, 0.924] | 0.645 [0.577, 0.708] | 0 / 0 |

- **vs RuleAgent: 新CI下限 0.310 > 旧CI上限 0.289（raw）/ 0.266（harness）** — どちらの
  ベースラインに対しても CI 非重複の統計的有意改善。
- **vs Random: 0.898** — 旧 0.645 から大幅改善（劣化なしの条件を大差でクリア）。
- 思考時間 max 402ms ≤ cap 5000ms、fault / 違法手 0（計800試合）。

## 25デッキ mirror ローテーション安全ゲート

`eval/bench_deck_rotation.py`（本Issueで新設）: 25デッキ全てで HarnessAgent ミラー2試合、
**fault 0 / invalid candidates 0**（計50試合・約1万決定）。特徴量はカードID非依存
(FEATURE_DIM=138) のため、学習に使ったローテーションデッキ群でも合法性ゲートが全デッキで
機能することを確認。

## 学習コーパス（ローテーション self-play データ）

- `eval/selfplay_runs/corpus_sot1695.jsonl` 260試合 + `corpus_sot1695_b.jsonl` 250試合
  = **計510試合 ≥ 500**（25デッキ × 20–21試合、rule ミラー）。全 114,954 レコードが
  `validate_record`（schema `ume-selfplay-v1`）を通過、fault 0。gitignore 対象
  （再生成可能）。再現: `venv/bin/python eval/selfplay.py --deck-dir decks/initial
  --games 250 --agents rule,rule --seed 5000 --out <path>`
- 学習中の on-policy 生成分（1,200試合・PPOミラー+スパーリング）は別途 `--data-dir` 配下に
  反復ごと JSONL で保存される。

## 受け入れ条件との対応

- [x] 25デッキローテーション自己対戦データ ≥500試合（510試合）が生成され
  `validate_record` 契約を満たす（invalid 0）
- [x] PPO 学習が反復ごとの評価ログ付きで完走し、新 `policy.json`（meta付き）をコミット
- [x] vs RuleAgent 勝率 0.355 [0.310, 0.403] — 旧 0.20 [0.133, 0.289] と CI 非重複の有意改善
- [x] vs Random 0.898（旧 0.645 から劣化なし）・25デッキ全てで fault 0

## 教訓

- マルチデッキPPOは **best_iter 昇格ゲートが必須**: 今回の学習曲線は iter 3 をピークに
  単調崩壊しており、「最後の重みを使う」旧方式なら旧 policy より弱い artifact を掴んでいた。
- 生の方策勝率（0.253）と提出構成の勝率（0.355）は乖離する — 重要局面MCTSの上乗せが
  あるため、**採用判定は必ず HarnessAgent 構成の N=400 ベンチで行う**。
