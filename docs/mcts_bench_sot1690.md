# SOT-1690 — 重要局面限定 determinized MCTS 補強の計測レポート

PPO方策（SOT-1689, `data/policy.json`）の第3段補強: 全局面を探索せず、**重要局面のみ**
エンジン公式 search API（`search_begin`/`search_step`、determinization対応）で
determinized MCTS を回し、明確に上回る手が見つかったときだけ方策の選択を上書きする
（実装: `agents/mcts.py`、組み込み: `agents/ppo_agent.py` の `mcts=True`）。

## 重要局面の判定基準（計測に基づく）

判定シグナルはPPO自身の出力のみ（追加コストは forward 1回の再利用で実質ゼロ）:

- **方策エントロピー** ≥ 1.9 nats（masked softmax、方策が選択肢を絞れていない局面）
- または **価値ヘッド** |value| ≤ 0.06（勝敗が拮抗している局面）
- 適用対象は単一選択（minCount ≤ 1 ≤ maxCount）かつ選択肢2以上、探索可能
  （`search_begin_input` あり）な決定のみ。

閾値は実対戦の計測から決定: committed policy.json を Random/Rule 相手に 12×2 試合
走らせた 432 決定の分布で、entropy p80≈1.92 / |value| p10≈0.064 を採り、
全決定の約25%（適格決定の約30%）が発動する組み合わせを選択した。

## 探索の構成（`MCTSConfig` 既定値）

| 項目 | 値 |
| --- | --- |
| 1決定あたり探索時間上限 | 0.5 s（ベンチは 0.4 s で実行） |
| determinization 数 | 3（隠れ情報は `UniformDeckPredictor` でサンプル、統計は全体でプール） |
| ルート候補数 | 上位prior 8 + 方策の選択（必ず最初に評価） |
| rollout | PPO方策サンプリングで深さ6、リーフはPPO価値ヘッド（終端は±1/0） |
| ルート選択 | PUCT（prior=PPO方策、ucb_c=1.0） |
| 上書き条件 | 探索平均リターンが方策の選択を **deviate_margin=0.1** 以上上回るときのみ（SOT-1672の教訓） |

失敗時は常に fail-closed（方策の選択を維持）。探索セッションは determinization ごとに
`finally` で `search_end`、生成状態は `search_release` で解放（リーク0）。

## 補強あり vs なし（N=200、seed=0、先後入替、per-move hard timeout 5s）

`venv/bin/python eval/bench_mcts_vs_ppo.py --n 200 --seed 0 --time-limit 0.4 --json eval/bench_mcts_vs_ppo.json`

| 指標 | 値 |
| --- | --- |
| 勝敗（mcts視点 W/D/L） | **126 / 0 / 74** |
| 勝率 | **0.630**、Wilson95 **[0.5612, 0.6939]**（下限 > 0.5 = 有意に改善） |
| fault（mcts側 / ppo側） | **0 / 0**（違法出力0） |
| MCTS発動率 | **30.1%**（1197 / 3979 全決定、適格 3391） |
| 探索成立 / 上書き / 失敗 | 1189 / 543 / 8 |
| determinization 成功/失敗 | 3552 / 39 |
| シミュレーション総数 | 171,768 |
| 探索時間 mean/max | **397 ms / 401 ms**（上限0.4sを遵守） |
| 1手思考時間（mcts側 mean/max） | **120 ms / 402 ms** |
| 1手思考時間（ppo側 mean/max） | 0.4 ms / 1.2 ms |

生データ: `eval/bench_mcts_vs_ppo.json`（arena run: `eval/arena_runs/bench_mcts_vs_ppo_n200/`）。

## 受け入れ条件との対応

- **重要局面のみMCTSが発動し、発動率・思考時間が計測されている** — 発動率 30.1%、
  探索時間 mean 397ms / max 401ms、1手思考時間 max 402ms（`MCTSStats` が常時計測）。
- **補強あり vs なし の比較結果（勝率+CI）がレポートとして残る** — 本ファイルと
  `eval/bench_mcts_vs_ppo.json`。
- **fault 0・違法出力0・1手max思考時間が持ち時間制約内** — 両側 fault 0、
  max 402ms は hard timeout 5s・Kaggle持ち時間（1試合≈10分、発動決定×0.4s ≈ 数秒/試合）
  に対し十分な余裕。

## 備考

- `mcts=False`（既定）の PPOAgent は SOT-1689 と decision path がバイト同一
  （`sample_action` の logits 再利用は同一乱数系列を保つ）。
- 上書きは 543/1197 発動（45%）。margin=0.1 を外すとノイズ上書きが増える
  （SOT-1672 で deviate_margin が決定打だった教訓をそのまま採用）。
- テスト: `eval/tests/test_mcts.py`（エンジン不要の判定・統計・logits再利用、
  fail-closed ゲート、実対戦での合法性・発動・時間上限）。
