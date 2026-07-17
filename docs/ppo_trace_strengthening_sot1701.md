# SOT-1701 対戦ログ駆動 PPO 推論強化

SOT-1701 の松竹梅 204 戦では梅が 22-114（勝率 0.162）で最下位だった。SOT-1700 の
強化ログ設計と既存ベンチの `harness_stats` を確認すると、提出構成では PPO が大半の決定を
担い、MCTS override は限定的だった。また SOT-1695/1699 の学習ログでは追加 PPO 更新が
ピーク後に崩壊し、リーグ学習候補も champion より有意に劣後した。このため、既存の最良
policy artifact は維持し、推論時の方策温度をログ駆動で較正した。

## 較正

同一 policy / HarnessAgent / RuleAgent、先後入替のプローブ（各 N=30、MCTS 0.15 秒）:

| 設定 | W-L | 勝率 | Wilson 95% CI | faults |
| --- | ---: | ---: | --- | ---: |
| deterministic | 10-20 | 0.333 | [0.192, 0.512] | 0 |
| temperature 0.25 | **18-12** | **0.600** | [0.423, 0.754] | 0 |
| temperature 0.50 | 15-15 | 0.500 | [0.332, 0.668] | 0 |

temperature 0.25 は PPO の確率的探索を残しつつ、弱い tail action の選択を抑える。完全な
argmax は明確に劣後したため採用しない。

## 本番予算ゲート

`venv/bin/python eval/bench_final_vs_rule.py --n 100 --seed 170100 --temperature 0.25`
を提出と同じ MCTS 0.4 秒で実行した。

- 候補: **54-46、勝率 0.540、Wilson 95% CI [0.443, 0.634]**
- faults / 違法手: **0**、最大思考時間 401.8 ms（上限 5,000 ms）
- 既存提出ベンチ: 89-311、勝率 0.223、95% CI [0.184, 0.266]
- 新旧 CI は非重複で、提出 PPO 構成の改善を確認
- 対 RuleAgent の優越（CI 下限 > 0.5）は未達。RuleAgent を統計的に上回ったとは主張しない

`main.py` と昇格ベンチの既定値をともに 0.25 に固定し、評価と提出の設定 drift をテストで防ぐ。
