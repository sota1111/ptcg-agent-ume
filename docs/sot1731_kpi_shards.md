# SOT-1731: N拡大・改善仮説の判定

## 独立seed計測

2026-07-19に提出構成（PPO temperature 0.25 + MCTS 0.4秒）を4つの
独立seed shardで再計測した。結果は `eval/kpi_history.jsonl` に追記済み。

| 対戦 | N | 勝率 | Wilson 95% CI | faults | 判定 |
|---|---:|---:|---:|---:|---|
| Rule | 200 | 0.560 | [0.4907, 0.6270] | 0 | CI下限が0.5以下のため昇格不可 |
| Random | 100 | 0.910 | [0.8377, 0.9519] | 0 | 既存0.8333から非劣化 |

決定時間は平均264.96ms、最大402.12msで、fault_totalは0だった。

## 仮説サイクル

### 松trace BC warm-start（棄却）

読み取り専用で `/workspaces/ptcg-agent-matsu` を確認したが、学習に利用できる
champion action trace corpusは存在しなかった。また松はplanner中心、梅は固定138次元
特徴量から離散actionを出すPPOであり、traceをそのまま教師データへ変換できる共通の
状態・action schemaもない。ラベル対応を推測した学習は誤教師による劣化を招くため、
候補policyを生成せず棄却した（したがって候補の対戦値はなし）。warm-startを各iterで
再適用する変更も行っていない。

特徴量拡充と報酬再調整は、それぞれ独立した学習・N=200評価を必要とし、このサイクル
では同時変更しなかった。現championは維持する。

## 結論

N拡大基盤、独立seed集約、履歴記録、Random非劣化、fault 0は達成した。一方、現champion
のCI下限は0.4907で、勝率改善の昇格ゲートは未達。測定結果に従いpolicy変更は採用しない。
