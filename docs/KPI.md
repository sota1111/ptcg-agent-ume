# 梅(ume) 対戦KPI定義 (SOT-1710)

対戦性能の向上を継続観測するためのKPI。ベンチ/評価を走らせるたびに
`eval/kpi.py` がKPIレコード(1計測=1行)を **`eval/kpi_history.jsonl`**
(コミット対象。`/data/` は二重gitignoreのため `eval/` 配下に置く)へ追記し、
`eval/kpi_report.py` が履歴と直近比較(改善/悪化)を表示する。

## KPI一覧

| # | KPI | 定義・測定方法 | 改善方向 |
|---|-----|---------------|---------|
| 1 | `winrate_vs_rule` | 提出構成そのもの(`agents.harness.HarnessAgent` = PPO temperature 0.25 + 重要局面MCTS 0.4s + 候補手ハーネス、`main.py` と同一)が固定ベースライン **RuleAgent** に `eval.arena.run_arena`(席交替)で勝つ率。Wilson 95% CI付き。**昇格ゲートの主指標**: champion交代は CI下限 > 0.50 が条件(`eval/bench_final_vs_rule.py` の PROMOTION 判定と同一)。 | **高いほど良い**。CIが前回と重ならない上昇のみ有意な改善とみなす |
| 2 | `winrate_vs_random` | 同じ提出構成が **RandomAgent** に勝つ率(Wilson 95% CI付き)。リグレッション検知用(SOT-1695時点 ≈ 0.90)。 | **高いほど良い**(大きく下がったら学習/推論の壊れを疑う) |
| 3 | `fault_total` | 提出側の fault(違法出力・agent exception・timeout 等、`run_arena` の `a_faults`)の全ベンチ合算。 | **0維持**(1でもあれば悪化=NG。トレンドではなくゲート) |
| 4 | `decision_time_mean_ms` | vs Rule ベンチでの提出側1手あたり平均決定時間(ms)。参考値として max と per-move timeout も記録。 | **低いほど良い**(ただし時間予算内なら探索を厚くする方が優先。max がハードタイムアウト未満であることが前提) |

## 測定方法(標準の計測コマンド)

フルKPI(vs Rule + vs Random)は `eval/kpi.py --measure` で計測する:

```bash
# vs Rule N=48 + vs Random N=24 を計測して履歴へ1行追記
venv/bin/python eval/kpi.py --measure --n-rule 48 --n-random 24 \
    --seed 20260718 --issue SOT-XXXX
```

既存の昇格ベンチ `eval/bench_final_vs_rule.py` からも `--kpi` フックで記録できる
(単発ベンチは片側の勝率KPIのみ埋まり、もう片方は null):

```bash
venv/bin/python eval/bench_final_vs_rule.py --n 400 --kpi SOT-XXXX
venv/bin/python eval/bench_final_vs_rule.py --aggregate final.json c0.json c1.json \
    --kpi SOT-XXXX                        # seedチャンク集約からも記録可
```

既存ベンチのJSON結果を後から変換して記録することもできる:

```bash
venv/bin/python eval/kpi.py --from-report rule_bench.json \
    --random-report random_bench.json --issue SOT-XXXX
```

履歴ファイルは `UME_KPI_HISTORY` 環境変数で差し替え可能(テスト/検証用)。

## 記録スキーマ (`ume-kpi-v1`)

1レコード=JSONL 1行。共通フィールド: `ts`(UTC) / `git_sha` / `issue` /
`source`(kpi-measure | bench_final_vs_rule | from-report) / `deck`(champion =
`deck.csv` 固定) / `n_rule` / `n_random` / `seed` / `temperature` /
`time_limit_s`、および `kpis.{各KPI}`(値+CI+内訳)。未計測のKPIは
`value: null`(比較時はスキップ)。

## トレンド確認

```bash
venv/bin/python eval/kpi_report.py
```

履歴テーブル(時系列)と、直近2計測の各KPIについて Δ と
改善/悪化/横ばい(`fault_total` は OK/NG)を表示する。微小変動は
FLAT_EPS(勝率±0.005、決定時間±5ms)以内なら横ばい扱い。**採用判断はCI非重複を
基準にする**(点推定の上下だけで昇格/棄却しない — p-hacking回避)。

## 運用ルール

- champion(`main.py` の提出構成、`data/policy.json` / `deck.csv` 含む)へ変更を
  入れたら、マージ前後どちらかでKPI計測を1件記録する(小規模Nでも可 — Nは
  レコードに残るのでCI幅で解釈)。
- `eval/kpi_history.jsonl` は追記のみ(書き換え・削除しない)。
- ベースライン(RuleAgent / RandomAgent)と提出構成のパラメータ
  (temperature 0.25 / MCTS 0.4s)は固定。変える場合は新KPI名を切る
  (履歴の連続性を壊さない)。
