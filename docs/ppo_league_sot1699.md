# PPO 対戦相手プール（リーグ）学習 + 報酬シェーピング (SOT-1699)

SOT-1695 の 25デッキ・マルチデッキ PPO（vs RuleAgent 0.355 [0.310, 0.403]）を、
**対戦相手プール（リーグ）self-play** と **報酬シェーピング** で強化しようとした記録。
学習=PPO / 推論=純Python / pip依存なしの梅スタイルは不変。採用判定は SOT-1695 と同じく
提出構成（`main.py` = HarnessAgent: PPO方策 + 重要局面MCTS 0.4s + 候補手ハーネス）の
N=400 ベンチで、旧 0.355 [0.310, 0.403] と **CI 非重複**の改善のときのみ policy.json を更新する。

## 追加した仕組み

### 1. 対戦相手プール（リーグ）self-play — `train/ppo.py` `_collect_selfplay`
各イテレーションの self-play 相手を「現policy mirror」だけでなく、
**過去イテレーションの policy スナップショットの有界プール**からもサンプリングする
（`--league-frac`, プール上限 `--league-pool-size`, 既定5, FIFO）。
- 学習者（現policy）は on-policy、対戦相手は過去スナップショット。
- **学習に使うのは学習者側の on-policy レコードのみ**（`run_selfplay(..., record_labels={"ppo"})`）。
  古い（弱い）方策を模倣しないため。RuleAgent スパーリング（`--sparring-frac`）は SOT-1695 の
  advantage-weighted imitation のまま併用。
- プールが空のイテレーション0はリーグ試合を mirror にフォールバック。
- 自分相手の過学習と搾取され耐性の低さ（SOT-1681 対竹0.167）への直接の手当て。

### 2. 報酬シェーピング — `train/ppo.py` `_shaped_rewards`
終端 ±1/0 に加えて（学習時のみ適用、ストア済み record の reward は ±1/0 のまま schema 不変）:
- **プライズ差分 potential-based shaping**（`--prize-shaping`）:
  ポテンシャル `φ_t = coef·(opp_残プライズ − 自分_残プライズ)/6`、
  `F_t = γ·φ_{t+1} − φ_t`（終端 φ=0）。potential-based なので方策不変性を保ち、
  終端 ±1 が支配したまま、プライズ獲得手へのクレジット割当だけを鋭くする。
  レコードに残プライズ枚数（`own_prizes`/`opp_prizes`, 追加フィールド）を焼き込み。
- **deck-out / no-active 敗北への負整形**（`--loss-shaping`）:
  engine end-reason code（2:山札切れ, 3:場ポケモン切れ）で終わった**敗北**にのみ小さな負を上乗せ。
  勝ち・prize-out 敗北には掛けない。`end_reason_code`（`MatchResult.detail` からパース）をレコードに保存。

### 3. best_iter 昇格ゲート（SOT-1695 の教訓 — 必須・維持）
イテレーションごとに vs RuleAgent / vs Random を評価し、**最良反復のスナップショットを採用**、
patience で停滞早期終了。iter -1（学習前）も候補に含むため eval を悪化させる run は regression を出さない。

## 学習構成の比較（in-training raw eval, n=150）

| 構成 | init | spar/league | shaping(prize/loss) | best_iter | best vs_rule (raw, in-train n=150) |
| --- | --- | --- | --- | --- | --- |
| W | champion warm-start | 0.4 / 0.3 | 0.1 / 0.1 | **-1**（=champion, 学習前）| 0.253（iter -1；全学習 iter は 0.093〜0.187 に劣化 → iter 9 で early-stop）|
| F | fresh | 0.4 / 0.3 | 0.1 / 0.1 | 18 | 0.193（20 iter 中のピーク；他 iter は 0.047〜0.153）|

（seed=1699, iters=20, games/iter=60, eval=150, patience 10-12。raw eval はシード感受性が高い
〈champion も seed により 0.167〜0.253〉ため、採用判定は下の N=400 Harness ベンチで行う。F2〈fresh,
spar0.5/league0.2, shaping0.05/0.05〉は W/F がいずれも champion を下回ったため未実行。）

**要点**: warm-start（W）はリーグ学習の各 iter が champion より **必ず悪化** し、昇格ゲートは学習前
スナップショット（=champion 本体, best_iter -1）を選んだ。fresh（F）はピークでも raw 0.193 で
champion raw 0.253 を下回った。→ どちらの構成でも「リーグ学習後の新 policy」は champion を超えない。

## 採用判定（HarnessAgent 構成 N=400, time_limit_s=0.4）

候補は **F（fresh, best_iter 18）** — 実際にリーグ学習で生成された新 policy。
（W は best_iter -1 = champion 本体なので N=400 は champion の再計測にしかならず省略。）

| 対戦相手 | N | 勝率 | Wilson 95% CI | 旧 champion | 判定 |
| --- | --- | --- | --- | --- | --- |
| RuleAgent | 400 | **0.2525** | **[0.212, 0.297]** | 0.355 [0.310, 0.403] | **FAIL（CI が champion より下方・非重複）** |

- 候補の N=400 は 4 チャンク（seed 0/100/200/300, 各 100）を集計（`--aggregate`）: 34/20/30/17 勝 = 101/400。
  seed 間分散が大きい（0.170〜0.340）が、集計 CI 上限 0.297 < champion CI 下限 0.310 で **有意に劣後**。
- **faults 0**（final 側, 400/400）・時間切れ 0（max think 402ms ≤ 5000ms）。25デッキ横断の学習 self-play /
  in-train eval も全 iter fault 0。
- vs Random は champion（policy.json 不変）が SOT-1695 実績 0.898 を保持。候補は昇格しないため再計測せず。

**結論: 昇格ゲート不成立 → champion（SOT-1695 の data/policy.json）を維持。** リーグ学習・報酬シェーピング・
best_iter 昇格ゲートの実装と、この否定的結果の再現可能なログ／ベンチを成果物として PR 化する
（policy.json は不変）。

## 受け入れ条件との対応

- [x] リーグ学習（相手プール）で学習した policy が生成されている（F: `/tmp/cand_F.json`, best_iter 18,
  records 161,853。過去 policy スナップショットの有界プール + RuleAgent スパーリングから対戦相手をサンプル）
- [x] vs RuleAgent N=400 の CI 付き比較があり、昇格判定が CI ゲート（0.355 と非重複改善）で行われている
  → 候補 0.2525 [0.212, 0.297] は champion に対し**非重複で劣後**、ゲート不成立で champion 維持
- [x] vs Random 非劣化・25デッキ fault 0 → champion（不変）が 0.898 を保持。候補も 400 harness + 全学習
  self-play/eval で fault 0
- [x] PPO スタイル不変（学習は PPO、推論は純 Python、pip 依存なし。requirements.txt 不変）
- [x] repo の venv での pytest green（`python -m pytest eval/tests/` → 371 passed）

## 教訓

- **リーグ学習 + 報酬シェーピングは、この規模（iters20×60games, seed1699）では SOT-1695 champion を
  超えなかった。** warm-start は毎 iter 劣化（強い champion からの追加 on-policy 学習が方策を崩す＝
  SOT-1695 の「iter ピーク後崩壊」と同種）、fresh はピークでも raw 0.193 < champion 0.253。
- 昇格ゲートは正しく機能: warm-start では best_iter=-1（学習前）を選び regression を出さず、fresh の
  候補は N=400 CI ゲートで正直に棄却。**非昇格でも infra/eval/docs を PR 化**（[[sot1698-prize-race-vs-matsu]]
  と同じ運用: behavior 変更を伴わない解析ハーネス＋ドキュメントとして残す）。
- harness（MCTS 0.4s）は raw を持ち上げる（F: raw 0.193 → harness 0.253）が、champion の raw→harness
  リフト（0.253 → 0.355）には届かず。採用判定を raw でなく HarnessAgent N=400 で行う方針は妥当。
- N=400 harness ベンチは seed チャンク（各 100, ~5min）+ `--aggregate` で分割集計するのが実用的
  （単一 run 20min は 1 セッションのタイムアウトに収まらない）。
