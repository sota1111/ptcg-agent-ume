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

| 構成 | init | spar/league | shaping(prize/loss) | best_iter | best vs_rule (raw) |
| --- | --- | --- | --- | --- | --- |
| W | champion warm-start | 0.4 / 0.3 | 0.1 / 0.1 | TBD | TBD |
| F | fresh | 0.4 / 0.3 | 0.1 / 0.1 | TBD | TBD |
| F2 | fresh | 0.5 / 0.2 | 0.05 / 0.05 | TBD | TBD |

（raw eval はシード感受性が高い〈champion も seed により 0.167〜0.253〉ため、採用判定は下の N=400 Harness ベンチで行う。）

## 採用判定（HarnessAgent 構成 N=400）

| 対戦相手 | N | 勝率 | Wilson 95% CI | 旧 champion | 判定 |
| --- | --- | --- | --- | --- | --- |
| RuleAgent | 400 | TBD | TBD | 0.355 [0.310, 0.403] | TBD |
| Random | 400 | TBD | TBD | 0.898 | TBD |

## 受け入れ条件との対応

- [ ] リーグ学習（相手プール）で学習した policy が生成されている
- [ ] vs RuleAgent N=400 の CI 付き比較があり、昇格判定が CI ゲート（0.355 と非重複改善）で行われている
- [ ] vs Random 非劣化・25デッキ fault 0
- [ ] PPO スタイル不変（学習は PPO、推論は純Python、pip 依存なし）
- [ ] repo の venv での pytest green

## 教訓

- TBD
