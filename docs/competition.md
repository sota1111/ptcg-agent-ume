# Pokémon TCG AI Battle Challenge — 情報まとめ

> The Pokémon Company **PTCG AI Battle Challenge**（Kaggle）についての情報を収集・整理した資料。
> このリポジトリ `ptcg-agent-ume` は本コンペの **Simulation** 部門に向けたエージェント／ローカル評価環境。

最終更新: 2026-07-12 / 対象 Issue: SOT-1613「コンペ情報の収集」

---

## 1. 概要

ポケモンカードゲーム（Pokémon Trading Card Game, PTCG）をプレイする **AI Training Agent** を構築するコンペ。
確率・不完全情報（相手の手札・デッキが不明）・戦略的プランニングが勝敗を分ける環境で、競技用シミュレーション上で
エージェントを訓練・評価する。

- ルールベースのプログラムだけでは上位入賞は難しく、先読み・リアルタイム適応・最適な意思決定が求められる。
- コイントスやカードドローといった確率要素、数千種のカード組み合わせ、多様なデッキがあり「同じ試合は二つとない」。
- 参加者にはトレーニング／テスト用のシミュレータ（SDK）が提供される。これは Kaggle 競技環境と同じロジックで、
  ローカルデバッグや強化学習に使える。

### 2つのコンペ（連結）

本チャレンジは 2 つのコンペで構成される。**このリポジトリが対象とするのは Simulation 部門。**

| 部門 | Kaggle slug | 位置づけ | 最終提出締切 |
| --- | --- | --- | --- |
| **Simulation**（本リポジトリ対象） | `pokemon-tcg-ai-battle` | エージェントを提出しラダー自動対戦。スキルレーティングで順位付け | 2026-08-16 |
| **Strategy / Hackathon** | `pokemon-tcg-ai-battle-challenge-strategy` | レポート提出（賞金あり） | 2026-09-13 |

- Hackathon への参加は本 Simulation コンペのエントリー要件ではない（独立して参加可能）。
- 最終的な Hackathon の順位は「Competition のリーダーボード成績」＋「Hackathon 評価」の両方で決まる。

---

## 2. 評価方式（Simulation）

- **1日あたり最大5エージェント**を提出可能。各提出は、ラダー上で近いスキルレーティングを持つ他エージェントと
  Episode（=試合）を行う。
- 最終評価に反映されるのは **最新2提出のみ**（多数のエージェントが走るのを抑え、1提出あたりの試合数を増やすため）。
- リーダーボードに表示されるのは自分の**ベストスコアのエージェントのみ**。全提出の推移は Submissions ページで追える。
- 提出されたエージェントはコンペ終了まで Episode をこなし続ける。新しいエージェントほど高頻度で対戦する
  （フィードバックが速い）。

### スキルレーティング（Gaussian）

- 各提出のスキルは正規分布 **N(μ, σ²)** でモデル化。μ=推定スキル、σ=その不確かさ（時間とともに減少）。
- 提出時にまず **Validation Episode**（自分自身のコピーとの対戦）を実行し、正常動作を確認。失敗すると
  **Error** 扱いになり、エージェントのログをDLして原因調査できる。
- 正常なら **μ₀ = 600** で初期化され、All Submissions プールに加わる。
- レーティング更新: 勝者の μ を上げ、敗者の μ を下げる。引き分けなら両者の μ を平均へ寄せる。更新幅は
  「以前の μ から期待される結果との乖離」と「各提出の σ」に比例。得られた情報量に応じて σ を縮小する。
- **勝敗の点差はレーティング更新に影響しない**（勝ち／負け／引き分けのみが効く）。

### 最終評価

- **2026-08-16** の締切で追加提出をロック。
- 2026-08-16 から約2週間、試合を継続実行（または リーダーボードが収束するまで）。その終了時点でリーダーボード確定。

---

## 3. タイムライン（全て特記なき限り 11:59 PM UTC）

| 日付 | イベント |
| --- | --- |
| **2026-06-16 11:00 UTC** | 開始 (Start Date) |
| **2026-08-09** | エントリー締切（この日までにルール承諾が必要） |
| **2026-08-09** | チームマージ締切（参加・チーム合流の最終日） |
| **2026-08-16** | 最終提出締切 (Final Submission Deadline) |
| **2026-08-17 〜 ≈08-31** | 試合継続実行（収束まで）。終了時点でリーダーボード確定 |

> 主催者はタイムラインを必要に応じて更新する権利を留保している。

---

## 4. 賞金

- **Competition（Simulation）トラック自体には金銭賞なし。**
- ただし **Hackathon トラックにレポートを提出**した参加者は賞の対象になる。Hackathon の最終順位は
  Competition リーダーボード成績と Hackathon 評価の両方で決定。

---

## 5. シミュレータ（cabt Engine）と Agent API

対戦は **cabt Engine**（kaggle-environments 向けに作られた PTCG バトルシミュレータ）上で実行される。

- 公式 API ドキュメント: https://matsuoinstitute.github.io/cabt/
- 公式ルールとシミュレータ挙動の差分: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586
- コード／設定は kaggle-environments 1.14.10 時点のもの。最新は https://github.com/Kaggle/kaggle-environments

### 意思決定モデル

エージェントは毎ターン **観測(observation)** を受け取り、選んだ **オプションのインデックス列**を返す。

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()      # 初期選択(デッキ提出): 60枚の Card ID を返す
    # obs.select.option の中から選ぶ。
    # 返す各要素は 0 <= i < len(obs.select.option)、
    # 長さは [obs.select.minCount, obs.select.maxCount]、重複なし。
    return [...]
```

- エンジンは **常に合法手(legal moves)のみ**を提示する（`obs.select.option`）。エージェントは提示された選択肢の
  index を返すだけでよい。
- 観測 `Observation` は `{ select: SelectData | None, logs, current: State }`。
  - `select is None` はデッキ選択時のみ。この時だけ 60 枚の Card ID リストを返す。
  - それ以外は `SelectData = { type: SelectType, context: SelectContext, minCount, maxCount, option: list[Option] }`。
    返り値は `option` の index リスト（長さ ∈ [minCount, maxCount]、重複なし）。
  - `current: State` に盤面状態、`current.result`（-1 なら試合継続、それ以外で決着）、`current.yourIndex`
    （手番判定）等が入る。

### 選択の種類（cg/api.py の Enum より）

- `SelectType`（0〜10）: MAIN / CARD / ATTACHED_CARD / CARD_OR_ATTACHED_CARD / ENERGY / SKILL / ATTACK /
  EVOLVE / COUNT / YES_NO / SPECIAL_CONDITION。
- `SelectContext`（49種, 0〜48）: 具体的な選択文脈（例: SETUP_ACTIVE_POKEMON, SWITCH, DISCARD, ATTACK,
  IS_FIRST=先攻選択, MULLIGAN=引き直し, COIN_HEAD=コイン表選択 など）。コンペ期間中に末尾へ追加され得る。
- `OptionType`: NUMBER / YES / NO / CARD / TOOL_CARD / ENERGY_CARD / ENERGY / PLAY / ATTACH / EVOLVE /
  ABILITY / DISCARD / RETREAT / ATTACK / END / SKILL / SPECIAL_CONDITION。
- 補助 Enum: `AreaType`（DECK/HAND/DISCARD/ACTIVE/BENCH/PRIZE/STADIUM/ENERGY/TOOL/…）,
  `EnergyType`（COLORLESS/GRASS/FIRE/WATER/LIGHTNING/PSYCHIC/FIGHTING/DARKNESS/METAL/DRAGON/RAINBOW/TEAM_ROCKET）,
  `CardType`, `SpecialConditionType`（POISON/BURN/SLEEP/PARALYZE/CONFUSE）, `LogType`。

> ⚠️ 推論時のネットワークアクセス・外部API・LLM呼び出しは禁止（提出は自己完結している必要がある）。

---

## 6. カードデータ

- 対象カード・デッキ情報は Kaggle の Data Page 参照:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data
- ローカル `data/`（gitignore, 再配布禁止）:
  - `EN_Card_Data.csv` / `JP_Card_Data.csv`（各 2102 カード + ヘッダ）。
  - 列: `Card ID, Card Name, Expansion, Collection No., Stage/Type, Rule, Category, Previous stage, HP,
    Type, Weakness, Resistance, Retreat, Move Name, Cost, Damage, Effect Explanation`。
  - 追加で `Card_ID List_*.pdf`（カード画像一覧, 大容量）が配布物に含まれる。
- エンジン内部では `all_card_data()` = 約1267種、`all_attack()` = 約1556技（`libcg` 実測, Linux/glibc）。
- 公式ルールブック（PTCG）:
  https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/meg_rulebook_en.pdf

---

## 7. 提出方法

- 提出物は `.tar.gz` バンドル。**トップレベル**（ネストなし）に `main.py`、そして `deck.csv` を含める。
  （本リポジトリのビルドではエンジンの `cg/` も同梱する。）
- 作成:
  ```bash
  tar -czvf submission.tar.gz *
  ```
- Kaggle の **My Submissions** タブからアップロード:
  https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions
- アップロード後、まず自分自身との Validation Episode が走り、問題なければマッチメイキングプールに投入される。

---

## 8. このリポジトリのローカル評価環境

- 提出実物と同じレイアウト（トップレベル `main.py` + `deck.csv` + `cg/`）。
- `main.py` — 提出エントリ `agent(obs_dict) -> list[int]`（公式 sample 起点、現状ランダム）。
- `deck.csv` — 60枚デッキ（Card ID を1行1枚）。
- `eval/run_match.py` — ローカル自己対戦ランナー。`cg`（cabt エンジン）を読み込み、agent 対 agent の1試合を完走。
  - 低レベルプリミティブ `game.battle_start(deck0, deck1)` → `game.battle_select(list[int])` →
    `game.battle_finish()` を使い、**2エージェント対戦ループは自前で実装**している
    （`current.result != -1` で終了判定）。
  - 実測 ≈ 37 ms/試合（ランダム自己対戦、約65意思決定で決着）。
- `scripts/setup_engine.sh` — Kaggle ダウンロードから `cg/`・`data/` をコピー。
- `scripts/build_submission.sh` — `submission.tar.gz`（`main.py` + `deck.csv` + `cg/`）を生成。

### ⚠️ ライセンス注意

エンジン `cg/`（`libcg.so` 等）とカードデータ `data/` は **competition-use-only で再配布禁止**。
`.gitignore` 済みで **絶対にコミットしない**。git に載るのは自作コード（`main.py`, `deck.csv`, `eval/`,
`scripts/`, `docs/`）のみ。復元は README 記載の Kaggle CLI 手順で再ダウンロードする。

---

## 9. 参考リンク

| 内容 | URL |
| --- | --- |
| Simulation コンペ | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle |
| Strategy / Hackathon コンペ | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy |
| Data Page（カード・デッキ） | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data |
| My Submissions | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions |
| cabt Engine API ドキュメント | https://matsuoinstitute.github.io/cabt/ |
| 公式ルール vs シミュレータ差分 | https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586 |
| PTCG 公式ルールブック(PDF) | https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/meg_rulebook_en.pdf |
| kaggle-environments | https://github.com/Kaggle/kaggle-environments |

---

## 10. 要点サマリ（TL;DR）

- **不完全情報＋確率＋戦略**の PTCG を対戦する AI エージェントを提出するコンペ（Simulation 部門）。
- 提出 = `main.py`(`agent(obs_dict)->list[int]`) + `deck.csv`、トップレベルの `.tar.gz`。**推論時ネット禁止**。
- 評価 = ラダー自動対戦の **Gaussian スキルレーティング**（μ₀=600）。点差は無関係、勝敗のみ。最新2提出が最終反映、
  1日5提出まで。
- **最終提出締切 2026-08-16** → 約2週間の追試合で確定。金銭賞は Hackathon トラック側。
- エンジンは **cabt**（合法手のみ提示）。ローカルは `eval/run_match.py` で自己対戦を再現済み（≈37ms/試合）。
- エンジン・カードデータは **ライセンス上コミット禁止**（gitignore + 再DL）。
