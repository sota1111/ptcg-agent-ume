# initial decks — 大会実績ベースの初期デッキ25組 (SOT-1684)

2026年国際大会の上位入賞デッキリストを cabt エンジンの Card ID 形式に変換した初期デッキ集。
松 (matsu) / 竹 (take) / 梅 (ume) の3リポジトリに同一内容で配布されている。

## 形式

- 各 `NN_<archetype>.csv` はルート `deck.csv` と同形式（カードID 60行・昇順）。
- `manifest.json` に各デッキのアーキタイプ・出典（大会・順位・プレイヤー・URL）・代替カード情報を記録。

## 出典

| 範囲 | 出典 |
|---|---|
| 01–13 | Special Event Turin (2026-06-06) 上位入賞リスト |
| 14–20 | NAIC 2026 (limitlesstcg.com/tournaments/518) 各アーキタイプ最上位リスト |
| 21–24 | NAIC 2026 優勝 (Lillie's Clefairy) / 準優勝 (Dragapult) / 4位 (Slowking) / 10位 (N's Zoroark, Tord Reklev) |
| 25 | Mega Lopunny ex (NAIC 2026, 現行メタ上位のMegaアーキタイプ) |

デッキリストは limitlesstcg.com から取得（カード名・枚数は事実情報）。
カード名→ID の突合は `data/EN_Card_Data.csv` の (Expansion, Collection No.) を第一キー、
正規化カード名を第二キーとして機械的に実施し、全カードで名前一致を検証済み。

## エンジン収録範囲による代替（重要な既知事実）

**cabt エンジンのカードプールには CRI 弾が存在しない**（`EN_Card_Data.csv` に CRI 該当 0 件）。
このため実リスト中の CRI カード（主に `Special Red Card` ×1）は、同カテゴリの既存採用カード
（4枚未満・非 ACE SPEC）を 1 枚増量する決定的規則で代替した。全 25 デッキ中 15 デッキ・計 20 枚。
詳細は `manifest.json` の `substitutions`。

- 24_n_s_zoroark_ex_naic_10th は `Transformation Tome` ×4 も CRI のため代替 5 枚と最も多い。
- 当初候補の Beedrill ex（メタ15位）は主軸ライン全て CRI のため収録不可能と判断し、
  CRI 非依存の Mega Lopunny ex に差し替えた（Charizard / Gardevoir / Gholdengo / Froslass は
  2026-07 時点の現行メタ圏外のため、メタ上位の Crustle / Rocket's Mewtwo / Rocket's Honchkrow /
  Ethan's Typhlosion / Cynthia's Garchomp を採用）。

## 検証

- 60枚 / 同名≤4（基本エネルギー除く）/ ACE SPEC ≤1 を機械検証。
- 25 デッキ全てで cabt エンジン `battle_start` が受理し、ランダム方策でのフルマッチが
  正常終了することを確認（2026-07-14, matsu リポジトリの `eval/run_match.py`）。

## 使い方

エージェントの対戦デッキ（ルート `deck.csv`）は変更していない。差し替える場合:

```bash
cp decks/initial/01_dragapult.csv deck.csv
```
