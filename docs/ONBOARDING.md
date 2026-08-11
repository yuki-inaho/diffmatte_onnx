# LLMオンボーディングサマリー

> 新任LLMエージェント向けの初期資料。記載内容は 2026-08-11 時点のローカル実装・実行結果に基づく。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** `diffmatte_onnx`。DiffMatte の ONNX Runtime 推論専用リポジトリ。
- **最終成果物:** PyTorch / Detectron2 を必要としない `diffmatte-infer-onnx` CLI、固定形状 ONNX モデル、公式デモ入力、検証記録。
- **ビジネス背景・価値:** 学習済み画像 matting モデルを軽量な本番推論環境へ分離する。
- **現時点の進捗サマリ:** CPU ONNX Runtime で公式 retriever サンプルを推論済み。PyTorch 参照との全画素比較も合格。

## 2. クリティカルな要求・制約

- 生成済みモデルの契約は `image=[1,3,643,960]`、`trimap=[1,1,643,960]`、`initial_noise=[1,1,643,960]`、`alpha=[1,1,643,960]`、すべて float32。batch/H/W は固定。
- image は RGB `[0,1]`。trimap は upstream と同じ閾値で `{0, 0.5, 1}` に量子化する。
- 再現比較では seed または `initial_noise.npy` を固定する。ノイズが異なる出力同士を数値比較しない。
- デフォルトは upstream 数値互換を優先し、既知 foreground の強制 clamp はしない。必要時のみ `--enforce-known-trimap` を使う。
- `models/*.onnx` は約119 MBで Git ignore 対象。JSON report は追跡対象で、ONNX 本体の存在・hash を別途確認する。
- 確認済み provider は `CPUExecutionProvider`。CUDA provider は未検証。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | セットアップ、推論 CLI、固定形状制約 |
| 要件定義書 | `../DiffMatte/docs/ONBOARDING.md` | checkpoint からの export と PyTorch 比較 |
| WBS / 進捗 | 未確認 | 専用 WBS はない。現状は本書の完了ゲートを使う |
| テスト資産 | `tests/test_runtime.py` | 前処理、ノイズ再現性、provider エラーの単体テスト |
| 検証記録 | `reports/retriever_cpu_comparison.json` | 実画像 PyTorch/ORT 誤差、hash、PNG 一致 |
| 生成記録 | `models/diffmatte_vits_1024_ddim10_643x960.onnx.json` | config、shape、step、export 検証誤差 |
| 既知課題リスト | 本書「制約」 | CUDA 未検証、モデル固定形状、モデル本体は未追跡 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- ONNX Runtime 前処理・入力検証・provider 選択の改善。
- 同じモデル契約を使う CLI/API テストの追加。
- 新しい固定 H/W モデルの配置と、hash/report 更新。

### 任せないタスク（例）

- このリポジトリへ PyTorch、Detectron2、学習コードを持ち込む変更。
- checkpoint と異なるノイズによる出力差を「変換誤差」と判断すること。
- 未検証 CUDA 実行や別解像度を確認済みとして記載すること。
- `../DiffMatte` や `../diffmatte_sandbox` の既存ユーザー変更を削除すること。

## 5. インタラクション方針

- **回答スタイル:** 日本語、結果を先に述べ、コマンド・shape・hash・誤差を具体的に示す。
- **回答手順:** 入出力契約確認 → 再現条件確認 → 実行 → report/画像/hash 確認。
- **禁止事項・注意:** ONNX ファイルの存在だけで完了としない。checker、実推論、参照比較を区別する。
- **秘匿情報の扱い:** checkpoint やローカルパスを外部送信しない。認証情報を文書へ記載しない。

## 6. 試行タスク（オンボーディング演習）

1. `uv run pytest` と `uv run ruff check src tests` を実行し、3 test と lint が成功することを確認する。
2. `uv run diffmatte-infer-onnx ... --noise-npy artifacts/reference_comparison/initial_noise.npy` を実行し、出力 PNG の SHA-256 が `f67a38...a953c` になることを確認する。
3. ONNX Runtime で input/output metadata を列挙し、本書の固定 shape と一致することを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** `確認済み` / `未検証` / `推定` を区別し、実行日・コマンド・入力・出力・誤差を残す。
- **TBDの扱い:** 不明点は `未確認` とし、次に確認するファイルまたはコマンドを併記する。
- **レビュー/承認フロー:** 未確認。少なくとも test/lint と実モデル smoke test を変更者が提示する。
- **その他の運用ルール:** ONNX を更新したら `.onnx.json`、comparison report、SHA-256 を同時更新する。

### 完了ゲート

- [x] `models/diffmatte_vits_1024_ddim10_643x960.onnx` がローカルに存在（118.9 MB、SHA-256 `d69827...acd6`）。
- [x] runtime test 3/3 と Ruff が成功。
- [x] 公式デモ画像/trimap の pairing、shape、SHA-256 を確認。
- [x] PyTorch vs ORT: max `4.589557647705078e-06`、mean `3.0285896457371564e-08`、allclose。
- [x] runtime PNG と exporter 側 ORT PNG が画素単位・SHA-256 とも一致。
- [ ] CUDA provider は未検証。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** 現在地が runtime、`../DiffMatte` が exporter、`../diffmatte_sandbox/checkpoints` が学習済み重み保管元。
- **代表的なコマンド:** `uv sync --extra dev`、`uv run pytest`、README の `diffmatte-infer-onnx` 例。
- **依存ライブラリ:** Python 3.11、NumPy 1.26、Pillow、ONNX Runtime 1.28（lock を正とする）。
- **連絡先/責任者:** 未確認。Git remote はまだ設定されていない。

