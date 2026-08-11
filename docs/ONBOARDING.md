# LLMオンボーディングサマリー

> 新任 LLM エージェント向けの初期資料。記載内容は 2026-08-11 時点のローカル実装、GTX 1070 実機実行、GitHub release 構成に基づく。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** `diffmatte_onnx`。DiffMatte の事前学習済み ViTS_1024 を ONNX Runtime で推論する専用リポジトリ。
- **最終成果物:** PyTorch / Detectron2 を必要としない `diffmatte-infer-onnx` CLI、CPU 用と GTX 1070 GPU 用の ONNX 配布物、公式 retriever デモ、再現可能な可視化 notebook。
- **ビジネス背景・価値:** 学習・export 系を [`../DiffMatte`](https://github.com/yuki-inaho/DiffMatte/tree/gtx1070) に分離したまま、軽量な実行環境だけを提供する。
- **現時点の進捗サマリ:** CPU は PyTorch 参照と全画素比較済み。GTX 1070 では ONNX Runtime GPU の `CUDAExecutionProvider` を明示的に使い、公式 retriever サンプルの alpha と合成画像を生成済み。

## 2. クリティカルな要求・制約

- モデル契約は `image=[1,3,643,960]`、`trimap=[1,1,643,960]`、`initial_noise=[1,1,643,960]`、`alpha=[1,1,643,960]` の float32 固定形状。batch/H/W を動的入力として扱わない。
- image は RGB `[0,1]`、trimap は upstream と同じ閾値で `{0, 0.5, 1}` に量子化する。再現比較では seed または `initial_noise.npy` を固定する。
- CPU 用元モデルは IR v10、GTX 1070 用モデルは IR v9。両者は opset 18 と演算グラフを保ち、互換版は ONNX IR version フィールドだけを変更したもの。
- GPU 確認済み組合せは GTX 1070 (Pascal / sm_61)、`onnxruntime-gpu==1.16.3`、CUDA 11.8、cuDNN 8。`runtime-gpu` extra 以外の ORT/CUDA 組合せは**未検証**。
- `--provider cuda` は CUDAExecutionProvider が有効にならなければエラーにする。CPU への黙ったフォールバックを成功と扱わない。CUDA 指定時は ORT 1.16 の FusedConv 問題を避けるためグラフ最適化を無効化する。
- `models/*.onnx`（各約 119 MB）は Git ignore。実体は GitHub Release から取得し、同梱または追跡された `.onnx.json` の SHA-256 と照合する。manifest だけで完了とせず、モデル本体・入力ペア・推論結果を確認する。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | CPU/GPU セットアップ、モデルの選択、CLI、可視化出力 |
| 要件定義書 | `../DiffMatte/docs/ONBOARDING.md` | checkpoint からの export、PyTorch 参照比較、GPU branch の境界 |
| WBS / 進捗 | `docs/ONBOARDING.md` の「完了ゲート」 | 専用 WBS は未確認。本書の実測済みゲートを使う |
| テスト資産 | `tests/test_runtime.py` | 前処理、ノイズ再現性、provider 選択の単体テスト |
| GPU 実行資産 | `notebooks/onnxruntime_gpu_retriever_visualization.ipynb` | 公式 retriever を CUDAExecutionProvider で実行し alpha / 合成を可視化 |
| GPU 実行記録 | `reports/retriever_gpu_gtx1070.json` | GPU / ORT / CUDA 構成、モデル hash、provider、CPU との PNG 比較 |
| 配布 manifest | `models/*.onnx.json` | SHA-256、IR/opset、固定 shape、検証ランタイム |
| 実画像比較 | `reports/retriever_cpu_comparison.json` | PyTorch vs CPU ORT の誤差、input hash、PNG 一致 |
| 可視化成果物 | `docs/assets/retriever_gpu_visualization.png` | 追跡済みの GTX 1070 推論プレビュー。再生成物は `artifacts/notebooks/` |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- ONNX Runtime 前処理・入力検証・provider 選択、および固定形状モデルの CLI/API テストを改善する。
- 対象 GPU 向けの互換モデルを追加し、Release、SHA-256 manifest、実画像 report、可視化を同時に更新する。
- `notebooks/` の GPU notebook を再実行し、`CUDAExecutionProvider` 表示、alpha PNG、合成 PNG を確認する。

### 任せないタスク（例）

- このリポジトリへ PyTorch、Detectron2、学習コードを持ち込む変更。
- `cu128` ブランチや隣接リポジトリの既存ユーザー変更を、本リポジトリの作業として変更・削除すること。
- CUDA provider が初期化されていない実行、別 GPU / 別解像度 / 別 ORT 世代を「確認済み」と記載すること。
- checkpoint と異なる初期ノイズの出力差を変換誤差と判断すること。

## 5. インタラクション方針

- **回答スタイル:** 日本語で結果を先に示し、モデル名、provider、shape、seed、SHA-256、誤差を具体的に記載する。
- **回答手順:** 配布物 hash 照合 → input/trimap pairing と shape 確認 → 実行 provider 確認 → alpha/合成と report 確認。
- **禁止事項・注意:** `CUDAExecutionProvider` が active providers に含まれない結果を GPU 成功と報告しない。未検証の依存バージョンや性能値を断定しない。
- **秘匿情報の扱い:** checkpoint のローカル絶対パス、認証情報、トークンを文書・report・Release note に含めない。

## 6. 試行タスク（オンボーディング演習）

1. **確認済み:** `uv sync --extra runtime-cpu --extra dev` の後、`uv run pytest` と `uv run ruff check src tests` を実行する。3 test と lint 成功を確認する。
2. **確認済み（GTX 1070）:** `uv sync --extra runtime-gpu --extra notebook --extra dev` の後、`uv run --extra runtime-gpu --extra notebook jupyter nbconvert --to notebook --execute notebooks/onnxruntime_gpu_retriever_visualization.ipynb --output executed.ipynb --output-dir artifacts/notebooks --ExecutePreprocessor.timeout=600` を実行する。report の provider に `CUDAExecutionProvider`、生成された `retriever_gpu_alpha.png` と `retriever_gpu_visualization.png` を確認する。
3. **確認済み（CPU 参照比較）:** `reports/retriever_cpu_comparison.json` の input hash、seed `0`、max `4.589557647705078e-06`、allclose `true` を確認し、異なる noise を比較していないことを確かめる。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** `確認済み` / `未検証` / `推定` を明示し、実行日、コマンド、モデル hash、入力、provider、結果を残す。
- **TBDの扱い:** 不明点は `未確認` とし、次に確認するファイル、GPU、またはコマンドを併記する。
- **レビュー/承認フロー:** 未確認。少なくとも test/lint、実モデル smoke test、Release asset hash、画像確認を変更者が提示する。
- **その他の運用ルール:** ONNX を追加・更新したら、本体を Release に配置し、`.onnx.json`、README のモデル表、実画像 report、可視化、SHA-256 を同じ変更で更新する。Release だけ、または manifest だけでは完了にしない。

### 完了ゲート

- [x] CPU 元モデル `diffmatte_vits_1024_ddim10_643x960.onnx` の SHA-256 `d69827e...acd6` と固定 shape を確認。
- [x] GTX 1070 互換モデル `diffmatte_vits_1024_ddim10_643x960_gtx1070.onnx` の SHA-256 `3eebe380...362b6` と IR v9 / opset 18 を確認。
- [x] 公式 `demo/retriever_rgb.png` / `demo/retriever_trimap.png` を seed `0` で推論し、`CUDAExecutionProvider` を確認。
- [x] GPU alpha と青背景合成を `docs/assets/retriever_gpu_visualization.png` として materialize し、目視確認（session 作成を含む実行時間 32.89 秒。性能ベンチマークではない）。
- [x] GPU/CPU の出力 PNG 比較: 最大画素差 `1`、平均差 `3.240020736132711e-06`。
- [x] CPU PyTorch vs ORT: max `4.589557647705078e-06`、mean `3.0285896457371564e-08`、allclose。
- [ ] GTX 1070 以外の GPU、動的 shape、ONNX Runtime GPU 1.18 以降の互換性は未検証。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** 現在地が runtime、`../DiffMatte` が exporter、`demo/` が公式 retriever 入力、`notebooks/` が GPU 可視化、`artifacts/` が再生成可能な非追跡出力。
- **代表的なコマンド:** `just sync-cpu`、`just sync-gpu`、`just test`、`just lint`、`just notebook-gpu`、README の `diffmatte-infer-onnx` 例。
- **依存ライブラリ:** Python 3.11、NumPy、Pillow、CPU は `onnxruntime`、GTX 1070 は `onnxruntime-gpu==1.16.3` と CUDA 11.8 / cuDNN 8。
- **配布先:** [yuki-inaho/diffmatte_onnx Releases](https://github.com/yuki-inaho/diffmatte_onnx/releases)。
- **連絡先/責任者:** GitHub owner は `yuki-inaho`。承認責任者は未確認。

> 本書はモデル実体の代わりにはならない。Release asset、SHA-256 manifest、実行 report、可視化 PNG の四点をそろえて初めて再現可能な配布物として扱う。
