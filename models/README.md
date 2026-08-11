# Local ONNX models

`.onnx` 本体（約 119 MB）は Git 管理対象外です。対応する `.onnx.json` manifest は Git 管理し、モデルの SHA-256、入出力契約、検証済みランタイムを記録します。

| モデル | 用途 | 取得先 |
|---|---|---|
| `diffmatte_vits_1024_ddim10_643x960.onnx` | CPU ONNX Runtime 用の元モデル（IR v10） | Release `gtx1070-onnx-v0.1.0` |
| `diffmatte_vits_1024_ddim10_643x960_gtx1070.onnx` | GTX 1070 / CUDA 11.8 / cuDNN 8 / ORT GPU 1.16.3 用（IR v9） | Release `gtx1070-onnx-v0.1.1` |

GTX 1070 互換版は演算グラフと opset 18 を保ったまま、ONNX IR version を 10 から 9 に変更したものです。`docs/ONBOARDING.md` と notebook の SHA-256 確認を通してから使ってください。
