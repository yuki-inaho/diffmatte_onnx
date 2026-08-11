# DiffMatte ONNX Runtime

事前生成済み DiffMatte ONNX モデルを、PyTorch・Detectron2 なしで実行するための uv プロジェクトです。モデルのエクスポートと PyTorch 比較は隣接する `../DiffMatte` の `cu128` ブランチ側で行います。

## セットアップ

```bash
uv sync --extra dev
```

## 推論

```bash
uv run diffmatte-infer-onnx \
  --model models/diffmatte_vits_1024_ddim10_643x960.onnx \
  --image demo/retriever_rgb.png \
  --trimap demo/retriever_trimap.png \
  --output artifacts/retriever_alpha.png \
  --provider cpu \
  --seed 0 \
  --report artifacts/inference.json
```

モデルの batch/H/W はエクスポート時に固定されます。生成済みモデルは `N=1, H=643, W=960` の公式デモ専用です。`--noise-npy` を指定すると、エクスポータ側の PyTorch 比較で使った初期ノイズをそのまま再利用できます。

`--enforce-known-trimap` は既知 foreground/background を後処理で 1/0 に固定します。指定しない場合は upstream の挙動を保ち、ONNX/PyTorch の数値比較が可能です。
