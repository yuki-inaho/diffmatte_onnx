"""ONNX Runtime single-image inference for a fixed-shape DiffMatte model."""
from __future__ import annotations

import argparse
import ctypes
import json
import site
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_NAMES = ("image", "trimap", "initial_noise")
OUTPUT_NAME = "alpha"


def preload_cuda_dependencies() -> list[str]:
    """Load pip-installed CUDA/cuDNN shared libraries before ORT loads its CUDA EP.

    This is a no-op for the CPU runtime. The GPU extra pins CUDA 11.8/cuDNN 8,
    the supported CUDA stack for the GTX 1070 (Pascal) notebook.
    """
    loaded: list[str] = []
    library_dirs = (
        Path(site.getsitepackages()[0]) / "nvidia" / "cuda_runtime" / "lib",
        Path(site.getsitepackages()[0]) / "nvidia" / "cuda_nvrtc" / "lib",
        Path(site.getsitepackages()[0]) / "nvidia" / "cublas" / "lib",
        Path(site.getsitepackages()[0]) / "nvidia" / "cufft" / "lib",
        Path(site.getsitepackages()[0]) / "nvidia" / "cudnn" / "lib",
    )
    library_names = (
        "libcudart.so.11.0",
        "libnvrtc.so.11.2",
        "libcublasLt.so.11",
        "libcublas.so.11",
        "libcufft.so.10",
        "libcudnn_ops_infer.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn.so.8",
    )
    for name in library_names:
        for directory in library_dirs:
            candidate = directory / name
            if candidate.is_file():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(candidate))
                break
    return loaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DiffMatte ONNX model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--trimap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-npy", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--enforce-known-trimap",
        action="store_true",
        help="Clamp known foreground to 1 and known background to 0 after inference",
    )
    return parser.parse_args()


def load_inputs(image_path: Path, trimap_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load files with the same RGB/trimap conventions as upstream run_one_image.py."""
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not trimap_path.is_file():
        raise FileNotFoundError(trimap_path)

    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(trimap_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != gray.shape:
        raise ValueError(f"Image and trimap sizes differ: {rgb.shape[:2]} vs {gray.shape}")

    quantized = np.empty_like(gray, dtype=np.float32)
    quantized[gray > 0.9] = 1.0
    quantized[(gray >= 0.1) & (gray <= 0.9)] = 0.5
    quantized[gray < 0.1] = 0.0

    image = np.transpose(rgb, (2, 0, 1))[None, ...]
    trimap = quantized[None, None, ...]
    return np.ascontiguousarray(image), np.ascontiguousarray(trimap)


def choose_providers(ort: Any, choice: str) -> list[str]:
    available = set(ort.get_available_providers())
    if choice == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(f"CPUExecutionProvider is unavailable: {sorted(available)}")
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install a compatible "
                f"onnxruntime-gpu environment. Available: {sorted(available)}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    raise RuntimeError(f"No supported execution provider is available: {sorted(available)}")


def validate_model_contract(session: Any, image: np.ndarray) -> None:
    inputs = {value.name: value for value in session.get_inputs()}
    missing = sorted(set(INPUT_NAMES) - inputs.keys())
    extra = sorted(inputs.keys() - set(INPUT_NAMES))
    outputs = {value.name for value in session.get_outputs()}
    if missing or extra or OUTPUT_NAME not in outputs:
        raise ValueError(
            f"Unexpected ONNX contract: missing={missing}, extra={extra}, outputs={sorted(outputs)}"
        )
    for name in INPUT_NAMES:
        if inputs[name].type != "tensor(float)":
            raise ValueError(f"Input {name!r} must be float32, got {inputs[name].type}")

    shape = inputs["image"].shape
    if len(shape) != 4:
        raise ValueError(f"Image input must be NCHW, got {shape}")
    expected_h = shape[2] if isinstance(shape[2], int) else None
    expected_w = shape[3] if isinstance(shape[3], int) else None
    actual_h, actual_w = image.shape[2:]
    if expected_h is not None and expected_h != actual_h:
        raise ValueError(f"ONNX model expects height={expected_h}, input has {actual_h}")
    if expected_w is not None and expected_w != actual_w:
        raise ValueError(f"ONNX model expects width={expected_w}, input has {actual_w}")


def make_noise(image: np.ndarray, seed: int, noise_path: Path | None) -> np.ndarray:
    expected_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
    if noise_path is not None:
        noise = np.load(noise_path, allow_pickle=False).astype(np.float32, copy=False)
        if noise.shape != expected_shape:
            raise ValueError(f"Noise shape must be {expected_shape}, got {noise.shape}")
        return np.ascontiguousarray(noise)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(expected_shape, dtype=np.float32)


def run_inference(
    model_path: Path,
    image: np.ndarray,
    trimap: np.ndarray,
    noise: np.ndarray,
    provider: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    try:
        if provider in {"auto", "cuda"}:
            preload_cuda_dependencies()
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "Install an execution runtime with `uv sync --extra runtime-cpu` or "
            "`uv sync --extra runtime-gpu`."
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    requested = choose_providers(ort, provider)
    session_options = ort.SessionOptions()
    if provider == "cuda":
        # ORT 1.16 is the CUDA 11.8/cuDNN 8 build that supports GTX 1070.
        # Its FusedConv optimizer cannot consume this modern opset-18 graph,
        # while the unfused CUDA kernels are compatible and numerically valid.
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=requested)
    active_providers = session.get_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but could not be initialized. "
            f"Active providers: {active_providers}. Check the ONNX Runtime/CUDA/cuDNN version match."
        )
    validate_model_contract(session, image)
    alpha = session.run(
        [OUTPUT_NAME],
        {"image": image, "trimap": trimap, "initial_noise": noise},
    )[0]
    if alpha.shape != trimap.shape:
        raise ValueError(f"Alpha shape must match trimap: {alpha.shape} vs {trimap.shape}")
    return alpha, active_providers


def _save_alpha(path: Path, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(output).save(path)


def main() -> None:
    args = _parse_args()
    image, trimap = load_inputs(args.image, args.trimap)
    noise = make_noise(image, args.seed, args.noise_npy)
    alpha, providers = run_inference(args.model, image, trimap, noise, args.provider)
    alpha_2d = np.clip(alpha[0, 0], 0.0, 1.0)
    if args.enforce_known_trimap:
        tri = trimap[0, 0]
        alpha_2d = np.where(tri == 0.0, 0.0, alpha_2d)
        alpha_2d = np.where(tri == 1.0, 1.0, alpha_2d)
    _save_alpha(args.output, alpha_2d)

    report = {
        "model": str(args.model.resolve()),
        "image": str(args.image.resolve()),
        "trimap": str(args.trimap.resolve()),
        "output": str(args.output.resolve()),
        "input_shape": list(image.shape),
        "seed": args.seed if args.noise_npy is None else None,
        "noise_npy": str(args.noise_npy.resolve()) if args.noise_npy else None,
        "providers": providers,
        "enforce_known_trimap": args.enforce_known_trimap,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
