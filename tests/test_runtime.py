from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from diffmatte_onnx.runtime import choose_providers, load_inputs, make_noise


class FakeOrt:
    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]


def test_load_inputs_matches_upstream_conventions(tmp_path: Path) -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    trimap = np.array([[0, 128]], dtype=np.uint8)
    image_path = tmp_path / "image.png"
    trimap_path = tmp_path / "trimap.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(trimap).save(trimap_path)

    image, tri = load_inputs(image_path, trimap_path)

    assert image.shape == (1, 3, 1, 2)
    np.testing.assert_array_equal(tri, np.array([[[[0.0, 0.5]]]], dtype=np.float32))


def test_seeded_noise_is_reproducible() -> None:
    image = np.zeros((1, 3, 4, 5), dtype=np.float32)
    first = make_noise(image, 7, None)
    second = make_noise(image, 7, None)
    np.testing.assert_array_equal(first, second)


def test_cuda_provider_requires_gpu_runtime() -> None:
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        choose_providers(FakeOrt, "cuda")
