"""
Smoke test for the FastVLM ONNX pipeline.

Skips if dependencies or model files are unavailable.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest


def _models_present(model_dir: Path) -> bool:
    onnx_dir = model_dir / "onnx"
    required = [
        "vision_encoder_q4.onnx",
        "embed_tokens_fp16.onnx",
        "decoder_model_merged_q4.onnx",
    ]
    return all((onnx_dir / name).exists() for name in required)


def _load_test_image(image_path: Path) -> np.ndarray:
    if image_path.exists():
        import cv2

        image = cv2.imread(str(image_path))
        if image is not None:
            return image

    height, width = 256, 256
    gradient = np.linspace(0, 255, num=width, dtype=np.uint8)
    plane = np.tile(gradient, (height, 1))
    image = np.stack([plane, np.flipud(plane), np.zeros_like(plane)], axis=-1)
    return image


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_fastvlm_describe_smoke() -> None:
    if not (_module_available("onnxruntime") and _module_available("regex") and _module_available("cv2")):
        pytest.skip("FastVLM dependencies are unavailable")

    from glados.utils.resources import resource_path
    from glados.vision.fastvlm import FastVLM

    model_dir = resource_path("models/Vision/FastVLM")
    if not _models_present(model_dir):
        pytest.skip("FastVLM model files not available")

    image_path = resource_path("data/Golden_Gate_Bridge.jpeg")
    image = _load_test_image(image_path)
    model = FastVLM(model_dir)
    result = model.describe(image, prompt="Describe the image briefly.", max_tokens=32)
    assert isinstance(result, str)
    assert result.strip()
