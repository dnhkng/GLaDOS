"""Hardware-independent tests for selecting audio execution providers."""

import pytest

from glados.onnx_runtime import audio_providers


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        (["CPUExecutionProvider", "MIGraphXExecutionProvider"], ["MIGraphXExecutionProvider", "CPUExecutionProvider"]),
        (["ROCMExecutionProvider", "CPUExecutionProvider"], ["ROCMExecutionProvider", "CPUExecutionProvider"]),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (["OpenVINOExecutionProvider", "CPUExecutionProvider"], ["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
        (
            ["OpenVINOExecutionProvider", "MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "MIGraphXExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            ["MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"],
        ),
        (["TensorrtExecutionProvider", "CoreMLExecutionProvider", "AzureExecutionProvider"], ["CPUExecutionProvider"]),
    ],
)
def test_audio_provider_selection(monkeypatch, available, expected):
    original = available.copy()
    monkeypatch.setattr("glados.onnx_runtime.ort.get_available_providers", lambda: available)
    assert audio_providers() == expected
    assert available == original
