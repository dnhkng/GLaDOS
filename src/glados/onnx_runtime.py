"""Consistent ONNX provider selection for the local audio models."""

from loguru import logger
import onnxruntime as ort

AUDIO_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
)


def audio_providers() -> list[str]:
    """Use an installed GPU backend with CPU fallback; exclude unsupported EPs."""
    available = ort.get_available_providers()
    return [provider for provider in AUDIO_PROVIDER_PRIORITY if provider in available] + ["CPUExecutionProvider"]


def report_session_providers(session: ort.InferenceSession, model: str) -> None:
    """Report the session's registered providers, including ORT's CPU fallback."""
    logger.info("{} ONNX session providers: {}", model, session.get_providers())
