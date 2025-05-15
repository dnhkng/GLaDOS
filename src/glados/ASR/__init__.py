"""ASR processing components."""

from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray

from .mel_spectrogram import MelSpectrogramCalculator
from .vad import VAD


class TranscriberProtocol(Protocol):
    def __init__(self, model_path: str, *args: str, **kwargs: dict[str, str]) -> None: ...
    def transcribe(self, audio_source: NDArray[Any]) -> str: ...
    def transcribe_file(self, audio_path: Path) -> str: ...


# Factory function
def get_audio_transcriber(
    engine_type: str = "ctc", **kwargs: dict[str, Any]
) -> TranscriberProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio transcriber.
    (docstring same as before)
    """
    if engine_type.lower() == "ctc":
        from .ctc_asr import AudioTranscriber as CTCTranscriber

        return CTCTranscriber()
    elif engine_type.lower() == "tdt":
        from .tdt_asr import AudioTranscriber as TDTTranscriber

        return TDTTranscriber()
    else:
        raise ValueError(f"Unsupported ASR engine type: {engine_type}")


__all__ = ["VAD", "MelSpectrogramCalculator", "TranscriberProtocol", "get_audio_transcriber"]
