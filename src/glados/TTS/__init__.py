"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
FRENCH_VOICE = "models/TTS/fr_FR-siwis-medium.onnx"


def get_speech_synthesizer(
    voice: str = "glados",
    language: str = "en",
) -> SpeechSynthesizerProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados

        if language == "fr":
            # The GLaDOS model only knows English and gives French a strong accent: use a native
            # French Piper voice, shaped towards GLaDOS.
            from ..utils.resources import resource_path
            from .voice_effect import GladosEffect

            return tts_glados.SpeechSynthesizer(
                model_path=resource_path(FRENCH_VOICE), language="fr", effect=GladosEffect()
            )
        return tts_glados.SpeechSynthesizer(language=language)

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
