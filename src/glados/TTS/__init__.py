from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
def get_speech_synthesizer(
    voice: str = "glados",
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

        return tts_glados.SpeechSynthesizer()

    else:
        from ..TTS import tts_kokoro

        assert voice in tts_kokoro.get_voices(), f"Voice '{voice}' not available"
        return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
