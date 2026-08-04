from functools import lru_cache
import io

import soundfile as sf

from glados.TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from glados.utils import spoken_text_converter

from .config import ApiConfig

_api_config = ApiConfig()


def configure_tts(config: ApiConfig) -> None:
    """Apply API TTS settings and warm models when reuse is enabled."""
    global _api_config
    _api_config = config
    if config.reuse_tts:
        warm_tts()


@lru_cache(maxsize=1)
def _get_synthesizer() -> SpeechSynthesizerProtocol:
    return get_speech_synthesizer("glados")


@lru_cache(maxsize=1)
def _get_text_converter() -> spoken_text_converter.SpokenTextConverter:
    return spoken_text_converter.SpokenTextConverter()


def warm_tts() -> None:
    """Load the ONNX session at startup without generating audio."""
    _get_synthesizer()
    _get_text_converter()


def write_glados_audio_file(f: str | io.BytesIO, text: str, *, format: str) -> None:
    """Generate GLaDOS-style speech audio from text and write to a file.

    Parameters:
        f: File path or BytesIO object to write the audio to
        text: Text to convert to speech
        format: Audio format (e.g., "mp3", "wav", "ogg")
    """
    if _api_config.reuse_tts:
        glados_tts = _get_synthesizer()
        converter = _get_text_converter()
    else:
        glados_tts = get_speech_synthesizer("glados")
        converter = spoken_text_converter.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    audio = glados_tts.generate_speech_audio(converted_text)
    sf.write(
        f,
        audio,
        glados_tts.sample_rate,
        format=format.upper(),
    )
