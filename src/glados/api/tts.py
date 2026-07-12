import io
from functools import lru_cache

import soundfile as sf

from glados.TTS import get_speech_synthesizer
from glados.utils import spoken_text_converter


@lru_cache(maxsize=1)
def _get_synthesizer():
    return get_speech_synthesizer("glados")


@lru_cache(maxsize=1)
def _get_text_converter() -> spoken_text_converter.SpokenTextConverter:
    return spoken_text_converter.SpokenTextConverter()


def warm_tts() -> None:
    """Load ONNX session and text converter at startup."""
    _get_synthesizer()
    _get_text_converter()


def write_glados_audio_file(f: str | io.BytesIO, text: str, *, format: str) -> None:
    """Generate GLaDOS-style speech audio from text and write to a file.

    Parameters:
        f: File path or BytesIO object to write the audio to
        text: Text to convert to speech
        format: Audio format (e.g., "mp3", "wav", "ogg")
    """
    glados_tts = _get_synthesizer()
    converted_text = _get_text_converter().text_to_spoken(text)
    audio = glados_tts.generate_speech_audio(converted_text)
    sf.write(
        f,
        audio,
        glados_tts.sample_rate,
        format=format.upper(),
    )
