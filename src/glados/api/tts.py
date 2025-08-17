import io

import soundfile as sf

from glados.TTS import tts_glados
from glados.utils import spoken_text_converter


def write_glados_audio_file(f: str | io.BytesIO, text: str, *, format: str) -> None:
    """Generate GLaDOS-style speech audio from text and write to a file.

    Parameters:
        f: File path or BytesIO object to write the audio to
        text: Text to convert to speech
        format: Audio format (e.g., "mp3", "wav", "ogg")
    """
    glados_tts = tts_glados.SpeechSynthesizer()
    converter = spoken_text_converter.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    audio = glados_tts.generate_speech_audio(converted_text)
    sf.write(
        f,
        audio,
        glados_tts.sample_rate,
        format=format.upper(),
    )
