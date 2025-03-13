import io

import soundfile as sf

from glados.TTS import tts_glados
from glados.utils import spoken_text_converter


def write_glados_audio_file(f: str | io.BytesIO, text: str) -> None:
    """Generate GLaDOS-style speech audio from text and write to a file.
    
    Parameters:
        f: File path or BytesIO object to write the audio to
        text: Text to convert to speech
    """
    glados_tts = tts_glados.Synthesizer()
    converter = spoken_text_converter.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    audio = glados_tts.generate_speech_audio(converted_text)
    sf.write(
        f,
        audio,
        glados_tts.sample_rate,
        format="MP3",
    )
