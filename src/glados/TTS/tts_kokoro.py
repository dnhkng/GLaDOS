from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from ..utils.resources import resource_path
from .phonemizer import Phonemizer

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


VOICES_PATH = resource_path("models/TTS/kokoro-voices-v1.0.bin")


def get_voices(path: Path = VOICES_PATH) -> list[str]:
    """Outside of the class to allow for easy access to the list of voices without
    creating an instance of the Synthesizer class
    """
    voices = np.load(path)
    return list(voices.keys())


class Synthesizer:
    MODEL_PATH: Path = resource_path("models/TTS/kokoro-v1.0.fp16.onnx")
    DEFAULT_VOICE: str = "af_alloy"

    MAX_PHONEME_LENGTH: int = 510
    SAMPLE_RATE: int = 24000

    def __init__(self, model_path: Path = MODEL_PATH, voice: str = DEFAULT_VOICE) -> None:
        self.sample_rate = self.SAMPLE_RATE
        self.voices: dict[str, NDArray[np.float32]] = np.load(VOICES_PATH)
        self.vocab = self._get_vocab()
        self.voice = voice

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = Phonemizer()

    def generate_speech_audio(self, text: str, voice: str | None = None) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Converts the input text to phonemes using the internal phonemizer, then generates audio from those phonemes.
        The result is returned as a NumPy array of 32-bit floating point audio samples.

        Parameters:
            text (str): The text to be converted to speech

        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech
        """
        if voice is None:
            voice = self.voice
        else:
            assert voice in self.voices.keys(), f"voice '{voice}' not found"
        phonemes = self.phonemizer.convert_to_phonemes([text], "en_us")
        phoneme_ids = self._phonemes_to_ids(phonemes[0])
        audio = self._synthesize_ids_to_audio(phoneme_ids, voice)
        return np.array(audio, dtype=np.float32)

    @staticmethod
    def _get_vocab() -> dict[str, int]:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = (
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
            "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        )
        symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i
        return dicts

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        if len(phonemes) > self.MAX_PHONEME_LENGTH:
            raise ValueError(f"text is too long, must be less than {self.MAX_PHONEME_LENGTH} phonemes")
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _synthesize_ids_to_audio(self, ids: list[int], voice: str | None = None) -> NDArray[np.float32]:
        if voice is None:
            voice = self.voice
        else:
            assert voice in self.voices, f"voice '{voice}' not found"
        voice_vector = self.voices[voice]
        voice_array = voice_vector[len(ids)]

        tokens = [[0, *ids, 0]]
        speed = 1.0
        audio = self.session.run(
            None,
            {
                "tokens": tokens,
                "style": voice_array,
                "speed": np.ones(1, dtype=np.float32) * speed,
            },
        )[0]
        return np.array(
            audio[:-8000], dtype=np.float32
        )  # Remove the last 1/3 of a second, as kokoro adds a lot of silence at the end

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess
