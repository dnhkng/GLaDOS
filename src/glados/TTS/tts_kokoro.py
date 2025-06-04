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
    """
    Get the list of available voices without creating a synthesizer instance.

    Outside of the class to allow for easy access to the list of voices without
    creating an instance of the Synthesizer class

    Args:
        path: Path to the voices binary file.

    Returns:
        List of available voice names.
    """
    voices = np.load(path)
    return list(voices.keys())


class SpeechSynthesizer:
    """
    Kokoro-based speech synthesizer for text-to-speech conversion.

    This class provides speech synthesis using the Kokoro TTS model with support
    for multiple voices. It converts text to phonemes and then to audio using
    an ONNX runtime model.

    Attributes:
        SAMPLE_RATE: Audio sample rate (24000 Hz)
        DEFAULT_VOICE: Default voice name
        MAX_PHONEME_LENGTH: Maximum allowed phoneme sequence length
    """

    MODEL_PATH: Path = resource_path("models/TTS/kokoro-v1.0.fp16.onnx")
    DEFAULT_VOICE: str = "af_alloy"
    MAX_PHONEME_LENGTH: int = 510
    SAMPLE_RATE: int = 24000

    def __init__(self, model_path: Path = MODEL_PATH, voice: str = DEFAULT_VOICE) -> None:
        self.sample_rate = self.SAMPLE_RATE
        self.voices: dict[str, NDArray[np.float32]] = np.load(VOICES_PATH)
        self.vocab = self._get_vocab()

        self.set_voice(voice)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_sess = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        self.phonemizer = Phonemizer()

    def set_voice(self, voice: str) -> None:
        """
        Set the voice for the synthesizer.

        Parameters:
            voice (str): The name of the voice to use for synthesis
        """
        if voice not in self.voices:
            raise ValueError(f"Voice '{voice}' not found. Available voices: {list(self.voices.keys())}")
        self.voice = voice

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Converts the input text to phonemes using the internal phonemizer, then generates audio from those phonemes.
        The result is returned as a NumPy array of 32-bit floating point audio samples.

        Parameters:
            text (str): The text to be converted to speech

        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech
        """
        phonemes = self.phonemizer.convert_to_phonemes([text], "en_us")
        phoneme_ids = self._phonemes_to_ids(phonemes[0])
        audio = self._synthesize_ids_to_audio(phoneme_ids)
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
        """
        Convert a string of phonemes to their corresponding integer IDs based on the vocabulary.

        Parameters:
            phonemes (str): A string of phonemes to be converted
        Returns:
            list[int]: A list of integer IDs corresponding to the phonemes
        Raises:
            ValueError: If the phoneme string exceeds the maximum length
        """
        if len(phonemes) > self.MAX_PHONEME_LENGTH:
            raise ValueError(f"text is too long, must be less than {self.MAX_PHONEME_LENGTH} phonemes")
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _synthesize_ids_to_audio(self, ids: list[int]) -> NDArray[np.float32]:
        """
        Convert a list of phoneme IDs to synthesized audio using the ONNX model.
        Parameters:
            ids (list[int]): A list of phoneme IDs to be converted to audio
        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech
        """
        voice_vector = self.voices[self.voice]
        voice_array = voice_vector[len(ids)]

        tokens = [[0, *ids, 0]]
        speed = 1.0
        audio = self.ort_sess.run(
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
