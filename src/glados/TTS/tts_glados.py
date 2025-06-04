from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from pickle import load
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from ..utils.resources import resource_path
from .phonemizer import Phonemizer

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


@dataclass
class PiperConfig:
    """Piper configuration"""

    num_symbols: int
    """Number of phonemes"""

    num_speakers: int
    """Number of speakers"""

    sample_rate: int
    """Sample rate of output audio"""

    espeak_voice: str
    """Name of espeak-ng voice or alphabet"""

    length_scale: float
    noise_scale: float
    noise_w: float

    phoneme_id_map: Mapping[str, Sequence[int]]
    """Phoneme -> [id,]"""

    speaker_id_map: dict[str, int] | None = None

    @staticmethod
    def from_dict(config: dict[str, Any]) -> "PiperConfig":
        """
        Create a PiperConfig instance from a configuration dictionary.

        This class method parses a configuration dictionary and constructs a PiperConfig object with specified
        parameters. It allows flexible configuration by providing default values for optional inference settings.

        Parameters:
            config (dict[str, Any]): A dictionary containing configuration parameters for the text-to-speech model.
                Required keys:
                    - "num_symbols": Number of unique phoneme symbols
                    - "num_speakers": Total number of available speakers
                    - "audio": Dictionary containing "sample_rate"
                    - "espeak": Dictionary containing "voice"
                    - "phoneme_id_map": Mapping of phonemes to their corresponding IDs

                Optional keys:
                    - "inference": Dictionary with optional scaling parameters
                        - "noise_scale": Controls audio noise (default: 0.667)
                        - "length_scale": Controls speech duration (default: 1.0)
                        - "noise_w": Additional noise parameter (default: 0.8)
                    - "speaker_id_map": Mapping of speaker names to IDs (default: empty dictionary)

        Returns:
            PiperConfig: A configured PiperConfig instance with the specified parameters.

        Example:
            config = {
                "num_symbols": 100,
                "num_speakers": 5,
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "phoneme_id_map": {...},
                "inference": {
                    "noise_scale": 0.5,
                    "length_scale": 1.2
                }
            }
            piper_config = PiperConfig.from_dict(config)
        """
        inference = config.get("inference", {})

        return PiperConfig(
            num_symbols=config["num_symbols"],
            num_speakers=config["num_speakers"],
            sample_rate=config["audio"]["sample_rate"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            espeak_voice=config["espeak"]["voice"],
            phoneme_id_map=config["phoneme_id_map"],
            speaker_id_map=config.get("speaker_id_map", {}),
        )


class SpeechSynthesizer:
    """Text to Synthesizer, based on the VITS model.

    Trained using the Piper project (https://github.com/rhasspy/piper)

    This class provides methods to convert text into speech audio using a pre-trained ONNX model.
    It supports phonemization of input text, synthesis of audio from phoneme IDs, and configuration
    for different speakers and audio parameters.
    It uses the espeak-ng phonemizer for converting text to phonemes and ONNX Runtime for audio synthesis.
    """

    # Constants
    MAX_WAV_VALUE = 32767.0

    # Settings
    MODEL_PATH = resource_path("models/TTS/glados.onnx")
    PHONEME_TO_ID_PATH = resource_path("models/TTS/phoneme_to_id.pkl")
    USE_CUDA = True

    # Conversions
    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def __init__(
        self, model_path: Path = MODEL_PATH, phoneme_path: Path = PHONEME_TO_ID_PATH, speaker_id: int | None = None
    ) -> None:
        """
        Initialize the text-to-speech synthesizer with a specified model and optional speaker configuration.

        Args:
            model_path (Path): Path to the ONNX model file. Defaults to MODEL_PATH.
            phoneme_path (Path): Path to the phoneme-to-ID mapping file. Defaults to PHONEME_TO_ID_PATH.
            speaker_id (int | None): Optional speaker ID for multi-speaker models. Defaults to None.
        """
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
        self.id_map = self._load_pickle(phoneme_path)

        try:
            # Load the configuration file
            config_file_path = model_path.with_suffix(".json")
            with open(config_file_path, encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at path: {config_file_path}") from None
        except json.JSONDecodeError as e:
            raise ValueError(f"Configuration file at path: {config_file_path} is not a valid JSON. Error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while reading the configuration at path: {config_file_path}. Error: {e}"
            ) from e
        self.config = PiperConfig.from_dict(config_dict)
        self.sample_rate = self.config.sample_rate
        self.speaker_id = (
            self.config.speaker_id_map.get(str(speaker_id), 0)
            if self.config.num_speakers > 1 and self.config.speaker_id_map is not None
            else None
        )

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """
        Load a pickled dictionary from the specified file path.

        Args:
            path (Path): The file path to the pickle file containing the dictionary.

        Returns:
            dict[str, Any]: A dictionary loaded from the pickle file, ensuring type consistency.

        Raises:
            FileNotFoundError: If the specified pickle file does not exist.
            PickleError: If there are issues during pickle deserialization.
        """
        with path.open("rb") as f:
            return dict(load(f))

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
        phonemes = self._phonemizer(text)
        phoneme_ids_list = [self._phonemes_to_ids(sentence) for sentence in phonemes]
        audio_chunks = [self._synthesize_ids_to_audio(phoneme_ids) for phoneme_ids in phoneme_ids_list]

        if audio_chunks:
            audio: NDArray[np.float32] = np.concatenate(audio_chunks, axis=1).T
            return audio
        return np.array([], dtype=np.float32)

    def _phonemizer(self, input_text: str) -> list[str]:
        """
        Convert input text to phonemes using espeak-ng phonemization.

        This method transforms plain text into a sequence of phonetic representations
        using the English (US) phoneme set. It leverages the pre-configured phonemizer
        to break down text into its constituent phonetic components.

        Parameters:
            input_text (str): The text to be converted into phonemes.

        Returns:
            list[str]: A list of phoneme strings representing the input text's pronunciation.

        Example:
            phonemes = synthesizer._phonemizer("Hello world")
            # Might return something like ['hh', 'AH0', 'l', 'oW1', 'r', 'AO1', 'l', 'd']
        """
        phonemes = self.phonemizer.convert_to_phonemes([input_text], "en_us")

        return phonemes

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """
        Convert a sequence of phonemes to their corresponding integer IDs.
        This method takes a string of phonemes and converts each phoneme into its corresponding
        integer ID based on the pre-defined phoneme ID mapping. It also adds special tokens
        for beginning of sentence (BOS) and end of sentence (EOS), as well as padding (PAD)
        tokens to ensure consistent input length for the model.

        Args:
            phonemes (str): A string of phonemes to be converted.
        Returns:
            list[int]: A list of integer IDs corresponding to the phonemes, with special tokens included.
        """

        ids: list[int] = list(self.id_map[self.BOS])

        for phoneme in phonemes:
            if phoneme not in self.id_map:
                continue

            ids.extend(self.id_map[phoneme])
            ids.extend(self.id_map[self.PAD])
        ids.extend(self.id_map[self.EOS])

        return ids

    def _synthesize_ids_to_audio(
        self,
        phoneme_ids: list[int],
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ) -> NDArray[np.float32]:
        """
        Synthesize raw audio from phoneme IDs using the VITS model.

        Converts a sequence of phoneme IDs into audio using the pre-trained ONNX model, with
        optional control over audio generation parameters.

        Parameters:
            phoneme_ids (list[int]): A list of integer phoneme identifiers to be converted to audio.
            length_scale (float, optional): Controls the duration of generated audio.
                Defaults to the configuration's default length scale.
            noise_scale (float, optional): Controls the randomness of audio generation.
                Defaults to the configuration's default noise scale.
            noise_w (float, optional): Controls the variance of audio generation.
                Defaults to the configuration's default noise width.

        Returns:
            NDArray[np.float32]: A numpy array containing the synthesized audio waveform.
        """
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)

        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        sid = None

        if self.speaker_id is not None:
            sid = np.array([self.speaker_id], dtype=np.int64)

        # Synthesize through Onnx
        audio: NDArray[np.float32] = self.ort_sess.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))

        return audio

    def __del__(self) -> None:
        """
        Clean up ONNX session to prevent context leaks.
        """
        if hasattr(self, "ort_sess"):
            del self.ort_sess
