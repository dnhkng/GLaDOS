from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore
import yaml

from ..utils.resources import resource_path
from .mel_spectrogram import MelSpectrogramCalculator, MelSpectrogramConfig

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


class AudioTranscriber:
    DEFAULT_MODEL_PATH = resource_path("models/ASR/nemo-parakeet_tdt_ctc_110m.onnx")
    DEFAULT_CONFIG_PATH = resource_path("models/ASR/parakeet-tdt_ctc-110m_model_config.yaml")

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initialize an AudioTranscriber with an ONNX speech recognition model.

        Parameters:
            model_path (Path, optional): Path to the ONNX model file. Defaults to the predefined MODEL_PATH.
            config_file (Path, optional): Path to the main YAML configuration file. Defaults
            to the predefined CONFIG_PATH.

        Initializes the transcriber by:
            - Configuring ONNX Runtime providers, excluding TensorRT if available
            - Creating an inference session with the specified model
            - Loading the vocabulary from the yaml file
            - Preparing a mel spectrogram calculator for audio preprocessing

        Note:
            - Removes TensorRT execution provider to ensure compatibility across different hardware
            - Uses default model and token paths if not explicitly specified
        """
        # 1. Load the main YAML configuration file
        if not config_path.exists():
            raise FileNotFoundError(f"Main YAML configuration file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            try:
                full_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {config_path}: {e}") from e

        # 2. Configure ONNX Runtime session
        providers = ort.get_available_providers()

        # Exclude providers known to cause issues or not desired
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        # Prioritize CUDA if available, otherwise CPU
        if "CUDAExecutionProvider" in providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        session_opts = ort.SessionOptions()

        # Enable memory pattern optimization for potential speedup
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.enable_mem_pattern = True  # Can uncomment if beneficial

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=providers,
        )

        # 3. Load the vocabulary from the YAML configuration file
        if "labels" not in full_config:
            raise ValueError("YAML missing 'labels' section for vocabulary configuration.")
        self.vocab = dict(enumerate(full_config["labels"]))
        self.vocab[1024] = "<blk>"  # Add blank token to vocab

        # 4. Initialize MelSpectrogramCalculator using the 'preprocessor' section
        if "preprocessor" not in full_config:
            raise ValueError("YAML missing 'preprocessor' section for mel spectrogram configuration.")

        preprocessor_conf_dict = full_config["preprocessor"]
        mel_config = MelSpectrogramConfig(**preprocessor_conf_dict)

        self.melspectrogram = MelSpectrogramCalculator.from_config(mel_config)

    def process_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute mel spectrogram from input audio with normalization and batch dimension preparation.

        This method transforms raw audio data into a normalized mel spectrogram suitable for machine learning
        model input. It performs the following key steps:
        - Converts audio to mel spectrogram using a pre-configured mel spectrogram calculator
        - Normalizes the spectrogram by centering and scaling using mean and standard deviation
        - Adds a batch dimension to make the tensor compatible with model inference requirements

        Parameters:
            audio (NDArray[np.float32]): Input audio time series data as a numpy float32 array

        Returns:
            NDArray[np.float32]: Processed mel spectrogram with shape [1, n_mels, time], normalized and batch-ready

        Notes:
            - Uses a small epsilon (1e-5) to prevent division by zero during normalization
            - Assumes self.melspectrogram is a pre-configured MelSpectrogramCalculator instance
        """

        mel_spec = self.melspectrogram.compute(audio)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)

        # Add batch dimension and ensure correct shape
        mel_spec = np.expand_dims(mel_spec, axis=0)  # [1, n_mels, time]

        return mel_spec

    def decode_output(self, output_logits: NDArray[np.float32]) -> list[str]:
        """
        Decodes model output logits into human-readable text by processing predicted token indices.

        This method transforms raw model predictions into coherent text by:
        - Filtering out blank tokens
        - Removing consecutive repeated tokens
        - Handling subword tokens with special prefix
        - Cleaning whitespace and formatting

        Parameters:
            output_logits (NDArray[np.float32]): Model output logits representing token probabilities
                with shape (batch_size, sequence_length, num_tokens)

        Returns:
            list[str]: A list of decoded text transcriptions, one for each batch entry

        Notes:
            - Uses argmax to select the most probable token at each timestep
            - Assumes tokens with '▁' prefix represent word starts
            - Skips tokens marked as '<blk>' (blank tokens)
            - Removes consecutive duplicate tokens
        """
        predictions = np.argmax(output_logits, axis=-1)

        decoded_texts = []
        for batch_idx in range(predictions.shape[0]):
            tokens = []
            prev_token = None

            for idx in predictions[batch_idx]:
                if idx in self.vocab:
                    token = self.vocab[idx]
                    # Skip <blk> tokens and repeated tokens
                    if token != "<blk>" and token != prev_token:
                        tokens.append(token)
                        prev_token = token

            # Combine tokens with improved handling
            text = ""
            for token in tokens:
                if token.startswith("▁"):
                    text += " " + token[1:]
                else:
                    text += token

            # Clean up the text
            text = text.strip()
            text = " ".join(text.split())  # Remove multiple spaces

            decoded_texts.append(text)

        return decoded_texts

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribes an audio signal to text using the pre-loaded ASR model.

        Converts the input audio into a mel spectrogram, runs inference through the ONNX Runtime session,
        and decodes the output logits into a human-readable transcription.

        Parameters:
            audio (NDArray[np.float32]): Input audio signal as a numpy float32 array.

        Returns:
            str: Transcribed text representation of the input audio.

        Notes:
            - Requires a pre-initialized ONNX Runtime session and loaded ASR model.
            - Assumes the input audio has been preprocessed to match model requirements.
        """

        # Process audio
        mel_spec = self.process_audio(audio)

        # Prepare length input
        length = np.array([mel_spec.shape[2]], dtype=np.int64)

        # Create input dictionary
        input_dict = {"audio_signal": mel_spec, "length": length}

        # Run inference
        outputs = self.session.run(None, input_dict)

        # Decode output
        transcription = self.decode_output(outputs[0])

        return transcription[0]

    def transcribe_file(self, audio_path: Path) -> str:
        """
        Transcribes an audio file to text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            A tuple containing:
            - str: Transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the audio file cannot be read or processed, or sample rate mismatch.
            sf.SoundFileError: If soundfile encounters an error reading the file.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load as float32, assume mono (take first channel if stereo)
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # Select first channel
            # Check sample rate and audio length
            if sr != self.melspectrogram.sample_rate:
                raise ValueError(f"Sample rate mismatch: expected {self.melspectrogram.sample_rate}Hz, got {sr}Hz")
            if len(audio) == 0:
                raise ValueError(f"Audio file {audio_path} is empty or has no valid samples.")
            if audio.ndim > 1 and audio.shape[1] > 1:
                raise ValueError(f"Audio file {audio_path} is not mono. Please provide a mono audio file.")
        except sf.SoundFileError as e:
            raise sf.SoundFileError(f"Error reading audio file {audio_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {e}") from e

        return self.transcribe(audio)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session"):
            del self.session
