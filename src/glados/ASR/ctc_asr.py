from pathlib import Path
import typing

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore
import yaml

from ..utils.resources import resource_path
from .mel_spectrogram import MelSpectrogramCalculator, MelSpectrogramConfig

# Default OnnxRuntime is way to verbose, only show fatal errors
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
            config_path (Path, optional): Path to the main YAML configuration file. Defaults
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
        self.config: dict[str, typing.Any]
        if not config_path.exists():
            raise FileNotFoundError(f"Main YAML configuration file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            try:
                self.config = yaml.safe_load(f)
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
        session_opts.enable_mem_pattern = True  # Memory pattern optimization enabled for better performance

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_opts,
            providers=providers,
        )

        # 3. Load the vocabulary from the YAML configuration file
        if "labels" not in self.config:
            raise ValueError("YAML missing 'labels' section for vocabulary configuration.")
        self.idx2token: dict[int, str] = dict(enumerate(self.config["labels"]))
        num_tokens = len(self.idx2token)
        if num_tokens != self.config["decoder"]["vocab_size"]:
            raise ValueError(
                f"Mismatch between number of tokens in vocabulary ({num_tokens}) "
                f"and decoder vocab size ({self.config['decoder']['vocab_size']})."
            )

        # Add blank token to vocab
        self.blank_idx = num_tokens  # Including blank toke
        self.idx2token[self.blank_idx] = "<blank>"  # Add blank token to vocab

        # 4. Initialize MelSpectrogramCalculator using the 'preprocessor' section
        if "preprocessor" not in self.config:
            raise ValueError("YAML missing 'preprocessor' section for mel spectrogram configuration.")

        preprocessor_conf_dict = self.config["preprocessor"]
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
            - Removes consecutive duplicate tokens
        """
        # Step 1: Greedy decoding to get the most probable token index at each time step
        predicted_indices_batch = np.argmax(output_logits, axis=-1)  # Shape: (batch_size, sequence_length)

        decoded_texts: list[str] = []
        for batch_idx in range(predicted_indices_batch.shape[0]):
            raw_indices_for_sample = predicted_indices_batch[batch_idx]

            # Step 2: CTC Collapse - Remove blanks and merge repeated non-blank tokens
            collapsed_indices: list[int] = []
            # Initialize to a non-valid token index to correctly handle the very first token (even blank)
            last_emitted_or_blank_idx = -1

            for current_idx in raw_indices_for_sample:
                if current_idx == last_emitted_or_blank_idx:
                    continue

                if current_idx == self.blank_idx:
                    last_emitted_or_blank_idx = self.blank_idx
                    continue

                collapsed_indices.append(current_idx)
                last_emitted_or_blank_idx = current_idx

            # Step 3: Convert collapsed_indices to string tokens
            # At this point, collapsed_indices should only contain valid, non-blank, de-duplicated token indices.
            tokens_str_list: list[str] = [self.idx2token.get(idx, "") for idx in collapsed_indices]

            # Handle SentencePiece style joining (replace "▁" with space)
            underline = "▁"
            text = "".join(tokens_str_list).replace(underline, " ").strip()

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
        except sf.SoundFileError as e:
            raise sf.SoundFileError(f"Error reading audio file {audio_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_path}: {e}") from e

        return self.transcribe(audio)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session"):
            del self.session
