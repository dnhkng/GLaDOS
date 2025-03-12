from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore

from ..utils.resources import resource_path
from .mel_spectrogram import MelSpectrogramCalculator

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


class AudioTranscriber:
    MODEL_PATH = resource_path("models/ASR/nemo-parakeet_tdt_ctc_110m.onnx")
    TOKEN_PATH = resource_path("models/ASR/nemo-parakeet_tdt_ctc_110m_tokens.txt")

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        tokens_file: Path = TOKEN_PATH,
    ) -> None:
        """
        Initialize an AudioTranscriber with an ONNX speech recognition model.

        Parameters:
            model_path (Path, optional): Path to the ONNX model file. Defaults to the predefined MODEL_PATH.
            tokens_file (Path, optional): Path to the file containing token mappings. Defaults
            to the predefined TOKEN_PATH.

        Initializes the transcriber by:
            - Configuring ONNX Runtime providers, excluding TensorRT if available
            - Creating an inference session with the specified model
            - Loading the vocabulary from the tokens file
            - Preparing a mel spectrogram calculator for audio preprocessing

        Note:
            - Removes TensorRT execution provider to ensure compatibility across different hardware
            - Uses default model and token paths if not explicitly specified
        """
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
        self.vocab = self._load_vocabulary(tokens_file)

        # Standard mel spectrogram parameters
        self.melspectrogram = MelSpectrogramCalculator()

    def _load_vocabulary(self, tokens_file: Path) -> dict[int, str]:
        """
        Load token vocabulary from a file mapping token indices to their string representations.

        Parameters:
            tokens_file (str): Path to the file containing token-to-index mappings.

        Returns:
            dict[int, str]: A dictionary where keys are integer token indices and values are
            corresponding token strings.

        Raises:
            FileNotFoundError: If the specified tokens file cannot be found.
            ValueError: If the tokens file is improperly formatted.

        Example:
            vocab = self._load_vocabulary('./models/ASR/tokens.txt')
            # Resulting vocab might look like: {0: '<blank>', 1: 'a', 2: 'b', ...}
        """
        vocab = {}
        with open(tokens_file, encoding="utf-8") as f:
            for line in f:
                token, index = line.strip().split()
                vocab[int(index)] = token
        return vocab

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

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text by reading the audio data and converting it to a textual representation.

        Parameters:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            str: The transcribed text content of the audio file.

        Raises:
            FileNotFoundError: If the specified audio file does not exist.
            ValueError: If the audio file cannot be read or processed.
        """

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")

        return self.transcribe(audio)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session"):
            del self.session
