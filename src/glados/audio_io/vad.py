from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


class VAD:
    VAD_MODEL: Path = resource_path("models/ASR/silero_vad_v5.onnx")
    SAMPLE_RATE: int = 16000  # or 8000 only!

    def __init__(self, model_path: Path = VAD_MODEL) -> None:
        """Initialize a Voice Activity Detection (VAD) model with an ONNX runtime inference session.

        Args:
            model_path (str, optional): Path to the ONNX VAD model. Defaults to VAD_MODEL.

        Notes:
            - Configures ONNX runtime providers, excluding TensorrtExecutionProvider
            - Sets up inference session with the specified model
            - Initializes internal state variables for processing audio chunks
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

        self.avaliable_sample_rates = [8000, 16000]

        self._state: NDArray[np.float32]
        self._context: NDArray[np.float32]
        self._last_sr: int
        self._last_batch_size: int

        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, audio_sample: NDArray[np.float32], sample_rate: int = SAMPLE_RATE) -> NDArray[np.float32]:
        """Process a batch of audio samples and return the VAD output.

        Args:
            x (NDArray[np.float32]): Audio samples with shape (batch_size, num_samples).
            sr (int): Sample rate of the audio samples.

        Returns:
            NDArray[np.float32]: VAD output with shape (batch_size, num_samples

        Raises:
            ValueError: If the number of samples is not supported.
        """
        num_samples = 512 if sample_rate == 16000 else 256

        if audio_sample.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {audio_sample.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = audio_sample.shape[0]
        context_size = 64 if sample_rate == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sample_rate):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        audio_sample = np.concatenate([self._context, audio_sample], axis=1)

        if sample_rate in [8000, 16000]:
            ort_inputs = {
                "input": audio_sample.astype(np.float32),
                "state": self._state,
                "sr": np.array(sample_rate, dtype=np.int64),
            }
            ort_outs = self.ort_sess.run(None, ort_inputs)
            out: NDArray[np.float32]
            state: NDArray[np.float32]
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = audio_sample[..., -context_size:]
        self._last_sr = sample_rate
        self._last_batch_size = batch_size

        return np.squeeze(out)

    def audio_forward(self, x: NDArray[np.float32], sample_rate: int = SAMPLE_RATE) -> NDArray[np.float32]:
        """Process an audio signal and return the VAD output.


        Args:
            x (NDArray[np.float32]): Audio samples with shape (num_channels, num_samples).
            sr (int): Sample rate of the audio samples.

        Returns:
            NDArray[np.float32]: VAD output with shape (num_channels, num_samples).

        Raises:
            ValueError: If the number of samples is not supported
        """
        outs = []
        self.reset_states()

        num_samples = 512 if sample_rate == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            pad_width = ((0, 0), (0, pad_num))
            x = np.pad(x, pad_width, mode="constant", constant_values=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(wavs_batch, sample_rate)
            outs.append(out_chunk)

        return np.stack(outs)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess
