"""
Use OpenWakeWord for wakeword detection/recognition.

Note: this module uses files from the openWakeWord python project: https://github.com/dscripka/openWakeWord
For the license of openWakeWord, see LICENSE.
Files from openWakeWord:
 - LICENSE
 - model.py
 - utils.py

This was done because
1. openWakeWord itself has functionalities and dependencies not needed by this project that cannot be deactivated when specified as a dependency
2. openWakeWord has a hard dependency on 'onnxruntime', which conflicts with the dependency on 'onnxruntime-gpu' when using '--extra cuda'
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .model import Model as _InternalModel
from ..utils.resources import resource_path


class Model:
    """
    Wrapper around OpenWakeWordModel to account for new model paths and GPU support
    """
    AUDIO_SAMPLE_RATE = 16000  # audio/microphone sample rate
    MIN_SAMPLES = 1280 # minimum amount of audio samples needed
    DEFAULT_MELSPEC_MODEL_PATH = resource_path("models/openwakeword/melspec.onnx")  # Model path to melspectogram model
    DEFAULT_EMBEDDING_MODEL_PATH = resource_path("models/openwakeword/embedding.onnx")  # Model path to embedding model
    DEFAULT_THRESHOLD = 0.7 # Default threshold for a successful prediction

    def __init__(self, wakeword_model_path: Path, melspec_model_path: Path | None = None, embedding_model_path: Path | None = None):
        """
        Create a new OpenWakeWordModel
        Args:
            wakeword_model_path: Path to the wake word model
        """
        self.model_path = wakeword_model_path
        self.model_name = wakeword_model_path.stem

        # Configure providers (same pattern as ASR)
        providers = ort.get_available_providers()
        for excluded in ["TensorrtExecutionProvider", "CoreMLExecutionProvider"]:
            if excluded in providers:
                providers.remove(excluded)

        if "CUDAExecutionProvider" in providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # configure model
        wakeword_models = [str(wakeword_model_path)]

        if melspec_model_path is None:
            melspec_model_path = self.DEFAULT_MELSPEC_MODEL_PATH
        if embedding_model_path is None:
            embedding_model_path = self.DEFAULT_EMBEDDING_MODEL_PATH

        self._model = _InternalModel(wakeword_models, onnx_providers=providers, melspec_model_path=str(melspec_model_path), embedding_model_path=str(embedding_model_path), sr=self.AUDIO_SAMPLE_RATE)

    def reset(self):
        """Reset the model state"""
        self._model.reset()

    @staticmethod
    def _np_f32_to_i16(x: np.ndarray) -> np.ndarray:
        """
        Convert to given np array with dtype f32 to a np array with dtype i16, convert data.
        Args:
            x: numpy array, dtype f32, data should be in range -1.0 to 1.0

        Returns:
            numpy array, dtype int16, data will be scaled from -32767 to 32767
        """
        return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)

    @classmethod
    def _validate_threshold(cls, threshold: float | None) -> float:
        """
        Validate the threshold: it should be None or in the range 0.0 to 1.0.

        Args:
            threshold: Threshold value

        Returns:
            Corrected threshold: if None is given, the default value is returned, if the value is out of range, a ValueError is raised

        Raises:
            ValueError: if the threshold is out of range
        """
        if threshold is None:
            return cls.DEFAULT_THRESHOLD
        if 0.0 <= threshold <= 1.0:
            return float(threshold)
        raise ValueError("threshold must be between 0.0 and 1.0")

    def predict(self, x: np.ndarray, patience: int | None = None, debounce_time: float| None = None, threshold: float | None = None) -> bool:
        """
        Predict if the wake word was spoken
        Args:
            x: Audio data: a multiple of 1280 samples with 16kHz sample rate, in f32 format
            patience: How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                      be observed before the current frame will be returned as non-zero.
                      By default, this behaviour is disabled.
            debounce_time: The time (in seconds) to wait before returning another non-zero prediction
                           By default, this behaviour is disabled.
            threshold: Threshold for the recognition score to be a successful prediction

        Returns:
            If the wake word was spoken in one of the samples.

        Raises:
            ValueError: if the threshold is out of range
        """

        threshold = self._validate_threshold(threshold)

        # convert to int16
        x = self._np_f32_to_i16(x)

        if patience is None:
            patience = {}
        else:
            patience = {
                self.model_name: patience,
            }

        if debounce_time is None:
            debounce_time = 0.0

        threshold_param = {
            self.model_name: threshold,
        }

        return self._model.predict(x, patience=patience, debounce_time=debounce_time, threshold=threshold_param)[self.model_name] >= threshold

    def predict_multi_sample(self, x: np.ndarray, threshold: float | None = None) -> bool:
        """
        Predict if the wake word was spoken in one of multiple 1280-sample-chunks.
        Resets the model before starting the prediction.

        Args:
            x: Audio data: a multiple of 1280 samples with 16kHz sample rate, in f32 format
            threshold: Threshold for the recognition score to be a successful prediction

        Returns:
            If the wake word was spoken in one of the sample chunks. If there are not enough sample chunks, return False.

        Raises:
            ValueError: if the threshold is out of range
        """

        self.reset()

        threshold = self._validate_threshold(threshold)

        # convert to int16
        x = self._np_f32_to_i16(x)

        count_samples = len(x) // self.MIN_SAMPLES
        prediction = 0
        for i in range(count_samples):
            current = x[i * self.MIN_SAMPLES: (i + 1) * self.MIN_SAMPLES]
            prediction = max(self._model.predict(current)[self.model_name], prediction)

        return prediction >= threshold


__all__ = ["Model"]

