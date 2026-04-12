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
        """

        threshold = self.DEFAULT_THRESHOLD if threshold is None else float(threshold)

        # convert to int16
        x = (x * 32767).astype(np.int16)

        if patience is None:
            patience = {}
        else:
            patience = {
                self.model_name: patience,
            }

        if threshold is None:
            threshold_param = {}
        else:
            threshold_param = {
                self.model_name: threshold,
            }

        if debounce_time is None:
            debounce_time = 0.0

        return self._model.predict(x, patience=patience, debounce_time=debounce_time, threshold=threshold_param)[self.model_name] > threshold

    def predict_multi_sample(self, x: np.ndarray, threshold: float | None = None) -> bool:
        """
        Predict if the wake word was spoken in one of multiple 1280-sample-chunks.
        Resets the model before starting the prediction.

        Args:
            x: Audio data: a multiple of 1280 samples with 16kHz sample rate, in f32 format
            threshold: Threshold for the recognition score to be a successful prediction

        Returns:
            If the wake word was spoken in one of the sample chunks. If there are not enough sample chunks, return False.
        """

        self.reset()

        threshold = self.DEFAULT_THRESHOLD if threshold is None else float(threshold)

        # convert to int16
        x = (x * 32767).astype(np.int16)

        prediction = 0
        while len(x) >= self.MIN_SAMPLES:
            current = x[:self.MIN_SAMPLES]
            x = x[self.MIN_SAMPLES:]
            prediction = max(self._model.predict(current)[self.model_name], prediction)

        return prediction > threshold


__all__ = ["Model"]

