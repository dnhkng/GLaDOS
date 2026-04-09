# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications by reisbauer03:
#   fixed imports to account for new module location
#   stripped VAD functionality
#   stripped tflite, forcing usage of onnxruntime
#   introduced 'onnx_providers' parameter on __init__ for onnx provider selection
#   stripped noise suppression
#   stripped pretrained models
#   added model_class_mappings from https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/__init__.py


# Imports
import numpy as np
from .utils import AudioFeatures, re_arg

import wave
import os
import functools
import pickle
from collections import deque, defaultdict
from functools import partial
import time
from typing import List, Union, DefaultDict, Dict

model_class_mappings = {
    "timer": {
        "1": "1_minute_timer",
        "2": "5_minute_timer",
        "3": "10_minute_timer",
        "4": "20_minute_timer",
        "5": "30_minute_timer",
        "6": "1_hour_timer"
    }
}

# Define main model class
class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    @re_arg({"wakeword_model_paths": "wakeword_models"})  # temporary handling of keyword argument change
    def __init__(
            self,
            wakeword_models: List[str],
            class_mapping_dicts: List[dict] = [],
            custom_verifier_models: dict = {},
            custom_verifier_threshold: float = 0.1,
            onnx_providers: list[str] = None,
            **kwargs
            ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            custom_verifier_models (dict): A dictionary of paths to custom verifier models, where
                                           the keys are the model names (corresponding to the openwakeword.MODELS
                                           attribute) and the values are the filepaths of the
                                           custom verifier models.
            custom_verifier_threshold (float): The score threshold to use a custom verifier model. If the score
                                               from a model for a given frame is greater than this value, the
                                               associated custom verifier model will also predict on that frame, and
                                               the verifier score will be returned.
            onnx_providers (list[str): list of ONNX providers to use. Default: ["CPUExecutionProvider"]
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        # Get model paths for pre-trained models if user doesn't provide models to load
        wakeword_model_names = []

        for ndx, i in enumerate(wakeword_models):
            if os.path.exists(i):
                wakeword_model_names.append(os.path.splitext(os.path.basename(i))[0])
            else:
                raise ValueError(f"Could not find model file '{i}'")

        # Create attributes to store models and metadata
        self.models = {}
        self.model_inputs = {}
        self.model_outputs = {}
        self.model_prediction_function = {}
        self.class_mapping = {}
        self.custom_verifier_models = {}
        self.custom_verifier_threshold = custom_verifier_threshold

        # Do imports for  inference framework
        try:
            import onnxruntime as ort

            def onnx_predict(onnx_model, x):
                return onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})

            if onnx_providers is None:
                onnx_providers = ["CPUExecutionProvider"]

        except ImportError:
            raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models
            if ".tflite" in mdl_path:
                raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

            sessionOptions = ort.SessionOptions()
            sessionOptions.inter_op_num_threads = 1
            sessionOptions.intra_op_num_threads = 1

            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                             providers=onnx_providers)

            self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
            self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
            pred_function = functools.partial(onnx_predict, self.models[mdl_name])
            self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            elif model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}

            # Load custom verifier models
            if isinstance(custom_verifier_models, dict):
                if custom_verifier_models.get(mdl_name, False):
                    self.custom_verifier_models[mdl_name] = pickle.load(open(custom_verifier_models[mdl_name], 'rb'))

            if len(self.custom_verifier_models.keys()) < len(custom_verifier_models.keys()):
                raise ValueError(
                    "Custom verifier models were provided, but some were not matched with a base model!"
                    " Make sure that the keys provided in the `custom_verifier_models` dictionary argument"
                    " exactly match that of the `.models` attribute of an instantiated openWakeWord Model object"
                    " that has the same base models but doesn't have custom verifier models."
                )

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(onnx_providers=onnx_providers, **kwargs)

    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    def reset(self):
        """Reset the prediction and audio feature buffers. Useful for re-initializing the model, though may not be efficient
        when called too frequently."""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))
        self.preprocessor.reset()

    def predict(self, x: np.ndarray, patience: dict = {},
                threshold: dict = {}, debounce_time: float = 0.0, timing: bool = False):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms
                                as the appropriate number of samples are accumulated.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` or `debounce_time` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            debounce_time (float): The time (in seconds) to wait before returning another non-zero prediction
                                   after a non-zero prediction. Can preven multiple detections of the same wake-word.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected. If the `timing` argument is true, returns a
                  tuple of dicts containing model predictions and timing information, respectively.
        """
        # Check input data type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input audio data (x) must by a Numpy array, instead received an object of type {type(x)}.")

        # Setup timing dict
        if timing:
            timing_dict: Dict[str, Dict] = {}
            timing_dict["models"] = {}
            feature_start = time.time()

        # Get audio features
        n_prepared_samples = self.preprocessor(x)

        if timing:
            timing_dict["models"]["preprocessor"] = time.time() - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            if timing:
                model_start = time.time()

            # Run model to get predictions
            if n_prepared_samples > 1280:
                group_predictions = []
                for i in np.arange(n_prepared_samples//1280-1, -1, -1):
                    group_predictions.extend(
                        self.model_prediction_function[mdl](
                            self.preprocessor.get_features(
                                    self.model_inputs[mdl],
                                    start_ndx=-self.model_inputs[mdl] - i
                            )
                        )
                    )
                prediction = np.array(group_predictions).max(axis=0)[None, ]
            elif n_prepared_samples == 1280:
                prediction = self.model_prediction_function[mdl](
                    self.preprocessor.get_features(self.model_inputs[mdl])
                )
            elif n_prepared_samples < 1280:  # get previous prediction if there aren't enough samples
                if self.model_outputs[mdl] == 1:
                    if len(self.prediction_buffer[mdl]) > 0:
                        prediction = [[[self.prediction_buffer[mdl][-1]]]]
                    else:
                        prediction = [[[0]]]
                elif self.model_outputs[mdl] != 1:
                    n_classes = max([int(i) for i in self.class_mapping[mdl].keys()])
                    prediction = [[[0]*(n_classes+1)]]

            if self.model_outputs[mdl] == 1:
                predictions[mdl] = prediction[0][0][0]
            else:
                for int_label, cls in self.class_mapping[mdl].items():
                    predictions[cls] = prediction[0][0][int(int_label)]

            # Update scores based on custom verifier model
            if self.custom_verifier_models != {}:
                for cls in predictions.keys():
                    if predictions[cls] >= self.custom_verifier_threshold:
                        parent_model = self.get_parent_model_from_label(cls)
                        if self.custom_verifier_models.get(parent_model, False):
                            verifier_prediction = self.custom_verifier_models[parent_model].predict_proba(
                                self.preprocessor.get_features(self.model_inputs[mdl])
                            )[0][-1]
                            predictions[cls] = verifier_prediction

            # Zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0

            # Get timing information
            if timing:
                timing_dict["models"][mdl] = time.time() - model_start

        # Update scores based on thresholds or patience arguments
        if patience != {} or debounce_time > 0:
            if threshold == {}:
                raise ValueError("Error! When using the `patience` argument, threshold "
                                 "values must be provided via the `threshold` argument!")
            if patience != {} and debounce_time > 0:
                raise ValueError("Error! The `patience` and `debounce_time` arguments cannot be used together!")
            for mdl in predictions.keys():
                parent_model = self.get_parent_model_from_label(mdl)
                if predictions[mdl] != 0.0:
                    if parent_model in patience.keys():
                        scores = np.array(self.prediction_buffer[mdl])[-patience[parent_model]:]
                        if (scores >= threshold[parent_model]).sum() < patience[parent_model]:
                            predictions[mdl] = 0.0
                    elif debounce_time > 0:
                        if parent_model in threshold.keys():
                            n_frames = int(np.ceil(debounce_time/(n_prepared_samples/16000)))
                            recent_predictions = np.array(self.prediction_buffer[mdl])[-n_frames:]
                            if predictions[mdl] >= threshold[parent_model] and \
                               (recent_predictions >= threshold[parent_model]).sum() > 0:
                                predictions[mdl] = 0.0

        # Update prediction buffer
        for mdl in predictions.keys():
            self.prediction_buffer[mdl].append(predictions[mdl])

        if timing:
            return predictions, timing_dict
        else:
            return predictions

    def predict_clip(self, clip: Union[str, np.ndarray], padding: int = 1, chunk_size=1280, **kwargs):
        """Predict on an full audio clip, simulating streaming prediction.
        The input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (Union[str, np.ndarray]): The path to a 16-bit PCM, 16 khz, single-channel WAV file,
                                           or an 1D array containing the same type of data
            padding (int): How many seconds of silence to pad the start/end of the clip with
                            to make sure that short clips can be processed correctly (default: 1)
            chunk_size (int): The size (in samples) of each chunk of audio to pass to the model
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            list: A list containing the frame-level prediction dictionaries for the audio clip
        """
        if isinstance(clip, str):
            # Load audio clip as 16-bit PCM data
            with wave.open(clip, mode='rb') as f:
                # Load WAV clip frames
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        elif isinstance(clip, np.ndarray):
            data = clip

        if padding:
            data = np.concatenate(
                (
                    np.zeros(16000*padding).astype(np.int16),
                    data,
                    np.zeros(16000*padding).astype(np.int16)
                )
            )

        # Iterate through clip, getting predictions
        predictions = []
        step_size = chunk_size
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self.predict(data[i:i+step_size], **kwargs))

        return predictions

    def _get_positive_prediction_frames(
            self,
            file: str,
            threshold: float = 0.5,
            return_type: str = "features",
            **kwargs
            ):
        """
        Gets predictions for the input audio data, and returns the audio features (embeddings)
        or audio data for all of the frames with a score above the `threshold` argument.
        Can be a useful way to collect false-positive predictions.

        Args:
            file (str): The path to a 16-bit 16khz WAV audio file to process
            threshold (float): The minimum score required for a frame of audio features
                               to be returned.
            return_type (str): The type of data to return when a positive prediction is
                               detected. Can be either 'features' or 'audio' to return
                               audio embeddings or raw audio data, respectively.
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            dict: A dictionary with filenames as keys and  N x M arrays as values,
                  where N is the number of examples and M is the number
                  of audio features, depending on the model input shape.
        """
        # Load audio clip as 16-bit PCM data
        with wave.open(file, mode='rb') as f:
            # Load WAV clip frames
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        # Iterate through clip, getting predictions
        positive_data = defaultdict(list)
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions = self.predict(data[i:i+step_size], **kwargs)
            for lbl in predictions.keys():
                if predictions[lbl] >= threshold:
                    mdl = self.get_parent_model_from_label(lbl)
                    features = self.preprocessor.get_features(self.model_inputs[mdl])
                    if return_type == 'features':
                        positive_data[lbl].append(features)
                    if return_type == 'audio':
                        context = data[max(0, i - 16000*3):i + 16000]
                        if len(context) == 16000*4:
                            positive_data[lbl].append(context)

        positive_data_combined = {}
        for lbl in positive_data.keys():
            positive_data_combined[lbl] = np.vstack(positive_data[lbl])

        return positive_data_combined
