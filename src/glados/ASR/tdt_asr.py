# tdt_transcriber.py
from pathlib import Path
import time
import typing

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore
import yaml

from ..utils.resources import resource_path
from .mel_spectrogram import MelSpectrogramCalculator, MelSpectrogramConfig

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(3)  # Slightly less verbose than error-only


class _OnnxTDTModel:
    """
    Internal helper class to manage the three ONNX sessions (Encoder, Decoder, Joiner)
    for the TDT ASR model and related metadata.
    """

    DEFAULT_ENCODER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_encoder.onnx")
    DEFAULT_DECODER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_decoder.onnx")
    DEFAULT_JOINER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_joiner.onnx")

    def __init__(
        self,
        providers: list[str],
        encoder_model_path: Path = DEFAULT_ENCODER_MODEL_PATH,
        decoder_model_path: Path = DEFAULT_DECODER_MODEL_PATH,
        joiner_model_path: Path = DEFAULT_JOINER_MODEL_PATH,
    ) -> None:
        """
        Initializes the ONNX model sessions and extracts necessary metadata.

        Args:
            config: The validated TDTConfig object containing model paths.
            providers: List of ONNX Runtime execution providers to use.
        """
        session_opts = ort.SessionOptions()

        # Enable memory pattern optimization for potential speedup
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.enable_mem_pattern = True  # Can uncomment if beneficial

        logger.info(f"Using ONNX providers: {providers}")
        self.encoder = self._init_session(encoder_model_path, session_opts, providers)
        self.decoder = self._init_session(decoder_model_path, session_opts, providers)
        self.joiner = self._init_session(joiner_model_path, session_opts, providers)

        logger.info("--- Encoder ---")
        self._log_model_io(self.encoder)
        logger.info("--- Decoder ---")
        self._log_model_io(self.decoder)
        logger.info("--- Joiner ---")
        self._log_model_io(self.joiner)

        # Extract metadata from encoder
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        self.normalize_type: str | None = encoder_meta.get("normalize_type")  # e.g., "per_feature" or None
        self.pred_rnn_layers = int(encoder_meta.get("pred_rnn_layers", 0))
        self.pred_hidden = int(encoder_meta.get("pred_hidden", 0))
        logger.info(f"Encoder metadata: {encoder_meta}")
        if not self.pred_rnn_layers or not self.pred_hidden:
            logger.warning(
                "Warning: Could not extract 'pred_rnn_layers' or 'pred_hidden' from encoder metadata. "
                "Decoder state initialization might fail."
            )

        # Get joiner output dimension to infer number of duration bins later
        self.joiner_output_total_dim = self.joiner.get_outputs()[0].shape[-1]
        if not isinstance(self.joiner_output_total_dim, int) or self.joiner_output_total_dim <= 0:
            # This is often symbolic ('unk__...') - handle validation later
            logger.warning(
                f"Warning: Joiner output dimension appears dynamic or invalid ({self.joiner_output_total_dim}). "
                "Will rely on token list size + config durations for validation."
            )
        else:
            logger.info(f"Joiner output total dimension: {self.joiner_output_total_dim}")

        # Store input/output names for clarity and robustness
        self.encoder_in_names = [i.name for i in self.encoder.get_inputs()]
        self.encoder_out_names = [o.name for o in self.encoder.get_outputs()]
        self.decoder_in_names = [i.name for i in self.decoder.get_inputs()]
        self.decoder_out_names = [o.name for o in self.decoder.get_outputs()]
        self.joiner_in_names = [i.name for i in self.joiner.get_inputs()]
        self.joiner_out_names = [o.name for o in self.joiner.get_outputs()]

    def _init_session(
        self, model_path: Path, sess_options: ort.SessionOptions, providers: list[str]
    ) -> ort.InferenceSession:
        """Initializes an ONNX Runtime Inference Session."""
        try:
            return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX session for {model_path}: {e}") from e

    def _log_model_io(self, session: ort.InferenceSession) -> None:
        """Logs the inputs and outputs of an ONNX model session."""
        logger.info("Inputs:")
        for i in session.get_inputs():
            logger.info(f"  - {i.name}: {i.shape}, Type: {i.type}")
        logger.info("Outputs:")
        for o in session.get_outputs():
            logger.info(f"  - {o.name}: {o.shape}, Type: {o.type}")

    def get_decoder_initial_state(self, batch_size: int = 1) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Generates the initial hidden states for the decoder RNN.

        Args:
            batch_size: The batch size for the states (usually 1 for inference).

        Returns:
            A tuple containing the two initial state arrays (h_state, c_state typically),
            initialized to zeros with the correct dimensions based on encoder metadata.
            Assumes float32 dtype, which is common.
        """
        if not self.pred_rnn_layers or not self.pred_hidden:
            raise ValueError("Cannot create decoder state. Missing 'pred_rnn_layers' or 'pred_hidden' metadata.")

        # Assuming states are float32 as explicit casting was needed in the original script
        # and ONNX often defaults to float32.
        dtype = np.float32
        logger.info(
            f"Initializing decoder state with shape: ({self.pred_rnn_layers}, {batch_size}, "
            f"{self.pred_hidden}), dtype: {dtype}"
        )

        state0 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=dtype)
        state1 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=dtype)
        return state0, state1

    def run_encoder(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Runs the encoder model.

        Args:
            features: Mel spectrogram features with shape [batch, n_mels, time].
                      Note: This expects the output of _process_audio.

        Returns:
            Encoder output tensor, typically [batch, channels, time_reduced].
        """
        # Need feature length for the second input
        feature_length = np.array([features.shape[2]], dtype=np.int64)

        # Ensure input names match the actual model
        # Assuming order: [audio_signal, length]
        if len(self.encoder_in_names) != 2:
            raise ValueError(f"Encoder expected 2 inputs, got {len(self.encoder_in_names)}")

        input_dict = {
            self.encoder_in_names[0]: features,
            self.encoder_in_names[1]: feature_length,
        }
        # We only need the main encoder output, typically the first one.
        encoder_out = self.encoder.run([self.encoder_out_names[0]], input_dict)[0]
        return np.asarray(encoder_out, dtype=np.float32)  # Shape [batch, channels, time_reduced]

    def run_decoder(
        self, token_input: int, state0: NDArray[np.float32], state1: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Runs the decoder model for one step.

        Args:
            token_input: The predicted token ID from the previous step (or blank for init).
            state0: The first hidden state from the previous step.
            state1: The second hidden state from the previous step.

        Returns:
            A tuple containing:
            - Decoder output tensor for this step.
            - Next state0.
            - Next state1.
        """
        # Decoder inputs typically: target, target_len, state0, state1
        if len(self.decoder_in_names) != 4:
            raise ValueError(f"Decoder expected 4 inputs, got {len(self.decoder_in_names)}")

        # Prepare inputs matching expected types
        target = np.array([[token_input]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)

        # Explicitly use float32 for states as determined earlier
        state0_fp32 = state0.astype(np.float32)
        state1_fp32 = state1.astype(np.float32)

        input_dict = {
            self.decoder_in_names[0]: target,
            self.decoder_in_names[1]: target_len,
            self.decoder_in_names[2]: state0_fp32,
            self.decoder_in_names[3]: state1_fp32,
        }

        # Decoder outputs typically: decoder_out, decoder_out_len, next_state0, next_state1
        if len(self.decoder_out_names) != 4:
            raise ValueError(f"Decoder expected 4 outputs, got {len(self.decoder_out_names)}")

        outputs = self.decoder.run(self.decoder_out_names, input_dict)
        decoder_out = outputs[0]
        next_state0 = outputs[2]
        next_state1 = outputs[3]

        return decoder_out, next_state0, next_state1

    def run_joiner(self, encoder_out_t: NDArray[np.float32], decoder_out: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Runs the joiner model.

        Args:
            encoder_out_t: Encoder output for the current time step `t`,
                           shape [batch, 1, channels] or similar (depends on model).
                           The script used [B, C, 1] -> [B, 1, C]? Check logs.
                           Log shows input: [1, 1, 512] - so [B, T=1, C]
            decoder_out: Decoder output for the current step, shape [batch, 1, channels].

        Returns:
            Logits from the joiner, shape [batch, 1, vocab_size + num_durations].
        """
        # Joiner inputs typically: encoder_out, decoder_out
        if len(self.joiner_in_names) != 2:
            raise ValueError(f"Joiner expected 2 inputs, got {len(self.joiner_in_names)}")

        input_dict = {
            self.joiner_in_names[0]: encoder_out_t,
            self.joiner_in_names[1]: decoder_out,
        }

        # Joiner usually has one output: logits
        if len(self.joiner_out_names) != 1:
            raise ValueError(f"Joiner expected 1 output, got {len(self.joiner_out_names)}")

        logits = self.joiner.run(self.joiner_out_names, input_dict)[0]
        return np.asarray(logits, dtype=np.float32)

    def __del__(self) -> None:
        """Clean up ONNX sessions (optional but good practice)."""
        # ONNX Runtime sessions are usually managed fine by Python's GC,
        # but explicit deletion can be added if needed.
        del self.encoder
        del self.decoder
        del self.joiner


class AudioTranscriber:
    """
    Transcribes audio using a TDT (Token and Duration Transducer) ASR model
    loaded from ONNX files.

    This class handles audio preprocessing (Mel Spectrogram), running the
    Encoder-Decoder-Joiner ONNX models, performing TDT decoding, and
    post-processing the results into text.
    """

    DEFAULT_CONFIG_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_model_config.yaml")

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initializes the AudioTranscriber with models and configurations.

        Args:
            config_path: Path to the YAML configuration file (validated by TDTConfig).
            Note: this config file is extracted with TAR from the original TDT-model NEMO file.

        Raises:
            FileNotFoundError: If the config file or specified model/token files don't exist.
            ValueError: If the configuration is invalid (YAML format, content validation).
            RuntimeError: If ONNX models fail to load.
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

        # Initialize the internal ONNX model handler
        self.model = _OnnxTDTModel(providers)

        # 4. Load the vocabulary from the YAML configuration file

        if "labels" not in self.config:
            raise ValueError("YAML missing 'labels' section for vocabulary configuration.")
        self.idx2token: dict[int, str] = dict(enumerate(self.config["labels"]))
        num_tokens = len(self.idx2token)
        if num_tokens != self.config["decoder"]['vocab_size']:
            raise ValueError(
                f"Mismatch between number of tokens in vocabulary ({num_tokens}) and decoder vocab size ({self.config['decoder']['vocab_size']})."
            )

        # Add blank token to vocab
        self.blank_id = num_tokens  # Including blank toke
        self.idx2token[self.blank_id] = "<blank>"  # Add blank token to vocab

        logger.info(f"Blank token ID: {self.blank_id} ('{self.idx2token[self.blank_id]}')")
        logger.info(f"Vocabulary size (incl. blank): {len(self.idx2token)}")
        logger.info(f"Vocabulary: {self.idx2token}")

        self.tdt_durations = self.config["model_defaults"]["tdt_durations"]
        if not self.tdt_durations:
            raise ValueError("TDT durations list is empty in the configuration.")
        logger.info(f"TDT durations: {self.tdt_durations}")

        # Initialize Mel Spectrogram calculator from config
        preprocessor_conf_dict = self.config["preprocessor"]
        self.preprocessor_conf = MelSpectrogramConfig(**preprocessor_conf_dict)

        self.melspectrogram = MelSpectrogramCalculator.from_config(self.preprocessor_conf)
        # self.melspectrogram = MelSpectrogramCalculator.from_config(self.config["preprocessor"])
        logger.info("MelSpectrogramCalculator initialized.")

        logger.info(f"config: {self.config}")

    def _process_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Preprocesses raw audio into Mel spectrogram features suitable for the encoder.

        Steps:
        1. Applies tail padding.
        2. Computes Mel spectrogram using MelSpectrogramCalculator.
        3. Applies normalization based on encoder metadata (`normalize_type`).
        4. Adds batch dimension.

        Args:
            audio: Input audio time series data as a numpy float32 array.

        Returns:
            Processed Mel spectrogram features with shape [1, n_mels, time].
        """
        # # 1. Apply tail padding
        # if self.config.tail_padding_seconds > 0:
        #     padding_samples = int(self.config.preprocessor.sample_rate * self.config.tail_padding_seconds)
        #     audio = np.pad(audio, (0, padding_samples), mode="constant", constant_values=0.0)

        # 2. Compute Mel spectrogram
        # Output shape: [n_mels, time]
        mel_spec = self.melspectrogram.compute(audio)

        # 3. Add batch dimension
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Shape: [1, n_mels, time]

        return mel_spec.astype(np.float32)  # Ensure float32 for ONNX

    def _decode_tdt(self, encoder_out: NDArray[np.float32]) -> list[int]:
        """
        Performs TDT greedy decoding using the Decoder and Joiner models.
        This function implements the TDT decoding loop, which iteratively predicts
        tokens and their durations based on the encoder output.

        Args:
            encoder_out: The output from the encoder model, shape [1, channels, time_reduced].

        Returns:
            A list of decoded token IDs (excluding blank tokens).
        """
        batch_size, _, max_encoder_t = encoder_out.shape
        if batch_size != 1:
            raise NotImplementedError("TDT decoding currently only supports batch size 1.")

        predicted_token_ids: list[int] = []
        last_emitted_token_for_decoder = self.blank_id  # Start with blank for decoder init
        state0, state1 = self.model.get_decoder_initial_state(batch_size=1)

        # Initial decoder run with blank token
        decoder_out, next_state0, next_state1 = self.model.run_decoder(last_emitted_token_for_decoder, state0, state1)

        current_t = 0
        loop_start_time = time.time()
        max_steps = max_encoder_t * 2  # Safety break for potential infinite loops
        steps_taken = 0

        logger.info(f"Starting TDT decoding loop for {max_encoder_t} encoder frames...")
        while current_t < max_encoder_t and steps_taken < max_steps:
            steps_taken += 1
            # Prepare encoder output for the current time step `t`
            # Joiner expects [B, T=1, C] based on logs/typical RNNT
            encoder_out_t = encoder_out[:, :, current_t : current_t + 1]

            # Run Joiner
            # Output shape: [1, 1, vocab_size + num_durations]
            joiner_logits = self.model.run_joiner(encoder_out_t, decoder_out)
            joiner_logits = joiner_logits.squeeze()  # -> [vocab + durations]

            # Split logits into token and duration parts
            token_logits = joiner_logits[: self.blank_id + 1]
            duration_logits = joiner_logits[self.blank_id + 1 :]

            # Argmax for token prediction
            predicted_token_idx = np.argmax(token_logits)
            predicted_duration_bin_idx = np.argmax(duration_logits)
            predicted_skip_amount = self.tdt_durations[predicted_duration_bin_idx]

            # Debugging log (optional)
            # predicted_token_char = self.idx2token.get(int(predicted_token_idx), f"UNK({predicted_token_idx})")
            # logger.trace(
            #     (
            #         f"t={current_t:04d}: Pred='{predicted_token_char}' ({predicted_token_idx}), "
            #         f"DurBin={predicted_duration_bin_idx}, Skip={predicted_skip_amount}"
            #     )
            # )

            # If a non-blank token is predicted:
            if predicted_token_idx != self.blank_id:
                # print(f"  --> Emitting: {predicted_token_char} ({predicted_token_idx})")
                predicted_token_ids.append(int(predicted_token_idx))
                last_emitted_token_for_decoder = int(predicted_token_idx)

                # Update decoder state and output for the *next* step
                state0 = next_state0
                state1 = next_state1
                decoder_out, next_state0, next_state1 = self.model.run_decoder(
                    last_emitted_token_for_decoder, state0, state1
                )
            # else: # Blank predicted, keep decoder state and output as is
            #     print("  --> Blank")
            #     pass

            # Advance time step based on predicted duration
            current_t += predicted_skip_amount

        loop_end_time = time.time()
        logger.info(f"TDT decoding loop finished in {loop_end_time - loop_start_time:.2f}s ({steps_taken} steps).")
        if steps_taken >= max_steps:
            logger.warning("Warning: TDT decoding loop hit maximum step limit. Result might be truncated.")

        return predicted_token_ids

    def _post_process_text(self, token_ids: list[int]) -> str:
        """
        Converts a list of token IDs into a human-readable string.

        Args:
            token_ids: List of predicted token IDs from the decoder.

        Returns:
            The final transcribed text.
        """
        if not token_ids:
            return ""

        # Convert IDs to tokens
        tokens_str_list = [self.idx2token.get(idx, "") for idx in token_ids]

        # Handle SentencePiece style joining (replace ' ' with space)
        underline = "▁"
        text = "".join(tokens_str_list).replace(underline, " ").strip()

        # Optional: Clean up multiple spaces if needed (though SentencePiece usually handles this)
        text = " ".join(text.split())

        return text

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribes an audio signal to text using the TDT model.

        Args:
            audio: Input audio signal as a numpy float32 array. Assumed mono and 16000Hz!

        Returns:
            str: Transcribed text.
        """
        start_time = time.time()
        audio_duration_sec = len(audio) / self.melspectrogram.sample_rate

        # 1. Preprocess audio -> Mel Spectrogram Features
        logger.info("Preprocessing audio...")
        preprocessing_start = time.time()
        features = self._process_audio(audio)
        preprocessing_end = time.time()
        logger.info(f"Preprocessing time: {preprocessing_end - preprocessing_start:.2f}s")

        # 2. Run Encoder
        logger.info("Running encoder...")
        encoder_start = time.time()
        encoder_out = self.model.run_encoder(features)
        encoder_end = time.time()
        logger.info(f"Encoder output shape: {encoder_out.shape}")  # [1, channels, time_reduced]
        logger.info(f"Encoder time: {encoder_end - encoder_start:.2f}s")

        # 3. Run TDT Decoding (Decoder + Joiner loop)
        logger.info("Running TDT decoding...")
        decoder_start = time.time()
        predicted_token_ids = self._decode_tdt(encoder_out)
        decoder_end = time.time()
        logger.info(f"Decoder time: {decoder_end - decoder_start:.2f}s")

        # 4. Post-process token IDs to text
        logger.info("Post-processing text...")
        text = self._post_process_text(predicted_token_ids)
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"Total processing time: {total_time:.2f}s (Audio duration: {audio_duration_sec:.2f}s)")
        logger.info(f"Transcribed text: '{text}'")

        return text

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
            logger.info(f"Loading audio file: {audio_path}")
            # Load as float32, assume mono (take first channel if stereo)
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # Select first channel
            logger.info(f"Audio loaded: duration={len(audio) / sr:.2f}s, sample_rate={sr}Hz")
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
        """Clean up internal ONNX model resources."""
        if hasattr(self, "model"):
            del self.model
