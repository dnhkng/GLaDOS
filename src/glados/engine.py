from pathlib import Path
import queue
import sys
import threading

from Levenshtein import distance
from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, HttpUrl
import sounddevice as sd  # type: ignore
from sounddevice import CallbackFlags
import yaml

from .ASR import VAD, TranscriberProtocol, get_audio_transcriber
from .core.audio_message import AudioMessage
from .core.llm_processor import LanguageModelProcessor
from .core.tts_synthesizer import TextToSpeechSynthesizer
from .core.voice_player import VoicePlayer
from .TTS import tts_glados, tts_kokoro
from .utils import spoken_text_converter as stc
from .utils.resources import resource_path

logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")


class PersonalityPrompt(BaseModel):
    system: str | None = None
    user: str | None = None
    assistant: str | None = None

    def to_chat_message(self) -> dict[str, str]:
        """Convert the prompt to a chat message format.

        Returns:
            dict[str, str]: A single chat message dictionary

        Raises:
            ValueError: If the prompt does not contain exactly one non-null field
        """
        for field, value in self.model_dump(exclude_none=True).items():
            return {"role": field, "content": value}
        raise ValueError("PersonalityPrompt must have exactly one non-null field")


class GladosConfig(BaseModel):
    completion_url: HttpUrl
    model: str
    api_key: str | None = None
    interruptible: bool = True
    asr_engine: str = "ctc"
    wake_word: str | None = None
    voice: str
    announcement: str | None = None
    personality_preprompt: list[PersonalityPrompt]

    @classmethod
    def from_yaml(cls, path: str | Path, key_to_config: tuple[str, ...] = ("Glados",)) -> "GladosConfig":
        """
        Load a GladosConfig instance from a YAML configuration file.

        Parameters:
            path: Path to the YAML configuration file
            key_to_config: Tuple of keys to navigate nested configuration

        Returns:
            GladosConfig: Configuration object with validated settings

        Raises:
            ValueError: If the YAML content is invalid
            OSError: If the file cannot be read
            pydantic.ValidationError: If the configuration is invalid
        """
        path = Path(path)

        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig"]:
            try:
                data = yaml.safe_load(path.read_text(encoding=encoding))
                break
            except UnicodeDecodeError:
                if encoding == "utf-8-sig":
                    raise

        # Navigate through nested keys
        config = data
        for key in key_to_config:
            config = config[key]

        return cls.model_validate(config)

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]


class Glados:
    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection
    BUFFER_SIZE: int = 800  # Milliseconds of buffer BEFORE VAD detection
    PAUSE_LIMIT: int = 640  # Milliseconds of pause allowed before processing
    SIMILARITY_THRESHOLD: int = 2  # Threshold for wake word similarity
    PUNCTUATION_SET: tuple[str, ...] = (".", "!", "?", ":", ";", "?!", "\n", "\n\n")  # Sentence splitting punctuation
    NEUROTOXIN_RELEASE_ALLOWED: bool = False  # preparation for function calling, see issue #13
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: TranscriberProtocol,
        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer,
        vad_model: VAD,
        completion_url: HttpUrl,
        model: str,
        api_key: str | None = None,
        interruptible: bool = True,
        wake_word: str | None = None,
        personality_preprompt: tuple[dict[str, str], ...] = DEFAULT_PERSONALITY_PREPROMPT,
        announcement: str | None = None,
    ) -> None:
        """
        Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Args:
            voice_model (str): Path to the voice model for text-to-speech synthesis.
            speaker_id (int | None): Identifier for the specific speaker voice, if applicable.
            completion_url (str): URL endpoint for language model completions.
            model (str): Identifier for the language model being used.
            api_key (str | None, optional): Authentication key for the language model API. Defaults to None.
            wake_word (str | None, optional): Activation word to trigger voice assistant. Defaults to None.
            personality_preprompt (list[dict[str, str]], optional): Initial context or personality
                configuration for the language model. Defaults to DEFAULT_PERSONALITY_PREPROMPT.
            announcement (str | None, optional): Initial announcement to be spoken upon initialization.
                Defaults to None.
            interruptible (bool, optional): Whether the assistant's speech can be interrupted.
                Defaults to True.
        """
        self.completion_url = completion_url
        self.model = model
        self.api_key = api_key
        self.wake_word = wake_word
        self.announcement = announcement
        self._vad_model = vad_model
        self._tts = tts_model
        self._asr_model = asr_model
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Initialize sample queues and state flags
        self._samples: list[NDArray[np.float32]] = []
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self._buffer: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=self.BUFFER_SIZE // self.VAD_SIZE)
        self._recording_started = False
        self._gap_counter = 0

        self._messages: list[dict[str, str]] = list(personality_preprompt)

        self.processing_active_event = threading.Event()
        self.interruptible = interruptible

        self.currently_speaking_event = threading.Event()
        self.shutdown_event = threading.Event()

        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()

        self.component_threads: list[threading.Thread] = []

        logger.info("Orchestrator: Initializing LanguageModelProcessor...")
        self.llm_processor = LanguageModelProcessor(
            llm_input_queue=self.llm_queue,
            tts_input_queue=self.tts_queue,
            conversation_history=self._messages,  # Shared
            completion_url=self.completion_url,
            model_name=self.model,
            api_key=self.api_key,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
        )

        logger.info("Orchestrator: Initializing TextToSpeechSynthesizer...")
        self.tts_synthesizer = TextToSpeechSynthesizer(
            tts_input_queue=self.tts_queue,
            audio_output_queue=self.audio_queue,
            tts_model=self._tts,
            stc_instance=self._stc,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
        )

        logger.info("Orchestrator: Initializing AudioPlayer...")
        self.audio_player = VoicePlayer(
            audio_output_queue=self.audio_queue,
            conversation_history=self._messages,  # Shared
            tts_sample_rate=self._tts.sample_rate,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        thread_targets = {
            # "AudioInput": self.audio_input_handler.run,
            "LLMProcessor": self.llm_processor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.audio_player.run,
        }

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")

        def audio_callback_for_sd_input_stream(
            indata: np.dtype[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: CallbackFlags,
        ) -> None:
            """
            Callback function for processing audio input from a sounddevice input stream.

            This method is responsible for handling incoming audio samples, performing voice activity detection (VAD),
            and queuing the processed audio data for further analysis.

            Parameters:
                indata (np.ndarray): Input audio data from the sounddevice stream
                frames (int): Number of audio frames in the current chunk
                time (sd.CallbackStop): Timing information for the audio callback
                status (CallbackFlags): Status flags for the audio callback

            Returns:
                None

            Notes:
                - Copies and squeezes the input data to ensure single-channel processing
                - Applies voice activity detection to determine speech presence
                - Puts processed audio samples and VAD confidence into a thread-safe queue
            """

            data = np.array(indata).copy().squeeze()  # Reduce to single channel if necessary
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.VAD_THRESHOLD
            self._sample_queue.put((data, bool(vad_confidence)))

        self.input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=audio_callback_for_sd_input_stream,
            blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
        )

    def play_announcement(self, interruptible: bool | None = None) -> None:
        """
        Play the announcement using text-to-speech (TTS) synthesis.

        Parameters:
            interruptible (bool | None): Whether the announcement can be interrupted. If None, uses the instance's
                interruptible setting.
        """
        if interruptible is None:
            interruptible = self.interruptible
        logger.success("Playing announcement...")
        if self.announcement:
            audio = self._tts.generate_speech_audio(self.announcement)
            logger.success(f"TTS text: {self.announcement}")
            sd.play(audio, self._tts.sample_rate)
            if not interruptible:
                sd.wait()

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Retrieve the current list of conversation messages.

        Returns:
            list[dict[str, str]]: A list of message dictionaries representing the conversation history.
        """
        return self._messages

    @classmethod
    def from_config(cls, config: GladosConfig) -> "Glados":
        """
        Create a Glados instance from a GladosConfig configuration object.

        Parameters:
            config (GladosConfig): Configuration object containing Glados initialization parameters

        Returns:
            Glados: A new Glados instance configured with the provided settings
        """

        asr_model = get_audio_transcriber(
            engine_type=config.asr_engine,
        )

        vad_model = VAD()

        tts_model: tts_glados.Synthesizer | tts_kokoro.Synthesizer
        if config.voice == "glados":
            tts_model = tts_glados.Synthesizer()
        else:
            assert config.voice in tts_kokoro.get_voices(), f"Voice '{config.voice}' not available"
            tts_model = tts_kokoro.Synthesizer(voice=config.voice)

        return cls(
            asr_model=asr_model,
            tts_model=tts_model,
            vad_model=vad_model,
            completion_url=config.completion_url,
            model=config.model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=config.announcement,
            personality_preprompt=tuple(config.to_chat_messages()),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Glados":
        """
        Create a Glados instance from a configuration file.

        Parameters:
            path (str): Path to the YAML configuration file containing Glados settings.

        Returns:
            Glados: A new Glados instance configured with settings from the specified YAML file.

        Example:
            glados = Glados.from_yaml('config/default.yaml')
        """
        return cls.from_config(GladosConfig.from_yaml(path))

    def start_listen_event_loop(self) -> None:
        """
        Start the voice assistant's listening event loop, continuously processing audio input.

        This method initializes the audio input stream and enters an infinite loop to handle incoming audio samples.
        The loop retrieves audio samples and their voice activity detection (VAD) confidence from a queue and processes
        each sample using the `_handle_audio_sample` method.

        Behavior:
        - Starts the audio input stream
        - Logs successful initialization of audio modules
        - Enters an infinite listening loop
        - Retrieves audio samples from a queue
        - Processes each audio sample with VAD confidence
        - Handles keyboard interrupts by stopping the input stream and setting a shutdown event

        Raises:
            KeyboardInterrupt: Allows graceful termination of the listening loop
        """
        self.input_stream.start()
        logger.success("Audio Modules Operational")
        logger.success("Listening...")
        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                try:
                    # Use a timeout for the queue get
                    sample, vad_confidence = self._sample_queue.get(timeout=self.PAUSE_TIME)
                    self._handle_audio_sample(sample, vad_confidence)
                except queue.Empty:
                    # Timeout occurred, loop again to check shutdown_event
                    continue
                except Exception as e:  # Catch other potential errors during get or handle
                    if not self.shutdown_event.is_set():  # Only log if not shutting down
                        logger.error(f"Error in listen loop: {e}")
                    continue

            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in listen loop.")
        finally:
            logger.info("Listen event loop is stopping/exiting.")
            self.stop_listen_event_loop()

    def stop_listen_event_loop(self) -> None:
        """
        Stop the voice assistant's listening event loop and clean up resources.

        This method stops the audio input stream and sets the shutdown event to terminate
        any ongoing processing threads. It ensures that all resources are released properly.

        Raises:
            No explicit exceptions raised; handles cleanup gracefully.
        """
        logger.info("Setting Shutdown event")
        self.shutdown_event.set()  # Set shutdown event first

        logger.info("Calling global sd.stop() to halt all sounddevice activity.")
        sd.stop()

        logger.info("Glados engine stop sequence initiated. Threads should terminate.")

    def _handle_audio_sample(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Handles the processing of each audio sample.

        If the recording has not started, the sample is added to the circular buffer.

        If the recording has started, the sample is added to the samples list, and the pause
        limit is checked to determine when to process the detected audio.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Manages the pre-activation audio buffer and handles voice activity detection.

        This method maintains a circular buffer of audio samples before voice activation,
        discarding the oldest sample when the buffer is full. When voice activity is detected,
        it stops the audio stream and prepares for audio processing.

        Args:
            sample (np.ndarray): The current audio sample to be added to the buffer.
            vad_confidence (bool): Indicates whether voice activity is detected in the sample.

        Side Effects:
            - Modifies the internal circular buffer
            - Stops the audio stream when voice is detected
            - Disables processing on LLM and TTS threads
            - Prepares samples for recording when voice is detected
        """
        if self._buffer.full():
            self._buffer.get()  # Discard the oldest sample to make room for new ones
        self._buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            if not self.interruptible and self.currently_speaking_event.is_set():
                logger.info("Interruption is disabled, and the assistant is currently speaking, ignoring new input.")
                return

            sd.stop()  # Stop the audio stream to prevent overlap
            self.processing_active_event.clear()  # Turns off processing on threads for the LLM and TTS!!!
            self._samples = list(self._buffer.queue)
            self._recording_started = True

    def _process_activated_audio(self, sample: NDArray[np.float32], vad_confidence: bool) -> None:
        """
        Process audio samples, tracking speech pauses to capture complete utterances.

        This method accumulates audio samples and monitors voice activity detection (VAD) confidence to determine
        when a complete speech segment has been captured. It appends incoming samples to the internal buffer and
        tracks silent gaps to trigger audio processing.

        Parameters:
            sample (np.ndarray): A single audio sample from the input stream
            vad_confidence (bool): Indicates whether voice activity is currently detected

        Side Effects:
            - Appends audio samples to self._samples
            - Increments or resets self._gap_counter
            - Triggers audio processing via self._process_detected_audio() when pause limit is reached
        """

        self._samples.append(sample)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Check if the detected text contains a close match to the wake word using Levenshtein distance.

        This method helps handle variations in wake word detection by calculating the minimum edit distance
        between detected words and the configured wake word. It accounts for potential misheard
        variations during speech recognition.

        Parameters:
            text (str): The transcribed text to check for wake word similarity

        Returns:
            bool: True if a word in the text is sufficiently similar to the wake word, False otherwise

        Raises:
            AssertionError: If the wake word is not configured (None)

        Notes:
            - Uses Levenshtein distance to measure text similarity
            - Compares each word in the text against the wake word
            - Considers a match if the distance is below a predefined similarity threshold
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min([distance(word.lower(), self.wake_word) for word in words])
        return bool(closest_distance < self.SIMILARITY_THRESHOLD)

    def reset(self) -> None:
        """
        Reset the voice recording state and clear all audio buffers.

        This method performs the following actions:
        - Logs a debug message indicating the reset process
        - Stops the current recording by setting `_recording_started` to False
        - Clears the collected audio samples
        - Resets the gap counter used for detecting speech pauses
        - Empties the thread-safe audio buffer queue

        Note:
            Uses a mutex lock to safely clear the shared buffer queue to prevent
            potential race conditions in multi-threaded audio processing.
        """
        logger.debug("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def _process_detected_audio(self) -> None:
        """
        Process detected audio and generate a response after speech pause.

        Transcribes audio samples and handles wake word detection and LLM processing. Manages the
        audio input stream and processing state throughout the interaction.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Stops the input audio stream
            - Performs automatic speech recognition (ASR)
            - Potentially sends text to LLM queue
            - Resets audio recording state
            - Restarts input audio stream

        Raises:
            No explicit exceptions raised
        """
        logger.debug("Detected pause after speech. Processing...")

        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                self.llm_queue.put(detected_text)
                self.processing_active_event.set()
                self.currently_speaking_event.set()

        self.reset()

    def asr(self, samples: list[NDArray[np.float32]]) -> str:
        """
        Perform automatic speech recognition (ASR) on the provided audio samples.

        Parameters:
            samples (list[np.dtype[np.float32]]): A list of numpy arrays containing audio samples to be transcribed.

        Returns:
            str: The transcribed text from the input audio samples.

        Notes:
            - Concatenates multiple audio samples into a single continuous audio array
            - Uses the pre-configured ASR model to transcribe the audio
        """
        audio = np.concatenate(samples)

        # Normalize audio to [-0.5, 0.5] range to prevent clipping and ensure consistent levels
        audio = audio / np.max(np.abs(audio)) / 2

        detected_text = self._asr_model.transcribe(audio)
        return detected_text


def start() -> None:
    """Set up the LLM server and start GlaDOS.

    This function reads the configuration file, initializes the Glados voice assistant,
    and starts the listening event loop.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there is an error parsing the YAML configuration file.
    """
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.play_announcement()
    glados.start_listen_event_loop()


if __name__ == "__main__":
    start()
