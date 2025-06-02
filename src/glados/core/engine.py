from pathlib import Path
import queue
import sys
import threading
import time

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, HttpUrl
import sounddevice as sd  # type: ignore
import yaml

from ..ASR import VAD, TranscriberProtocol, get_audio_transcriber
from ..audio_io.sounddevice_io import SoundDeviceAudioIO
from ..TTS import tts_glados, tts_kokoro
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from .audio_data import AudioMessage
from .llm_processor import LanguageModelProcessor
from .speech_listener import SpeechListener
from .speech_player import SpeechPlayer
from .tts_synthesizer import TextToSpeechSynthesizer

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


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
        # self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
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

        # Initialize audio I/O
        self.audio_io = SoundDeviceAudioIO(
            sample_rate=self.SAMPLE_RATE,
            vad_size=self.VAD_SIZE,
            vad_model=self._vad_model,
            vad_threshold=self.VAD_THRESHOLD,
        )
        self._sample_queue = self.audio_io._get_sample_queue()
        logger.info("Audio input started successfully.")

        self.component_threads: list[threading.Thread] = []

        logger.info("Orchestrator: Initializing SpeechListener...")
        self.speech_listener = SpeechListener(
            audio_io=self.audio_io,
            llm_queue=self.llm_queue,
            asr_model=self._asr_model,
            wake_word=self.wake_word,
            interruptible=self.interruptible,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
        )

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
        self.audio_player = SpeechPlayer(
            audio_io=self.audio_io,
            audio_output_queue=self.audio_queue,
            conversation_history=self._messages,  # Shared
            tts_sample_rate=self._tts.sample_rate,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        # logger.info("Orchestrator: Initializing Speech Input Handler...")
        # self.speech_

        thread_targets = {
            "SpeechListener": self.speech_listener.run,
            "LLMProcessor": self.llm_processor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.audio_player.run,
        }

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")

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

    def run(self) -> None:
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

        # self.input_stream.start()
        self.audio_io.start_listening()

        logger.success("Audio Modules Operational")
        logger.success("Listening...")

        logger.info(f"Shutdown event: {self.shutdown_event.is_set()}")
        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                time.sleep(self.PAUSE_TIME)
            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in listen loop.")
            self.shutdown_event.set()
        finally:
            logger.info("Listen event loop is stopping/exiting.")


if __name__ == "__main__":
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)

    glados.run()
