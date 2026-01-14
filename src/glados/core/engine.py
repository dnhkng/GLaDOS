"""
Core engine module for the Glados voice assistant.

This module provides the main orchestration classes including the Glados assistant,
configuration management, and component coordination.
"""

from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, HttpUrl
import yaml

from ..ASR import TranscriberProtocol, get_audio_transcriber
from ..audio_io import AudioProtocol, get_audio_system
from ..TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from .audio_data import AudioMessage
from .llm_processor import LanguageModelProcessor
from .speech_listener import SpeechListener
from .speech_player import SpeechPlayer
from .tool_executor import ToolExecutor
from .tts_synthesizer import TextToSpeechSynthesizer

logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")


class PersonalityPrompt(BaseModel):
    """
    Represents a single personality prompt message for the assistant.

    Contains exactly one of: system, user, or assistant message content.
    Used to configure the assistant's personality and behavior.
    """

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
        fields = self.model_dump(exclude_none=True)
        if len(fields) != 1:
            raise ValueError("PersonalityPrompt must have exactly one non-null field")

        field, value = next(iter(fields.items()))
        return {"role": field, "content": value}


class GladosConfig(BaseModel):
    """
    Configuration model for the Glados voice assistant.

    Defines all necessary parameters for initializing the assistant including
    LLM settings, audio I/O backend, ASR/TTS engines, and personality configuration.
    Supports loading from YAML files with nested key navigation.
    """

    llm_model: str
    completion_url: HttpUrl
    api_key: str | None
    interruptible: bool
    audio_io: str
    asr_engine: str
    wake_word: str | None
    voice: str
    announcement: str | None
    personality_preprompt: list[PersonalityPrompt]
    slow_clap_audio_path: str = "data/slow-clap.mp3"
    tool_timeout: float = 30.0

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
                    raise ValueError(f"Could not decode YAML file {path} with any supported encoding")

        # Navigate through nested keys
        config = data
        for key in key_to_config:
            config = config[key]

        return cls.model_validate(config)

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        return [prompt.to_chat_message() for prompt in self.personality_preprompt]


class Glados:
    """
    Glados voice assistant orchestrator.
    This class manages the components of the Glados voice assistant, including speech recognition,
    language model processing, text-to-speech synthesis, and audio playback.
    It initializes the necessary components, starts background threads for processing, and provides
    methods for interaction with the assistant.
    """

    PAUSE_TIME: float = 0.05  # Time to wait between processing loops
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
        tts_model: SpeechSynthesizerProtocol,
        audio_io: AudioProtocol,
        completion_url: HttpUrl,
        llm_model: str,
        api_key: str | None = None,
        interruptible: bool = True,
        wake_word: str | None = None,
        announcement: str | None = None,
        personality_preprompt: tuple[dict[str, str], ...] = DEFAULT_PERSONALITY_PREPROMPT,
        tool_config: dict[str, Any] | None = None,
        tool_timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Glados voice assistant with configuration parameters.

        This method sets up the voice recognition system, including voice activity detection (VAD),
        automatic speech recognition (ASR), text-to-speech (TTS), and language model processing.
        The initialization configures various components and starts background threads for
        processing LLM responses and TTS output.

        Args:
            asr_model (TranscriberProtocol): The ASR model for transcribing audio input.
            tts_model (SpeechSynthesizerProtocol): The TTS model for synthesizing spoken output.
            audio_io (AudioProtocol): The audio input/output system to use.
            completion_url (HttpUrl): The URL for the LLM completion endpoint.
            llm_model (str): The name of the LLM model to use.
            api_key (str | None): API key for accessing the LLM service, if required.
            interruptible (bool): Whether the assistant can be interrupted while speaking.
            wake_word (str | None): Optional wake word to trigger the assistant.
            announcement (str | None): Optional announcement to play on startup.
            personality_preprompt (tuple[dict[str, str], ...]): Initial personality preprompt messages.
            tool_config (dict[str, Any] | None): Configuration for tools (e.g., audio paths).
            tool_timeout (float): Timeout in seconds for tool execution.
        """
        self._asr_model = asr_model
        self._tts = tts_model
        self.completion_url = completion_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.interruptible = interruptible
        self.wake_word = wake_word
        self.announcement = announcement
        self.tool_config = tool_config or {}
        self.tool_timeout = tool_timeout
        self._messages: list[dict[str, str]] = list(personality_preprompt)

        # Initialize spoken text converter, that converts text to spoken text. eg. 12 -> "twelve"
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model, this is needed to avoid long pauses on first request
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Initialize events for thread synchronization
        self.processing_active_event = threading.Event()  # Indicates if input processing is active (ASR + LLM + TTS)
        self.currently_speaking_event = threading.Event()  # Indicates if the assistant is currently speaking
        self.shutdown_event = threading.Event()  # Event to signal shutdown of all threads

        # Initialize queues for inter-thread communication
        self.llm_queue: queue.Queue[dict[str, Any]] = queue.Queue()  # Data from SpeechListener and ToolExecutor to LLMProcessor
        self.tool_calls_queue: queue.Queue[dict[str, Any]] = queue.Queue()  # Tool calls from LLMProcessor to ToolExecutor
        self.tts_queue: queue.Queue[str] = queue.Queue()  # Text from LLMProcessor to TTSynthesizer
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()  # AudioMessages from TTSSynthesizer to AudioPlayer

        # Initialize audio input/output system
        self.audio_io: AudioProtocol = audio_io
        logger.info("Audio input started successfully.")

        # Initialize threads for each component
        self.component_threads: list[threading.Thread] = []

        self.speech_listener = SpeechListener(
            audio_io=self.audio_io,
            llm_queue=self.llm_queue,
            asr_model=self._asr_model,
            wake_word=self.wake_word,
            interruptible=self.interruptible,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        self.llm_processor = LanguageModelProcessor(
            llm_input_queue=self.llm_queue,
            tool_calls_queue=self.tool_calls_queue,
            tts_input_queue=self.tts_queue,
            conversation_history=self._messages,  # Shared, to be refactored
            completion_url=self.completion_url,
            model_name=self.llm_model,
            api_key=self.api_key,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
        )

        self.tool_executor = ToolExecutor(
            llm_queue=self.llm_queue,
            tool_calls_queue=self.tool_calls_queue,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            tool_config=self.tool_config,
            tool_timeout=self.tool_timeout,
            pause_time=self.PAUSE_TIME,
        )

        self.tts_synthesizer = TextToSpeechSynthesizer(
            tts_input_queue=self.tts_queue,
            audio_output_queue=self.audio_queue,
            tts_model=self._tts,
            stc_instance=self._stc,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
        )

        self.speech_player = SpeechPlayer(
            audio_io=self.audio_io,
            audio_output_queue=self.audio_queue,
            conversation_history=self._messages,  # Shared, to be refactored
            tts_sample_rate=self._tts.sample_rate,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME,
        )

        thread_targets = {
            "SpeechListener": self.speech_listener.run,
            "LLMProcessor": self.llm_processor.run,
            "ToolExecutor": self.tool_executor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.speech_player.run,
        }

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")

    def play_announcement(self, interruptible: bool | None = None) -> None:
        """
        Play the announcement using text-to-speech (TTS) synthesis.

        This method checks if an announcement is set and, if so, places it in the TTS queue for processing.
        If the `interruptible` parameter is set to `True`, it allows the announcement to be interrupted by other
        audio playback. If `interruptible` is `None`, it defaults to the instance's `interruptible` setting.

        Args:
            interruptible (bool | None): Whether the announcement can be interrupted by other audio playback.
                If `None`, it defaults to the instance's `interruptible` setting.
        """

        if interruptible is None:
            interruptible = self.interruptible
        logger.success("Playing announcement...")
        if self.announcement:
            self.tts_queue.put(self.announcement)
            self.processing_active_event.set()

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

        tts_model: SpeechSynthesizerProtocol
        tts_model = get_speech_synthesizer(config.voice)

        audio_io = get_audio_system(backend_type=config.audio_io)

        return cls(
            asr_model=asr_model,
            tts_model=tts_model,
            audio_io=audio_io,
            completion_url=config.completion_url,
            llm_model=config.llm_model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=config.announcement,
            personality_preprompt=tuple(config.to_chat_messages()),
            tool_config={"slow_clap_audio_path": config.slow_clap_audio_path},
            tool_timeout=config.tool_timeout,
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
        This method initializes the audio input system, starts listening for audio samples,
        and enters a loop that waits for audio input until a shutdown event is triggered.
        It handles keyboard interrupts gracefully and ensures that all components are properly shut down.

        This method is the main entry point for running the Glados voice assistant.
        """
        self.audio_io.start_listening()

        logger.success("Audio Modules Operational")
        logger.success("Listening...")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while not self.shutdown_event.is_set():  # Check event BEFORE blocking get
                time.sleep(self.PAUSE_TIME)
            logger.info("Shutdown event detected in listen loop, exiting loop.")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt in main run loop.")
            # Make sure any ongoing audio playback is stopped
            if self.currently_speaking_event.is_set():
                for component in self.component_threads:
                    if component.name == "AudioPlayer":
                        self.audio_io.stop_speaking()
                        self.currently_speaking_event.clear()
                        break
            self.shutdown_event.set()
            # Give threads a moment to notice the shutdown event
            time.sleep(self.PAUSE_TIME)
        finally:
            logger.info("Listen event loop is stopping/exiting.")
            sys.exit(0)


if __name__ == "__main__":
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.run()
