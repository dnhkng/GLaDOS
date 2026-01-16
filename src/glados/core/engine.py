"""
Core engine module for the Glados voice assistant.

This module provides the main orchestration classes including the Glados assistant,
configuration management, and component coordination.
"""

from dataclasses import dataclass
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel, HttpUrl
import yaml

from ..ASR import TranscriberProtocol, get_audio_transcriber
from ..audio_io import AudioProtocol, get_audio_system
from ..TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from ..autonomy import AutonomyConfig, AutonomyLoop, EventBus, InteractionState, TaskManager, TaskSlotStore
from ..autonomy.events import TimeTickEvent
from ..autonomy.jobs import BackgroundJobScheduler, build_jobs
from ..mcp import MCPManager, MCPServerConfig
from ..observability import MindRegistry, ObservabilityBus, trim_message
from ..vision import VisionConfig, VisionState
from ..vision.constants import SYSTEM_PROMPT_VISION_HANDLING
from .audio_data import AudioMessage
from .audio_state import AudioState
from .knowledge_store import KnowledgeStore
from .llm_processor import LanguageModelProcessor
from .speech_listener import SpeechListener
from .speech_player import SpeechPlayer
from .text_listener import TextListener
from .tool_executor import ToolExecutor
from .tts_synthesizer import TextToSpeechSynthesizer

logger.remove(0)
logger.add(sys.stderr, level="SUCCESS")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    handler: Callable[[list[str]], str]
    usage: str | None = None
    aliases: tuple[str, ...] = ()


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
    input_mode: Literal["audio", "text", "both"] = "audio"
    tts_enabled: bool = True
    asr_muted: bool = False
    asr_engine: str
    wake_word: str | None
    voice: str
    announcement: str | None
    llm_headers: dict[str, str] | None = None
    personality_preprompt: list[PersonalityPrompt]
    slow_clap_audio_path: str = "data/slow-clap.mp3"
    tool_timeout: float = 30.0
    vision: VisionConfig | None = None
    autonomy: AutonomyConfig | None = None
    mcp_servers: list[MCPServerConfig] | None = None

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
        vision_config: VisionConfig | None = None,
        autonomy_config: AutonomyConfig | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        input_mode: Literal["audio", "text", "both"] = "audio",
        tts_enabled: bool = True,
        asr_muted: bool = False,
        llm_headers: dict[str, str] | None = None,
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
            vision_config (VisionConfig | None): Optional vision configuration.
            autonomy_config (AutonomyConfig | None): Optional autonomy configuration.
            mcp_servers (list[MCPServerConfig] | None): Optional MCP server configurations.
            tts_enabled (bool): Whether TTS audio output is enabled at startup.
            asr_muted (bool): Whether ASR starts muted.
            llm_headers (dict[str, str] | None): Extra headers for LLM requests.
        """
        self._asr_model = asr_model
        self._tts = tts_model
        self.input_mode = input_mode
        self.completion_url = completion_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.interruptible = interruptible
        self.wake_word = wake_word
        self.announcement = announcement
        self.tool_config = tool_config or {}
        self.tool_timeout = tool_timeout
        self.mcp_servers = mcp_servers or []
        self._messages: list[dict[str, Any]] = list(personality_preprompt)
        self.vision_config = vision_config
        self.autonomy_config = autonomy_config or AutonomyConfig()
        self.vision_state: VisionState | None = VisionState() if self.vision_config else None
        self.vision_request_queue: queue.Queue | None = queue.Queue() if self.vision_config else None
        self.autonomy_event_bus: EventBus | None = None
        self.autonomy_loop: AutonomyLoop | None = None
        self.autonomy_slots: TaskSlotStore | None = None
        self.autonomy_tasks: TaskManager | None = None
        self.autonomy_job_runner: BackgroundJobScheduler | None = None
        self.observability_bus = ObservabilityBus()
        self.mind_registry = MindRegistry()
        self.interaction_state = InteractionState()
        self.asr_muted_event = threading.Event()
        if asr_muted:
            self.asr_muted_event.set()
        self.tts_muted_event = threading.Event()
        if not tts_enabled:
            self.tts_muted_event.set()
        self.audio_state = AudioState()
        self.knowledge_store = KnowledgeStore(resource_path("data/knowledge.json"))
        self._command_registry, self._command_order = self._build_command_registry()
        # Initialize events for thread synchronization
        self.processing_active_event = threading.Event()  # Indicates if input processing is active (ASR + LLM + TTS + VLM)
        self.currently_speaking_event = threading.Event()  # Indicates if the assistant is currently speaking
        self.shutdown_event = threading.Event()  # Event to signal shutdown of all threads
        if self.autonomy_config.enabled:
            self.autonomy_event_bus = EventBus()
            self.autonomy_slots = TaskSlotStore(observability_bus=self.observability_bus)
            self.autonomy_tasks = TaskManager(self.autonomy_slots, self.autonomy_event_bus)
            if self.autonomy_config.jobs.enabled:
                jobs = build_jobs(self.autonomy_config.jobs, observability_bus=self.observability_bus)
                if jobs:
                    self.autonomy_job_runner = BackgroundJobScheduler(
                        jobs=jobs,
                        task_manager=self.autonomy_tasks,
                        shutdown_event=self.shutdown_event,
                        observability_bus=self.observability_bus,
                        poll_interval_s=self.autonomy_config.jobs.poll_interval_s,
                    )

        if self.vision_config:
            # Add instructions to system prompt to correctly handle [vision] marked messages
            for message in self._messages:
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    message["content"] = f"{message['content']} {SYSTEM_PROMPT_VISION_HANDLING}"
                    break
            else:
                self._messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_VISION_HANDLING})


        # Initialize spoken text converter, that converts text to spoken text. eg. 12 -> "twelve"
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model, this is needed to avoid long pauses on first request
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Initialize queues for inter-thread communication
        self.llm_queue: queue.Queue[dict[str, Any]] = queue.Queue()  # Data from SpeechListener and ToolExecutor to LLMProcessor
        self.tool_calls_queue: queue.Queue[dict[str, Any]] = queue.Queue()  # Tool calls from LLMProcessor to ToolExecutor
        self.tts_queue: queue.Queue[str] = queue.Queue()  # Text from LLMProcessor to TTSynthesizer
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()  # AudioMessages from TTSSynthesizer to AudioPlayer

        self.mcp_manager: MCPManager | None = None
        if self.mcp_servers:
            self.mcp_manager = MCPManager(
                self.mcp_servers,
                tool_timeout=self.tool_timeout,
                observability_bus=self.observability_bus,
            )
            self.mcp_manager.start()

        # Initialize audio input/output system
        self.audio_io: AudioProtocol = audio_io
        logger.info("Audio input started successfully.")

        # Initialize threads for each component
        self.component_threads: list[threading.Thread] = []

        self.speech_listener: SpeechListener | None = None
        self.text_listener: TextListener | None = None
        if self.input_mode in {"audio", "both"}:
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
                interaction_state=self.interaction_state,
                observability_bus=self.observability_bus,
                asr_muted_event=self.asr_muted_event,
                audio_state=self.audio_state,
            )
        if self.input_mode in {"text", "both"}:
            if self.input_mode == "text":
                logger.info("Text input mode enabled. ASR is disabled.")
            self.text_listener = TextListener(
                llm_queue=self.llm_queue,
                processing_active_event=self.processing_active_event,
                shutdown_event=self.shutdown_event,
                pause_time=self.PAUSE_TIME,
                interaction_state=self.interaction_state,
                observability_bus=self.observability_bus,
                command_handler=self.handle_command,
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
            vision_state=self.vision_state,
            slot_store=self.autonomy_slots,
            autonomy_system_prompt=self.autonomy_config.system_prompt if self.autonomy_config.enabled else None,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
            extra_headers=llm_headers,
        )

        self.tool_executor = ToolExecutor(
            llm_queue=self.llm_queue,
            tool_calls_queue=self.tool_calls_queue,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            tool_config={
                **self.tool_config,
                "vision_request_queue": self.vision_request_queue,
                "vision_tool_timeout": self.tool_timeout,
                "tts_queue": self.tts_queue,
            },
            tool_timeout=self.tool_timeout,
            pause_time=self.PAUSE_TIME,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
        )

        self.tts_synthesizer = TextToSpeechSynthesizer(
            tts_input_queue=self.tts_queue,
            audio_output_queue=self.audio_queue,
            tts_model=self._tts,
            stc_instance=self._stc,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
            tts_muted_event=self.tts_muted_event,
            observability_bus=self.observability_bus,
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
            tts_muted_event=self.tts_muted_event,
            interaction_state=self.interaction_state,
            observability_bus=self.observability_bus,
        )

        self.vision_processor = None
        if self.vision_config:
            from ..vision import VisionProcessor
            self.vision_processor = VisionProcessor(
                vision_state=self.vision_state,
                processing_active_event=self.processing_active_event,
                shutdown_event=self.shutdown_event,
                config=self.vision_config,
                request_queue=self.vision_request_queue,
                event_bus=self.autonomy_event_bus,
                observability_bus=self.observability_bus,
            )

        self.autonomy_ticker_thread: threading.Thread | None = None
        if self.autonomy_config.enabled:
            assert self.autonomy_event_bus is not None
            assert self.autonomy_slots is not None
            self.autonomy_loop = AutonomyLoop(
                config=self.autonomy_config,
                event_bus=self.autonomy_event_bus,
                interaction_state=self.interaction_state,
                vision_state=self.vision_state,
                slot_store=self.autonomy_slots,
                llm_queue=self.llm_queue,
                processing_active_event=self.processing_active_event,
                currently_speaking_event=self.currently_speaking_event,
                shutdown_event=self.shutdown_event,
                observability_bus=self.observability_bus,
                pause_time=self.PAUSE_TIME,
            )
            if not self.vision_config:
                self.autonomy_ticker_thread = threading.Thread(
                    target=self._run_autonomy_ticker,
                    name="AutonomyTicker",
                    daemon=True,
                )

        thread_targets = {
            "LLMProcessor": self.llm_processor.run,
            "ToolExecutor": self.tool_executor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.speech_player.run,
        }
        if self.speech_listener:
            thread_targets["SpeechListener"] = self.speech_listener.run
        if self.text_listener:
            thread_targets["TextListener"] = self.text_listener.run
        if self.autonomy_loop:
            thread_targets["AutonomyLoop"] = self.autonomy_loop.run
        if self.vision_processor:
            thread_targets["VisionProcessor"] = self.vision_processor.run
        if self.autonomy_job_runner:
            thread_targets["BackgroundJobs"] = self.autonomy_job_runner.run
        if self.autonomy_ticker_thread:
            self.component_threads.append(self.autonomy_ticker_thread)
            self.autonomy_ticker_thread.start()
            logger.info("Orchestrator: AutonomyTicker thread started.")
            self.mind_registry.register(
                "AutonomyTicker",
                title="Autonomy Ticker",
                status="running",
                summary="Periodic autonomy ticks",
            )

        for name in thread_targets:
            self.mind_registry.register(name, title=name, status="starting", summary="Initializing")

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")
            self.mind_registry.update(name, "running", summary="Thread active")

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
    def messages(self) -> list[dict[str, Any]]:
        """
        Retrieve the current list of conversation messages.

        Returns:
            list[dict[str, Any]]: A list of message dictionaries representing the conversation history.
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
            vision_config=config.vision,
            autonomy_config=config.autonomy,
            mcp_servers=config.mcp_servers,
            input_mode=config.input_mode,
            tts_enabled=config.tts_enabled,
            asr_muted=config.asr_muted,
            llm_headers=config.llm_headers,
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
        if self.input_mode in {"audio", "both"}:
            self.audio_io.start_listening()
        else:
            logger.info("Text input mode active. Audio input is disabled.")

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
            for component in self.component_threads:
                self.mind_registry.update(component.name, "stopped", summary="Shutdown")
            if self.autonomy_ticker_thread:
                self.mind_registry.update("AutonomyTicker", "stopped", summary="Shutdown")
            if self.autonomy_tasks:
                self.autonomy_tasks.shutdown()
            if self.mcp_manager:
                self.mcp_manager.shutdown()
            sys.exit(0)

    def set_asr_muted(self, muted: bool) -> None:
        if muted:
            self.asr_muted_event.set()
        else:
            self.asr_muted_event.clear()
        if self.speech_listener:
            self.speech_listener.reset()
        self.audio_state.reset()
        if self.observability_bus:
            state = "muted" if muted else "unmuted"
            self.observability_bus.emit(
                source="asr",
                kind="mute",
                message=f"ASR {state}",
                meta={"muted": muted},
            )

    def toggle_asr_muted(self) -> bool:
        muted = not self.asr_muted_event.is_set()
        self.set_asr_muted(muted)
        return muted

    def set_tts_muted(self, muted: bool) -> None:
        if muted:
            self.tts_muted_event.set()
            self.audio_io.stop_speaking()
            self.currently_speaking_event.clear()
        else:
            self.tts_muted_event.clear()
        if self.observability_bus:
            state = "muted" if muted else "unmuted"
            self.observability_bus.emit(
                source="tts",
                kind="mute",
                message=f"TTS {state}",
                meta={"muted": muted},
            )

    def toggle_tts_muted(self) -> bool:
        muted = not self.tts_muted_event.is_set()
        self.set_tts_muted(muted)
        return muted

    def command_specs(self) -> list[CommandSpec]:
        return [self._command_registry[name] for name in self._command_order]

    def submit_text_input(self, text: str, source: str = "text") -> bool:
        text = text.strip()
        if not text:
            return False
        if self.observability_bus:
            self.observability_bus.emit(
                source=source,
                kind="user_input",
                message=trim_message(text),
            )
        self.llm_queue.put({"role": "user", "content": text})
        if self.interaction_state:
            self.interaction_state.mark_user()
        self.processing_active_event.set()
        return True

    def handle_command(self, command: str) -> str:
        text = command.strip()
        if not text:
            return "No command entered."
        if text.startswith("/"):
            text = text[1:]
        parts = text.split()
        if not parts:
            return "No command entered."
        cmd = parts[0].lower()
        args = parts[1:]
        spec = self._command_registry.get(cmd)
        if not spec:
            return f"Unknown command: /{cmd}. Try /help."
        return spec.handler(args)

    def _build_command_registry(self) -> tuple[dict[str, CommandSpec], list[str]]:
        registry: dict[str, CommandSpec] = {}
        order: list[str] = []

        def register(spec: CommandSpec) -> None:
            registry[spec.name] = spec
            order.append(spec.name)
            for alias in spec.aliases:
                registry[alias] = spec

        register(
            CommandSpec(
                name="help",
                description="Show available commands",
                usage="/help",
                handler=self._cmd_help,
                aliases=("?",),
            )
        )
        register(
            CommandSpec(
                name="status",
                description="Show engine status",
                usage="/status",
                handler=self._cmd_status,
            )
        )
        register(
            CommandSpec(
                name="tts",
                description="Control TTS output",
                usage="/tts on|off|toggle",
                handler=self._cmd_tts,
            )
        )
        register(
            CommandSpec(
                name="mute-tts",
                description="Mute TTS output",
                usage="/mute-tts",
                handler=self._cmd_mute_tts,
                aliases=("tts-mute",),
            )
        )
        register(
            CommandSpec(
                name="unmute-tts",
                description="Unmute TTS output",
                usage="/unmute-tts",
                handler=self._cmd_unmute_tts,
                aliases=("tts-unmute",),
            )
        )
        register(
            CommandSpec(
                name="quit",
                description="Quit GLaDOS",
                usage="/quit",
                handler=self._cmd_quit,
                aliases=("exit",),
            )
        )
        register(
            CommandSpec(
                name="asr",
                description="Control ASR input",
                usage="/asr on|off|toggle",
                handler=self._cmd_asr,
            )
        )
        register(
            CommandSpec(
                name="mute-asr",
                description="Mute ASR input",
                usage="/mute-asr",
                handler=self._cmd_mute_asr,
                aliases=("asr-mute",),
            )
        )
        register(
            CommandSpec(
                name="unmute-asr",
                description="Unmute ASR input",
                usage="/unmute-asr",
                handler=self._cmd_unmute_asr,
                aliases=("asr-unmute",),
            )
        )
        register(
            CommandSpec(
                name="observe",
                description="Open observability screen (TUI)",
                usage="/observe",
                handler=self._cmd_observe,
                aliases=("observability",),
            )
        )
        register(
            CommandSpec(
                name="slots",
                description="Show autonomy slots",
                usage="/slots",
                handler=self._cmd_slots,
            )
        )
        register(
            CommandSpec(
                name="minds",
                description="Show active minds",
                usage="/minds",
                handler=self._cmd_minds,
            )
        )
        register(
            CommandSpec(
                name="vision",
                description="Show latest vision snapshot",
                usage="/vision",
                handler=self._cmd_vision,
            )
        )
        register(
            CommandSpec(
                name="config",
                description="Show config summary",
                usage="/config",
                handler=self._cmd_config,
            )
        )
        register(
            CommandSpec(
                name="knowledge",
                description="Manage local knowledge notes",
                usage="/knowledge add|list|set|delete|clear",
                handler=self._cmd_knowledge,
            )
        )
        return registry, order

    def _cmd_help(self, _args: list[str]) -> str:
        lines = ["Commands:"]
        for name in self._command_order:
            spec = self._command_registry[name]
            usage = spec.usage or f"/{spec.name}"
            lines.append(f"- {usage}: {spec.description}")
        return "\n".join(lines)

    def _cmd_status(self, _args: list[str]) -> str:
        autonomy_enabled = self.autonomy_config.enabled
        vision_enabled = self.vision_config is not None
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"asr_muted={self.asr_muted_event.is_set()}, "
            f"tts_muted={self.tts_muted_event.is_set()}, "
            f"autonomy_enabled={autonomy_enabled}, "
            f"vision_enabled={vision_enabled}, "
            f"jobs_enabled={jobs_enabled}"
        )

    def _cmd_quit(self, _args: list[str]) -> str:
        self.shutdown_event.set()
        return "Shutting down."

    def _cmd_asr(self, args: list[str]) -> str:
        if not args:
            return f"ASR is {'muted' if self.asr_muted_event.is_set() else 'active'}."
        arg = args[0].lower()
        if arg in {"on", "unmute", "active"}:
            self.set_asr_muted(False)
            return "ASR unmuted."
        if arg in {"off", "mute"}:
            self.set_asr_muted(True)
            return "ASR muted."
        if arg in {"toggle", "swap"}:
            muted = self.toggle_asr_muted()
            return f"ASR {'muted' if muted else 'unmuted'}."
        return "Usage: /asr on|off|toggle"

    def _cmd_tts(self, args: list[str]) -> str:
        if not args:
            return f"TTS is {'muted' if self.tts_muted_event.is_set() else 'active'}."
        arg = args[0].lower()
        if arg in {"on", "unmute", "active"}:
            self.set_tts_muted(False)
            return "TTS unmuted."
        if arg in {"off", "mute"}:
            self.set_tts_muted(True)
            return "TTS muted."
        if arg in {"toggle", "swap"}:
            muted = self.toggle_tts_muted()
            return f"TTS {'muted' if muted else 'unmuted'}."
        return "Usage: /tts on|off|toggle"

    def _cmd_mute_asr(self, _args: list[str]) -> str:
        self.set_asr_muted(True)
        return "ASR muted."

    def _cmd_unmute_asr(self, _args: list[str]) -> str:
        self.set_asr_muted(False)
        return "ASR unmuted."

    def _cmd_mute_tts(self, _args: list[str]) -> str:
        self.set_tts_muted(True)
        return "TTS muted."

    def _cmd_unmute_tts(self, _args: list[str]) -> str:
        self.set_tts_muted(False)
        return "TTS unmuted."

    def _cmd_observe(self, _args: list[str]) -> str:
        return "Observability is available in the TUI via /observe."

    def _cmd_slots(self, _args: list[str]) -> str:
        if not self.autonomy_slots:
            return "Autonomy slots are unavailable."
        slots = self.autonomy_slots.list_slots()
        if not slots:
            return "No active slots."
        lines = ["Slots:"]
        for slot in slots[:20]:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- {slot.title}: {slot.status}{summary_text}")
        if len(slots) > 20:
            lines.append(f"... {len(slots) - 20} more")
        return "\n".join(lines)

    def _cmd_minds(self, _args: list[str]) -> str:
        minds = self.mind_registry.snapshot()
        if not minds:
            return "No minds registered."
        lines = ["Minds:"]
        for mind in minds[:20]:
            summary = mind.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- {mind.title}: {mind.status}{summary_text}")
        if len(minds) > 20:
            lines.append(f"... {len(minds) - 20} more")
        return "\n".join(lines)

    def _cmd_vision(self, _args: list[str]) -> str:
        if not self.vision_state:
            return "Vision is disabled."
        snapshot = self.vision_state.snapshot()
        return snapshot or "Vision has no snapshot yet."

    def _cmd_config(self, _args: list[str]) -> str:
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"autonomy.enabled={self.autonomy_config.enabled}, "
            f"autonomy.jobs.enabled={jobs_enabled}, "
            f"vision.enabled={self.vision_config is not None}"
        )

    def _cmd_knowledge(self, args: list[str]) -> str:
        if not args or args[0] == "list":
            entries = self.knowledge_store.list_entries()
            if not entries:
                return "Knowledge: no entries."
            lines = ["Knowledge:"]
            for entry in entries[:20]:
                text = entry.text.strip()
                preview = (text[:120] + "...") if len(text) > 120 else text
                lines.append(f"- {entry.entry_id}: {preview}")
            if len(entries) > 20:
                lines.append(f"... {len(entries) - 20} more")
            return "\n".join(lines)

        action = args[0].lower()
        if action == "add":
            text = " ".join(args[1:]).strip()
            if not text:
                return "Usage: /knowledge add <text>"
            entry = self.knowledge_store.add(text)
            return f"Added knowledge #{entry.entry_id}."

        if action in {"set", "update"}:
            if len(args) < 3:
                return "Usage: /knowledge set <id> <text>"
            try:
                entry_id = int(args[1])
            except ValueError:
                return "Knowledge id must be a number."
            text = " ".join(args[2:]).strip()
            if not text:
                return "Usage: /knowledge set <id> <text>"
            updated = self.knowledge_store.update(entry_id, text)
            if not updated:
                return f"Knowledge #{entry_id} not found."
            return f"Updated knowledge #{entry_id}."

        if action in {"delete", "remove"}:
            if len(args) < 2:
                return "Usage: /knowledge delete <id>"
            try:
                entry_id = int(args[1])
            except ValueError:
                return "Knowledge id must be a number."
            removed = self.knowledge_store.delete(entry_id)
            if not removed:
                return f"Knowledge #{entry_id} not found."
            return f"Deleted knowledge #{entry_id}."

        if action == "clear":
            removed = self.knowledge_store.clear()
            return f"Cleared {removed} knowledge entr{'y' if removed == 1 else 'ies'}."

        return "Usage: /knowledge add|list|set|delete|clear"

    def _run_autonomy_ticker(self) -> None:
        assert self.autonomy_event_bus is not None
        logger.info("AutonomyTicker thread started.")
        while not self.shutdown_event.is_set():
            self.autonomy_event_bus.publish(TimeTickEvent(ticked_at=time.time()))
            self.shutdown_event.wait(timeout=self.autonomy_config.tick_interval_s)
        logger.info("AutonomyTicker thread finished.")


if __name__ == "__main__":
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.run()
