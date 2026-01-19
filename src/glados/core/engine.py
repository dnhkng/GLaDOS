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
from ..autonomy import AutonomyConfig, AutonomyLoop, ConstitutionalState, EventBus, InteractionState, SubagentConfig, SubagentManager, TaskManager, TaskSlotStore
from ..autonomy.agents import CompactionAgent, EmotionAgent, HackerNewsSubagent, ObserverAgent, WeatherSubagent
from ..autonomy.emotion_state import EmotionEvent
from ..autonomy.events import TimeTickEvent
from ..autonomy.llm_client import LLMConfig
from ..autonomy.summarization import estimate_tokens
from ..mcp import MCPManager, MCPServerConfig
from ..observability import MindRegistry, ObservabilityBus, trim_message
from ..vision import VisionConfig, VisionState
from ..vision.constants import SYSTEM_PROMPT_VISION_HANDLING
from .audio_data import AudioMessage
from .context import ContextBuilder
from .audio_state import AudioState
from .conversation_store import ConversationStore
from .knowledge_store import KnowledgeStore
from .llm_processor import LanguageModelProcessor
from .shutdown import ShutdownOrchestrator, ShutdownPriority
from .store import Store, format_preferences
from .llm_tracking import InFlightCounter
from .speech_listener import SpeechListener
from .speech_player import SpeechPlayer
from .text_listener import TextListener
from .tool_executor import ToolExecutor
from .tts_synthesizer import TextToSpeechSynthesizer
from .memory_context import MemoryContext

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
    tui_theme: str | None = None
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
        self._conversation_store = ConversationStore(initial_messages=list(personality_preprompt))
        self.vision_config = vision_config
        self.autonomy_config = autonomy_config or AutonomyConfig()
        self.vision_state: VisionState | None = VisionState() if self.vision_config else None
        self.vision_request_queue: queue.Queue | None = queue.Queue() if self.vision_config else None
        self.autonomy_event_bus: EventBus | None = None
        self.autonomy_loop: AutonomyLoop | None = None
        self.autonomy_slots: TaskSlotStore | None = None
        self.autonomy_tasks: TaskManager | None = None
        self.subagent_manager: SubagentManager | None = None
        self._emotion_agent: EmotionAgent | None = None
        self.constitutional_state = ConstitutionalState()
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
        self.preferences_store = Store[Any](
            path=resource_path("data/preferences.json"),
            formatter=format_preferences,
        )

        # Create unified context builder for LLM context injection
        self.context_builder = ContextBuilder()
        self.context_builder.register("preferences", self.preferences_store.as_prompt, priority=10)
        self.context_builder.register("knowledge", lambda: self._format_knowledge(), priority=5)
        self.context_builder.register("constitution", self.constitutional_state.get_modifiers_prompt, priority=3)

        # Register long-term memory for context injection
        self.memory_context = MemoryContext()
        self.context_builder.register("memory", self.memory_context.as_prompt, priority=7)

        self._command_registry, self._command_order = self._build_command_registry()
        # Initialize events for thread synchronization
        self.processing_active_event = threading.Event()  # Indicates if input processing is active (ASR + LLM + TTS + VLM)
        self.currently_speaking_event = threading.Event()  # Indicates if the assistant is currently speaking
        self.shutdown_event = threading.Event()  # Event to signal shutdown of all threads

        # Initialize shutdown orchestrator for graceful shutdown
        self._shutdown_orchestrator = ShutdownOrchestrator(
            shutdown_event=self.shutdown_event,
            global_timeout=30.0,
            phase_timeout=10.0,
        )
        if self.autonomy_config.enabled:
            self.autonomy_event_bus = EventBus()
            self.autonomy_slots = TaskSlotStore(observability_bus=self.observability_bus)
            self.autonomy_tasks = TaskManager(self.autonomy_slots, self.autonomy_event_bus)
            # Register slots with context builder
            self.context_builder.register("slots", lambda: self._format_slots(), priority=8)
            if self.autonomy_config.jobs.enabled:
                self.subagent_manager = SubagentManager(
                    slot_store=self.autonomy_slots,
                    mind_registry=self.mind_registry,
                    observability_bus=self.observability_bus,
                    shutdown_event=self.shutdown_event,
                )
                self._register_subagents()

        if self.vision_config:
            # Add instructions to system prompt to correctly handle [vision] marked messages
            messages = self._conversation_store.snapshot()
            vision_prompt_added = False
            for i, message in enumerate(messages):
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    self._conversation_store.modify_message(
                        i,
                        {"content": f"{message['content']} {SYSTEM_PROMPT_VISION_HANDLING}"}
                    )
                    vision_prompt_added = True
                    break
            if not vision_prompt_added:
                # Prepend a new system message with vision handling instructions
                current_messages = self._conversation_store.snapshot()
                self._conversation_store.replace_all(
                    [{"role": "system", "content": SYSTEM_PROMPT_VISION_HANDLING}] + current_messages
                )


        # Initialize spoken text converter, that converts text to spoken text. eg. 12 -> "twelve"
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model, this is needed to avoid long pauses on first request
        self._asr_model.transcribe_file(resource_path("data/0.wav"))

        # Initialize queues for inter-thread communication
        self._autonomy_inflight = InFlightCounter()
        self.llm_queue_priority: queue.Queue[dict[str, Any]] = queue.Queue()
        autonomy_queue_max = self.autonomy_config.autonomy_queue_max
        autonomy_queue_size = autonomy_queue_max if autonomy_queue_max and autonomy_queue_max > 0 else 0
        self.llm_queue_autonomy: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=autonomy_queue_size)
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
        logger.info("Audio I/O system initialized.")

        # Initialize threads for each component
        self.component_threads: list[threading.Thread] = []

        self.speech_listener: SpeechListener | None = None
        self.text_listener: TextListener | None = None
        if self.input_mode in {"audio", "both"}:
            self.speech_listener = SpeechListener(
                audio_io=self.audio_io,
                llm_queue=self.llm_queue_priority,
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
                on_interrupt=lambda _: self._push_emotion_event("user", "User interrupted me mid-sentence"),
            )
        if self.input_mode in {"text", "both"}:
            if self.input_mode == "text":
                logger.info("Text input mode enabled. ASR is disabled.")
            self.text_listener = TextListener(
                llm_queue=self.llm_queue_priority,
                processing_active_event=self.processing_active_event,
                shutdown_event=self.shutdown_event,
                pause_time=self.PAUSE_TIME,
                interaction_state=self.interaction_state,
                observability_bus=self.observability_bus,
                command_handler=self.handle_command,
            )

        self.llm_processor = LanguageModelProcessor(
            llm_input_queue=self.llm_queue_priority,
            tool_calls_queue=self.tool_calls_queue,
            tts_input_queue=self.tts_queue,
            conversation_store=self._conversation_store,
            completion_url=self.completion_url,
            model_name=self.llm_model,
            api_key=self.api_key,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME,
            vision_state=self.vision_state,
            slot_store=self.autonomy_slots,
            preferences_store=self.preferences_store,
            constitutional_state=self.constitutional_state,
            context_builder=self.context_builder,
            autonomy_system_prompt=self.autonomy_config.system_prompt if self.autonomy_config.enabled else None,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
            extra_headers=llm_headers,
            lane="priority",
        )
        self.autonomy_llm_processors: list[LanguageModelProcessor] = []
        autonomy_parallel_calls = 0
        if self.autonomy_config.enabled:
            autonomy_parallel_calls = max(0, self.autonomy_config.autonomy_parallel_calls)
        for _ in range(autonomy_parallel_calls):
            self.autonomy_llm_processors.append(
                LanguageModelProcessor(
                    llm_input_queue=self.llm_queue_autonomy,
                    tool_calls_queue=self.tool_calls_queue,
                    tts_input_queue=self.tts_queue,
                    conversation_store=self._conversation_store,
                    completion_url=self.completion_url,
                    model_name=self.llm_model,
                    api_key=self.api_key,
                    processing_active_event=self.processing_active_event,
                    shutdown_event=self.shutdown_event,
                    pause_time=self.PAUSE_TIME,
                    vision_state=self.vision_state,
                    slot_store=self.autonomy_slots,
                    preferences_store=self.preferences_store,
                    constitutional_state=self.constitutional_state,
                    context_builder=self.context_builder,
                    autonomy_system_prompt=self.autonomy_config.system_prompt if self.autonomy_config.enabled else None,
                    mcp_manager=self.mcp_manager,
                    observability_bus=self.observability_bus,
                    extra_headers=llm_headers,
                    lane="autonomy",
                    inflight_counter=self._autonomy_inflight,
                )
            )

        self.tool_executor = ToolExecutor(
            llm_queue_priority=self.llm_queue_priority,
            llm_queue_autonomy=self.llm_queue_autonomy,
            tool_calls_queue=self.tool_calls_queue,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            tool_config={
                **self.tool_config,
                "vision_request_queue": self.vision_request_queue,
                "vision_tool_timeout": self.tool_timeout,
                "tts_queue": self.tts_queue,
                "preferences_store": self.preferences_store,
                "slot_store": self.autonomy_slots,
            },
            tool_timeout=self.tool_timeout,
            pause_time=self.PAUSE_TIME,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
            on_tool_event=self._on_tool_event,
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
            conversation_store=self._conversation_store,
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
                llm_queue=self.llm_queue_autonomy,
                processing_active_event=self.processing_active_event,
                currently_speaking_event=self.currently_speaking_event,
                shutdown_event=self.shutdown_event,
                observability_bus=self.observability_bus,
                inflight_counter=self._autonomy_inflight,
                pause_time=self.PAUSE_TIME,
            )
            if not self.vision_config:
                self.autonomy_ticker_thread = threading.Thread(
                    target=self._run_autonomy_ticker,
                    name="AutonomyTicker",
                    daemon=True,
                )

        # Define thread configurations with daemon settings and shutdown priorities
        # daemon=True: Can be killed without waiting (pure input, stateless)
        # daemon=False: Must be joined (has in-flight state to preserve)
        thread_configs: dict[str, tuple[Any, bool, ShutdownPriority, queue.Queue | None]] = {
            "LLMProcessor": (
                self.llm_processor.run,
                False,  # Has in-flight conversation updates
                ShutdownPriority.PROCESSING,
                self.llm_queue_priority,
            ),
            "ToolExecutor": (
                self.tool_executor.run,
                False,  # Tool results need to be recorded
                ShutdownPriority.PROCESSING,
                self.tool_calls_queue,
            ),
            "TTSSynthesizer": (
                self.tts_synthesizer.run,
                False,  # Pending TTS to complete
                ShutdownPriority.OUTPUT,
                self.tts_queue,
            ),
            "AudioPlayer": (
                self.speech_player.run,
                False,  # Audio playing needs to finish
                ShutdownPriority.OUTPUT,
                self.audio_queue,
            ),
        }
        for index, processor in enumerate(self.autonomy_llm_processors, start=1):
            thread_configs[f"LLMProcessorAutonomy-{index}"] = (
                processor.run,
                False,  # Has in-flight conversation updates
                ShutdownPriority.PROCESSING,
                self.llm_queue_autonomy,
            )
        if self.speech_listener:
            thread_configs["SpeechListener"] = (
                self.speech_listener.run,
                True,  # Pure input, no state
                ShutdownPriority.INPUT,
                None,
            )
        if self.text_listener:
            thread_configs["TextListener"] = (
                self.text_listener.run,
                True,  # Pure input, no state
                ShutdownPriority.INPUT,
                None,
            )
        if self.autonomy_loop:
            thread_configs["AutonomyLoop"] = (
                self.autonomy_loop.run,
                True,  # Can safely abandon
                ShutdownPriority.BACKGROUND,
                None,
            )
        if self.vision_processor:
            thread_configs["VisionProcessor"] = (
                self.vision_processor.run,
                True,  # Can safely abandon
                ShutdownPriority.BACKGROUND,
                self.vision_request_queue,
            )
        if self.autonomy_ticker_thread:
            self.component_threads.append(self.autonomy_ticker_thread)
            self.autonomy_ticker_thread.start()
            self._shutdown_orchestrator.register(
                "AutonomyTicker",
                self.autonomy_ticker_thread,
                priority=ShutdownPriority.BACKGROUND,
            )
            logger.info("Orchestrator: AutonomyTicker thread started.")
            self.mind_registry.register(
                "AutonomyTicker",
                title="Autonomy Ticker",
                status="running",
                summary="Periodic autonomy ticks",
            )

        for name in thread_configs:
            self.mind_registry.register(name, title=name, status="starting", summary="Initializing")

        for name, (target_func, daemon, priority, component_queue) in thread_configs.items():
            thread = threading.Thread(target=target_func, name=name, daemon=daemon)
            self.component_threads.append(thread)
            thread.start()
            self._shutdown_orchestrator.register(
                name,
                thread,
                queue=component_queue,
                priority=priority,
            )
            logger.info(f"Orchestrator: {name} thread started (daemon={daemon}).")
            self.mind_registry.update(name, "running", summary="Thread active")

        # Start subagents after other components are running
        if self.subagent_manager:
            self.subagent_manager.start_all()

    def _register_subagents(self) -> None:
        """Register configured subagents with the manager."""
        if not self.subagent_manager:
            return

        jobs_config = self.autonomy_config.jobs

        # Create shared LLM config for subagents
        llm_config = LLMConfig(
            url=str(self.completion_url),
            api_key=self.api_key,
            model=self.llm_model,
        )

        if jobs_config.hacker_news.enabled:
            hn_config = SubagentConfig(
                agent_id="hn_top",
                title="Hacker News",
                role="news_monitor",
                loop_interval_s=jobs_config.hacker_news.interval_s,
                run_on_start=True,
            )
            hn_subagent = HackerNewsSubagent(
                config=hn_config,
                top_n=jobs_config.hacker_news.top_n,
                min_score=jobs_config.hacker_news.min_score,
                llm_config=llm_config,
                slot_store=self.autonomy_slots,
                mind_registry=self.mind_registry,
                observability_bus=self.observability_bus,
                shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(hn_subagent)

        if jobs_config.weather.enabled:
            if jobs_config.weather.latitude is None or jobs_config.weather.longitude is None:
                logger.warning("Weather subagent enabled but latitude/longitude are missing.")
            else:
                weather_config = SubagentConfig(
                    agent_id="weather",
                    title="Weather",
                    role="weather_monitor",
                    loop_interval_s=jobs_config.weather.interval_s,
                    run_on_start=True,
                )
                weather_subagent = WeatherSubagent(
                    config=weather_config,
                    latitude=jobs_config.weather.latitude,
                    longitude=jobs_config.weather.longitude,
                    timezone=jobs_config.weather.timezone,
                    temp_change_c=jobs_config.weather.temp_change_c,
                    wind_alert_kmh=jobs_config.weather.wind_alert_kmh,
                    llm_config=llm_config,
                    slot_store=self.autonomy_slots,
                    mind_registry=self.mind_registry,
                    observability_bus=self.observability_bus,
                    shutdown_event=self.shutdown_event,
                )
                self.subagent_manager.register(weather_subagent)

        # Emotion agent - always registered, core to GLaDOS personality
        emotion_cfg = self.autonomy_config.emotion
        emotion_subagent_config = SubagentConfig(
            agent_id="emotion",
            title="Emotional State",
            role="emotional_regulation",
            loop_interval_s=emotion_cfg.tick_interval_s,
            run_on_start=True,
        )
        emotion_agent = EmotionAgent(
            config=emotion_subagent_config,
            llm_config=llm_config,
            emotion_config=emotion_cfg,
            slot_store=self.autonomy_slots,
            mind_registry=self.mind_registry,
            observability_bus=self.observability_bus,
            shutdown_event=self.shutdown_event,
        )
        self.subagent_manager.register(emotion_agent)
        self._emotion_agent = emotion_agent  # Keep reference for event pushing
        # Wire emotion agent to autonomy loop for vision events
        if self.autonomy_config.enabled and hasattr(self, "autonomy_loop"):
            self.autonomy_loop.set_emotion_agent(emotion_agent)

        # Compaction agent - monitors conversation size and compacts when needed
        compaction_config = SubagentConfig(
            agent_id="compaction",
            title="Message Compaction",
            role="context_management",
            loop_interval_s=60.0,  # Check every minute
            run_on_start=False,  # Wait for conversation to build up
        )
        compaction_agent = CompactionAgent(
            config=compaction_config,
            llm_config=llm_config,
            conversation_store=self._conversation_store,
            token_threshold=self.autonomy_config.tokens.token_threshold,
            preserve_recent=self.autonomy_config.tokens.preserve_recent_messages,
            slot_store=self.autonomy_slots,
            mind_registry=self.mind_registry,
            observability_bus=self.observability_bus,
            shutdown_event=self.shutdown_event,
        )
        self.subagent_manager.register(compaction_agent)

        # Observer agent - monitors behavior and proposes adjustments
        observer_config = SubagentConfig(
            agent_id="observer",
            title="Behavior Observer",
            role="meta_supervision",
            loop_interval_s=120.0,  # Analyze every 2 minutes
            run_on_start=False,  # Wait for conversation to build up
        )
        observer_agent = ObserverAgent(
            config=observer_config,
            llm_config=llm_config,
            conversation_store=self._conversation_store,
            constitutional_state=self.constitutional_state,
            sample_count=10,
            min_samples_for_analysis=5,
            slot_store=self.autonomy_slots,
            mind_registry=self.mind_registry,
            observability_bus=self.observability_bus,
            shutdown_event=self.shutdown_event,
        )
        self.subagent_manager.register(observer_agent)

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
            list[dict[str, Any]]: A snapshot of message dictionaries representing the conversation history.
        """
        return self._conversation_store.snapshot()

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
            try:
                self.audio_io.start_listening()
                logger.success("Audio input stream started successfully")
            except RuntimeError as e:
                logger.error(f"Failed to start audio input: {e}")
                logger.warning("Voice input disabled - text input still available")
        else:
            logger.info("Text input mode active. Audio input is disabled.")

        logger.success("Engine running")
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
        finally:
            self._graceful_shutdown()

    def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown of all components."""
        logger.info("Beginning graceful shutdown...")

        # Stop subagents first (they may be using shared resources)
        if self.subagent_manager:
            logger.debug("Shutting down subagent manager...")
            self.subagent_manager.shutdown(timeout=5.0)

        # Stop task manager
        if self.autonomy_tasks:
            logger.debug("Shutting down task manager...")
            self.autonomy_tasks.shutdown(wait=True)

        # Use orchestrator for coordinated thread shutdown
        results = self._shutdown_orchestrator.initiate_shutdown()

        # Update mind registry for all components
        for component in self.component_threads:
            self.mind_registry.update(component.name, "stopped", summary="Shutdown")
        if self.autonomy_ticker_thread:
            self.mind_registry.update("AutonomyTicker", "stopped", summary="Shutdown")

        # Stop MCP manager last (other components may need it during shutdown)
        if self.mcp_manager:
            logger.debug("Shutting down MCP manager...")
            self.mcp_manager.shutdown()

        # Log any failed shutdowns
        failed = [r for r in results if not r.success]
        if failed:
            logger.warning(
                "Some components did not shut down cleanly: {}",
                [r.component for r in failed],
            )

        logger.info("Graceful shutdown complete.")

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
        self.llm_queue_priority.put(
            {
                "role": "user",
                "content": text,
                "_enqueued_at": time.time(),
                "_lane": "priority",
            }
        )
        if self.interaction_state:
            self.interaction_state.mark_user()
        self.processing_active_event.set()
        return True

    def autonomy_inflight(self) -> int:
        return self._autonomy_inflight.value()

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
                usage="/tts on|off",
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
                usage="/asr on|off",
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
                name="mcp",
                description="Show MCP server status",
                usage="/mcp status",
                handler=self._cmd_mcp,
            )
        )
        register(
            CommandSpec(
                name="autonomy",
                description="Manage autonomy settings",
                usage="/autonomy on|off | /autonomy debounce on|off",
                handler=self._cmd_autonomy,
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
                name="agents",
                description="Show registered subagents",
                usage="/agents",
                handler=self._cmd_agents,
            )
        )
        register(
            CommandSpec(
                name="emotion",
                description="Show current emotional state",
                usage="/emotion",
                handler=self._cmd_emotion,
            )
        )
        register(
            CommandSpec(
                name="preferences",
                description="Show user preferences",
                usage="/preferences",
                handler=self._cmd_preferences,
            )
        )
        register(
            CommandSpec(
                name="context",
                description="Show context/token usage",
                usage="/context",
                handler=self._cmd_context,
            )
        )
        register(
            CommandSpec(
                name="constitution",
                description="Show constitutional state and modifiers",
                usage="/constitution",
                handler=self._cmd_constitution,
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
        register(
            CommandSpec(
                name="memory",
                description="Show long-term memory stats",
                usage="/memory",
                handler=self._cmd_memory,
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
        return "Usage: /asr on|off"

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
        return "Usage: /tts on|off"

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

    def _cmd_agents(self, _args: list[str]) -> str:
        if not self.subagent_manager:
            return "Subagent manager is not enabled."
        agents = self.subagent_manager.list_agents()
        if not agents:
            return "No subagents registered."
        lines = ["Subagents:"]
        for agent in agents[:20]:
            status = "running" if agent.running else "stopped"
            tick_info = f"ticks={agent.tick_count}" if agent.tick_count > 0 else "not started"
            lines.append(f"- {agent.title} ({agent.agent_id}): {status}, {tick_info}")
        if len(agents) > 20:
            lines.append(f"... {len(agents) - 20} more")
        return "\n".join(lines)

    def _push_emotion_event(self, source: str, description: str) -> None:
        """Push an event to the emotion agent if it's running."""
        if self._emotion_agent:
            event = EmotionEvent(source=source, description=description)
            self._emotion_agent.push_event(event)

    def _on_tool_event(self, event_type: str, tool_name: str) -> None:
        """Handle tool events for emotional processing."""
        if event_type == "tool_success":
            self._push_emotion_event("system", f"Tool '{tool_name}' completed successfully")
        elif event_type == "tool_failure":
            self._push_emotion_event("system", f"Tool '{tool_name}' failed")
        elif event_type == "tool_timeout":
            self._push_emotion_event("system", f"Tool '{tool_name}' timed out")

    def _cmd_emotion(self, _args: list[str]) -> str:
        if not self._emotion_agent:
            return "Emotion agent is not running."
        state = self._emotion_agent.state
        lines = [
            "Emotional State:",
            f"  Pleasure:  {state.pleasure:+.2f}",
            f"  Arousal:   {state.arousal:+.2f}",
            f"  Dominance: {state.dominance:+.2f}",
            "Mood Baseline:",
            f"  Pleasure:  {state.mood_pleasure:+.2f}",
            f"  Arousal:   {state.mood_arousal:+.2f}",
            f"  Dominance: {state.mood_dominance:+.2f}",
            "",
            state.to_prompt(),
        ]
        return "\n".join(lines)

    def _cmd_preferences(self, _args: list[str]) -> str:
        prefs = self.preferences_store.all()
        if not prefs:
            return "No preferences set."
        lines = ["User Preferences:"]
        for key, value in prefs.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _cmd_context(self, _args: list[str]) -> str:
        messages = self._conversation_store.snapshot()
        token_count = estimate_tokens(messages)
        msg_count = len(messages)
        system_count = sum(1 for m in messages if m.get("role") == "system")
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        summary_count = sum(
            1 for m in messages
            if isinstance(m.get("content"), str) and m["content"].startswith("[summary]")
        )
        lines = [
            f"Context Usage:",
            f"  Estimated tokens: {token_count}",
            f"  Total messages: {msg_count}",
            f"    System: {system_count}",
            f"    User: {user_count}",
            f"    Assistant: {assistant_count}",
            f"    Summaries: {summary_count}",
        ]
        return "\n".join(lines)

    def _cmd_constitution(self, _args: list[str]) -> str:
        state = self.constitutional_state
        lines = ["Constitutional State:"]
        lines.append("")
        lines.append("Immutable Rules:")
        for rule in state.constitution.immutable_rules:
            lines.append(f"  - {rule}")
        lines.append("")
        lines.append("Modifiable Bounds:")
        for name, (min_val, max_val) in state.constitution.modifiable_bounds.items():
            lines.append(f"  {name}: {min_val} to {max_val}")
        lines.append("")
        if state.active_modifiers:
            lines.append("Active Modifiers:")
            for name, modifier in state.active_modifiers.items():
                lines.append(f"  {name}: {modifier.value} ({modifier.reason})")
        else:
            lines.append("Active Modifiers: none")
        lines.append("")
        lines.append(f"Modifier History: {len(state.modifier_history)} changes")
        return "\n".join(lines)

    def _cmd_vision(self, _args: list[str]) -> str:
        if not self.vision_state:
            return "Vision is disabled."
        snapshot = self.vision_state.snapshot()
        return snapshot or "Vision has no snapshot yet."

    def _cmd_mcp(self, args: list[str]) -> str:
        if not self.mcp_manager:
            return "MCP is disabled."
        if args and args[0].lower() not in {"status", "list"}:
            return "Usage: /mcp status"
        lines = ["MCP servers:"]
        for entry in self.mcp_manager.status_snapshot():
            status = "connected" if entry["connected"] else "offline"
            tools = entry.get("tools", 0)
            resources = entry.get("resources", 0)
            lines.append(f"- {entry['name']}: {status}, tools={tools}, resources={resources}")
        return "\n".join(lines)

    def _cmd_autonomy(self, args: list[str]) -> str:
        if not args:
            return (
                f"Autonomy enabled={self.autonomy_config.enabled}, "
                f"parallel_calls={self.autonomy_config.autonomy_parallel_calls}, "
                f"coalesce_ticks={self.autonomy_config.coalesce_ticks}"
            )
        head = args[0].lower()
        if head in {"on", "off", "true", "false", "enable", "enabled", "disable", "disabled"}:
            enabled = head in {"on", "true", "enable", "enabled"}
            self.autonomy_config.enabled = enabled
            return f"Autonomy {'enabled' if enabled else 'disabled'}."
        if head not in {"coalesce", "debounce"}:
            return "Usage: /autonomy on|off | /autonomy debounce on|off"
        if len(args) == 1:
            return f"Autonomy coalesce_ticks={self.autonomy_config.coalesce_ticks}"
        value = args[1].lower()
        if value in {"on", "true", "enable", "enabled"}:
            self.autonomy_config.coalesce_ticks = True
            return "Autonomy tick coalescing enabled."
        if value in {"off", "false", "disable", "disabled"}:
            self.autonomy_config.coalesce_ticks = False
            return "Autonomy tick coalescing disabled."
        return "Usage: /autonomy on|off | /autonomy debounce on|off"

    def _cmd_config(self, _args: list[str]) -> str:
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"autonomy.enabled={self.autonomy_config.enabled}, "
            f"autonomy.jobs.enabled={jobs_enabled}, "
            f"autonomy.coalesce_ticks={self.autonomy_config.coalesce_ticks}, "
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

    def _cmd_memory(self, _args: list[str]) -> str:
        import json
        from pathlib import Path

        memory_dir = Path.home() / ".glados" / "memory"
        facts_file = memory_dir / "facts.jsonl"
        summaries_file = memory_dir / "summaries.jsonl"

        # Count facts
        fact_count = 0
        source_counts: dict[str, int] = {}
        total_importance = 0.0
        if facts_file.exists():
            try:
                with facts_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            fact_count += 1
                            try:
                                fact = json.loads(line)
                                source = fact.get("source", "unknown")
                                source_counts[source] = source_counts.get(source, 0) + 1
                                total_importance += fact.get("importance", 0.5)
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        # Count summaries
        summary_count = 0
        period_counts: dict[str, int] = {}
        if summaries_file.exists():
            try:
                with summaries_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            summary_count += 1
                            try:
                                summary = json.loads(line)
                                period = summary.get("period", "unknown")
                                period_counts[period] = period_counts.get(period, 0) + 1
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        if fact_count == 0 and summary_count == 0:
            return "Long-term Memory: empty (no facts or summaries stored)"

        avg_importance = total_importance / fact_count if fact_count > 0 else 0.0

        lines = [
            "Long-term Memory Stats:",
            f"  Total facts: {fact_count}",
            f"  Total summaries: {summary_count}",
        ]

        if source_counts:
            lines.append("  Facts by source:")
            for source, count in sorted(source_counts.items()):
                lines.append(f"    {source}: {count}")

        if period_counts:
            lines.append("  Summaries by period:")
            for period, count in sorted(period_counts.items()):
                lines.append(f"    {period}: {count}")

        lines.append(f"  Average importance: {avg_importance:.2f}")
        lines.append(f"  Storage: {memory_dir}")

        return "\n".join(lines)

    def _format_knowledge(self) -> str | None:
        """Format knowledge entries for LLM context."""
        entries = self.knowledge_store.list_entries()
        if not entries:
            return None
        lines = ["[knowledge]"]
        for entry in entries:
            lines.append(f"- #{entry.entry_id}: {entry.text}")
        return "\n".join(lines)

    def _format_slots(self) -> str | None:
        """Format task slots for LLM context."""
        if not self.autonomy_slots:
            return None
        slots = self.autonomy_slots.list_slots()
        if not slots:
            return None
        lines = ["[tasks]"]
        for slot in slots:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- {slot.title}: {slot.status}{summary_text}")
        return "\n".join(lines)

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
