"""
Core engine module for the Glados voice assistant.

This module provides the main orchestration classes including the Glados assistant,
configuration management, and component coordination.
"""

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, model_validator
import yaml

from ..ASR import TranscriberProtocol, get_audio_transcriber
from ..audio_io import AudioProtocol, get_audio_system
from ..autonomy import (
    AutonomyConfig,
    AutonomyLoop,
    ConstitutionalState,
    EventBus,
    InteractionState,
    MindScheduler,
    SubagentConfig,
    TaskManager,
    TaskSlotStore,
)
from ..autonomy.agents import CompactionAgent, EmotionAgent, HackerNewsSubagent, ObserverAgent, WeatherSubagent
from ..autonomy.agents.health_agent import HealthAgent, HealthConfig
from ..autonomy.agents.search_agent import SearchAgent, SearchConfig
from ..autonomy.emotion_state import EmotionEvent
from ..autonomy.events import TimeTickEvent
from ..autonomy.llm_client import LLMConfig
from ..autonomy.mind_schedule import FixedInterval, OnDemand, RandomAdaptive
from ..autonomy.summarization import estimate_tokens
from ..mcp import MCPManager, MCPServerConfig
from ..observability import MindRegistry, ObservabilityBus, trim_message
from ..tools.safe_command import SafeCommandRunner
from ..TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from ..vision import VisionConfig, VisionState
from ..vision.constants import SYSTEM_PROMPT_VISION_HANDLING
from ..webapp import WebappConfig
from .audio_data import AudioMessage
from .audio_state import AudioState
from .context import ContextBuilder
from .conversation_store import ConversationStore
from .decision_lists import DecisionListStore
from .inference import InferenceConfig, InferenceScheduler
from .knowledge_store import KnowledgeStore
from .llm_processor import LanguageModelProcessor
from .llm_tracking import InFlightCounter
from .native_audio import NativeAudioConfig, NativeAudioInput
from .operator_state import OperatorState
from .routing import DecisionRouter, RoutingConfig
from .shutdown import ShutdownOrchestrator, ShutdownPriority
from .speech_animation import SpeechAnimationState
from .speech_listener import SpeechListener
from .speech_markup import SpeechText
from .speech_player import SpeechPlayer
from .store import Store, format_preferences
from .text_listener import TextListener
from .tool_executor import ToolExecutor
from .tts_synthesizer import TextToSpeechSynthesizer

try:
    logger.remove(0)
except ValueError:
    pass  # Handler already removed (e.g., by TUI)
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
    audio_io_options: dict[str, Any] | None = None
    input_mode: Literal["audio", "text", "both"] = "audio"
    tts_enabled: bool = True
    asr_muted: bool = False
    asr_engine: str
    native_audio: NativeAudioConfig = Field(default_factory=NativeAudioConfig)
    wake_word: str | None
    voice: str
    announcement: str | None
    llm_headers: dict[str, str] | None = None
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    llm_request_options: dict[str, Any] | None = None
    tui_theme: str | None = None
    personality_preprompt: list[PersonalityPrompt]
    slow_clap_audio_path: str = "data/slow-clap.mp3"
    tool_timeout: float = 30.0
    vision: VisionConfig | None = None
    autonomy: AutonomyConfig | None = None
    mcp_servers: list[MCPServerConfig] | None = None
    webapp: WebappConfig | None = None

    @model_validator(mode="after")
    def _validate_native_audio(self) -> "GladosConfig":
        if self.native_audio.enabled:
            if not str(self.completion_url).rstrip("/").endswith("/v1/chat/completions"):
                raise ValueError("Native audio requires a multimodal /v1/chat/completions endpoint (e.g. llama.cpp)")
            if self.wake_word:
                raise ValueError("Native audio does not support transcript-based wake words; use Parakeet mode")
        return self

    @model_validator(mode="after")
    def _resolve_api_key_from_env(self) -> "GladosConfig":
        """Fall back to MINIMAX_API_KEY environment variable when api_key is not set."""
        if self.api_key is None:
            env_key = os.environ.get("MINIMAX_API_KEY")
            if env_key:
                self.api_key = env_key
        return self

    @model_validator(mode="after")
    def _apply_webapp_env(self) -> "GladosConfig":
        """Enable/configure the webapp console from GLADOS_WEBAPP_* env vars.

        Lets the console be switched on without editing any YAML::

            GLADOS_WEBAPP_ENABLED=1 GLADOS_WEBAPP_PORT=8050 glados webapp
        """
        flag = os.environ.get("GLADOS_WEBAPP_ENABLED")
        host = os.environ.get("GLADOS_WEBAPP_HOST")
        port = os.environ.get("GLADOS_WEBAPP_PORT")
        if flag is None and not host and not port:
            return self
        base = self.webapp or WebappConfig()
        try:
            env_port = int(port or "")
        except ValueError:
            env_port = None
        self.webapp = WebappConfig.model_validate(
            {
                **base.model_dump(),
                "enabled": base.enabled if flag is None else flag.strip().lower() in ("1", "true", "yes"),
                "host": host or base.host,
                "port": env_port or base.port,
            }
        )
        return self

    @classmethod
    def from_yaml(
        cls, paths: str | Path | list[str] | list[Path], key_to_config: tuple[str, ...] = ("Glados",)
    ) -> "GladosConfig":
        """
        Load a GladosConfig instance from one or more configuration files.
        Explicitly specified options in later configuration files override options specified in earlier files.

        Parameters:
            paths: Path to one or multiple YAML configuration files
            key_to_config: Tuple of keys to navigate nested configuration

        Returns:
            GladosConfig: Configuration object with validated settings

        Raises:
            ValueError: If the YAML content is invalid
            OSError: If a file cannot be read
            pydantic.ValidationError: If the configuration is invalid
        """

        # if config is a single path, create a single-element list
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]

        config = dict()

        for path in paths:
            path = Path(path)

            # Try different encodings
            for encoding in ["utf-8", "utf-8-sig"]:
                try:
                    data = yaml.safe_load(path.read_text(encoding=encoding))
                    break
                except UnicodeDecodeError:
                    if encoding == "utf-8-sig":
                        raise ValueError(f"Could not decode YAML file {path} with any supported encoding")

            data = data or dict()

            # Navigate through nested keys
            for key in key_to_config:
                data = data[key]

            # Update config dict - config from later paths overrides earlier config
            config = GladosConfig._dict_deep_merge(config, data)

        return cls.model_validate(config)

    @staticmethod
    def _dict_deep_merge(a: dict, b: dict) -> dict:
        """
        Recursively merge two dictionaries into one.
        Values from the second dictionary override values from the first.
        Note: mutates the first dictionary.

        Returns:
            The merged dictionary.
        """
        for key in b:
            if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
                GladosConfig._dict_deep_merge(a[key], b[key])
            else:
                a[key] = b[key]
        return a

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
    DEFAULT_PERSONALITY_PREPROMPT: tuple[dict[str, str], ...] = (
        {
            "role": "system",
            "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
        },
    )

    def __init__(
        self,
        asr_model: TranscriberProtocol | None,
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
        native_audio_config: NativeAudioConfig | None = None,
        llm_request_options: dict[str, Any] | None = None,
        inference_config: InferenceConfig | None = None,
        routing_config: RoutingConfig | None = None,
        health_config: HealthConfig | None = None,
        search_config: SearchConfig | None = None,
    ) -> None:
        """Wire injected models/backends, then start components in dependency order."""
        self._asr_model = asr_model
        self.started_at = time.time()
        self.quiet_event = threading.Event()
        self._quiet_lock = threading.RLock()
        self._quiet_saved_pauses: dict[str, bool] = {}
        self._quiet_generation = 0
        self._autonomy_generation = 0
        self.operator_state = OperatorState(path=resource_path("data/operator_settings.yaml"))
        self.inference_scheduler = InferenceScheduler(inference_config)
        native_audio = (
            NativeAudioInput(native_audio_config) if native_audio_config and native_audio_config.enabled else None
        )
        self.native_audio = native_audio
        self._tts = tts_model
        self.input_mode = input_mode
        self.completion_url = completion_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.llm_request_options = dict(llm_request_options or {})
        self.interruptible = interruptible
        self.wake_word = wake_word
        self.announcement = announcement
        self.tool_config = tool_config or {}
        self.tool_timeout = tool_timeout
        self.mcp_servers = mcp_servers or []
        self.autonomy_config = autonomy_config or AutonomyConfig()
        self.health_config = health_config or HealthConfig()
        self.health_agent: HealthAgent | None = None
        self.search_config = search_config or SearchConfig()
        self.search_agent: SearchAgent | None = None
        self._init_context_and_state(personality_preprompt, vision_config, asr_muted, tts_enabled)

        self._init_background_cores()

        # Initialize spoken text converter, that converts text to spoken text. eg. 12 -> "twelve"
        self._stc = stc.SpokenTextConverter()

        # warm up onnx ASR model, this is needed to avoid long pauses on first request
        if self._asr_model is not None:
            self._asr_model.transcribe_file(resource_path("data/0.wav"))

        self._init_queues_and_mcp()

        # Initialize audio input/output system
        self.audio_io: AudioProtocol = audio_io
        logger.info("Audio I/O system initialized.")

        # Initialize threads for each component
        self.component_threads: list[threading.Thread] = []

        self._init_listeners()

        self._init_primary_processor(llm_headers, llm_request_options, routing_config)

        self._init_autonomy_processors(llm_headers, llm_request_options)

        self._init_tools_and_speech()

        self._init_autonomy_loop()

        self._start_components()

    def _init_context_and_state(self, personality_preprompt, vision_config, asr_muted, tts_enabled) -> None:
        history_path = self.autonomy_config.tokens.state_path
        self._conversation_store = ConversationStore(initial_messages=list(personality_preprompt),
                                                     path=Path(history_path) if history_path else None)
        self.vision_config = vision_config
        self.vision_state: VisionState | None = VisionState() if self.vision_config and self.vision_config.enabled else None
        self.vision_agent = None
        self.autonomy_event_bus: EventBus | None = None
        self.autonomy_loop: AutonomyLoop | None = None
        self.autonomy_slots: TaskSlotStore | None = None
        self.autonomy_tasks: TaskManager | None = None
        self.subagent_manager: MindScheduler | None = None
        self._emotion_agent: EmotionAgent | None = None
        self.compaction_agent: CompactionAgent | None = None
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
        self.speech_animation = SpeechAnimationState(self.observability_bus)
        self.knowledge_store = KnowledgeStore(resource_path("data/knowledge.json"))
        self.preferences_store = Store[Any](
            path=resource_path("data/preferences.yaml"),
            formatter=format_preferences,
        )

        # Create unified context builder for LLM context injection
        self.context_builder = ContextBuilder()
        self.context_builder.register("operator", self.operator_state.as_prompt, priority=20)
        self.context_builder.register("preferences", self.preferences_store.as_prompt, priority=10)
        self.context_builder.register("knowledge", lambda: self._format_knowledge(), priority=5)
        self.context_builder.register("constitution", self.constitutional_state.get_modifiers_prompt, priority=3)

        # Long-term recall is published by the Memory Core through its slot.
        self.context_builder.register("emotion", self._emotion_prompt, priority=15, volatile=True)
        self.context_builder.register("health", lambda: self.health_agent.as_prompt() if self.health_agent else None,
                                      priority=12, volatile=True)

        self._command_registry, self._command_order = self._build_command_registry()
        # Initialize events for thread synchronization
        self.processing_active_event = (
            threading.Event()
        )  # Indicates if input processing is active (ASR + LLM + TTS + VLM)
        self.currently_speaking_event = threading.Event()  # Indicates if the assistant is currently speaking
        self.shutdown_event = threading.Event()  # Event to signal shutdown of all threads

        # Initialize shutdown orchestrator for graceful shutdown
        self._shutdown_orchestrator = ShutdownOrchestrator(
            shutdown_event=self.shutdown_event,
            global_timeout=30.0,
            phase_timeout=10.0,
        )

    def _init_background_cores(self) -> None:
        # The shared task board is useful even with background autonomy disabled.
        self.autonomy_slots = TaskSlotStore(observability_bus=self.observability_bus)
        self.context_builder.register("slots", lambda: self._format_slots(), priority=8, volatile=True)
        self.autonomy_event_bus = EventBus()
        self.autonomy_tasks = TaskManager(self.autonomy_slots, self.autonomy_event_bus)
        if self.search_config.enabled or self.health_config.enabled or self.vision_state is not None or self.autonomy_config.emotion.enabled or self.autonomy_config.tokens.enabled or self.autonomy_config.tokens.recall.enabled or (
            self.autonomy_config.enabled and self.autonomy_config.jobs.enabled
        ):
            self.subagent_manager = MindScheduler(
                slot_store=self.autonomy_slots,
                mind_registry=self.mind_registry,
                observability_bus=self.observability_bus,
                shutdown_event=self.shutdown_event,
            )
            self._register_subagents()

        if self.vision_state is not None:
            # Add instructions to system prompt to correctly handle [vision] marked messages
            messages = self._conversation_store.snapshot()
            vision_prompt_added = False
            for i, message in enumerate(messages):
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    self._conversation_store.modify_message(
                        i, {"content": f"{message['content']} {SYSTEM_PROMPT_VISION_HANDLING}"}
                    )
                    vision_prompt_added = True
                    break
            if not vision_prompt_added:
                # Prepend a new system message with vision handling instructions
                current_messages = self._conversation_store.snapshot()
                self._conversation_store.replace_all(
                    [{"role": "system", "content": SYSTEM_PROMPT_VISION_HANDLING}] + current_messages
                )


    def _init_queues_and_mcp(self) -> None:
        # Initialize queues for inter-thread communication
        self._priority_inflight = InFlightCounter()
        self._autonomy_inflight = InFlightCounter()
        self.llm_queue_priority: queue.Queue[dict[str, Any]] = queue.Queue()
        autonomy_queue_max = self.autonomy_config.autonomy_queue_max
        autonomy_queue_size = autonomy_queue_max if autonomy_queue_max and autonomy_queue_max > 0 else 0
        self.llm_queue_autonomy: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=autonomy_queue_size)
        self.tool_calls_queue: queue.Queue[dict[str, Any]] = (
            queue.Queue()
        )  # Tool calls from LLMProcessor to ToolExecutor
        self.tts_queue: queue.Queue[str | SpeechText] = queue.Queue()  # Text from LLMProcessor to TTSynthesizer
        self.audio_queue: queue.Queue[AudioMessage] = queue.Queue()  # AudioMessages from TTSSynthesizer to AudioPlayer

        self.mcp_manager: MCPManager | None = None
        if self.mcp_servers:
            self.mcp_manager = MCPManager(
                self.mcp_servers,
                tool_timeout=self.tool_timeout,
                observability_bus=self.observability_bus,
            )
            self.mcp_manager.start()


    def _init_listeners(self) -> None:
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
                native_audio=self.native_audio,
                begin_user_turn=self._begin_user_turn,
                end_user_turn=self.inference_scheduler.end_interaction,
                turn_is_current=lambda generation: generation == self._quiet_generation,
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
                begin_user_turn=self._begin_user_turn,
            )


    def _init_primary_processor(self, llm_headers, llm_request_options, routing_config) -> None:
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
            autonomy_system_prompt=self.autonomy_config.system_prompt,
            autonomy_enabled=lambda: self.autonomy_config.enabled and not self.quiet_event.is_set(),
            autonomy_generation=lambda: self._autonomy_generation,
            on_autonomy_done=self._on_autonomy_done,
            autonomy_request_current=self._autonomy_request_current,
            quiet_mode=self.quiet_event.is_set,
            set_quiet_mode=self.set_quiet_mode,
            quiet_generation=lambda: self._quiet_generation,
            before_reply=self._react_to_input,
            before_context=self._clear_input_context,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
            extra_headers=llm_headers,
            lane="priority",
            inflight_counter=self._priority_inflight,
            inference_scheduler=self.inference_scheduler,
            native_audio=self.native_audio,
            request_options=llm_request_options,
        )
        self.decision_lists = DecisionListStore(
            resource_path("data/decision_lists.yaml"), lambda: self.llm_processor._build_tools(False),
            enabled=(routing_config or RoutingConfig()).enabled_for(
                str(self.completion_url), self.llm_processor.prompt_headers),
            backend_key=hashlib.sha256(f"{self.completion_url}|{self.llm_model}".encode()).hexdigest(),
        )
        self.router = DecisionRouter(
            self.decision_lists, self.inference_scheduler, str(self.completion_url), self.llm_model,
            self.llm_processor.prompt_headers, routing_config or RoutingConfig(),
            self.observability_bus, self.shutdown_event,
            mcp_catalog=self.mcp_manager.get_routing_catalog if self.mcp_manager else None,
            health_metrics=lambda: self.health_agent.covered_metrics() if self.health_agent else set(),
            recalled_topic=lambda: self.compaction_agent.recalled_topic if self.compaction_agent else None,
        )
        self.llm_processor.router = self.router

    def _init_autonomy_processors(self, llm_headers, llm_request_options) -> None:
        self.autonomy_llm_processors: list[LanguageModelProcessor] = []
        # Idle workers allow the console to enable autonomy without restarting.
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
                    autonomy_system_prompt=self.autonomy_config.system_prompt,
                    autonomy_enabled=lambda: self.autonomy_config.enabled and not self.quiet_event.is_set(),
                    autonomy_generation=lambda: self._autonomy_generation,
                    on_autonomy_done=self._on_autonomy_done,
                    on_autonomy_prompt=self._on_autonomy_prompt,
                    autonomy_thinking=self.autonomy_config.decision_thinking,
                    quiet_mode=self.quiet_event.is_set,
                    quiet_generation=lambda: self._quiet_generation,
                    mcp_manager=self.mcp_manager,
                    observability_bus=self.observability_bus,
                    extra_headers=llm_headers,
                    lane="autonomy",
                    inflight_counter=self._autonomy_inflight,
                    inference_scheduler=self.inference_scheduler,
                    request_options=llm_request_options,
                )
            )


    def _init_tools_and_speech(self) -> None:
        self.command_runner = SafeCommandRunner(self.observability_bus)
        self.tool_executor = ToolExecutor(
            end_user_turn=self.inference_scheduler.end_interaction,
            autonomy_enabled=lambda: self.autonomy_config.enabled and not self.quiet_event.is_set(),
            autonomy_generation=lambda: self._autonomy_generation,
            on_autonomy_done=self._on_autonomy_done,
            quiet_mode=self.quiet_event.is_set,
            quiet_generation=lambda: self._quiet_generation,
            decision_store=self.decision_lists,
            llm_queue_priority=self.llm_queue_priority,
            llm_queue_autonomy=self.llm_queue_autonomy,
            tool_calls_queue=self.tool_calls_queue,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            tool_config={
                **self.tool_config,
                "command_runner": self.command_runner,
                "vision_agent": self.vision_agent,
                "tts_queue": self.tts_queue,
                "preferences_store": self.preferences_store,
                "slot_store": self.autonomy_slots,
                "task_manager": self.autonomy_tasks,
                "search_agent": self.search_agent,
                "memory_agent": self.compaction_agent,
            },
            tool_timeout=self.tool_timeout,
            pause_time=self.PAUSE_TIME,
            mcp_manager=self.mcp_manager,
            observability_bus=self.observability_bus,
            on_tool_event=self._on_tool_event,
        )

        self.tts_synthesizer = TextToSpeechSynthesizer(
            on_response_ready=self.inference_scheduler.end_interaction,
            autonomy_enabled=lambda: self.autonomy_config.enabled,
            autonomy_generation=lambda: self._autonomy_generation,
            quiet_mode=self.quiet_event.is_set,
            quiet_generation=lambda: self._quiet_generation,
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
            playback_lock=self._quiet_lock,
            on_response_started=self.speech_listener.response_started if self.speech_listener else None,
            on_autonomy_done=self._on_autonomy_done,
            autonomy_enabled=lambda: self.autonomy_config.enabled,
            autonomy_generation=lambda: self._autonomy_generation,
            quiet_mode=self.quiet_event.is_set,
            quiet_generation=lambda: self._quiet_generation,
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
            speech_animation=self.speech_animation,
        )


    def _init_autonomy_loop(self) -> None:
        self.autonomy_ticker_thread: threading.Thread | None = None
        if self.autonomy_event_bus is not None:
            assert self.autonomy_event_bus is not None
            assert self.autonomy_slots is not None
            self.autonomy_loop = AutonomyLoop(
                quiet_mode=self.quiet_event.is_set,
                user_busy=self._autonomy_user_busy,
                quiet_generation=lambda: self._quiet_generation,
                autonomy_generation=lambda: self._autonomy_generation,
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
            self.autonomy_ticker_thread = threading.Thread(
                target=self._run_autonomy_ticker,
                name="AutonomyTicker",
                daemon=True,
            )


    def _start_components(self) -> None:
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


    def _health_runtime_status(self) -> dict[str, Any]:
        """Read existing in-process status without issuing tools or inference."""
        state = self.inference_scheduler.snapshot()
        now = time.time()
        waiting = state['waiting']
        audio = getattr(self, 'audio_io', None)
        capture = audio.capture_health() if audio and hasattr(audio, 'capture_health') else None
        if capture:
            capture = {**capture, 'expected': self.input_mode in {'audio', 'both'}
                       and not self.asr_muted_event.is_set()}
        manager = getattr(self, 'mcp_manager', None)
        return {'inference': {'capacity': state['capacity'], 'active': len(state['active']),
                              'waiting': len(waiting), 'oldest_wait_s':
                              round(max((now-r['queued_at'] for r in waiting), default=0), 1)},
                'mcp': [{'name': server['name'], 'connected': server['connected']}
                        for server in manager.status_snapshot()[:16]] if manager else [],
                'audio': {key: capture.get(key) for key in ('enabled', 'expected', 'connected', 'overflows', 'recoveries')}
                         if capture else None}

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
            request_options=self.llm_request_options,
            scheduler=self.inference_scheduler,
            shutdown_event=self.shutdown_event,
            cancelled=self.quiet_event.is_set,
        )

        health = getattr(self, 'health_config', None)
        if health and health.enabled:
            self.health_agent = HealthAgent(
                health_config=health, completion_url=str(self.completion_url),
                runtime_status=self._health_runtime_status, llm_config=llm_config,
                interactive_busy=lambda: self.quiet_event.is_set() or bool(
                    getattr(self, '_priority_inflight', None) and self._priority_inflight.value()
                    or getattr(self, 'llm_queue_priority', None) and not self.llm_queue_priority.empty()),
                slot_store=self.autonomy_slots, mind_registry=self.mind_registry,
                observability_bus=self.observability_bus, shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(self.health_agent, FixedInterval(health.interval_s))

        search = getattr(self, 'search_config', None)
        if search and search.enabled:
            self.search_agent = SearchAgent(
                settings=search, llm_config=llm_config,
                settings_path=resource_path("data/search_settings.yaml"),
                search=lambda arguments, timeout: self.mcp_manager.call_tool(
                    "mcp.internet_search.web_search_exa", arguments, timeout=timeout),
                slot_store=self.autonomy_slots, mind_registry=self.mind_registry,
                observability_bus=self.observability_bus, shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(self.search_agent, OnDemand())

        # Vision stays on E4B even when conversation uses a remote API model.
        if self.vision_state is not None and self.vision_config is not None:
            from ..vision.vision_mind import VisionMind

            vision = self.vision_config
            self.vision_agent = VisionMind(
                vision_config=vision,
                llm_config=LLMConfig(
                    url=vision.completion_url, model=vision.model, api_key=vision.api_key,
                    timeout=vision.timeout_s, scheduler=self.inference_scheduler,
                    shutdown_event=self.shutdown_event, owner="Vision", cancelled=self.quiet_event.is_set,
                ),
                vision_state=self.vision_state, slot_store=self.autonomy_slots,
                mind_registry=self.mind_registry, observability_bus=self.observability_bus,
                shutdown_event=self.shutdown_event,
                settings_path=resource_path("data/vision_settings.yaml"),
                greetings_path=resource_path("data/vision_greetings.json"),
            )
            self.subagent_manager.register(self.vision_agent, RandomAdaptive(
                bounds=lambda: (self.vision_agent.settings.interval_min_s, self.vision_agent.settings.interval_max_s),
                activity=lambda: self.vision_agent.camera.motion.snapshot()["activity"],
            ))

        # Emotional regulation is independent of background job scheduling.
        if self.autonomy_config.emotion.enabled:
            emotion_cfg = self.autonomy_config.emotion
            emotion_subagent_config = SubagentConfig(
                agent_id="emotion",
                title="Emotion Core",
                role="emotional_regulation",
            )
            emotion_agent = EmotionAgent(
                config=emotion_subagent_config,
                llm_config=replace(llm_config, owner="emotion"),
                emotion_config=emotion_cfg,
                slot_store=self.autonomy_slots,
                mind_registry=self.mind_registry,
                observability_bus=self.observability_bus,
                shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(emotion_agent, FixedInterval(emotion_cfg.tick_interval_s))
            self._emotion_agent = emotion_agent  # Keep reference for event pushing

        # Context maintenance is a background Mind even with autonomous speech OFF.
        if self.autonomy_config.tokens.enabled or self.autonomy_config.tokens.recall.enabled:
            tokens = self.autonomy_config.tokens
            threshold = tokens.token_threshold
            if tokens.model_context_window:
                threshold = min(threshold, int(tokens.model_context_window * tokens.target_utilization))
            vision = self.vision_config if self.vision_config and self.vision_config.enabled else None
            memory_llm = (
                LLMConfig(
                    url=vision.completion_url, model=vision.model, api_key=vision.api_key,
                    timeout=vision.timeout_s, owner="Compaction", scheduler=self.inference_scheduler,
                    shutdown_event=self.shutdown_event, cancelled=self.quiet_event.is_set,
                )
                if vision else replace(llm_config, owner="Compaction")
            )
            self.compaction_agent = CompactionAgent(
                config=SubagentConfig(agent_id="compaction", title="Memory Core", role="Recall and context management"),
                llm_config=memory_llm,
                conversation_store=self._conversation_store, token_threshold=threshold,
                preserve_recent=tokens.preserve_recent_messages, summary_max_tokens=tokens.summary_max_tokens,
                summary_input_tokens=tokens.summary_input_tokens,
                recall_config=tokens.recall, compaction_enabled=tokens.enabled,
                interactive_busy=lambda: bool(getattr(self, "_priority_inflight", None) and
                    (self._priority_inflight.value() or not self.llm_queue_priority.empty())) or self.quiet_event.is_set(),
                slot_store=self.autonomy_slots, mind_registry=self.mind_registry,
                observability_bus=self.observability_bus, shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(self.compaction_agent, FixedInterval(tokens.tick_interval_s))

        if not (self.autonomy_config.enabled and jobs_config.enabled):
            return

        if jobs_config.hacker_news.enabled:
            hn_config = SubagentConfig(
                agent_id="hn_top",
                title="Hacker News",
                role="news_monitor",
            )
            hn_subagent = HackerNewsSubagent(
                config=hn_config,
                top_n=jobs_config.hacker_news.top_n,
                min_score=jobs_config.hacker_news.min_score,
                llm_config=replace(llm_config, owner="hn_top"),
                slot_store=self.autonomy_slots,
                mind_registry=self.mind_registry,
                observability_bus=self.observability_bus,
                shutdown_event=self.shutdown_event,
            )
            self.subagent_manager.register(hn_subagent, FixedInterval(jobs_config.hacker_news.interval_s))

        if jobs_config.weather.enabled:
            if jobs_config.weather.latitude is None or jobs_config.weather.longitude is None:
                logger.warning("Weather subagent enabled but latitude/longitude are missing.")
            else:
                weather_config = SubagentConfig(
                    agent_id="weather",
                    title="Weather",
                    role="weather_monitor",
                )
                weather_subagent = WeatherSubagent(
                    config=weather_config,
                    latitude=jobs_config.weather.latitude,
                    longitude=jobs_config.weather.longitude,
                    timezone=jobs_config.weather.timezone,
                    temp_change_c=jobs_config.weather.temp_change_c,
                    wind_alert_kmh=jobs_config.weather.wind_alert_kmh,
                    llm_config=replace(llm_config, owner="weather"),
                    slot_store=self.autonomy_slots,
                    mind_registry=self.mind_registry,
                    observability_bus=self.observability_bus,
                    shutdown_event=self.shutdown_event,
                )
                self.subagent_manager.register(weather_subagent, FixedInterval(jobs_config.weather.interval_s))

        # Observer agent - monitors behavior and proposes adjustments
        observer_config = SubagentConfig(
            agent_id="observer",
            title="Behavior Observer",
            role="meta_supervision",
        )
        observer_agent = ObserverAgent(
            config=observer_config,
            llm_config=replace(llm_config, owner="observer"),
            conversation_store=self._conversation_store,
            constitutional_state=self.constitutional_state,
            sample_count=10,
            min_samples_for_analysis=5,
            slot_store=self.autonomy_slots,
            mind_registry=self.mind_registry,
            observability_bus=self.observability_bus,
            shutdown_event=self.shutdown_event,
        )
        self.subagent_manager.register(observer_agent, FixedInterval(120), run_on_start=False)

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

        asr_model = None
        if config.input_mode != "text" and not config.native_audio.enabled:
            asr_model = get_audio_transcriber(engine_type=config.asr_engine)

        tts_model: SpeechSynthesizerProtocol
        tts_model = get_speech_synthesizer(config.voice)

        audio_io = get_audio_system(
            backend_type=config.audio_io,
            backend_options=config.audio_io_options,
        )

        try:
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
                native_audio_config=config.native_audio,
                llm_request_options=config.llm_request_options,
                inference_config=config.inference,
                routing_config=config.routing,
                health_config=config.health,
                search_config=config.search,
            )
        except Exception:
            cls._close_audio_backend(audio_io)
            raise

    @staticmethod
    def _close_audio_backend(audio_io: AudioProtocol) -> None:
        """Close a backend without breaking legacy structural implementations."""
        close = getattr(audio_io, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close audio I/O backend")

    @classmethod
    def from_yaml(cls, path: str | Path | list[str] | list[Path]) -> "Glados":
        """
        Create a Glados instance from one or more configuration files.
        Explicitly specified options in later configuration files override options specified in earlier files.

        Parameters:
            path: One or multiple paths to the YAML configuration file(s) containing Glados settings.

        Returns:
            Glados: A new Glados instance configured with settings from the specified YAML file(s).

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
        self.inference_scheduler.end_interaction(self._quiet_generation, "shutdown")

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

        self._close_audio_backend(self.audio_io)

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
        generation = self._begin_user_turn()
        if self.observability_bus:
            self.observability_bus.emit(
                source=source,
                kind="user_input",
                message=trim_message(text),
            )
        self.processing_active_event.set()
        self.llm_queue_priority.put(
            {
                "role": "user",
                "content": text,
                "_enqueued_at": time.time(),
                "_lane": "priority",
                "_quiet_generation": generation,
            }
        )
        if self.interaction_state:
            self.interaction_state.mark_user()
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
                usage="/autonomy on|off",
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
        vision_enabled = self.vision_state is not None
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
        self.inference_scheduler.end_interaction(self._quiet_generation, "shutdown")
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
            lines.append(f"- [{slot.slot_id}] {slot.title}: {slot.status}{summary_text}")
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

    def _react_to_input(self, message: dict) -> None:
        self._recall_input(message)
        if self._emotion_agent and not self.quiet_event.is_set():
            self._emotion_agent.react(str(message.get("content", "")), message.get("_native_audio"))

    def _clear_input_context(self, message: dict) -> None:
        """Clear the previous topic before speculative inference; ignored speech cannot launch recall."""
        if getattr(self, "search_agent", None):
            self.search_agent.clear_context()
        if self.compaction_agent:
            self.compaction_agent.request_recall("", turn_id=str(message.get("_quiet_generation", self._quiet_generation)))

    def _recall_input(self, message: dict) -> None:
        if self.compaction_agent and not self.quiet_event.is_set():
            users = [m["content"] for m in self._conversation_store.snapshot()
                     if m.get("role") == "user" and isinstance(m.get("content"), str)]
            # Raw audio has no reliable text query when optional transcripts are off.
            query = message.get("content", "")
            if message.get("_native_audio") and str(query).startswith("[User spoke via audio"):
                query = ""
            self.compaction_agent.request_recall(
                query if isinstance(query, str) else "", users[-1] if users else "",
                turn_id=str(message.get("_quiet_generation", self._quiet_generation)),
                audio=message.get("_native_audio"),
            )

    def _emotion_prompt(self) -> str | None:
        if not self._emotion_agent:
            return None
        state = self._emotion_agent.state
        return state.to_prompt() + "\nCurrent tone guidance: " + state.response_instructions()

    def _begin_user_turn(self) -> int:
        """Invalidate prior work permanently, including speech still being synthesized."""
        with self._quiet_lock:
            self._quiet_generation += 1
            self.inference_scheduler.begin_interaction(self._quiet_generation)
            if getattr(self, "autonomy_loop", None):
                self.autonomy_loop.reset()
            self.processing_active_event.clear()
            self.audio_io.stop_speaking()
            for pending in (self.llm_queue_priority, self.tts_queue, self.audio_queue, self.tool_calls_queue):
                while True:
                    try:
                        pending.get_nowait()
                    except queue.Empty:
                        break
            return self._quiet_generation

    def set_quiet_mode(self, enabled: bool) -> None:
        with self._quiet_lock:
            if enabled == self.quiet_event.is_set():
                return
            self.inference_scheduler.end_interaction(self._quiet_generation, "quiet_changed")
            self._quiet_generation += 1
            if getattr(self, "autonomy_loop", None):
                self.autonomy_loop.reset()
            if enabled:
                self.quiet_event.set()
                self.processing_active_event.clear()
                self.audio_io.stop_speaking()
                self.currently_speaking_event.clear()
                self.speech_animation.set(False)
                for pending in (self.tts_queue, self.audio_queue, self.llm_queue_autonomy, self.tool_calls_queue):
                    while True:
                        try:
                            pending.get_nowait()
                        except queue.Empty:
                            break
                if self.subagent_manager:
                    for status in self.subagent_manager.list_agents():
                        agent = self.subagent_manager.get(status.agent_id)
                        self._quiet_saved_pauses[status.agent_id] = agent.paused
                        self.subagent_manager.pause(status.agent_id)
            else:
                self.quiet_event.clear()
                if self.subagent_manager:
                    for agent_id, paused in self._quiet_saved_pauses.items():
                        agent = self.subagent_manager.get(agent_id)
                        if agent:
                            self.subagent_manager.pause(agent_id, paused)
                self._quiet_saved_pauses.clear()
            self.observability_bus.emit("quiet", "control", "Sleeping; listening only for wake requests" if enabled else "Awake")

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
            1 for m in messages if isinstance(m.get("content"), str) and m["content"].startswith("[summary]")
        )
        lines = [
            "Context Usage:",
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
                "coalescing=automatic"
            )
        head = args[0].lower()
        if head in {"on", "off", "true", "false", "enable", "enabled", "disable", "disabled"}:
            enabled = head in {"on", "true", "enable", "enabled"}
            self.set_autonomy_enabled(enabled)
            return f"Autonomy {'enabled' if enabled else 'disabled'}."
        return "Usage: /autonomy on|off (idle checks coalesce automatically)"

    def set_autonomy_enabled(self, enabled: bool) -> None:
        with self._quiet_lock:
            if enabled == self.autonomy_config.enabled:
                return
            self._autonomy_generation += 1
            self.autonomy_config.enabled = enabled
            if self.autonomy_loop:
                self.autonomy_loop.reset()
            if not enabled:
                while True:
                    try:
                        self.llm_queue_autonomy.get_nowait()
                    except queue.Empty:
                        break
                if self.speech_player.autonomy_speaking:
                    self.audio_io.stop_speaking()
        self.observability_bus.emit("autonomy", "control", "Enabled" if enabled else "Disabled")

    def _autonomy_user_busy(self) -> bool:
        return (bool(getattr(getattr(self, "speech_listener", None), "_recording_started", False))
                or self.llm_processor._request_active.is_set()
                or self._priority_inflight.value() > 0 or not self.llm_queue_priority.empty()
                or not self.tts_queue.empty() or not self.audio_queue.empty())

    def _on_autonomy_done(self, cycle: str, outcome: str, reason: str) -> None:
        if self.autonomy_loop:
            self.autonomy_loop.finish_cycle(cycle, outcome, reason)

    def _on_autonomy_prompt(self, decision: dict, meta: dict) -> bool:
        with self._quiet_lock:
            if (meta.get("_quiet_generation") != self._quiet_generation or
                    meta.get("_autonomy_generation") != self._autonomy_generation or not self.autonomy_loop):
                return False
            return self.autonomy_loop.prompt_main(meta["_autonomy_cycle"], decision, self.llm_queue_priority, meta.get("_evidence_versions"))

    def _autonomy_request_current(self, meta: dict) -> bool:
        return bool(self.autonomy_loop and self.autonomy_loop.request_current(meta["_autonomy_cycle"]))

    def _cmd_config(self, _args: list[str]) -> str:
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"autonomy.enabled={self.autonomy_config.enabled}, "
            f"autonomy.jobs.enabled={jobs_enabled}, "
            f"vision.enabled={self.vision_state is not None}"
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
        # These cores already have dedicated context: observation, PAD and compacted history.
        slots = [slot for slot in self.autonomy_slots.list_slots()
                 if slot.slot_id not in {"vision", "emotion", "compaction", "health"} and not slot.handled]
        memory = self.autonomy_slots.get_slot("compaction")
        recall = memory.context if memory else None
        if not slots and not recall:
            return None
        lines = ["[tasks]"]
        if recall:
            lines.append(recall)
        for slot in slots:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- [{slot.slot_id}] {slot.title}: {slot.status}{summary_text}")
            if slot.context and slot.owner_id is None:
                lines.append(slot.context[:1200])
        return "\n".join(lines)

    def _run_autonomy_ticker(self) -> None:
        assert self.autonomy_event_bus is not None
        logger.info("AutonomyTicker thread started.")
        while not self.shutdown_event.is_set():
            if self.autonomy_config.enabled and not self.quiet_event.is_set():
                self.autonomy_event_bus.publish(TimeTickEvent(ticked_at=time.time()))
            self.shutdown_event.wait(timeout=self.autonomy_config.tick_interval_s)
        logger.info("AutonomyTicker thread finished.")


if __name__ == "__main__":
    glados_config = GladosConfig.from_yaml("glados_config.yaml")
    glados = Glados.from_config(glados_config)
    glados.run()
