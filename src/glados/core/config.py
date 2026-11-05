"""Configuration for the Glados assistant: the YAML profile model and its validation."""

import os
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, model_validator
import yaml

from ..autonomy import AutonomyConfig
from ..autonomy.agents.health_agent import HealthConfig
from ..autonomy.agents.nexus_agent import NexusConfig
from ..autonomy.agents.search_agent import SearchConfig
from ..mcp import MCPServerConfig
from ..tools.agent import AgentConfig
from ..vision import VisionConfig
from ..webapp import WebappConfig
from .hardware import detect_hardware, get_adaptive_llm_settings, probe_model_spec
from .inference import InferenceConfig
from .native_audio import NativeAudioConfig
from .routing import RoutingConfig

FRENCH_TURN_REMINDER = "Réponds en français, même si la demande, le contexte ou les outils sont en anglais."
FRENCH_REPLY_INSTRUCTION = (
    "Language: the user speaks French. Always reply in natural spoken French, keeping your GLaDOS personality, "
    "whatever the language of these instructions, tool results or memory. Write numbers, times and units the way "
    "they are said aloud in French."
)


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
    wake_word_follow_up_s: float = Field(
        default=0.0,
        ge=0,
        le=120,
        description=("After GLaDOS speaks, how long a reply is accepted without repeating the wake word."),
    )
    wake_word_session_s: float = Field(
        default=0.0,
        ge=0,
        le=86400,
        description="After the wake word is heard, how long speech is accepted without it; saying it again restarts.",
    )
    voice: str
    announcement: str | None
    llm_headers: dict[str, str] | None = None
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    nexus: NexusConfig = Field(default_factory=NexusConfig)
    llm_request_options: dict[str, Any] | None = None
    llm_timeout: float | None = None
    adaptive_hardware: bool = True
    hardware_profile: dict[str, Any] | None = None
    tui_theme: str | None = None
    personality_preprompt: list[PersonalityPrompt]
    end_of_speech_ms: int = Field(
        default=416,
        ge=160,
        le=3000,
        description=(
            "Silence that ends an utterance. Lower replies sooner but may cut slow speakers off mid-sentence."
        ),
    )
    language: Literal["en", "fr"] = Field(
        default="en",
        description=("Spoken language: replies, ASR expectations and GLaDOS-voice phonemization (fr uses espeak-ng)."),
    )
    slow_clap_audio_path: str = "data/slow-clap.mp3"
    tool_timeout: float = 30.0
    agent: AgentConfig = Field(default_factory=AgentConfig)
    vision: VisionConfig | None = None
    autonomy: AutonomyConfig | None = None
    mcp_servers: list[MCPServerConfig] | None = None
    webapp: WebappConfig | None = None

    @model_validator(mode="after")
    def _french_speech_is_addressed(self) -> "GladosConfig":
        # A profile that ignores non-English speech would otherwise discard every French request.
        if self.language == "fr" and (self.routing.addressed_language or "").lower() == "english":
            self.routing.addressed_language = "French"
        return self

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

    @model_validator(mode="after")
    def _apply_hardware_profile(self) -> "GladosConfig":
        """Adapt LLM parameters (context size, tokens, timeouts, workers) dynamically to host hardware."""
        if not self.adaptive_hardware:
            return self
        is_local = any(h in str(self.completion_url).lower() for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434"))
        profile = detect_hardware()
        spec = probe_model_spec(str(self.completion_url), self.llm_model) if is_local else None
        adaptive = get_adaptive_llm_settings(profile, spec=spec, is_local=is_local)
        self.hardware_profile = adaptive
        if self.llm_timeout is None:
            self.llm_timeout = adaptive["llm_timeout"]
        if is_local:
            req_opts = dict(self.llm_request_options or {})
            has_ctx = "num_ctx" in req_opts or (
                "options" in req_opts and isinstance(req_opts["options"], dict) and "num_ctx" in req_opts["options"]
            )
            opts = dict(req_opts.get("options") or {})
            if not has_ctx:
                opts["num_ctx"] = adaptive["num_ctx"]
            if "num_predict" not in opts and "num_predict" not in req_opts:
                opts["num_predict"] = adaptive["num_predict"]
            if "num_thread" not in opts and "num_thread" not in req_opts:
                opts["num_thread"] = adaptive["num_thread"]
            req_opts["options"] = opts
            self.llm_request_options = req_opts
            if self.autonomy and self.autonomy.autonomy_parallel_calls == 2:
                self.autonomy.autonomy_parallel_calls = adaptive["autonomy_parallel_calls"]
        logger.info(
            "Adaptive hardware: {} -> Context: {} tok, Timeout: {}s, Autonomy workers: {}",
            adaptive["hardware_summary"],
            adaptive["num_ctx"],
            self.llm_timeout,
            self.autonomy.autonomy_parallel_calls if self.autonomy else 1,
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

        files = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(path) for path in paths]
        config: dict[str, Any] = {}

        for path in files:
            # Try different encodings
            for encoding in ["utf-8", "utf-8-sig"]:
                try:
                    data = yaml.safe_load(path.read_text(encoding=encoding))
                    break
                except UnicodeDecodeError as exc:
                    if encoding == "utf-8-sig":
                        raise ValueError(f"Could not decode YAML file {path} with any supported encoding") from exc

            data = data or {}

            # Navigate through nested keys
            for key in key_to_config:
                data = data[key]

            # Update config dict - config from later paths overrides earlier config
            config = GladosConfig._dict_deep_merge(config, data)

        return cls.model_validate(config)

    @staticmethod
    def _dict_deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merge two dictionaries into one.
        Values from the second dictionary override values from the first.
        Note: mutates the first dictionary.

        Returns:
            The merged dictionary.
        """
        for key, value in b.items():
            if isinstance(a.get(key), dict) and isinstance(value, dict):
                GladosConfig._dict_deep_merge(a[key], value)
            else:
                a[key] = value
        return a

    def to_chat_messages(self) -> list[dict[str, str]]:
        """Convert personality preprompt to chat message format."""
        messages = [prompt.to_chat_message() for prompt in self.personality_preprompt]
        if self.language == "fr":
            messages.append({"role": "system", "content": FRENCH_REPLY_INSTRUCTION})
        return messages
