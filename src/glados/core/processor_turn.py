"""Request and stream data owned by one processor turn, never persisted as chat."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .speech_markup import SpeechMarkupParser

if TYPE_CHECKING:
    from .inference import InferenceRequest
    from .speculative import SpeculativeStream


@dataclass
class ProcessorTurn:
    llm_input: dict[str, Any] = field(default_factory=dict)
    llm_message: dict[str, Any] = field(default_factory=dict)
    audio_content: list[dict[str, Any]] | None = None
    autonomy_mode: bool = False
    turn_generation: int | None = None
    inflight_guard: bool = False
    inference_lease: InferenceRequest | None = None
    draft: SpeculativeStream | None = None
    draft_messages: list[dict[str, Any]] = field(default_factory=list)
    draft_sources: dict[int, str] = field(default_factory=dict)
    route: dict[str, Any] | None = None
    routing_permit: dict[str, Any] | None = None
    read_only: bool = False
    allow_tools: bool = False
    available_tools: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_names: set[str] = field(default_factory=set)
    base_messages: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    search_planning: bool = False
    wait_s: float | None = None
    queue_depth: int | None = None


@dataclass
class ResponseState:
    tool_calls_buffer: list[dict[str, Any]] = field(default_factory=list)
    autonomy_json: list[str] = field(default_factory=list)
    sentence_buffer: list[str] = field(default_factory=list)
    speech_parser: SpeechMarkupParser = field(default_factory=SpeechMarkupParser)
    sentence_emotion: str | None = None
    thinking_buffer: list[str] = field(default_factory=list)
    in_thinking: bool = False
    harmony_mode: bool = False
    http_error_detail: tuple[str | int, str] | None = None
