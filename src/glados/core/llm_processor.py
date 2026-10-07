from __future__ import annotations

from collections.abc import Callable
import copy

# --- llm_processor.py ---
import json
import queue
import re
import threading
import time
from typing import Any, ClassVar
from urllib.parse import urlparse
import uuid

from loguru import logger
from pydantic import HttpUrl  # If HttpUrl is used by config
import requests

from ..autonomy import ConstitutionalState, TaskSlotStore
from ..autonomy.context import chat_evidence, slot_evidence, slot_version
from ..autonomy.decision import decision_prompt, decision_schema, parse_decision
from ..mcp import MCPManager
from ..observability import ObservabilityBus, trim_message
from ..tools import tool_definitions
from ..vision.vision_state import VisionState
from .clock import clock_context, current_time
from .context import ContextBuilder
from .context_budget import reduce_request_context
from .context_inspection import describe_context
from .conversation_store import ConversationStore
from .inference import InferenceCancelledError, InferenceScheduler
from .llm_tracking import InFlightCounter
from .native_audio import NativeAudioInput
from .operator_state import CONSOLE_PROMPT
from .processor_turn import ProcessorTurn, ResponseState
from .prompts import (
    ROUTING_FALLBACK_INSTRUCTIONS,
    SEARCH_AVAILABLE_INSTRUCTIONS,
    SEARCH_COMPLETED_INSTRUCTIONS,
    SEARCH_FINDINGS_INSTRUCTIONS,
    SEARCH_PENDING_INSTRUCTIONS,
)
from .speculative import SpeculativeStream
from .speech_chunking import split_speech_clauses
from .speech_markup import SPEECH_DIRECTION_PROMPT, SpeechText
from .store import Store

INTERNET_SEARCH_TOOL = "mcp.internet_search.web_search_exa"


class LanguageModelProcessor:
    """
    A thread that processes text input for a language model, streaming responses and sending them to TTS.

    Supports multiple lanes (priority/autonomy) - instantiate once per lane for parallel inference.
    Handles streaming with thinking tag extraction for reasoning models.
    """

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n", "\n\n"}

    # Standard thinking tags (GLM-4.7, MiniMax M2.7/M3, DeepSeek, etc.)
    THINKING_OPEN_TAGS: ClassVar[tuple[str, ...]] = ("<think>", "<thinking>", "<reasoning>")
    THINKING_CLOSE_TAGS: ClassVar[tuple[str, ...]] = ("</think>", "</thinking>", "</reasoning>")

    # GPT-OSS harmony format channel markers
    HARMONY_CHANNEL_MARKER: ClassVar[str] = "<|channel|>"
    HARMONY_ANALYSIS_CHANNELS: ClassVar[tuple[str, ...]] = ("analysis", "commentary")
    HARMONY_FINAL_CHANNEL: ClassVar[str] = "final"
    HARMONY_MESSAGE_MARKER: ClassVar[str] = "<|message|>"
    HARMONY_END_MARKER: ClassVar[str] = "<|end|>"

    def __init__(
        self,
        llm_input_queue: queue.Queue[dict[str, Any]],
        tool_calls_queue: queue.Queue[dict[str, Any]],
        tts_input_queue: queue.Queue[str | SpeechText],
        conversation_store: ConversationStore,
        completion_url: HttpUrl,
        model_name: str,  # Renamed from 'model' to avoid conflict
        api_key: str | None,
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
        vision_state: VisionState | None = None,
        slot_store: TaskSlotStore | None = None,
        preferences_store: Store[Any] | None = None,
        constitutional_state: ConstitutionalState | None = None,
        context_builder: ContextBuilder | None = None,
        autonomy_system_prompt: str | None = None,
        mcp_manager: MCPManager | None = None,
        observability_bus: ObservabilityBus | None = None,
        extra_headers: dict[str, str] | None = None,
        lane: str = "priority",
        inflight_counter: InFlightCounter | None = None,
        inference_scheduler: InferenceScheduler | None = None,
        native_audio: NativeAudioInput | None = None,
        request_options: dict[str, Any] | None = None,
        autonomy_enabled: Callable[[], bool] = lambda: True,
        quiet_mode: Callable[[], bool] = lambda: False,
        quiet_generation: Callable[[], int] = lambda: 0,
        set_quiet_mode: Callable[[bool], None] | None = None,
        before_reply: Callable[[dict], None] | None = None,
        before_context: Callable[[dict], None] | None = None,
        autonomy_generation: Callable[[], int] = lambda: 0,
        on_autonomy_done: Callable[[str, str, str], None] | None = None,
        on_autonomy_prompt: Callable[[dict, dict], bool] | None = None,
        autonomy_thinking: bool = False,
        autonomy_request_current: Callable[[dict], bool] = lambda meta: True,
    ) -> None:
        self.llm_input_queue = llm_input_queue
        self._autonomy_enabled = autonomy_enabled
        self._autonomy_generation, self._on_autonomy_done = autonomy_generation, on_autonomy_done
        self._on_autonomy_prompt = on_autonomy_prompt
        self._autonomy_thinking = autonomy_thinking
        self._autonomy_request_current = autonomy_request_current
        self._autonomy_response = False
        self._response_has_text = False
        self._autonomy_context: list[dict] = []
        self._autonomy_meta: dict = {}
        self._autonomy_handoff = False
        self._autonomy_offered_tools: list[dict] = []
        self._autonomy_error = ""
        self._quiet_mode, self._set_quiet_mode, self._before_reply = quiet_mode, set_quiet_mode, before_reply
        self._before_context = before_context
        self._quiet_generation = quiet_generation
        self._reply_generation = quiet_generation()
        self.tool_calls_queue = tool_calls_queue
        self.tts_input_queue = tts_input_queue
        self._conversation_store = conversation_store
        self.completion_url = completion_url
        self.model_name = model_name
        self.api_key = api_key
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time
        self.vision_state = vision_state
        self.slot_store = slot_store
        self.preferences_store = preferences_store
        self.constitutional_state = constitutional_state
        self.context_builder = context_builder
        self.autonomy_system_prompt = autonomy_system_prompt
        self.mcp_manager = mcp_manager
        self._observability_bus = observability_bus
        self._lane = lane
        self._inflight_counter = inflight_counter
        self._inference_scheduler = inference_scheduler
        self._native_audio = native_audio
        self._request_options = request_options or {}
        self.router = None
        self._context_lock = threading.Lock()
        self._request_active = threading.Event()
        self._last_context: dict[str, Any] | None = None
        self._context_sources: dict[int, str] = {}
        self._ollama_mode = self._is_ollama_endpoint()

        self.prompt_headers = {"Content-Type": "application/json"}
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            self.prompt_headers.update(extra_headers)

    def _is_ollama_endpoint(self) -> bool:
        try:
            parsed = urlparse(str(self.completion_url))
        except Exception:
            return False
        path = (parsed.path or "").rstrip("/")
        return path.endswith("/api/chat")

    def _autonomy_cancelled(self) -> bool:
        return (self._lane == "autonomy" or self._autonomy_response) and (
            not self._autonomy_enabled()
            or self._autonomy_meta.get("_autonomy_generation", self._autonomy_generation())
            != self._autonomy_generation()
        )

    @staticmethod
    def _sanitize_messages_for_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_keys = {"role", "content", "name", "tool_calls", "tool_call_id", "images", "function_call"}
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            cleaned = {key: value for key, value in message.items() if key in allowed_keys}
            tool_calls = cleaned.get("tool_calls")
            if isinstance(tool_calls, list):
                normalized_calls: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            function["arguments"] = {}
                    tool_call["function"] = function
                    normalized_calls.append(tool_call)
                cleaned["tool_calls"] = normalized_calls
            sanitized.append(cleaned)
        return sanitized

    @staticmethod
    def _sanitize_messages_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_keys = {"role", "content", "name", "tool_calls", "tool_call_id"}
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            cleaned = {key: value for key, value in message.items() if key in allowed_keys}
            tool_calls = cleaned.get("tool_calls")
            if isinstance(tool_calls, list):
                normalized_calls: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        function["arguments"] = json.dumps(arguments or {})
                    tool_call_id = tool_call.get("id") or ""
                    normalized_calls.append(
                        {
                            "id": tool_call_id,
                            "type": tool_call.get("type", "function"),
                            "function": function,
                        }
                    )
                cleaned["tool_calls"] = normalized_calls
            sanitized.append(cleaned)
        return sanitized

    def _clean_raw_bytes(self, line: bytes) -> dict[str, str] | None:
        """
        Clean and parse a raw byte line from the LLM response.
        Handles both OpenAI and Ollama formats, returning a dictionary or None if parsing fails.

        Args:
            line (bytes): The raw byte line from the LLM response.
        Returns:
            dict[str, str] | None: Parsed JSON dictionary or None if parsing fails.
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]
                if json_str.strip() == "[DONE]":  # Handle OpenAI [DONE] marker
                    return {"done_marker": "True"}
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except json.JSONDecodeError:
            # If it's not JSON, it might be Ollama's final summary object which isn't part of the stream
            # Or just noise.
            logger.trace(
                "LLM Processor: Failed to parse non-JSON server response line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )  # Log only a part
            return None
        except Exception as e:
            logger.warning(
                "LLM Processor: Failed to parse server response: "
                f"{e} for line: {line[:100].decode('utf-8', errors='replace')}"
            )
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | list[dict[str, Any]] | None:
        # Copy from Glados._process_chunk
        if not line or not isinstance(line, dict):
            return None
        try:
            # Handle OpenAI format
            if line.get("done_marker"):  # Handle [DONE] marker
                return None
            elif "choices" in line:  # OpenAI format
                delta = line.get("choices", [{}])[0].get("delta", {})
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    return tool_calls

                content = delta.get("content")
                return str(content) if content else None
            # Handle Ollama format
            else:
                message = line.get("message", {})
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return tool_calls

                content = message.get("content")
                return content if content else None
        except Exception as e:
            logger.error(f"LLM Processor: Error processing chunk: {e}, chunk: {line}")
            return None

    def _process_tool_chunks(
        self,
        tool_calls_buffer: list[dict[str, Any]],
        tool_chunks: list[dict[str, Any]],
    ) -> None:
        """
        Extract tool call data from chunks to populate final tool_calls_buffer.

        Args:
            tool_calls_buffer: List of tool calls to be run.
            tool_chunks: List of streaming tool call data split into chunks.
        """
        for tool_chunk in tool_chunks:
            tool_chunk_index = tool_chunk.get("index", 0)
            try:
                tool_chunk_index = int(tool_chunk_index)
            except (TypeError, ValueError):
                tool_chunk_index = 0
            if tool_chunk_index < 0:
                tool_chunk_index = 0
            while tool_chunk_index >= len(tool_calls_buffer):
                # we have a new tool call to initialize
                tool_calls_buffer.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            tool_call = tool_calls_buffer[tool_chunk_index]

            tool_id = tool_chunk.get("id")
            name = tool_chunk.get("function", {}).get("name")
            arguments = tool_chunk.get("function", {}).get("arguments")

            if tool_id:
                tool_call["id"] += tool_id
            if name:
                tool_call["function"]["name"] += name
            if arguments:
                if isinstance(arguments, str):
                    # OpenAI format
                    tool_call["function"]["arguments"] += arguments
                else:
                    # Ollama format
                    tool_call["function"]["arguments"] = arguments

    @staticmethod
    def _sanitize_tool_name(name: str) -> str:
        return "".join(ch for ch in name.casefold() if ch.isalnum())

    def _normalize_tool_name(self, name: str, known_names: set[str]) -> str:
        if not name or not known_names:
            return name
        if name in known_names:
            return name
        for candidate in known_names:
            if candidate.casefold() == name.casefold():
                return candidate
        if name.startswith("mcp."):
            candidates = [candidate for candidate in known_names if candidate.startswith("mcp.")]
        else:
            candidates = [candidate for candidate in known_names if not candidate.startswith("mcp.")]
        if not candidates:
            candidates = list(known_names)
        normalized = self._sanitize_tool_name(name)
        if normalized:
            normalized_candidates = [(candidate, self._sanitize_tool_name(candidate)) for candidate in candidates]
            exact = [candidate for candidate, norm in normalized_candidates if norm == normalized]
            if len(exact) == 1:
                return exact[0]
            substring = [candidate for candidate, norm in normalized_candidates if norm and norm in normalized]
            if substring:
                return max(substring, key=len)
            superstrings = [candidate for candidate, norm in normalized_candidates if normalized and normalized in norm]
            if len(superstrings) == 1:
                return superstrings[0]
        return name

    def _normalize_tool_calls(self, tool_calls: list[dict[str, Any]], tool_names: set[str]) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            if not tool_name:
                continue
            tool_call["function"]["name"] = self._normalize_tool_name(tool_name, tool_names)

    @staticmethod
    def _filter_tools_for_message(tools: list[dict[str, Any]], content: str) -> list[dict[str, Any]]:
        text = content.casefold()
        wants_system = any(
            keyword in text
            for keyword in (
                "system",
                "status",
                "cpu",
                "memory",
                "ram",
                "disk",
                "storage",
                "network",
                "ip",
                "uptime",
                "temperature",
                "temp",
                "process",
                "battery",
                "power",
                "load",
            )
        )
        wants_clap = "clap" in text
        filtered: list[dict[str, Any]] = []
        for tool in tools:
            name = tool.get("function", {}).get("name", "")
            if name == "slow clap" and not wants_clap:
                continue
            if name.startswith("mcp.") and name != INTERNET_SEARCH_TOOL and not wants_system:
                continue
            filtered.append(tool)
        return filtered

    @staticmethod
    def _measurement_hint(content: str) -> str:
        """Make structured CPU readings explicit without confusing periods with values."""
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            return ""
        if not isinstance(result, dict):
            return ""
        if all(key in result for key in ("load_1m", "load_5m", "load_15m")):
            values = [str(result[key]) for key in ("load_1m", "load_5m", "load_15m")]
        elif result.get("task") == "cpu_load" and result.get("ok") is True:
            values = str(result.get("stdout", "")).split()[:3]
        else:
            return ""
        if len(values) != 3 or any(not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v) for v in values):
            return ""
        readings = [f"{float(v):.2f}" for v in values]
        return (
            " Verified measurement, rounded to two decimals: the CPU load averages are "
            + ", ".join(readings) + " for the one-minute, five-minute and fifteen-minute periods respectively. "
            "Use this exact factual sentence in your reply. The period names are not the measured values."
        )

    def _process_tool_call(
        self,
        tool_calls: list[dict[str, Any]],
        autonomy_mode: bool,
        tool_names: set[str],
        read_only: bool = False,
        routing_permit: dict | None = None,
    ) -> None:
        """
        Add tool calls to conversation history and send each to the tool calls queue.

        Args:
            tool_calls: List of tool calls to be run.
        """
        if autonomy_mode or self._autonomy_response:
            self._autonomy_error = "Autonomy notifications cannot execute tools"
            return
        self._normalize_tool_calls(tool_calls, tool_names)
        # A generated name cannot expand the capabilities offered for this request.
        tool_calls = [call for call in tool_calls if call.get("function", {}).get("name") in tool_names]
        if not tool_calls:
            if not autonomy_mode:
                self._process_sentence_for_tts(["I don't have an available tool for that action."])
            return
        for tool_call in tool_calls:
            tool_call.setdefault("type", "function")
            if not tool_call.get("id"):
                tool_call["id"] = f"toolcall_{uuid.uuid4().hex}"
        message = {"role": "assistant", "index": 0, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
        self._conversation_store.append(message)
        tool_labels = [call.get("function", {}).get("name", "unknown") for call in tool_calls]
        tool_label_text = ", ".join(tool_labels)
        suffix = " (autonomy)" if autonomy_mode else ""
        logger.success("LLM tool calls queued: {}{}", tool_label_text, suffix)
        for tool_call in tool_calls:
            logger.debug("LLM Processor: Sending to tool calls queue: '{}'", tool_call)
            queued_call = {**tool_call, "_quiet_generation": self._reply_generation}
            if read_only:
                queued_call["_read_only_tools"] = True
            if routing_permit:
                queued_call["_routing_permit"] = routing_permit
            self._tool_handoff = True
            self.tool_calls_queue.put(queued_call)
        if self._observability_bus:
            tool_names = [call.get("function", {}).get("name", "unknown") for call in tool_calls]
            self._observability_bus.emit(
                source="llm",
                kind="tool_calls",
                message=",".join(tool_names),
                meta={"count": len(tool_names), "autonomy": autonomy_mode},
            )

    def _process_sentence_for_tts(self, current_sentence_parts: list[str], emotion: str | None = None) -> None:
        """
        Process the current sentence parts and send the complete sentence to the TTS queue.
        Cleans up the sentence by removing unwanted characters and formatting it for TTS.
        Args:
            current_sentence_parts (list[str]): List of sentence parts to be processed.
        """
        if (self._lane == "autonomy" or self._quiet_mode() or self._reply_generation != self._quiet_generation()
                or self._autonomy_cancelled() or (self._autonomy_response and self._autonomy_error)):
            return
        sentence = "".join(current_sentence_parts)
        # Preserve research citations through the existing aside/punctuation cleanup.
        sentence = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1: \2", sentence)
        sentence = re.sub(r"\((https?://[^\s)]+)\)", r"\1", sentence)
        urls: list[str] = []

        def protect_url(match: re.Match[str]) -> str:
            urls.append(match.group())
            return f"GLADOSCITATIONURL{len(urls) - 1}TOKEN"

        sentence = re.sub(r"https?://[^\s<>\])]+", protect_url, sentence)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")
        for index, url in enumerate(urls):
            sentence = sentence.replace(f"GLADOSCITATIONURL{index}TOKEN", url)

        if sentence.strip() and sentence.strip() != ".":  # Avoid empty fragments and lone periods
            logger.info(f"LLM Processor: Sending to TTS queue: '{sentence}'")
            self.tts_input_queue.put(SpeechText(sentence, emotion, self._reply_generation,
                                              self._autonomy_meta.get("_autonomy_generation")
                                              if self._autonomy_response else None,
                                              self._autonomy_meta.get("_autonomy_cycle")
                                              if self._autonomy_response else None))
            self._response_has_text = True

    def _extract_thinking(
        self,
        chunk: str,
        in_thinking: bool,
        thinking_buffer: list[str],
        harmony_mode: bool = False,
    ) -> tuple[str, bool, bool]:
        """
        Extract thinking tags from streaming chunk, returning only speakable content.

        Supports two formats:
        1. Standard: <think>...</think> (GLM-4.7, MiniMax M2.7/M3, DeepSeek)
        2. Harmony: <|channel|>analysis vs <|channel|>final (GPT-OSS-120B)

        Args:
            chunk: The current text chunk from the stream
            in_thinking: Whether we're currently inside a thinking block
            thinking_buffer: Buffer to accumulate thinking content (for logging)
            harmony_mode: Whether we've detected harmony format

        Returns:
            Tuple of (speakable_content, still_in_thinking, is_harmony_mode)
        """
        # Auto-detect harmony format
        if not harmony_mode and self.HARMONY_CHANNEL_MARKER in chunk:
            harmony_mode = True

        if harmony_mode:
            return self._extract_thinking_harmony(chunk, in_thinking, thinking_buffer)

        return (*self._extract_thinking_standard(chunk, in_thinking, thinking_buffer), False)

    def _extract_thinking_standard(
        self,
        chunk: str,
        in_thinking: bool,
        thinking_buffer: list[str],
    ) -> tuple[str, bool]:
        """Extract thinking using standard <think>...</think> tags."""
        result: list[str] = []
        i = 0
        text = chunk

        while i < len(text):
            if in_thinking:
                # Look for closing tag
                close_idx = -1
                close_tag = ""
                for tag in self.THINKING_CLOSE_TAGS:
                    idx = text.find(tag, i)
                    if idx != -1 and (close_idx == -1 or idx < close_idx):
                        close_idx = idx
                        close_tag = tag

                if close_idx != -1:
                    # Found closing tag - buffer thinking content, exit thinking mode
                    thinking_buffer.append(text[i:close_idx])
                    i = close_idx + len(close_tag)
                    in_thinking = False
                    # Log thinking content for debugging
                    if thinking_buffer:
                        thinking_content = "".join(thinking_buffer)
                        if thinking_content.strip():
                            logger.debug(f"LLM thinking: {thinking_content[:200]}...")
                        thinking_buffer.clear()
                else:
                    # Still in thinking block, buffer everything
                    thinking_buffer.append(text[i:])
                    break
            else:
                # Look for opening tag
                open_idx = -1
                open_tag = ""
                for tag in self.THINKING_OPEN_TAGS:
                    idx = text.find(tag, i)
                    if idx != -1 and (open_idx == -1 or idx < open_idx):
                        open_idx = idx
                        open_tag = tag

                if open_idx != -1:
                    # Found opening tag - emit content before it, enter thinking mode
                    result.append(text[i:open_idx])
                    i = open_idx + len(open_tag)
                    in_thinking = True
                else:
                    # No thinking tag, emit everything
                    result.append(text[i:])
                    break

        return "".join(result), in_thinking

    def _extract_thinking_harmony(
        self,
        chunk: str,
        in_thinking: bool,
        thinking_buffer: list[str],
    ) -> tuple[str, bool, bool]:
        """
        Extract thinking using GPT-OSS harmony format.

        Format: <|channel|>analysis<|message|>... for thinking
                <|channel|>final<|message|>... for output
        """
        result: list[str] = []
        text = chunk
        i = 0

        while i < len(text):
            # Look for channel marker
            channel_idx = text.find(self.HARMONY_CHANNEL_MARKER, i)

            if channel_idx == -1:
                # No more channel markers
                if in_thinking:
                    thinking_buffer.append(text[i:])
                else:
                    # Strip harmony end markers from output
                    content = text[i:].replace(self.HARMONY_END_MARKER, "")
                    result.append(content)
                break

            # Found a channel marker - check what type
            if not in_thinking:
                # Emit content before the marker
                result.append(text[i:channel_idx])

            # Find channel name (between <|channel|> and <|message|>)
            channel_start = channel_idx + len(self.HARMONY_CHANNEL_MARKER)
            message_idx = text.find(self.HARMONY_MESSAGE_MARKER, channel_start)

            if message_idx == -1:
                # Incomplete marker, wait for more data
                if in_thinking:
                    thinking_buffer.append(text[i:])
                break

            channel_name = text[channel_start:message_idx].strip().split()[0]  # e.g., "final" or "analysis"
            i = message_idx + len(self.HARMONY_MESSAGE_MARKER)

            if channel_name == self.HARMONY_FINAL_CHANNEL:
                # Switch to output mode
                if in_thinking and thinking_buffer:
                    thinking_content = "".join(thinking_buffer)
                    if thinking_content.strip():
                        logger.debug(f"LLM thinking (harmony): {thinking_content[:200]}...")
                    thinking_buffer.clear()
                in_thinking = False
            elif channel_name in self.HARMONY_ANALYSIS_CHANNELS:
                # Switch to thinking mode
                in_thinking = True

        return "".join(result), in_thinking, True

    def _build_context_entries(self, autonomy_mode: bool, *, preview: bool = False) -> list[dict[str, Any]]:
        """Assemble real inference messages with provenance kept out of the payload."""
        messages = self._conversation_store.snapshot()
        if autonomy_mode:
            slots = self.slot_store.list_slots() if self.slot_store else []
            evidence = slot_evidence(slots)
            offered = {r["slot_id"] for r in evidence if "slot_id" in r}
            if not preview:
                self._autonomy_meta["_evidence_versions"] = {s.slot_id: slot_version(s) for s in slots if s.slot_id in offered}
            entries = [
                {"source": "autonomy", "message": {"role": "system", "content": self.autonomy_system_prompt or
                    "Review core slots and conversation context; prompt Central Core only for useful new information."}},
                {"source": "autonomy_decision", "message": {"role": "system", "content": decision_prompt()}},
                {"source": "autonomy_chat", "message": {"role": "system", "content": chat_evidence(self._conversation_store)}},
                {"source": "autonomy_slots", "message": {"role": "system", "content": "[All core and task slots; quoted evidence]\n"
                    + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}},
                {"source": "clock", "message": {"role": "system", "content": clock_context(current_time())}},
            ]
            return entries
        stable: list[dict[str, Any]] = []
        live: list[dict[str, Any]] = []

        def add(source: str, message: dict[str, Any] | None, volatile: bool = False) -> None:
            if message:
                (live if volatile else stable).append({"source": source, "message": message})

        if not autonomy_mode:
            add("speech", {"role": "system", "content": SPEECH_DIRECTION_PROMPT})
            add("console", {"role": "system", "content": CONSOLE_PROMPT})

        if self.context_builder:
            for entry in self.context_builder.build_system_entries():
                (live if entry["volatile"] else stable).append(entry)
        else:
            if self.slot_store:
                add("slots", self.slot_store.as_message(), volatile=True)
            if self.preferences_store:
                prompt = self.preferences_store.as_prompt()
                if prompt:
                    add("preferences", {"role": "system", "content": prompt})
            if self.constitutional_state:
                prompt = self.constitutional_state.get_modifiers_prompt()
                if prompt:
                    add("constitution", {"role": "system", "content": prompt})
        if self.mcp_manager:
            try:
                for message in self.mcp_manager.get_context_messages(block=False):
                    add("mcp", message, volatile=True)
            except Exception as exc:
                logger.warning(f"LLM Processor: Failed to load MCP context messages: {exc}")
        add("clock", {"role": "system", "content": clock_context(current_time())}, volatile=True)
        if self.vision_state:
            add("vision", self.vision_state.as_message(), volatile=True)

        split = 0
        while split < len(messages) and messages[split].get("role") == "system":
            split += 1
        # Keep the current user turn and all its tool calls/results together.
        # Completed history is stable; live state belongs immediately before the
        # pending turn, never between an assistant tool call and its tool result.
        turn_start = self._pending_turn_start(messages, split)
        return ([{"source": "system", "message": m} for m in messages[:split]] + stable
                + [{"source": "history", "message": m} for m in messages[split:turn_start]] + live
                + [{"source": "history", "message": m} for m in messages[turn_start:]])

    @staticmethod
    def _pending_turn_start(messages: list[dict[str, Any]], start: int = 0) -> int:
        """Find an unfinished user/tool exchange; completed history stays intact."""
        if not messages or (messages[-1].get("role") not in {"user", "tool"}
                            and not messages[-1].get("tool_calls")):
            return len(messages)
        user = next((i for i in range(len(messages) - 1, start - 1, -1)
                     if messages[i].get("role") == "user"), None)
        if user is not None:
            return user
        position = len(messages) - 1
        while position > start and (messages[position - 1].get("role") == "tool"
                                    or messages[position - 1].get("tool_calls")):
            position -= 1
        return position

    def _add_request_context(self, messages: list[dict[str, Any]], content: str) -> None:
        """Attach per-request routing/tool data without changing the stable prefix."""
        position = self._pending_turn_start(messages)
        # Keep the latest camera observation next to the pending input.
        while position > 0 and self._context_sources.get(id(messages[position - 1])) in {"vision", "request"}:
            position -= 1
        message = {"role": "system", "content": content}
        messages.insert(position, message)
        self._context_sources[id(message)] = "request"

    def _build_messages(self, autonomy_mode: bool) -> list[dict[str, Any]]:
        entries = self._build_context_entries(autonomy_mode)
        self._context_sources = {id(entry["message"]): entry["source"] for entry in entries}
        return [entry["message"] for entry in entries]

    def context_preview(self, autonomy_mode: bool = False) -> dict[str, Any]:
        """Read current inputs without inference, queue work or conversation changes."""
        entries = self._build_context_entries(autonomy_mode, preview=True)
        return describe_context([e["message"] for e in entries], [e["source"] for e in entries],
                                [] if autonomy_mode else self._build_tools(False), model=self.model_name,
                                mode="autonomy" if autonomy_mode else "user", kind="preview")

    def last_context(self) -> dict[str, Any] | None:
        with self._context_lock:
            return self._last_context

    def _post_with_context_recovery(
        self, request_url: str, data: dict[str, Any], originals: list[dict[str, Any]], autonomy_mode: bool,
    ) -> requests.Response:
        for attempt in range(3):
            response = requests.post(request_url, headers=self.prompt_headers, json=data, stream=True, timeout=30)
            if response.status_code != 400 or attempt == 2:
                return response
            try:
                error = response.json().get("error", {})
                if error.get("type") != "exceed_context_size_error":
                    return response
                prompt_tokens, context_tokens = int(error["n_prompt_tokens"]), int(error["n_ctx"])
            except (ValueError, TypeError, KeyError, AttributeError):
                return response
            reduced = reduce_request_context(originals, prompt_tokens, context_tokens)
            if reduced == originals:
                return response
            response.close()
            logger.debug("Shortening inference context after overflow: {} tokens / {} available",
                         prompt_tokens, context_tokens)
            # Only the request copy changes; persisted history and tool execution stay intact.
            originals = reduced
            data["messages"] = (
                self._sanitize_messages_for_ollama(originals)
                if self._ollama_mode and not request_url.endswith("/v1/chat/completions")
                else self._sanitize_messages_for_openai(originals)
            )
            self._record_context(data, originals, autonomy_mode)
            if self._quiet_mode() or self._reply_generation != self._quiet_generation() or self.shutdown_event.is_set():
                raise requests.exceptions.ConnectionError("Inference cancelled during context recovery")
        raise RuntimeError("Context recovery attempts exhausted")  # pragma: no cover

    def _record_context(self, data: dict[str, Any], originals: list[dict[str, Any]],
                        autonomy_mode: bool, sources: dict[int, str] | None = None) -> None:
        sources = self._context_sources if sources is None else sources
        labels = [sources.get(id(m), "request" if m.get("role") == "system" else "history")
                  for m in originals]
        # The latest user message is the input for this request, including tool continuations.
        users = [i for i, m in enumerate(originals) if m.get("role") == "user"]
        if users:
            labels[users[-1]] = "input"
        snapshot = describe_context(data["messages"], labels,
                                    self._autonomy_offered_tools if autonomy_mode else data.get("tools", []),
                                    model=self.model_name, mode="autonomy" if autonomy_mode else "user",
                                    kind="request")
        with self._context_lock:
            self._last_context = snapshot

    def _add_reply_instructions(self, messages: list[dict[str, Any]], tool_names: set[str]) -> None:
        """Apply the same request instructions to routed drafts and normal replies."""
        if INTERNET_SEARCH_TOOL in tool_names:
            self._add_request_context(messages,
                SEARCH_AVAILABLE_INSTRUCTIONS)

    def _build_tools(self, autonomy_mode: bool) -> list[dict[str, Any]]:
        """Return the tool list for the LLM request."""
        tools = list(tool_definitions)
        if self.vision_state is None:
            tools = [tool for tool in tools if tool.get("function", {}).get("name") != "vision_look"]
        if self.mcp_manager:
            try:
                tools.extend(self.mcp_manager.get_tool_definitions())
            except Exception as e:
                logger.warning(f"LLM Processor: Failed to load MCP tool definitions: {e}")
        return tools

    def _reply_tools(self) -> list[dict[str, Any]]:
        # Replies can need current facts or a fresh visual inspection.
        return [tool for tool in self._build_tools(False)
                if tool.get("function", {}).get("name") == INTERNET_SEARCH_TOOL
                or (self.vision_state is not None and tool.get("function", {}).get("name") == "vision_look")]

    def _reset_request(self) -> None:
        self._autonomy_handoff = False
        self._autonomy_meta = {}
        self._autonomy_error = ""
        self._autonomy_response = False
        self._response_has_text = False
        self._tool_handoff = False

    def _accept_turn(self, turn: ProcessorTurn) -> bool:
        self._request_active.set()
        turn.autonomy_mode = bool(turn.llm_input.get("autonomy", False))
        self._autonomy_response = bool(turn.llm_input.get("_autonomy_response"))
        if turn.autonomy_mode or self._autonomy_response:
            self._autonomy_meta = {k: turn.llm_input[k] for k in (
                "_autonomy_cycle", "_autonomy_generation", "_autonomy_steps", "_autonomy_reason",
            ) if k in turn.llm_input}
            self._autonomy_context = copy.deepcopy(turn.llm_input.get("_autonomy_context", []))
        self._reply_generation = turn.llm_input.get("_quiet_generation", self._quiet_generation())
        if self._lane == "priority" and not turn.autonomy_mode and not self._autonomy_response:
            turn.turn_generation = self._reply_generation
        if self._reply_generation != self._quiet_generation():
            return False
        if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
            logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
            # Ensure EOS is sent if a previous stream was cut short by this interruption
            # This logic might need refinement based on state. For now, assume no prior stream.
            return False

        enqueued_at = turn.llm_input.get("_enqueued_at")
        turn.wait_s = None
        if isinstance(enqueued_at, (int, float)):
            turn.wait_s = time.time() - float(enqueued_at)
        turn.queue_depth = None
        try:
            turn.queue_depth = self.llm_input_queue.qsize()
        except NotImplementedError:
            turn.queue_depth = None
        turn.autonomy_mode = bool(turn.llm_input.get("autonomy", False))
        if self._autonomy_cancelled():
            return False
        if self._autonomy_response and not self._autonomy_request_current(self._autonomy_meta):
            self._autonomy_error = "Notification source changed before Central Core could respond"
            return False
        if self._quiet_mode() and turn.llm_input.get("role") != "user":
            return False
        if turn.llm_input.get("_voice_continuation") and isinstance(turn.llm_input.get("_voice_turn_id"), str):
            self._conversation_store.remove_voice_input(turn.llm_input["_voice_turn_id"])
        return True

    def _transcribe_turn(self, turn: ProcessorTurn) -> bool:
        turn.audio_content = turn.llm_input.get("_native_audio")
        if turn.audio_content and self._native_audio and self._native_audio.config.user_transcripts:
            transcript_lease = None
            try:
                if self._inference_scheduler:
                    transcript_lease = self._inference_scheduler.acquire(
                        "Speech transcript", "priority", self.model_name,
                        lambda: self.shutdown_event.is_set() or self._quiet_mode()
                        or self._reply_generation != self._quiet_generation()
                        or not self.processing_active_event.is_set(),
                    )
                transcript = self._native_audio.transcribe(
                    turn.audio_content, str(self.completion_url), self.model_name, self.prompt_headers,
                    cancelled=lambda: self.shutdown_event.is_set() or self._quiet_mode()
                    or self._reply_generation != self._quiet_generation()
                    or not self.processing_active_event.is_set(),
                )
                if transcript:
                    turn.llm_input = {**turn.llm_input, "content": transcript}
                    turn.llm_input.pop("_native_audio", None)
                    if self._observability_bus:
                        self._observability_bus.emit("asr", "transcript", trim_message(transcript),
                                                     meta={"backend": "gemma"})
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                logger.warning("Optional Gemma transcript failed: {}", type(exc).__name__)
            finally:
                if transcript_lease is not None:
                    self._inference_scheduler.release(transcript_lease)
            if self._quiet_mode() or self._reply_generation != self._quiet_generation():
                return False
        return True

    def _describe_input(self, turn: ProcessorTurn) -> None:
        turn.llm_message = {
            key: value
            for key, value in turn.llm_input.items()
            if key != "autonomy" and not key.startswith("_")
        }
        logger.info(f"LLM Processor: Received input for LLM: '{turn.llm_message}'")
        if self._observability_bus:
            message_text = turn.llm_message.get("content", "")
            self._observability_bus.emit(
                source="llm",
                kind="request",
                message=trim_message(str(message_text)),
                meta={"autonomy": turn.autonomy_mode, "lane": self._lane},
            )
            if turn.wait_s is not None:
                self._observability_bus.emit(
                    source="llm",
                    kind="queue",
                    message=self._lane,
                    level="debug",
                    meta={
                        "lane": self._lane,
                        "wait_s": round(turn.wait_s, 3),
                        "queue_depth": turn.queue_depth,
                    },
                )

    def _route_turn(self, turn: ProcessorTurn) -> bool:
        turn.route = None
        if (self.router and self._set_quiet_mode and turn.llm_message.get("role") == "user"
                and self.router.store.snapshot().get("enabled", True)):
            active_decision = self.router.store.get(active=True)
            was_quiet = self._quiet_mode()
            if was_quiet or not active_decision or active_decision.strategy == "flat":
                gate = self.router.quiet_score(was_quiet, str(turn.llm_message.get("content", "")),
                                              turn.llm_input.get("_native_audio"), bool(turn.llm_input.get("_spoken")))
                if self._reply_generation != self._quiet_generation():
                    return False
                if was_quiet:
                    if not gate["accepted"] or gate["action"] != "wake":
                        return False
                    self._set_quiet_mode(False)
                    self._reply_generation = self._quiet_generation()
                    turn.turn_generation = self._reply_generation
                    if self._inference_scheduler:
                        self._inference_scheduler.begin_interaction(turn.turn_generation)
                    turn.route = {"action": "plan"}  # Interpret any follow-up in the original wake request.
                elif gate["accepted"] and gate["action"] == "quiet":
                    self._set_quiet_mode(True)
                    return False
                elif gate["accepted"] and gate["action"] == "ignore":
                    return False
        if self._before_context and not turn.autonomy_mode and turn.llm_message.get("role") == "user":
            self._before_context(turn.llm_input)
        if turn.route is None and self.router and self._lane == "priority" and turn.llm_message.get("role") == "user":
            decision = self.router.store.get(active=True)
            if decision:
                if (self._inference_scheduler and self._inference_scheduler.config.slots > 1
                        and not self._ollama_mode and not turn.llm_input.get("_native_audio")
                        and (not self._before_reply or self.context_builder)):
                    draft_turn = ProcessorTurn(llm_input=turn.llm_input, llm_message=turn.llm_message,
                                               route={"action": "reply"})
                    self._build_request(draft_turn, draft=True)
                    turn.draft_messages = draft_turn.base_messages
                    turn.draft_sources = dict(self._context_sources)
                    generation = self._reply_generation
                    turn.draft = SpeculativeStream(
                        self._inference_scheduler, str(self.completion_url), self.prompt_headers,
                        {**draft_turn.data, "chat_template_kwargs": {"enable_thinking": False},
                         "messages": self._sanitize_messages_for_openai(turn.draft_messages)},
                        self.shutdown_event, self.processing_active_event,
                        cancelled_if=lambda generation=generation: (
                            self._quiet_mode() or generation != self._quiet_generation()),
                    )
                try:
                    turn.route = self.router.score(
                        decision, str(turn.llm_message.get("content", "")), turn.llm_input.get("_native_audio"),
                        context=self._conversation_store.snapshot(),
                        cancelled=lambda: self.shutdown_event.is_set() or self._quiet_mode() or self._reply_generation != self._quiet_generation() or not self.processing_active_event.is_set(),
                        spoken=bool(turn.llm_input.get("_spoken")),
                        on_admitted=turn.draft.start if turn.draft else None,
                    )
                except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
                    logger.warning("Routing unavailable: {}", type(exc).__name__)
                    if self._observability_bus:
                        self._observability_bus.emit("routing", "error", "Routing failed; no action executed", level="warning")
                    turn.route = {"action": "assist"}
                if turn.draft and (turn.route["action"] != "reply" or turn.route.get("context_source") == "clock"):
                    turn.draft.cancel()
                    turn.draft = None
                if turn.route["action"] == "quiet" and turn.route.get("accepted") and self._set_quiet_mode:
                    self._set_quiet_mode(True)
                    return False
                if turn.route["action"] == "wake" and turn.route.get("accepted") and self._set_quiet_mode:
                    self._set_quiet_mode(False)
                    self._reply_generation = self._quiet_generation()
                    turn.turn_generation = self._reply_generation
                    if self._inference_scheduler:
                        self._inference_scheduler.begin_interaction(turn.turn_generation)
                    turn.route = {"action": "plan"}
                if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                    return False
                if turn.route["action"] == "ignore":
                    return False
        if self._quiet_mode() or self._reply_generation != self._quiet_generation() or self._autonomy_cancelled():
            return False
        return True

    def _prepare_reply(self, turn: ProcessorTurn) -> bool:
        if (not turn.autonomy_mode and not self._autonomy_response and turn.llm_message.get("role") == "tool"
                and turn.llm_input.get("_tool_reply_context", {}).get("name") == INTERNET_SEARCH_TOOL):
            try:
                pending = json.loads(str(turn.llm_message.get("content", "")))
            except ValueError:
                pending = None
            if isinstance(pending, dict) and pending.get("status") in {"queued", "running"}:
                # A progress acknowledgement needs no inference or unrelated context.
                self._conversation_store.append(turn.llm_message)
                acknowledgement = ("Your search is queued." if pending["status"] == "queued"
                                   else "I'm checking that now.")
                self._process_sentence_for_tts([acknowledgement])
                self.tts_input_queue.put(SpeechText("<EOS>", generation=self._reply_generation))
                return False
        if self._before_reply and not turn.autonomy_mode and turn.llm_message.get("role") == "user":
            self._before_reply(turn.llm_input)
        if self._quiet_mode() or self._reply_generation != self._quiet_generation() or self._autonomy_cancelled():
            return False
        return True

    def _admit_turn(self, turn: ProcessorTurn) -> None:
        admission_started = time.perf_counter()
        if self._inference_scheduler and not (turn.draft and turn.draft.started):
            turn.inference_lease = self._inference_scheduler.acquire(
                "Central Core notification" if self._autonomy_response else
                "GLaDOS" if self._lane == "priority" else "autonomy",
                "autonomy" if self._autonomy_response else self._lane, self.model_name,
                lambda: self.shutdown_event.is_set() or self._quiet_mode() or self._reply_generation != self._quiet_generation() or self._autonomy_cancelled() or not self.processing_active_event.is_set(),
            )
        if self._observability_bus:
            self._observability_bus.emit("llm", "admitted", "Inference admitted", level="debug",
                meta={"generation": self._reply_generation, "lane": self._lane,
                      "wait_ms": round((time.perf_counter() - admission_started) * 1000, 1)})
        if self._inflight_counter is not None:
            self._inflight_counter.increment()
            turn.inflight_guard = True
        else:
            turn.inflight_guard = False

    def _record_input(self, turn: ProcessorTurn) -> None:
        turn.audio_content = turn.llm_input.get("_native_audio")
        if turn.audio_content and turn.route and turn.route["action"] == "tool":
            # Preserve the interpreted request, never raw audio or a fabricated transcript.
            turn.llm_message["content"] = "[Spoken request interpreted by routing: " + json.dumps({
                "tool": turn.route["tool"], "arguments": turn.route["arguments"],
            }) + ". This is an action summary, not a transcript.]"
        if turn.autonomy_mode or self._autonomy_response:
            self._autonomy_context.append(turn.llm_message)
        else:
            voice_turn_id = turn.llm_input.get("_voice_turn_id")
            if turn.llm_message.get("role") == "user" and isinstance(voice_turn_id, str):
                self._conversation_store.append_voice_input(turn.llm_message, voice_turn_id)
            else:
                self._conversation_store.append(turn.llm_message)


    def _dispatch_fixed_action(self, turn: ProcessorTurn) -> bool:
        if turn.route and turn.route["action"] == "tool":
            if not self.router.store.authorize(turn.route):
                self._process_sentence_for_tts(["That action changed while I was checking. Please ask again."])
                self.tts_input_queue.put(SpeechText("<EOS>", generation=self._reply_generation))
                return True
            call = {"id": f"route_{uuid.uuid4().hex}", "type": "function",
                    "function": {"name": turn.route["tool"], "arguments": json.dumps(turn.route["arguments"])}}
            self._conversation_store.append({"role": "assistant", "tool_calls": [call]})
            self._tool_handoff = True
            self.tool_calls_queue.put({**call, "_quiet_generation": self._reply_generation,
                                      "_decision_permit": {
                k: turn.route[k] for k in ("list_id", "revision", "option_id")}})
            return True
        return False

    def _select_tools(self, turn: ProcessorTurn) -> None:
        turn.routing_permit = None
        if turn.route and turn.route.get("strategy") == "hierarchical" and turn.route["action"] == "plan":
            turn.routing_permit = {k: turn.route[k] for k in ("list_id", "revision", "settings_revision", "tool_scope")}
            if not self.router.store.authorize_scope(turn.routing_permit):
                turn.route = {"action": "assist"}
                turn.routing_permit = None
        turn.read_only = bool(turn.route and turn.route["action"] in {"assist", "reply"})
        turn.allow_tools = not self._autonomy_response and bool(turn.llm_input.get("_allow_tools", True)) and not (
            turn.route and (turn.route.get("context_source") == "clock" or turn.route["action"] == "clarify" or (
                turn.route["action"] == "reply" and not self._reply_tools()
            ))
        )
        turn.tools = self._build_tools(False) if turn.allow_tools and not turn.autonomy_mode else []
        if turn.route and turn.route["action"] == "reply":
            turn.tools = self._reply_tools() if turn.allow_tools else []
        if turn.routing_permit:
            turn.tools = [tool for tool in turn.tools if tool.get("function", {}).get("name") in turn.routing_permit["tool_scope"]]
        if turn.read_only:
            turn.tools = [tool for tool in turn.tools if tool.get("function", {}).get("name") in {
                "get_time", "run_safe_command", "get_report", "get_preferences", "vision_look",
                INTERNET_SEARCH_TOOL,
            }]
        if turn.tools and not turn.route and not turn.autonomy_mode and turn.llm_message.get("role") == "user" and not turn.audio_content:
            content = str(turn.llm_message.get("content", ""))
            turn.tools = self._filter_tools_for_message(turn.tools, content)
        turn.tool_names = {
            tool.get("function", {}).get("name", "")
            for tool in turn.tools
            if tool.get("function", {}).get("name")
        }
        if turn.autonomy_mode:
            self._autonomy_offered_tools = turn.tools

    def _add_turn_instructions(self, turn: ProcessorTurn) -> None:
        self._add_reply_instructions(turn.base_messages, turn.tool_names)
        if turn.routing_permit:
            self._add_request_context(turn.base_messages,
                "[Capability routing for this request]\n" + json.dumps({
                    "category": turn.route.get("category") or "general tool planning",
                    "server": turn.route.get("server"), "tool_scope": turn.routing_permit["tool_scope"],
                }))
        if turn.route and turn.route["action"] in {"assist", "clarify"}:
            self._add_request_context(turn.base_messages,
                ROUTING_FALLBACK_INSTRUCTIONS)
        if turn.llm_input.get("_tool_reply_context") and not turn.autonomy_mode:
            self._add_request_context(turn.base_messages,
                "The performed action and tool result for this request are: "
                + json.dumps(turn.llm_input["_tool_reply_context"])
                + self._measurement_hint(str(turn.llm_message.get("content", ""))))
            if turn.llm_input["_tool_reply_context"].get("name") == INTERNET_SEARCH_TOOL:
                try:
                    search_payload = json.loads(str(turn.llm_message.get("content", "")))
                    search_running = search_payload.get("status") in {"queued", "running"}
                except (ValueError, AttributeError):
                    search_payload = {}
                    search_running = False
                self._add_request_context(turn.base_messages,
                    SEARCH_PENDING_INSTRUCTIONS
                    if search_running else
                    SEARCH_COMPLETED_INSTRUCTIONS)
                if isinstance(search_payload, dict) and "findings" in search_payload:
                    self._add_request_context(turn.base_messages,
                        SEARCH_FINDINGS_INSTRUCTIONS)

    def _request_options_for_turn(self, turn: ProcessorTurn) -> None:
        turn.data = {
            **self._request_options,
            "model": self.model_name,
            "stream": True,
            # Add other parameters like temperature, max_tokens if needed from config
        }
        if turn.autonomy_mode:
            limit = 512 if self._autonomy_thinking else 256
            turn.data["max_tokens"] = limit
            turn.data["chat_template_kwargs"] = {"enable_thinking": self._autonomy_thinking}
            # Gemma's structured-output grammar permits a thought channel even
            # when the template disables thinking. Bound that channel separately
            # so it cannot consume the entire decision budget.
            if not self._ollama_mode:
                turn.data["reasoning_budget_tokens"] = 128 if self._autonomy_thinking else 0
            for key in ("tools", "tool_choice", "parallel_tool_calls", "format", "response_format"):
                turn.data.pop(key, None)
        # Gemma 4 defaults to thinking in recent Ollama releases. Voice
        # responses need the final answer promptly, including its directions.
        if self._ollama_mode and self.model_name.lower().startswith("gemma4"):
            turn.data["think"] = self._autonomy_thinking if turn.autonomy_mode else False
        if turn.audio_content:
            turn.data["chat_template_kwargs"] = {"enable_thinking": False}
        turn.search_planning = bool(turn.allow_tools and turn.tools and not turn.autonomy_mode and turn.routing_permit
                               and turn.routing_permit["tool_scope"] == [INTERNET_SEARCH_TOOL])
        if turn.allow_tools and turn.tools and not turn.autonomy_mode:
            turn.data["tools"] = turn.tools
            if turn.search_planning:
                # An explicit search route must perform a search, including repeated questions.
                turn.data["tool_choice"] = "required"


    def _build_request(self, turn: ProcessorTurn, *, draft: bool = False) -> None:
        """Resolve context once and share instructions/tool policy between reply and draft."""
        self._select_tools(turn)
        turn.base_messages = self._build_messages(turn.autonomy_mode)
        if draft:
            turn.base_messages.append(turn.llm_message)
        elif turn.autonomy_mode or self._autonomy_response:
            turn.base_messages += self._autonomy_context
            self._context_sources.update({id(m): "input" for m in self._autonomy_context})
        self._add_turn_instructions(turn)
        if turn.audio_content:
            turn.base_messages = [
                {**message, "content": turn.audio_content} if message is turn.llm_message else message
                for message in turn.base_messages
            ]
        self._request_options_for_turn(turn)

    def _consume_stream(self, response: requests.Response, turn: ProcessorTurn, state: ResponseState, stream_started: float) -> None:
        first_token = True
        for line in response.iter_lines(chunk_size=1):
            if self._quiet_mode() or self._reply_generation != self._quiet_generation() or self._autonomy_cancelled() or not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                logger.info("LLM Processor: Interruption or shutdown detected during LLM stream.")
                break  # Stop processing stream

            if line:
                cleaned_line_data = self._clean_raw_bytes(line)
                if cleaned_line_data:
                    chunk = self._process_chunk(cleaned_line_data)
                    if chunk:
                        if first_token:
                            first_token = False
                            if self._observability_bus:
                                self._observability_bus.emit("llm", "first_token", "First response token",
                                    level="debug", meta={"generation": self._reply_generation,
                                        "lane": self._lane,
                                        "elapsed_ms": round((time.perf_counter() - stream_started) * 1000, 1)})
                        if isinstance(chunk, list):
                            if not turn.autonomy_mode:
                                self._process_tool_chunks(state.tool_calls_buffer, chunk)
                        elif turn.autonomy_mode:
                            speakable, state.in_thinking, state.harmony_mode = self._extract_thinking(
                                chunk, state.in_thinking, state.thinking_buffer, state.harmony_mode)
                            state.autonomy_json.append(speakable)
                        elif not turn.autonomy_mode:
                            # Extract thinking tags before TTS (auto-detects format)
                            speakable, state.in_thinking, state.harmony_mode = self._extract_thinking(
                                chunk, state.in_thinking, state.thinking_buffer, state.harmony_mode
                            )
                            # A required search call is argument planning, not a spoken reply.
                            # Models may emit a date question before the tool call despite the clock.
                            if speakable and not turn.search_planning:
                                for segment in state.speech_parser.feed(speakable):
                                    if segment.emotion != state.sentence_emotion and state.sentence_buffer:
                                        self._process_sentence_for_tts(
                                            state.sentence_buffer, state.sentence_emotion
                                        )
                                        state.sentence_buffer = []
                                    state.sentence_emotion = segment.emotion
                                    state.sentence_buffer.append(segment.text)
                                    clauses, remainder = split_speech_clauses("".join(state.sentence_buffer))
                                    for clause in clauses:
                                        self._process_sentence_for_tts([clause], state.sentence_emotion)
                                    state.sentence_buffer = [remainder] if remainder else []
                    elif cleaned_line_data.get("done_marker"):
                        break
                    elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                        break


    def _finish_response(self, turn: ProcessorTurn, state: ResponseState) -> None:
        if turn.autonomy_mode and not self._autonomy_cancelled() and self._reply_generation == self._quiet_generation() and self.processing_active_event.is_set():
            try:
                offered_slots = set(self._autonomy_meta.get("_evidence_versions", {}))
                offered_slots -= set(turn.llm_input.get("_autonomy_announced_slots", []))
                decision = parse_decision("".join(state.autonomy_json), offered_slots)
                if decision is None and self._on_autonomy_done:
                    self._on_autonomy_done(self._autonomy_meta.get("_autonomy_cycle", ""),
                                           "silent", "No useful new intervention")
                    self._autonomy_handoff = True
                elif decision is not None and self._on_autonomy_prompt:
                    self._autonomy_handoff = self._on_autonomy_prompt(decision, {
                        **self._autonomy_meta, "_quiet_generation": self._reply_generation})
                    if not self._autonomy_handoff:
                        self._autonomy_error = "Slot evidence or user activity changed before handoff"
                else:
                    self._autonomy_error = "Central Core handoff is unavailable"
            except ValueError as exc:
                self._autonomy_error = str(exc)
                logger.warning("Autonomy decision rejected: {}", exc)
        if not self._autonomy_cancelled() and self._reply_generation == self._quiet_generation() and self.processing_active_event.is_set() and state.tool_calls_buffer and turn.allow_tools:
            self._process_tool_call(state.tool_calls_buffer, turn.autonomy_mode, turn.tool_names, read_only=turn.read_only,
                                    routing_permit=turn.routing_permit)
        elif self.processing_active_event.is_set() and turn.search_planning:
            self._process_sentence_for_tts(["I couldn't start the internet search."])
        elif self.processing_active_event.is_set() and not turn.autonomy_mode:
            for segment in state.speech_parser.feed("", final=True):
                if segment.emotion != state.sentence_emotion and state.sentence_buffer:
                    self._process_sentence_for_tts(state.sentence_buffer, state.sentence_emotion)
                    state.sentence_buffer = []
                state.sentence_emotion = segment.emotion
                state.sentence_buffer.append(segment.text)
            if state.sentence_buffer:
                self._process_sentence_for_tts(state.sentence_buffer, state.sentence_emotion)

    def _stream_request(self, turn: ProcessorTurn) -> None:
        state = ResponseState()
        try:
            state.http_error_detail: tuple[str | int, str] | None = None
            request_urls = [str(self.completion_url)]
            if self._ollama_mode:
                fallback_url = str(self.completion_url).replace("/api/chat", "/v1/chat/completions")
                if fallback_url != request_urls[0]:
                    request_urls.append(fallback_url)

            for attempt, request_url in enumerate(request_urls):
                if turn.autonomy_mode:
                    turn.data.pop("format", None)
                    turn.data.pop("response_format", None)
                    offered_slots = set(self._autonomy_meta.get("_evidence_versions", {}))
                    offered_slots -= set(turn.llm_input.get("_autonomy_announced_slots", []))
                    schema = decision_schema(offered_slots)
                    if request_url.rstrip("/").endswith("/api/chat"):
                        turn.data["format"] = schema
                    else:
                        turn.data["response_format"] = {"type": "json_schema", "json_schema": {
                            "name": "autonomy_decision", "strict": False, "schema": schema}}
                if request_url.endswith("/v1/chat/completions"):
                    turn.data["messages"] = self._sanitize_messages_for_openai(turn.base_messages)
                elif self._ollama_mode:
                    turn.data["messages"] = self._sanitize_messages_for_ollama(turn.base_messages)
                else:
                    turn.data["messages"] = self._sanitize_messages_for_openai(turn.base_messages)
                if turn.draft and turn.draft.started:
                    self._record_context(turn.draft.data, turn.draft_messages, False, turn.draft_sources)
                else:
                    self._record_context(turn.data, turn.base_messages, turn.autonomy_mode)
                try:
                    stream_started = time.perf_counter()
                    with (turn.draft if turn.draft and turn.draft.started else self._post_with_context_recovery(
                        request_url, turn.data, turn.base_messages, turn.autonomy_mode,
                    )) as response:
                        if response.status_code >= 400:
                            response_text = response.text.strip()
                            state.http_error_detail = (response.status_code, response_text)
                            logger.error(
                                "LLM Processor: HTTP error {} from LLM service: {}",
                                response.status_code,
                                response_text or response.reason,
                            )
                            if turn.audio_content:
                                logger.error(
                                    "LLM Processor: native audio request failed (audio payload omitted)"
                                )
                            else:
                                logger.error(
                                    "LLM Processor: LLM payload (truncated): {}",
                                    json.dumps(turn.data)[:1200],
                                )
                            response.raise_for_status()
                        logger.debug("LLM Processor: Request to LLM successful, processing stream...")
                        self._consume_stream(response, turn, state, stream_started)
                        self._finish_response(turn, state)
                    break
                except requests.exceptions.HTTPError as e:
                    response = getattr(e, "response", None)
                    status_code = response.status_code if response is not None else "unknown"
                    response_text = ""
                    if response is not None:
                        response_text = response.text.strip()
                    state.http_error_detail = (status_code, response_text or str(e))
                    if attempt < len(request_urls) - 1:
                        logger.warning(
                            "LLM Processor: Retrying with fallback endpoint {}",
                            request_urls[attempt + 1],
                        )
                        continue
                    raise

        except requests.exceptions.ConnectionError as e:
            self._autonomy_error = type(e).__name__
            logger.error(f"LLM Processor: Connection error to LLM service: {e}")
            self._process_sentence_for_tts([
                "I'm unable to connect to my thinking module. Please check the LLM service connection."
            ])
        except requests.exceptions.Timeout as e:
            self._autonomy_error = type(e).__name__
            logger.error(f"LLM Processor: Request to LLM timed out: {e}")
            self._process_sentence_for_tts(["My brain seems to be taking too long to respond. It might be overloaded."])
        except requests.exceptions.HTTPError as e:
            self._autonomy_error = type(e).__name__
            if state.http_error_detail:
                status_code, detail = state.http_error_detail
                logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {detail}")
                self._process_sentence_for_tts([f"I received an error from my thinking module. HTTP status {status_code}."])
            else:
                status_code = (
                    e.response.status_code
                    if hasattr(e, "response") and hasattr(e.response, "status_code")
                    else "unknown"
                )
                logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                self._process_sentence_for_tts([f"I received an error from my thinking module. HTTP status {status_code}."])
        except requests.exceptions.RequestException as e:
            self._autonomy_error = type(e).__name__
            logger.error(f"LLM Processor: Request to LLM failed: {e}")
            self._process_sentence_for_tts(["Sorry, I encountered an error trying to reach my brain."])
        except Exception as e:
            self._autonomy_error = type(e).__name__
            logger.exception(f"LLM Processor: Unexpected error during LLM request/streaming: {e}")
            self._process_sentence_for_tts(["I'm having a little trouble thinking right now."])
        finally:
            if not turn.autonomy_mode and self.processing_active_event.is_set():
                logger.debug("LLM Processor: Sending EOS token to TTS queue.")
                if self._autonomy_response and self._response_has_text and not self._autonomy_error:
                    self._autonomy_handoff = True  # SpeechPlayer acknowledges actual delivery at EOS.
                self.tts_input_queue.put(SpeechText("<EOS>", generation=self._reply_generation,
                    autonomy_generation=self._autonomy_meta.get("_autonomy_generation")
                    if self._autonomy_response else None,
                    autonomy_cycle=self._autonomy_meta.get("_autonomy_cycle")
                    if self._autonomy_response and not self._autonomy_error else None))
            else:
                logger.info("LLM Processor: Interrupted, not sending EOS from LLM processing.")
                # The AudioPlayer will handle clearing its state.
                # If an EOS was already sent by TTS from a *previous* partial sentence,
                # this could lead to an early clear of currently_speaking.
                # The `processing_active_event` is key to synchronize.


    def _release_turn(self, turn: ProcessorTurn) -> None:
        if (turn.turn_generation is not None and self._inference_scheduler
                and not self._response_has_text and not self._tool_handoff):
            self._inference_scheduler.end_interaction(turn.turn_generation, "no_response")
        if turn.draft:
            turn.draft.cancel()
        if turn.inference_lease is not None:
            self._inference_scheduler.release(turn.inference_lease)
        if turn.inflight_guard:
            self._inflight_counter.decrement()
        if (turn.autonomy_mode or self._autonomy_response) and not self._autonomy_handoff and self._on_autonomy_done:
            cycle = self._autonomy_meta.get("_autonomy_cycle")
            if cycle:
                cancelled = (self._autonomy_cancelled() or self.shutdown_event.is_set()
                             or self._reply_generation != self._quiet_generation()
                             or not self.processing_active_event.is_set())
                outcome = ("cancelled" if cancelled else "error" if self._autonomy_error or
                           not self._response_has_text else "response")
                self._on_autonomy_done(cycle, outcome,
                                       "Check cancelled" if cancelled else
                                       self._autonomy_error or self._autonomy_meta.get("_autonomy_reason")
                                       or "Autonomy request returned no valid decision")
        self._autonomy_context = []
        self._autonomy_offered_tools = []
        self._request_active.clear()

    def _process_turn(self, turn: ProcessorTurn) -> None:
        if not self._accept_turn(turn) or not self._transcribe_turn(turn):
            return
        self._describe_input(turn)
        if not self._route_turn(turn) or not self._prepare_reply(turn):
            return
        self._admit_turn(turn)
        self._record_input(turn)
        if self._dispatch_fixed_action(turn):
            return
        self._build_request(turn)
        self._stream_request(turn)

    def run(self) -> None:
        """Dispatch queued turns; all exit paths release inference and handoff state."""
        logger.info("LanguageModelProcessor thread started.")
        while not self.shutdown_event.is_set():
            turn = ProcessorTurn()
            self._reset_request()
            try:
                turn.llm_input = self.llm_input_queue.get(timeout=self.pause_time)
                self._process_turn(turn)
            except (InferenceCancelledError, queue.Empty):
                pass
            except Exception as exc:
                logger.exception("LLM Processor: Unexpected error in main run loop: {}", exc)
                time.sleep(0.1)
            finally:
                self._release_turn(turn)
        logger.info("LanguageModelProcessor thread finished.")
