from __future__ import annotations

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
from ..autonomy import ConstitutionalState, PreferencesStore, TaskSlotStore
from .context import ContextBuilder
from .llm_tracking import InFlightCounter
from ..mcp import MCPManager
from ..observability import ObservabilityBus, trim_message
from ..tools import tool_definitions
from ..vision.vision_state import VisionState

class LanguageModelProcessor:
    """
    A thread that processes text input for a language model, streaming responses and sending them to TTS.
    This class is designed to run in a separate thread, continuously checking for new text to process
    until a shutdown event is set. It handles conversation history, manages streaming responses,
    and sends synthesized sentences to a TTS queue.
    """

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n", "\n\n"}

    def __init__(
        self,
        llm_input_queue: queue.Queue[dict[str, Any]],
        tool_calls_queue: queue.Queue[dict[str, Any]],
        tts_input_queue: queue.Queue[str],
        conversation_history: list[dict[str, Any]],  # Shared
        completion_url: HttpUrl,
        model_name: str,  # Renamed from 'model' to avoid conflict
        api_key: str | None,
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
        vision_state: VisionState | None = None,
        slot_store: TaskSlotStore | None = None,
        preferences_store: PreferencesStore | None = None,
        constitutional_state: ConstitutionalState | None = None,
        context_builder: ContextBuilder | None = None,
        autonomy_system_prompt: str | None = None,
        mcp_manager: MCPManager | None = None,
        observability_bus: ObservabilityBus | None = None,
        extra_headers: dict[str, str] | None = None,
        conversation_lock: threading.Lock | None = None,
        lane: str = "priority",
        inflight_counter: InFlightCounter | None = None,
    ) -> None:
        self.llm_input_queue = llm_input_queue
        self.tool_calls_queue = tool_calls_queue
        self.tts_input_queue = tts_input_queue
        self.conversation_history = conversation_history
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
        self._conversation_lock = conversation_lock
        self._lane = lane
        self._inflight_counter = inflight_counter
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
            if name.startswith("mcp.") and not wants_system:
                continue
            filtered.append(tool)
        return filtered

    def _process_tool_call(
        self,
        tool_calls: list[dict[str, Any]],
        autonomy_mode: bool,
        tool_names: set[str],
    ) -> None:
        """
        Add tool calls to conversation history and send each to the tool calls queue.

        Args:
            tool_calls: List of tool calls to be run.
        """
        self._normalize_tool_calls(tool_calls, tool_names)
        for tool_call in tool_calls:
            tool_call.setdefault("type", "function")
            if not tool_call.get("id"):
                tool_call["id"] = f"toolcall_{uuid.uuid4().hex}"
        if self._conversation_lock:
            with self._conversation_lock:
                self.conversation_history.append(
                    {"role": "assistant", "index": 0, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
                )
        else:
            self.conversation_history.append(
                {"role": "assistant", "index": 0, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
            )
        tool_labels = [call.get("function", {}).get("name", "unknown") for call in tool_calls]
        tool_label_text = ", ".join(tool_labels)
        suffix = " (autonomy)" if autonomy_mode else ""
        logger.success("LLM tool calls queued: {}{}", tool_label_text, suffix)
        for tool_call in tool_calls:
            if autonomy_mode:
                tool_call["autonomy"] = True
            logger.debug("LLM Processor: Sending to tool calls queue: '{}'", tool_call)
            self.tool_calls_queue.put(tool_call)
        if self._observability_bus:
            tool_names = [call.get("function", {}).get("name", "unknown") for call in tool_calls]
            self._observability_bus.emit(
                source="llm",
                kind="tool_calls",
                message=",".join(tool_names),
                meta={"count": len(tool_names), "autonomy": autonomy_mode},
            )

    def _process_sentence_for_tts(self, current_sentence_parts: list[str]) -> None:
        """
        Process the current sentence parts and send the complete sentence to the TTS queue.
        Cleans up the sentence by removing unwanted characters and formatting it for TTS.
        Args:
            current_sentence_parts (list[str]): List of sentence parts to be processed.
        """
        sentence = "".join(current_sentence_parts)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")

        if sentence and sentence != ".":  # Avoid sending just a period
            logger.info(f"LLM Processor: Sending to TTS queue: '{sentence}'")
            self.tts_input_queue.put(sentence)

    def _build_messages(self, autonomy_mode: bool) -> list[dict[str, Any]]:
        """Build the message list for the LLM request, injecting context from registered sources."""
        if self._conversation_lock:
            with self._conversation_lock:
                messages = list(self.conversation_history)
        else:
            messages = list(self.conversation_history)
        extra_messages: list[dict[str, Any]] = []

        if autonomy_mode and self.autonomy_system_prompt:
            extra_messages.append({"role": "system", "content": self.autonomy_system_prompt})

        # Use ContextBuilder if available (new pattern)
        if self.context_builder:
            extra_messages.extend(self.context_builder.build_system_messages())
        else:
            # Fallback to old pattern for backward compatibility
            if self.slot_store:
                slot_message = self.slot_store.as_message()
                if slot_message:
                    extra_messages.append(slot_message)
            if self.preferences_store:
                prefs_prompt = self.preferences_store.as_prompt()
                if prefs_prompt:
                    extra_messages.append({"role": "system", "content": prefs_prompt})
            if self.constitutional_state:
                modifiers_prompt = self.constitutional_state.get_modifiers_prompt()
                if modifiers_prompt:
                    extra_messages.append({"role": "system", "content": modifiers_prompt})

        # MCP context is handled separately (returns list of messages)
        if self.mcp_manager:
            try:
                extra_messages.extend(self.mcp_manager.get_context_messages(block=False))
            except Exception as e:
                logger.warning(f"LLM Processor: Failed to load MCP context messages: {e}")

        # Vision context is handled separately (has special formatting)
        if self.vision_state:
            vision_message = self.vision_state.as_message()
            if vision_message:
                extra_messages.append(vision_message)

        if extra_messages:
            insert_index = 0
            while insert_index < len(messages) and messages[insert_index].get("role") == "system":
                insert_index += 1
            for offset, message in enumerate(extra_messages):
                messages.insert(insert_index + offset, message)

        return messages

    def _build_tools(self, autonomy_mode: bool) -> list[dict[str, Any]]:
        """Return the tool list for the LLM request."""
        tools = list(tool_definitions)
        if self.vision_state is None:
            tools = [tool for tool in tools if tool.get("function", {}).get("name") != "vision_look"]
        if not autonomy_mode:
            tools = [
                tool
                for tool in tools
                if tool.get("function", {}).get("name") not in {"speak", "do_nothing"}
            ]
            tools = [tool for tool in tools if tool.get("function", {}).get("name") != "vision_look"]
        if not autonomy_mode:
            tools = [
                tool
                for tool in tools
                if tool.get("function", {}).get("name") not in {"speak", "do_nothing"}
            ]
        if self.mcp_manager:
            try:
                tools.extend(self.mcp_manager.get_tool_definitions())
            except Exception as e:
                logger.warning(f"LLM Processor: Failed to load MCP tool definitions: {e}")
        return tools

    def run(self) -> None:
        """
        Starts the main loop for the LanguageModelProcessor thread.

        This method continuously checks the LLM input queue for text to process.
        It processes the text, sends it to the LLM API, and streams the response.
        It handles conversation history, manages streaming responses, and sends synthesized sentences
        to a TTS queue. The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("LanguageModelProcessor thread started.")
        while not self.shutdown_event.is_set():
            try:
                llm_input = self.llm_input_queue.get(timeout=self.pause_time)
                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
                    # Ensure EOS is sent if a previous stream was cut short by this interruption
                    # This logic might need refinement based on state. For now, assume no prior stream.
                    continue

                inflight_guard = False
                enqueued_at = llm_input.get("_enqueued_at")
                wait_s = None
                if isinstance(enqueued_at, (int, float)):
                    wait_s = time.time() - float(enqueued_at)
                queue_depth = None
                try:
                    queue_depth = self.llm_input_queue.qsize()
                except NotImplementedError:
                    queue_depth = None
                autonomy_mode = bool(llm_input.get("autonomy", False))
                llm_message = {
                    key: value
                    for key, value in llm_input.items()
                    if key != "autonomy" and not key.startswith("_")
                }
                logger.info(f"LLM Processor: Received input for LLM: '{llm_message}'")
                if self._observability_bus:
                    message_text = llm_message.get("content", "")
                    self._observability_bus.emit(
                        source="llm",
                        kind="request",
                        message=trim_message(str(message_text)),
                        meta={"autonomy": autonomy_mode, "lane": self._lane},
                    )
                    if wait_s is not None:
                        self._observability_bus.emit(
                            source="llm",
                            kind="queue",
                            message=self._lane,
                            meta={
                                "lane": self._lane,
                                "wait_s": round(wait_s, 3),
                                "queue_depth": queue_depth,
                            },
                        )
                if self._inflight_counter and self._lane == "autonomy":
                    self._inflight_counter.increment()
                    inflight_guard = True
                else:
                    inflight_guard = False
                if self._conversation_lock:
                    with self._conversation_lock:
                        self.conversation_history.append(llm_message)
                else:
                    self.conversation_history.append(llm_message)

                allow_tools = bool(llm_input.get("_allow_tools", True))
                tools = self._build_tools(autonomy_mode) if allow_tools else []
                if tools and not autonomy_mode and llm_message.get("role") == "user":
                    content = str(llm_message.get("content", ""))
                    tools = self._filter_tools_for_message(tools, content)
                tool_names = {
                    tool.get("function", {}).get("name", "")
                    for tool in tools
                    if tool.get("function", {}).get("name")
                }
                base_messages = self._build_messages(autonomy_mode)
                data = {
                    "model": self.model_name,
                    "stream": True,
                    # Add other parameters like temperature, max_tokens if needed from config
                }
                if allow_tools and tools:
                    data["tools"] = tools

                tool_calls_buffer: list[dict[str, Any]] = []
                sentence_buffer: list[str] = []
                try:
                    http_error_detail: tuple[str | int, str] | None = None
                    request_urls = [str(self.completion_url)]
                    if self._ollama_mode:
                        fallback_url = str(self.completion_url).replace("/api/chat", "/v1/chat/completions")
                        if fallback_url != request_urls[0]:
                            request_urls.append(fallback_url)

                    for attempt, request_url in enumerate(request_urls):
                        if request_url.endswith("/v1/chat/completions"):
                            data["messages"] = self._sanitize_messages_for_openai(base_messages)
                        elif self._ollama_mode:
                            data["messages"] = self._sanitize_messages_for_ollama(base_messages)
                        else:
                            data["messages"] = self._sanitize_messages_for_openai(base_messages)
                        try:
                            with requests.post(
                                request_url,
                                headers=self.prompt_headers,
                                json=data,
                                stream=True,
                                timeout=30,  # Add a timeout for the request itself
                            ) as response:
                                if response.status_code >= 400:
                                    response_text = response.text.strip()
                                    http_error_detail = (response.status_code, response_text)
                                    logger.error(
                                        "LLM Processor: HTTP error {} from LLM service: {}",
                                        response.status_code,
                                        response_text or response.reason,
                                    )
                                    logger.error(
                                        "LLM Processor: LLM payload (truncated): {}",
                                        json.dumps(data)[:1200],
                                    )
                                    response.raise_for_status()
                                logger.debug("LLM Processor: Request to LLM successful, processing stream...")
                                for line in response.iter_lines():
                                    if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                                        logger.info("LLM Processor: Interruption or shutdown detected during LLM stream.")
                                        break  # Stop processing stream

                                    if line:
                                        cleaned_line_data = self._clean_raw_bytes(line)
                                        if cleaned_line_data:
                                            chunk = self._process_chunk(cleaned_line_data)
                                            if chunk:
                                                if isinstance(chunk, list):
                                                    self._process_tool_chunks(tool_calls_buffer, chunk)
                                                elif not autonomy_mode:
                                                    sentence_buffer.append(chunk)
                                                    if chunk.strip() in self.PUNCTUATION_SET and (
                                                        len(sentence_buffer) < 2
                                                        or not sentence_buffer[-2].strip().isdigit()
                                                    ):
                                                        self._process_sentence_for_tts(sentence_buffer)
                                                        sentence_buffer = []
                                            elif cleaned_line_data.get("done_marker"):
                                                break
                                            elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                                                break

                                if self.processing_active_event.is_set() and tool_calls_buffer:
                                    self._process_tool_call(tool_calls_buffer, autonomy_mode, tool_names)
                                elif self.processing_active_event.is_set() and sentence_buffer:
                                    self._process_sentence_for_tts(sentence_buffer)
                            break
                        except requests.exceptions.HTTPError as e:
                            response = getattr(e, "response", None)
                            status_code = response.status_code if response is not None else "unknown"
                            response_text = ""
                            if response is not None:
                                response_text = response.text.strip()
                            http_error_detail = (status_code, response_text or str(e))
                            if attempt < len(request_urls) - 1:
                                logger.warning(
                                    "LLM Processor: Retrying with fallback endpoint {}",
                                    request_urls[attempt + 1],
                                )
                                continue
                            raise

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"LLM Processor: Connection error to LLM service: {e}")
                    self.tts_input_queue.put(
                        "I'm unable to connect to my thinking module. Please check the LLM service connection."
                    )
                except requests.exceptions.Timeout as e:
                    logger.error(f"LLM Processor: Request to LLM timed out: {e}")
                    self.tts_input_queue.put("My brain seems to be taking too long to respond. It might be overloaded.")
                except requests.exceptions.HTTPError as e:
                    if http_error_detail:
                        status_code, detail = http_error_detail
                        logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {detail}")
                        self.tts_input_queue.put(
                            f"I received an error from my thinking module. HTTP status {status_code}."
                        )
                    else:
                        status_code = (
                            e.response.status_code
                            if hasattr(e, "response") and hasattr(e.response, "status_code")
                            else "unknown"
                        )
                        logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                        self.tts_input_queue.put(
                            f"I received an error from my thinking module. HTTP status {status_code}."
                        )
                except requests.exceptions.RequestException as e:
                    logger.error(f"LLM Processor: Request to LLM failed: {e}")
                    self.tts_input_queue.put("Sorry, I encountered an error trying to reach my brain.")
                except Exception as e:
                    logger.exception(f"LLM Processor: Unexpected error during LLM request/streaming: {e}")
                    self.tts_input_queue.put("I'm having a little trouble thinking right now.")
                finally:
                    if self.processing_active_event.is_set():  # Only send EOS if not interrupted
                        logger.debug("LLM Processor: Sending EOS token to TTS queue.")
                        self.tts_input_queue.put("<EOS>")
                    else:
                        logger.info("LLM Processor: Interrupted, not sending EOS from LLM processing.")
                        # The AudioPlayer will handle clearing its state.
                        # If an EOS was already sent by TTS from a *previous* partial sentence,
                        # this could lead to an early clear of currently_speaking.
                        # The `processing_active_event` is key to synchronize.
                    if inflight_guard:
                        self._inflight_counter.decrement()

            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"LLM Processor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("LanguageModelProcessor thread finished.")
