# --- llm_processor.py ---
import json
import queue
import re
import threading
import time
from typing import Any, ClassVar

from loguru import logger
from pydantic import HttpUrl  # If HttpUrl is used by config
import requests
from ..autonomy import TaskSlotStore
from ..mcp import MCPManager
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
        autonomy_system_prompt: str | None = None,
        mcp_manager: MCPManager | None = None,
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
        self.autonomy_system_prompt = autonomy_system_prompt
        self.mcp_manager = mcp_manager

        self.prompt_headers = {"Content-Type": "application/json"}
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"

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
            if tool_chunk_index >= len(tool_calls_buffer):
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

    def _process_tool_call(self, tool_calls: list[dict[str, Any]], autonomy_mode: bool) -> None:
        """
        Add tool calls to conversation history and send each to the tool calls queue.

        Args:
            tool_calls: List of tool calls to be run.
        """
        self.conversation_history.append(
            {"role": "assistant", "index": 0, "tool_calls": tool_calls, "finish_reason": "tool_calls"}
        )
        for tool_call in tool_calls:
            if autonomy_mode:
                tool_call["autonomy"] = True
            logger.info(f"LLM Processor: Sending to tool calls queue: '{tool_call}'")
            self.tool_calls_queue.put(tool_call)

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
        """Build the message list for the LLM request, injecting vision context if available."""
        messages = list(self.conversation_history)
        extra_messages: list[dict[str, Any]] = []

        if autonomy_mode and self.autonomy_system_prompt:
            extra_messages.append({"role": "system", "content": self.autonomy_system_prompt})

        if self.slot_store:
            slot_message = self.slot_store.as_message()
            if slot_message:
                extra_messages.append(slot_message)
        if self.mcp_manager:
            try:
                extra_messages.extend(self.mcp_manager.get_context_messages())
            except Exception as e:
                logger.warning(f"LLM Processor: Failed to load MCP context messages: {e}")

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

                autonomy_mode = bool(llm_input.get("autonomy", False))
                llm_message = {key: value for key, value in llm_input.items() if key != "autonomy"}
                logger.info(f"LLM Processor: Received input for LLM: '{llm_message}'")
                self.conversation_history.append(llm_message)

                data = {
                    "model": self.model_name,
                    "stream": True,
                    "messages": self._build_messages(autonomy_mode),
                    "tools": self._build_tools(autonomy_mode),
                    # Add other parameters like temperature, max_tokens if needed from config
                }

                tool_calls_buffer: list[dict[str, Any]] = []
                sentence_buffer: list[str] = []
                try:
                    with requests.post(
                        str(self.completion_url),
                        headers=self.prompt_headers,
                        json=data,
                        stream=True,
                        timeout=30,  # Add a timeout for the request itself
                    ) as response:
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
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
                                                len(sentence_buffer) < 2 or not sentence_buffer[-2].strip().isdigit()
                                            ):
                                                self._process_sentence_for_tts(sentence_buffer)
                                                sentence_buffer = []
                                    elif cleaned_line_data.get("done_marker"):
                                        break
                                    elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                                        break

                        if self.processing_active_event.is_set() and tool_calls_buffer:
                            self._process_tool_call(tool_calls_buffer, autonomy_mode)
                        elif self.processing_active_event.is_set() and sentence_buffer:
                            self._process_sentence_for_tts(sentence_buffer)

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"LLM Processor: Connection error to LLM service: {e}")
                    self.tts_input_queue.put(
                        "I'm unable to connect to my thinking module. Please check the LLM service connection."
                    )
                except requests.exceptions.Timeout as e:
                    logger.error(f"LLM Processor: Request to LLM timed out: {e}")
                    self.tts_input_queue.put("My brain seems to be taking too long to respond. It might be overloaded.")
                except requests.exceptions.HTTPError as e:
                    status_code = (
                        e.response.status_code
                        if hasattr(e, "response") and hasattr(e.response, "status_code")
                        else "unknown"
                    )
                    logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                    self.tts_input_queue.put(f"I received an error from my thinking module. HTTP status {status_code}.")
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

            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"LLM Processor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("LanguageModelProcessor thread finished.")
