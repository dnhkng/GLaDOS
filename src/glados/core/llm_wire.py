"""Wire formats of chat-completion backends: request cleanup and streamed-response parsing.

Mixed into :class:`~glados.core.llm_processor.LanguageModelProcessor`.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from loguru import logger


class LLMWireFormat:
    """Ollama and OpenAI-compatible message, chunk, tool-call and thinking-block handling."""

    # Standard thinking tags (GLM-4.7, MiniMax M2.7/M3, DeepSeek, etc.)
    THINKING_OPEN_TAGS: ClassVar[tuple[str, ...]] = ("<think>", "<thinking>", "<reasoning>")
    THINKING_CLOSE_TAGS: ClassVar[tuple[str, ...]] = ("</think>", "</thinking>", "</reasoning>")

    # GPT-OSS harmony format channel markers
    HARMONY_CHANNEL_MARKER: ClassVar[str] = "<|channel|>"
    HARMONY_ANALYSIS_CHANNELS: ClassVar[tuple[str, ...]] = ("analysis", "commentary")
    HARMONY_FINAL_CHANNEL: ClassVar[str] = "final"
    HARMONY_MESSAGE_MARKER: ClassVar[str] = "<|message|>"
    HARMONY_END_MARKER: ClassVar[str] = "<|end|>"

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
                if not line["choices"]:
                    # The final streamed chunk of a remote base reports token usage.
                    if self._remote_backend and (nexus := self._nexus()):
                        nexus.observe_usage(self.model_name, line.get("usage"))
                    return None
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
        # OpenAI-compatible requests carry MCP tools as "mcp_server_tool" (no dots allowed).
        if name.startswith(("mcp.", "mcp_")):
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
