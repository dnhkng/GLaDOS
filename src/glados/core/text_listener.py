"""
Text listener module for the Glados voice assistant.

This module provides a TextListener class that reads text input from a stream
and forwards it to the LLM queue as an alternative to ASR.
"""

from __future__ import annotations

import queue
import selectors
import sys
import threading
from typing import Any, Callable, TextIO

from loguru import logger

from ..autonomy.interaction_state import InteractionState
from ..observability import ObservabilityBus, trim_message


class TextListener:
    """
    Reads text input from a stream and forwards it to the LLM queue.

    This can be used as an alternative to ASR, e.g., for quick manual testing.
    """

    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        processing_active_event: threading.Event,
        shutdown_event: threading.Event,
        pause_time: float,
        interaction_state: InteractionState | None = None,
        observability_bus: ObservabilityBus | None = None,
        input_stream: TextIO | None = None,
        command_handler: Callable[[str], str] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time
        self._interaction_state = interaction_state
        self._observability_bus = observability_bus
        self._input_stream = input_stream or sys.stdin
        self._command_handler = command_handler
        self._selector: selectors.BaseSelector | None = None

        try:
            selector = selectors.DefaultSelector()
            selector.register(self._input_stream, selectors.EVENT_READ)
            self._selector = selector
        except Exception:
            self._selector = None

    def run(self) -> None:
        logger.info("TextListener thread started.")
        try:
            while not self.shutdown_event.is_set():
                line = self._read_line()
                if line is None:
                    continue
                if line == "":
                    logger.info("TextListener: input stream closed.")
                    break
                text = line.strip()
                if not text:
                    continue
                logger.success(f"Text input: '{text}'")
                if text.startswith("/") and self._command_handler:
                    response = self._command_handler(text)
                    logger.success("Command: {} -> {}", text, response)
                    continue
                if self._observability_bus:
                    self._observability_bus.emit(
                        source="text",
                        kind="user_input",
                        message=trim_message(text),
                    )
                self.llm_queue.put({"role": "user", "content": text})
                if self._interaction_state:
                    self._interaction_state.mark_user()
                self.processing_active_event.set()
        finally:
            if self._selector:
                self._selector.close()
            logger.info("TextListener thread finished.")

    def _read_line(self) -> str | None:
        if self._selector is None:
            return self._input_stream.readline()
        events = self._selector.select(timeout=self.pause_time)
        if not events:
            return None
        return self._input_stream.readline()
