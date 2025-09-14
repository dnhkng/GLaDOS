# --- tool_executor.py ---
import json
import queue
import threading
import time
from typing import Any

from loguru import logger
from ..tools import all_tools, tool_classes


class ToolExecutor:
    """
    A thread that executes tool calls from the LLM.
    This class is designed to run in a separate thread, continuously checking
    for new tool calls until a shutdown event is set.
    """

    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_calls_queue: queue.Queue[dict[str, Any]],
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
    ) -> None:
        self.llm_queue = llm_queue
        self.tool_calls_queue = tool_calls_queue
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

    def run(self) -> None:
        """
        Starts the main loop for the ToolExecutor thread.

        This method continuously checks the tool calls queue for tool calls to
        run. It processes the tool arguments, sends them to the tool and
        streams the response. The thread will run until the shutdown event is
        set, at which point it will exit gracefully.
        """
        logger.info("ToolExecutor thread started.")
        while not self.shutdown_event.is_set():
            try:
                tool_call = self.tool_calls_queue.get(timeout=self.pause_time)
                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("ToolExecutor: Interruption signal active, discarding tool call.")
                    continue

                logger.info(f"ToolExecutor: Received tool call: '{tool_call}'")
                tool = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]

                try:
                    raw_args = tool_call["function"]["arguments"]
                    if isinstance(raw_args, str):
                        # OpenAI format
                        args = json.loads(raw_args)
                    else:
                        # Ollama format
                        args = raw_args
                except json.JSONDecodeError:
                    logger.trace(
                        f"ToolExecutor: Failed to parse non-JSON tool call args: "
                        f"{tool_call["function"]["arguments"]}"
                    )
                    args = {}

                if tool in all_tools:
                    tool_instance = tool_classes.get(tool)(
                        llm_queue=self.llm_queue
                    )
                    tool_instance.run(tool_call_id, args)
                else:
                    tool_error = f"error: no tool named {tool} is available"
                    logger.error(f"ToolExecutor: {tool_error}")
                    # Let the LLM know of the error
                    self.llm_queue.put({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_error,
                        "type": "function_call_output"
                    })
            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"ToolExecutor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("ToolExecutor thread finished.")
