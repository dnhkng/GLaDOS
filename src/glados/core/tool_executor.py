# --- tool_executor.py ---
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from loguru import logger
from ..mcp import MCPManager
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
        tool_config: dict[str, Any] | None = None,
        tool_timeout: float = 30.0,
        pause_time: float = 0.05,
        mcp_manager: MCPManager | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.tool_calls_queue = tool_calls_queue
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.tool_config = tool_config or {}
        self.tool_timeout = tool_timeout
        self.pause_time = pause_time
        self.mcp_manager = mcp_manager

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
                autonomy_mode = bool(tool_call.get("autonomy", False))
                autonomy_flag = {"autonomy": True} if autonomy_mode else {}
                llm_queue = self._wrap_llm_queue(self.llm_queue) if autonomy_mode else self.llm_queue

                try:
                    raw_args = tool_call["function"]["arguments"]
                    if isinstance(raw_args, str):
                        args = json.loads(raw_args)
                    else:
                        args = raw_args
                except json.JSONDecodeError:
                    logger.trace(
                        "ToolExecutor: Failed to parse non-JSON tool call args: "
                        f"{tool_call['function']['arguments']}"
                    )
                    args = {}

                if tool.startswith("mcp."):
                    if not self.mcp_manager:
                        tool_error = "error: MCP tools are unavailable"
                        logger.error(f"ToolExecutor: {tool_error}")
                        llm_queue.put(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_error,
                                "type": "function_call_output",
                                **autonomy_flag,
                            }
                        )
                        continue
                    try:
                        result = self.mcp_manager.call_tool(tool, args, timeout=self.tool_timeout)
                        llm_queue.put(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(result),
                                "type": "function_call_output",
                                **autonomy_flag,
                            }
                        )
                    except Exception as e:
                        tool_error = f"error: MCP tool '{tool}' failed - {e}"
                        logger.error(f"ToolExecutor: {tool_error}")
                        llm_queue.put(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_error,
                                "type": "function_call_output",
                                **autonomy_flag,
                            }
                        )
                    continue

                if tool in all_tools:
                    tool_instance = tool_classes.get(tool)(
                        llm_queue=llm_queue,
                        tool_config=self.tool_config,
                    )
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(tool_instance.run, tool_call_id, args)
                        try:
                            future.result(timeout=self.tool_timeout)
                        except FuturesTimeoutError:
                            timeout_error = f"error: tool '{tool}' timed out after {self.tool_timeout}s"
                            logger.error(f"ToolExecutor: {timeout_error}")
                            self.llm_queue.put(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": timeout_error,
                                    "type": "function_call_output",
                                    **autonomy_flag,
                                }
                            )
                else:
                    tool_error = f"error: no tool named {tool} is available"
                    logger.error(f"ToolExecutor: {tool_error}")
                    self.llm_queue.put(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_error,
                            "type": "function_call_output",
                            **autonomy_flag,
                        }
                    )
            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"ToolExecutor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("ToolExecutor thread finished.")

    @staticmethod
    def _wrap_llm_queue(llm_queue: queue.Queue[dict[str, Any]]) -> "queue.Queue[dict[str, Any]]":
        class AutonomyQueue:
            def __init__(self, base_queue: queue.Queue[dict[str, Any]]) -> None:
                self._base_queue = base_queue

            def put(self, item: dict[str, Any]) -> None:
                if "autonomy" not in item:
                    item = {**item, "autonomy": True}
                self._base_queue.put(item)

            def put_nowait(self, item: dict[str, Any]) -> None:
                self.put(item)

        return AutonomyQueue(llm_queue)
