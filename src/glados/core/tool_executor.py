# --- tool_executor.py ---
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import json
import queue
import threading
import time
from typing import Any, Callable
import uuid

from loguru import logger

from ..autonomy.task_manager import TaskResult
from ..mcp import MCPManager
from ..observability import ObservabilityBus, trim_message
from ..tools import all_tools, tool_classes

# Callback signature: (event_type: str, tool_name: str) -> None
ToolEventCallback = Callable[[str, str], None]


class ToolExecutor:
    """
    A thread that executes tool calls from the LLM.
    This class is designed to run in a separate thread, continuously checking
    for new tool calls until a shutdown event is set.
    """

    def __init__(
        self,
        llm_queue_priority: queue.Queue[dict[str, Any]],
        llm_queue_autonomy: queue.Queue[dict[str, Any]],
        tool_calls_queue: queue.Queue[dict[str, Any]],
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        tool_config: dict[str, Any] | None = None,
        tool_timeout: float = 30.0,
        pause_time: float = 0.05,
        mcp_manager: MCPManager | None = None,
        observability_bus: ObservabilityBus | None = None,
        on_tool_event: ToolEventCallback | None = None,
        decision_store=None,
        autonomy_enabled: Callable[[], bool] = lambda: True,
        quiet_mode: Callable[[], bool] = lambda: False,
        quiet_generation: Callable[[], int] = lambda: 0,
        end_user_turn: Callable[[int, str], None] = lambda generation, reason: None,
        autonomy_generation: Callable[[], int] = lambda: 0,
        on_autonomy_done: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.llm_queue_priority = llm_queue_priority
        self.llm_queue_autonomy = llm_queue_autonomy
        self.tool_calls_queue = tool_calls_queue
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.tool_config = tool_config or {}
        self.tool_timeout = tool_timeout
        self.pause_time = pause_time
        self.mcp_manager = mcp_manager
        self._observability_bus = observability_bus
        self._on_tool_event = on_tool_event
        self.decision_store = decision_store
        self._autonomy_enabled = autonomy_enabled
        self._end_user_turn = end_user_turn
        self._quiet_mode, self._quiet_generation = quiet_mode, quiet_generation
        self._autonomy_generation, self._on_autonomy_done = autonomy_generation, on_autonomy_done

    def _emit_tool_event(self, event_type: str, tool_name: str) -> None:
        """Emit a tool event to the callback if registered."""
        if self._on_tool_event:
            self._on_tool_event(event_type, tool_name)

    @staticmethod
    def _search_status(result: str) -> str:
        if str(result).lower().startswith("error:"):
            return "error"
        try:
            payload = json.loads(result)
            if isinstance(payload, dict) and payload.get("status") in {"done", "partial", "error", "cancelled"}:
                return payload["status"]
        except (ValueError, TypeError):
            pass
        return "done"

    @staticmethod
    def _search_failed(result: str) -> bool:
        return ToolExecutor._search_status(result) in {"error", "cancelled"}

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
            generation = None
            autonomy_mode = False
            try:
                tool_call = self.tool_calls_queue.get(timeout=self.pause_time)
                generation = tool_call.get("_quiet_generation", self._quiet_generation())
                if self._quiet_mode() or generation != self._quiet_generation():
                    continue
                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("ToolExecutor: Interruption signal active, discarding tool call.")
                    continue

                logger.info(f"ToolExecutor: Received tool call: '{tool_call}'")
                tool = tool_call["function"]["name"]
                logger.success("ToolExecutor: executing {}", tool)
                tool_call_id = tool_call["id"]
                started_at = time.perf_counter()
                autonomy_mode = bool(tool_call.get("autonomy", False))
                autonomy_epoch = tool_call.get("_autonomy_generation", self._autonomy_generation())
                if autonomy_mode and (not self._autonomy_enabled() or autonomy_epoch != self._autonomy_generation()):
                    continue
                cancelled = lambda g=generation, a=autonomy_epoch, mode=autonomy_mode: (
                    self.shutdown_event.is_set() or self._quiet_mode() or g != self._quiet_generation()
                    or (mode and (not self._autonomy_enabled() or a != self._autonomy_generation()))
                )
                terminal = autonomy_mode and tool in {"speak", "do_nothing"}
                autonomy_flag = {"autonomy": True} if autonomy_mode else {}
                base_queue = self.llm_queue_autonomy if autonomy_mode else self.llm_queue_priority
                lane = "autonomy" if autonomy_mode else "priority"
                terminal_result = queue.Queue() if terminal else None
                llm_queue = (terminal_result if terminal else
                             self._wrap_llm_queue(base_queue) if autonomy_mode else base_queue)
                permit = tool_call.get("_decision_permit")
                if permit:
                    if self.decision_store is None or not self.decision_store.authorize(permit):
                        base_queue.put({"role": "tool", "tool_call_id": tool_call_id,
                                        "content": "Action cancelled: its decision settings changed. Ask the user to retry.",
                                        "_allow_tools": False, "_quiet_generation": generation})
                        continue
                routing_permit = tool_call.get("_routing_permit")
                if routing_permit and (self.decision_store is None
                                       or not self.decision_store.authorize_scope(routing_permit, tool)):
                    base_queue.put({"role": "tool", "tool_call_id": tool_call_id,
                                    "content": "Action cancelled: its routing settings or available tools changed.",
                                    "_allow_tools": False, "_quiet_generation": generation})
                    continue
                llm_queue = _ToolResultQueue(llm_queue, tool_call, bound=bool(permit or routing_permit or tool_call.get("_read_only_tools")),
                    cancelled=cancelled)
                if self._observability_bus:
                    self._observability_bus.emit(
                        source="tool",
                        kind="start",
                        message=tool,
                        meta={"tool_call_id": tool_call_id, "autonomy": autonomy_mode},
                    )

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
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tool",
                                kind="error",
                                message=tool_error,
                                level="error",
                                meta={"tool": tool, "tool_call_id": tool_call_id},
                            )
                        self._enqueue(
                            llm_queue,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_error,
                                "type": "function_call_output",
                                **autonomy_flag,
                            },
                            lane=lane,
                        )
                        continue
                    tasks = self.tool_config.get("task_manager")
                    if tool == "mcp.internet_search.web_search_exa" and tasks:
                        background_search = self._autonomy_enabled() and not autonomy_mode
                        search_cancelled = threading.Event()
                        slot_id = "task_search_" + uuid.uuid4().hex[:10]
                        query = str(args.get("query") or args.get("search_query") or args.get("objective") or "Web search")[:160]
                        def search_result(tool_name: str = tool, parameters: dict = args,
                                          requested_query: str = query, call_id: str = tool_call_id,
                                          task_id: str = slot_id, request_cancelled: Callable[[], bool] = cancelled,
                                          cancel_event: threading.Event = search_cancelled,
                                          background: bool = background_search) -> TaskResult:
                            search_started = time.perf_counter()
                            try:
                                core = self.tool_config.get("search_agent")
                                result = (core.research(parameters,
                                                      cancelled=lambda: cancel_event.is_set() or self.shutdown_event.is_set() or self._quiet_mode(),
                                                      context_current=lambda: not request_cancelled(), task_id=task_id,
                                                      inference_lane="autonomy" if background else "priority") if core else
                                          self.mcp_manager.call_tool(tool_name, parameters, timeout=self.tool_timeout))
                            except Exception:
                                if self._observability_bus:
                                    self._observability_bus.emit("tool", "error", "Background web search failed",
                                        level="error", meta={"tool": tool_name, "tool_call_id": call_id,
                                                              "slot_id": task_id})
                                self._emit_tool_event("tool_failure", tool_name)
                                raise
                            failed = self._search_failed(result)
                            if self._observability_bus:
                                self._observability_bus.emit("tool", "error" if failed else "finish", tool_name,
                                    level="error" if failed else "info", meta={"tool_call_id": call_id,
                                        "slot_id": task_id, "elapsed_s": round(time.perf_counter() - search_started, 3)})
                            self._emit_tool_event("tool_failure" if failed else "tool_success", tool_name)
                            status = self._search_status(result)
                            description = {"done": "is ready", "partial": "has partial findings",
                                           "error": "failed", "cancelled": "was cancelled"}[status]
                            return TaskResult(status, "Requested web search " + description + ": " + requested_query,
                                              report=str(result), importance=0.8 if failed else 0.7,
                                              update_priority="important")
                        def search_progress(parameters: dict = args) -> str:
                            core = self.tool_config.get("search_agent")
                            state = core.snapshot() if core else {}
                            if state.get("requested_query", state.get("query")) == parameters.get("query"):
                                return (f"Researching {state.get('current_query', state['query'])}; "
                                        f"{state.get('sources', 0)} sources, {state.get('findings', 0)} findings")
                            return "Waiting for search results"
                        try:
                            handle = tasks.submit(slot_id, "Web search: " + query, search_result,
                                                  progress=search_progress, group="search", cancelled=search_cancelled)
                        except ValueError as exc:
                            self._enqueue(llm_queue, {"role": "tool", "tool_call_id": tool_call_id,
                                                      "content": json.dumps({"error": str(exc)})}, lane=lane)
                            continue
                        if not background_search:
                            while not handle.future.done() and not cancelled():
                                self.shutdown_event.wait(.05)
                            if not cancelled():
                                completed = handle.future.result()
                                slot = tasks._slot_store.get_slot(slot_id)
                                tasks._slot_store.mark_handled(slot_id, slot.revision)
                                self._enqueue(llm_queue, {"role": "tool", "tool_call_id": tool_call_id,
                                    "content": completed.report or json.dumps({"status": completed.status,
                                                                            "summary": completed.summary})}, lane=lane)
                            continue
                        self._enqueue(llm_queue, {
                            "role": "tool", "tool_call_id": tool_call_id,
                            "content": json.dumps({"status": handle.status if isinstance(handle.status, str) else "queued", "task_id": slot_id, "query": query,
                                "instruction": "Give only one short acknowledgement of the stated search status (queued or running). "
                                "Do not add commentary, camera observations, questions or invented findings. "
                                "Autonomy Core will ask Central Core to report the saved result when ready."}),
                        }, lane=lane)
                        if self._observability_bus:
                            self._observability_bus.emit("tool", "background", "Web search started",
                                                         meta={"slot_id": slot_id, "query": query})
                        continue
                    try:
                        core = self.tool_config.get("search_agent") if tool == "mcp.internet_search.web_search_exa" else None
                        result = (core.research(args, cancelled=cancelled, context_current=lambda: not cancelled())
                                  if core else self.mcp_manager.call_tool(tool, args, timeout=self.tool_timeout))
                        if cancelled():
                            continue
                        failed = self._search_failed(result) if core else str(result).lower().startswith("error:")
                        if self._observability_bus:
                            elapsed = time.perf_counter() - started_at
                            self._observability_bus.emit(
                                source="tool",
                                kind="error" if failed else "finish",
                                message=tool,
                                level="error" if failed else "info",
                                meta={"tool_call_id": tool_call_id, "elapsed_s": round(elapsed, 3)},
                            )
                        logger.log("ERROR" if failed else "SUCCESS", "ToolExecutor: finished {}", tool)
                        self._emit_tool_event("tool_failure" if failed else "tool_success", tool)
                        self._enqueue(
                            llm_queue,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(result),
                                "type": "function_call_output",
                                **autonomy_flag,
                            },
                            lane=lane,
                        )
                    except Exception as e:
                        tool_error = f"error: MCP tool '{tool}' failed - {e}"
                        self._emit_tool_event("tool_failure", tool)
                        logger.error(f"ToolExecutor: {tool_error}")
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tool",
                                kind="error",
                                message=trim_message(tool_error),
                                level="error",
                                meta={"tool": tool, "tool_call_id": tool_call_id},
                            )
                        self._enqueue(
                            llm_queue,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_error,
                                "type": "function_call_output",
                                **autonomy_flag,
                            },
                            lane=lane,
                        )
                    continue

                if tool in all_tools:
                    tool_instance = tool_classes.get(tool)(
                        llm_queue=llm_queue,
                        tool_config={**self.tool_config, "_quiet_generation": generation,
                                     "_autonomy_generation": autonomy_epoch if autonomy_mode else None,
                                     "_cancelled": cancelled},
                    )
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(tool_instance.run, tool_call_id, args)
                        try:
                            future.result(timeout=self.tool_timeout)
                            if self._observability_bus:
                                elapsed = time.perf_counter() - started_at
                                self._observability_bus.emit(
                                    source="tool",
                                    kind="finish",
                                    message=tool,
                                    meta={"tool_call_id": tool_call_id, "elapsed_s": round(elapsed, 3)},
                                )
                            logger.success("ToolExecutor: finished {}", tool)
                            if not terminal:
                                self._emit_tool_event("tool_success", tool)
                        except FuturesTimeoutError:
                            timeout_error = f"error: tool '{tool}' timed out after {self.tool_timeout}s"
                            self._emit_tool_event("tool_timeout", tool)
                            logger.error(f"ToolExecutor: {timeout_error}")
                            if self._observability_bus:
                                self._observability_bus.emit(
                                    source="tool",
                                    kind="timeout",
                                    message=timeout_error,
                                    level="warning",
                                    meta={"tool": tool, "tool_call_id": tool_call_id},
                                )
                            self._enqueue(
                                llm_queue,
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": timeout_error,
                                    "type": "function_call_output",
                                    **autonomy_flag,
                                },
                                lane=lane,
                            )
                        except Exception as exc:
                            self._emit_tool_event("tool_failure", tool)
                            self._enqueue(llm_queue, {"role": "tool", "tool_call_id": tool_call_id,
                                                      "content": f"error: tool '{tool}' failed - {exc}",
                                                      **autonomy_flag}, lane=lane)
                    if terminal and self._on_autonomy_done and not cancelled():
                        try:
                            content = terminal_result.get_nowait().get("content", "")
                        except queue.Empty:
                            content = "error: tool returned no result"
                        outcome = ("error" if str(content).startswith("error:") else
                                   "speak" if tool == "speak" else "silent")
                        reason = str(tool_call.get("_autonomy_reason") or args.get("reason")
                                     or args.get("text") or "No useful action needed")
                        self._on_autonomy_done(tool_call.get("_autonomy_cycle", ""), outcome,
                                               str(content) if outcome == "error" else reason)
                else:
                    tool_error = f"error: no tool named {tool} is available"
                    logger.error(f"ToolExecutor: {tool_error}")
                    if self._observability_bus:
                        self._observability_bus.emit(
                            source="tool",
                            kind="error",
                            message=trim_message(tool_error),
                            level="error",
                            meta={"tool": tool, "tool_call_id": tool_call_id},
                        )
                    self._enqueue(
                        llm_queue,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_error,
                            "type": "function_call_output",
                            **autonomy_flag,
                        },
                        lane=lane,
                    )
            except queue.Empty:
                pass  # Normal
            except Exception as e:
                if generation is not None and not autonomy_mode:
                    self._end_user_turn(generation, "tool_error")
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
                if "_enqueued_at" not in item:
                    item = {**item, "_enqueued_at": time.time(), "_lane": "autonomy"}
                if item.get("role") == "tool" and "_allow_tools" not in item:
                    item = {**item, "_allow_tools": "_autonomy_context" in item}
                try:
                    self._base_queue.put_nowait(item)
                except queue.Full:
                    logger.warning("ToolExecutor: dropped autonomy tool output because LLM queue is full.")

            def put_nowait(self, item: dict[str, Any]) -> None:
                self.put(item)

        return AutonomyQueue(llm_queue)

    @staticmethod
    def _enqueue(
        target_queue: queue.Queue[dict[str, Any]],
        item: dict[str, Any],
        lane: str = "priority",
    ) -> None:
        try:
            if "_enqueued_at" not in item:
                item = {**item, "_enqueued_at": time.time(), "_lane": lane}
            if lane != "autonomy" and item.get("role") == "tool" and "_allow_tools" not in item:
                item = {**item, "_allow_tools": False}
            target_queue.put_nowait(item)
        except queue.Full:
            logger.warning("ToolExecutor: dropped tool output because LLM queue is full.")


class _ToolResultQueue:
    """Carry the performed action into the reply, including transcript-free voice turns."""
    def __init__(self, target, tool_call, bound=False, cancelled=lambda: False):
        self.target = target
        self.context = tool_call["function"]
        self.generation = tool_call.get("_quiet_generation")
        self.autonomy_context = {key: tool_call[key] for key in (
            "_autonomy_context", "_autonomy_steps", "_autonomy_generation", "_autonomy_cycle",
        ) if key in tool_call}
        self.bound = bound
        self.cancelled = cancelled

    def _message(self, message):
        result = {**message, "_tool_reply_context": self.context, **self.autonomy_context}
        if self.generation is not None:
            result["_quiet_generation"] = self.generation
        if self.bound:
            result["_allow_tools"] = False
        return result

    def put(self, message, *args, **kwargs):
        if not self.cancelled():
            self.target.put(self._message(message), *args, **kwargs)

    def put_nowait(self, message):
        if not self.cancelled():
            self.target.put_nowait(self._message(message))
