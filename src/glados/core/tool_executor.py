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
from .tool_invocation import ToolInvocation

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
        self._tool_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="native-tool")
        self._tool_capacity = threading.BoundedSemaphore(4)
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

    def _prepare_call(self, call: ToolInvocation) -> bool:
        call.generation = call.tool_call.get("_quiet_generation", self._quiet_generation())
        if self._quiet_mode() or call.generation != self._quiet_generation():
            return False
        if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
            logger.info("ToolExecutor: Interruption signal active, discarding tool call.")
            return False

        logger.info(f"ToolExecutor: Received tool call: '{call.tool_call}'")
        call.tool = call.tool_call["function"]["name"]
        logger.success("ToolExecutor: executing {}", call.tool)
        call.tool_call_id = call.tool_call["id"]
        call.started_at = time.perf_counter()
        call.autonomy_mode = bool(call.tool_call.get("autonomy", False))
        call.autonomy_epoch = call.tool_call.get("_autonomy_generation", self._autonomy_generation())
        if call.autonomy_mode and (not self._autonomy_enabled() or call.autonomy_epoch != self._autonomy_generation()):
            return False
        call.cancelled = lambda g=call.generation, a=call.autonomy_epoch, mode=call.autonomy_mode: (
            self.shutdown_event.is_set() or self._quiet_mode() or g != self._quiet_generation()
            or (mode and (not self._autonomy_enabled() or a != self._autonomy_generation()))
        )
        call.autonomy_flag = {"autonomy": True} if call.autonomy_mode else {}
        call.base_queue = self.llm_queue_autonomy if call.autonomy_mode else self.llm_queue_priority
        call.lane = "autonomy" if call.autonomy_mode else "priority"
        call.llm_queue = self._wrap_llm_queue(call.base_queue) if call.autonomy_mode else call.base_queue
        permit = call.tool_call.get("_decision_permit")
        if permit:
            if self.decision_store is None or not self.decision_store.authorize(permit):
                call.base_queue.put({"role": "tool", "tool_call_id": call.tool_call_id,
                                "content": "Action cancelled: its decision settings changed. Ask the user to retry.",
                                "_allow_tools": False, "_quiet_generation": call.generation})
                return False
        routing_permit = call.tool_call.get("_routing_permit")
        if routing_permit and (self.decision_store is None
                               or not self.decision_store.authorize_scope(routing_permit, call.tool)):
            call.base_queue.put({"role": "tool", "tool_call_id": call.tool_call_id,
                            "content": "Action cancelled: its routing settings or available tools changed.",
                            "_allow_tools": False, "_quiet_generation": call.generation})
            return False
        call.llm_queue = _ToolResultQueue(call.llm_queue, call.tool_call, bound=bool(permit or routing_permit or call.tool_call.get("_read_only_tools")),
            cancelled=call.cancelled)
        if self._observability_bus:
            self._observability_bus.emit(
                source="tool",
                kind="start",
                message=call.tool,
                meta={"tool_call_id": call.tool_call_id, "autonomy": call.autonomy_mode},
            )

        try:
            raw_args = call.tool_call["function"]["arguments"]
            if isinstance(raw_args, str):
                call.args = json.loads(raw_args)
            else:
                call.args = raw_args
        except json.JSONDecodeError:
            logger.trace(
                "ToolExecutor: Failed to parse non-JSON tool call args: "
                f"{call.tool_call['function']['arguments']}"
            )
            call.args = {}

        return True

    def _execute_search(self, call: ToolInvocation, tasks) -> None:
        background_search = self._autonomy_enabled() and not call.autonomy_mode
        search_cancelled = threading.Event()
        slot_id = "task_search_" + uuid.uuid4().hex[:10]
        query = str(call.args.get("query") or call.args.get("search_query") or call.args.get("objective") or "Web search")[:160]
        def search_result(tool_name: str = call.tool, parameters: dict = call.args,
                          requested_query: str = query, call_id: str = call.tool_call_id,
                          task_id: str = slot_id, request_cancelled: Callable[[], bool] = call.cancelled,
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
        def search_progress(parameters: dict = call.args) -> str:
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
            self._enqueue(call.llm_queue, {"role": "tool", "tool_call_id": call.tool_call_id,
                                      "content": json.dumps({"error": str(exc)})}, lane=call.lane)
            return
        if not background_search:
            while not handle.future.done() and not call.cancelled():
                self.shutdown_event.wait(.05)
            if not call.cancelled():
                completed = handle.future.result()
                slot = tasks._slot_store.get_slot(slot_id)
                tasks._slot_store.mark_handled(slot_id, slot.revision)
                self._enqueue(call.llm_queue, {"role": "tool", "tool_call_id": call.tool_call_id,
                    "content": completed.report or json.dumps({"status": completed.status,
                                                            "summary": completed.summary})}, lane=call.lane)
            return
        self._enqueue(call.llm_queue, {
            "role": "tool", "tool_call_id": call.tool_call_id,
            "content": json.dumps({"status": handle.status if isinstance(handle.status, str) else "queued", "task_id": slot_id, "query": query,
                "instruction": "Give only one short acknowledgement of the stated search status (queued or running). "
                "Do not add commentary, camera observations, questions or invented findings. "
                "Autonomy Core will ask Central Core to report the saved result when ready."}),
        }, lane=call.lane)
        if self._observability_bus:
            self._observability_bus.emit("tool", "background", "Web search started",
                                         meta={"slot_id": slot_id, "query": query})
        return

    def _execute_mcp(self, call: ToolInvocation) -> None:
        if not self.mcp_manager:
            tool_error = "error: MCP tools are unavailable"
            logger.error(f"ToolExecutor: {tool_error}")
            if self._observability_bus:
                self._observability_bus.emit(
                    source="tool",
                    kind="error",
                    message=tool_error,
                    level="error",
                    meta={"tool": call.tool, "tool_call_id": call.tool_call_id},
                )
            self._enqueue(
                call.llm_queue,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": tool_error,
                    "type": "function_call_output",
                    **call.autonomy_flag,
                },
                lane=call.lane,
            )
            return
        tasks = self.tool_config.get("task_manager")
        if call.tool == "mcp.internet_search.web_search_exa" and tasks:
            self._execute_search(call, tasks)
            return
        try:
            core = self.tool_config.get("search_agent") if call.tool == "mcp.internet_search.web_search_exa" else None
            result = (core.research(call.args, cancelled=call.cancelled, context_current=lambda: not call.cancelled())
                      if core else self.mcp_manager.call_tool(call.tool, call.args, timeout=self.tool_timeout))
            if call.cancelled():
                return
            failed = self._search_failed(result) if core else str(result).lower().startswith("error:")
            if self._observability_bus:
                elapsed = time.perf_counter() - call.started_at
                self._observability_bus.emit(
                    source="tool",
                    kind="error" if failed else "finish",
                    message=call.tool,
                    level="error" if failed else "info",
                    meta={"tool_call_id": call.tool_call_id, "elapsed_s": round(elapsed, 3)},
                )
            logger.log("ERROR" if failed else "SUCCESS", "ToolExecutor: finished {}", call.tool)
            self._emit_tool_event("tool_failure" if failed else "tool_success", call.tool)
            self._enqueue(
                call.llm_queue,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": str(result),
                    "type": "function_call_output",
                    **call.autonomy_flag,
                },
                lane=call.lane,
            )
        except Exception as e:
            tool_error = f"error: MCP tool '{call.tool}' failed - {e}"
            self._emit_tool_event("tool_failure", call.tool)
            logger.error(f"ToolExecutor: {tool_error}")
            if self._observability_bus:
                self._observability_bus.emit(
                    source="tool",
                    kind="error",
                    message=trim_message(tool_error),
                    level="error",
                    meta={"tool": call.tool, "tool_call_id": call.tool_call_id},
                )
            self._enqueue(
                call.llm_queue,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": tool_error,
                    "type": "function_call_output",
                    **call.autonomy_flag,
                },
                lane=call.lane,
            )
        return


    def _execute_native(self, call: ToolInvocation) -> None:
        if not self._tool_capacity.acquire(blocking=False):
            self._enqueue(call.llm_queue, {"role": "tool", "tool_call_id": call.tool_call_id,
                "content": "error: native tool capacity exhausted; earlier calls are still running"}, lane=call.lane)
            return
        timed_out = threading.Event()
        result_queue = call.llm_queue
        result_queue.single_result = True
        result_queue.cancelled = lambda expired=timed_out, stale=call.cancelled: expired.is_set() or stale()
        try:
            tool_instance = tool_classes[call.tool](
                llm_queue=result_queue,
                tool_config={**self.tool_config, "_quiet_generation": call.generation,
                             "_autonomy_generation": call.autonomy_epoch if call.autonomy_mode else None,
                             "_cancelled": result_queue.cancelled},
            )
            future = self._tool_pool.submit(tool_instance.run, call.tool_call_id, call.args)
        except Exception:
            self._tool_capacity.release()
            raise
        future.add_done_callback(lambda done: self._tool_capacity.release())
        try:
            future.result(timeout=self.tool_timeout)
            self._emit_tool_event("tool_success", call.tool)
            if self._observability_bus:
                self._observability_bus.emit("tool", "finish", call.tool,
                    meta={"tool_call_id": call.tool_call_id, "elapsed_s": round(time.perf_counter() - call.started_at, 3)})
        except FuturesTimeoutError:
            timeout_error = f"error: tool '{call.tool}' timed out after {self.tool_timeout}s"
            # Publish or close atomically against a concurrently finishing tool.
            self._finish(result_queue, {"role": "tool", "tool_call_id": call.tool_call_id,
                "content": timeout_error}, call.lane)
            timed_out.set()
            future.cancel()
            self._emit_tool_event("tool_timeout", call.tool)
            if self._observability_bus:
                self._observability_bus.emit("tool", "timeout", timeout_error, level="warning",
                    meta={"tool": call.tool, "tool_call_id": call.tool_call_id})
        except Exception as exc:
            self._finish(result_queue, {"role": "tool", "tool_call_id": call.tool_call_id,
                "content": f"error: tool '{call.tool}' failed - {exc}"}, call.lane)
            self._emit_tool_event("tool_failure", call.tool)

    def _unknown_tool(self, call: ToolInvocation) -> None:
        tool_error = f"error: no tool named {call.tool} is available"
        logger.error(f"ToolExecutor: {tool_error}")
        if self._observability_bus:
            self._observability_bus.emit(
                source="tool",
                kind="error",
                message=trim_message(tool_error),
                level="error",
                meta={"tool": call.tool, "tool_call_id": call.tool_call_id},
            )
        self._enqueue(
            call.llm_queue,
            {
                "role": "tool",
                "tool_call_id": call.tool_call_id,
                "content": tool_error,
                "type": "function_call_output",
                **call.autonomy_flag,
            },
            lane=call.lane,
        )

    def _dispatch_call(self, call: ToolInvocation) -> None:
        if not self._prepare_call(call):
            return
        if call.tool.startswith("mcp."):
            self._execute_mcp(call)
        elif call.tool in all_tools:
            self._execute_native(call)
        else:
            self._unknown_tool(call)

    def run(self) -> None:
        """Dispatch tools without tying worker lifetime to the queue-consumer thread."""
        logger.info("ToolExecutor thread started.")
        try:
            while not self.shutdown_event.is_set():
                call = ToolInvocation()
                try:
                    call.tool_call = self.tool_calls_queue.get(timeout=self.pause_time)
                    self._dispatch_call(call)
                except queue.Empty:
                    pass
                except Exception as exc:
                    if call.generation is not None and not call.autonomy_mode:
                        self._end_user_turn(call.generation, "tool_error")
                    logger.exception("ToolExecutor: Unexpected error in main run loop: {}", exc)
                    time.sleep(0.1)
        finally:
            self._tool_pool.shutdown(wait=False, cancel_futures=True)
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
    def _finish(result_queue: "_ToolResultQueue", item: dict[str, Any], lane: str) -> None:
        """Terminal timeout/failure results get the same lane metadata and tool policy as _enqueue."""
        item = {"type": "function_call_output", **item, "_enqueued_at": time.time(), "_lane": lane}
        if lane != "autonomy":
            item.setdefault("_allow_tools", False)
        result_queue.finish(item)

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
    def __init__(self, target, tool_call, bound=False, cancelled=lambda: False, single_result=False):
        self.target = target
        self.context = tool_call["function"]
        self.generation = tool_call.get("_quiet_generation")
        self.autonomy_context = {key: tool_call[key] for key in (
            "_autonomy_context", "_autonomy_steps", "_autonomy_generation", "_autonomy_cycle",
        ) if key in tool_call}
        self.bound = bound
        self.cancelled = cancelled
        self.single_result = single_result
        self._result_lock = threading.Lock()
        self._finished = False

    def _message(self, message):
        result = {**message, "_tool_reply_context": self.context, **self.autonomy_context}
        if self.generation is not None:
            result["_quiet_generation"] = self.generation
        if self.bound:
            result["_allow_tools"] = False
        return result

    def put(self, message, *args, **kwargs):
        with self._result_lock:
            if not self.cancelled() and not self._finished:
                self.target.put(self._message(message), *args, **kwargs)
                if self.single_result:
                    self._finished = True

    def put_nowait(self, message):
        with self._result_lock:
            if not self.cancelled() and not self._finished:
                self.target.put_nowait(self._message(message))
                if self.single_result:
                    self._finished = True

    def finish(self, message):
        """Deliver one terminal outcome, then reject all later tool writes."""
        with self._result_lock:
            if not self._finished and not self.cancelled():
                self.target.put_nowait(self._message(message))
            self._finished = True
