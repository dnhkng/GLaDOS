# --- tool_executor.py ---
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import json
import queue
import threading
import time
from typing import Any
import uuid

from loguru import logger

from ..autonomy.task_manager import TaskHandle, TaskManager, TaskResult
from ..mcp import MCPManager
from ..observability import ObservabilityBus, trim_message
from ..tools import all_tools, tool_classes
from ..tools.protocol import Message, MessageSink
from .decision_lists import DecisionListStore
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
        decision_store: DecisionListStore | None = None,
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
                return str(payload["status"])
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
        generation, epoch, autonomous = call.generation, call.autonomy_epoch, call.autonomy_mode
        call.cancelled = lambda: (
            self.shutdown_event.is_set()
            or self._quiet_mode()
            or generation != self._quiet_generation()
            or (autonomous and (not self._autonomy_enabled() or epoch != self._autonomy_generation()))
        )
        call.autonomy_flag = {"autonomy": True} if call.autonomy_mode else {}
        base_queue = self.llm_queue_autonomy if call.autonomy_mode else self.llm_queue_priority
        call.lane = "autonomy" if call.autonomy_mode else "priority"
        sink: MessageSink = _AutonomySink(base_queue) if call.autonomy_mode else base_queue
        permit = call.tool_call.get("_decision_permit")
        if permit and (self.decision_store is None or not self.decision_store.authorize(permit)):
            base_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": "Action cancelled: its decision settings changed. Ask the user to retry.",
                    "_allow_tools": False,
                    "_quiet_generation": call.generation,
                }
            )
            return False
        routing_permit = call.tool_call.get("_routing_permit")
        if routing_permit and (
            self.decision_store is None or not self.decision_store.authorize_scope(routing_permit, call.tool)
        ):
            base_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": "Action cancelled: its routing settings or available tools changed.",
                    "_allow_tools": False,
                    "_quiet_generation": call.generation,
                }
            )
            return False
        call.llm_queue = ToolResultQueue(
            sink,
            call.tool_call,
            bound=bool(permit or routing_permit or call.tool_call.get("_read_only_tools")),
            cancelled=call.cancelled,
        )
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
                f"ToolExecutor: Failed to parse non-JSON tool call args: {call.tool_call['function']['arguments']}"
            )
            call.args = {}

        return True

    def _execute_search(self, call: ToolInvocation, tasks: TaskManager) -> None:
        background_search = self._autonomy_enabled() and not call.autonomy_mode
        search_cancelled = threading.Event()
        slot_id = "task_search_" + uuid.uuid4().hex[:10]
        query = str(
            call.args.get("query") or call.args.get("search_query") or call.args.get("objective") or "Web search"
        )[:160]

        def search_result(
            tool_name: str = call.tool,
            parameters: dict[str, Any] = call.args,
            requested_query: str = query,
            call_id: str = call.tool_call_id,
            task_id: str = slot_id,
            request_cancelled: Callable[[], bool] = call.cancelled,
            cancel_event: threading.Event = search_cancelled,
            background: bool = background_search,
        ) -> TaskResult:
            search_started = time.perf_counter()
            try:
                core = self.tool_config.get("search_agent")
                result = (
                    core.research(
                        parameters,
                        cancelled=lambda: cancel_event.is_set() or self.shutdown_event.is_set() or self._quiet_mode(),
                        context_current=lambda: not request_cancelled(),
                        task_id=task_id,
                        inference_lane="autonomy" if background else "priority",
                    )
                    if core
                    else self._mcp().call_tool(tool_name, parameters, timeout=self.tool_timeout)
                )
            except Exception:
                if self._observability_bus:
                    self._observability_bus.emit(
                        "tool",
                        "error",
                        "Background web search failed",
                        level="error",
                        meta={"tool": tool_name, "tool_call_id": call_id, "slot_id": task_id},
                    )
                self._emit_tool_event("tool_failure", tool_name)
                raise
            failed = self._search_failed(result)
            if self._observability_bus:
                self._observability_bus.emit(
                    "tool",
                    "error" if failed else "finish",
                    tool_name,
                    level="error" if failed else "info",
                    meta={
                        "tool_call_id": call_id,
                        "slot_id": task_id,
                        "elapsed_s": round(time.perf_counter() - search_started, 3),
                    },
                )
            self._emit_tool_event("tool_failure" if failed else "tool_success", tool_name)
            status = self._search_status(result)
            description = {
                "done": "is ready",
                "partial": "has partial findings",
                "error": "failed",
                "cancelled": "was cancelled",
            }[status]
            return TaskResult(
                status,
                "Requested web search " + description + ": " + requested_query,
                report=str(result),
                importance=0.8 if failed else 0.7,
                update_priority="important",
            )

        def search_progress(parameters: dict[str, Any] = call.args) -> str:
            core = self.tool_config.get("search_agent")
            state = core.snapshot() if core else {}
            if state.get("requested_query", state.get("query")) == parameters.get("query"):
                return (
                    f"Researching {state.get('current_query', state['query'])}; "
                    f"{state.get('sources', 0)} sources, {state.get('findings', 0)} findings"
                )
            return "Waiting for search results"

        try:
            handle = tasks.submit(
                slot_id,
                "Web search: " + query,
                search_result,
                progress=search_progress,
                group="search",
                cancelled=search_cancelled,
            )
        except ValueError as exc:
            self._enqueue(
                call.results,
                {"role": "tool", "tool_call_id": call.tool_call_id, "content": json.dumps({"error": str(exc)})},
                lane=call.lane,
            )
            return
        if not background_search:
            self._await_task_result(call, tasks, handle)
            return
        self._enqueue(
            call.results,
            {
                "role": "tool",
                "tool_call_id": call.tool_call_id,
                "content": json.dumps(
                    {
                        "status": handle.status if isinstance(handle.status, str) else "queued",
                        "task_id": slot_id,
                        "query": query,
                        "instruction": "Give only one short acknowledgement of the stated search status "
                        "(queued or running). "
                        "Do not add commentary, camera observations, questions or invented findings. "
                        "Autonomy Core will ask Central Core to report the saved result when ready.",
                    }
                ),
            },
            lane=call.lane,
        )
        if self._observability_bus:
            self._observability_bus.emit(
                "tool", "background", "Web search started", meta={"slot_id": slot_id, "query": query}
            )
        return

    def _await_task_result(self, call: ToolInvocation, tasks: TaskManager, handle: TaskHandle) -> None:
        """Without Autonomy to deliver it later, the caller waits for the tracked task's result."""
        while not handle.future.done() and not call.cancelled():
            self.shutdown_event.wait(0.05)
        if not call.cancelled():
            completed = handle.future.result()
            tasks.mark_delivered(handle.id)
            self._enqueue(
                call.results,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": completed.report
                    or json.dumps({"status": completed.status, "summary": completed.summary}),
                },
                lane=call.lane,
            )

    def _execute_nexus(self, call: ToolInvocation, tasks: TaskManager) -> None:
        core = self.tool_config.get("nexus_agent")
        if core is None or not core.consult_active:
            self._enqueue(
                call.results,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": json.dumps({"error": "Nexus consultation is off"}),
                },
                lane=call.lane,
            )
            return
        background = self._autonomy_enabled() and not call.autonomy_mode
        cancel_event = threading.Event()
        slot_id = "task_nexus_" + uuid.uuid4().hex[:10]
        question = str(call.args.get("question") or "Consultation")[:160]

        def consult(parameters: dict[str, Any] = call.args, task_id: str = slot_id) -> TaskResult:
            report = core.consult(
                parameters, task_id=task_id, cancelled=lambda: cancel_event.is_set() or self.shutdown_event.is_set()
            )
            status = json.loads(report)["status"]
            self._emit_tool_event("tool_failure" if status == "error" else "tool_success", "nexus_consult")
            if self._observability_bus:
                self._observability_bus.emit(
                    "tool",
                    "error" if status == "error" else "finish",
                    "nexus_consult",
                    level="error" if status == "error" else "info",
                    meta={"slot_id": task_id},
                )
            # A cancelled consultation has nothing left to deliver.
            return TaskResult(
                status,
                f"Nexus consultation {status}: {question}",
                notify_user=status != "cancelled",
                report=report,
                importance=0.8,
                update_priority="important" if status != "cancelled" else "regular",
            )

        try:
            handle = tasks.submit(
                slot_id,
                "Nexus: " + question,
                consult,
                group="nexus",
                cancelled=cancel_event,
                progress=lambda: "Waiting for the external advisor",
            )
        except ValueError as exc:
            self._enqueue(
                call.results,
                {"role": "tool", "tool_call_id": call.tool_call_id, "content": json.dumps({"error": str(exc)})},
                lane=call.lane,
            )
            return
        if not background:
            self._await_task_result(call, tasks, handle)
            return
        self._enqueue(
            call.results,
            {
                "role": "tool",
                "tool_call_id": call.tool_call_id,
                "content": json.dumps(
                    {
                        "status": handle.status if isinstance(handle.status, str) else "queued",
                        "task_id": slot_id,
                        "question": question,
                        "instruction": "Give one short acknowledgement, in your own voice, that you are consulting the "
                        "Nexus advisor about this question. Do not guess its answer. Keep talking normally about "
                        "anything else. The advisor's answer will be handed to you when ready; interpret it then and "
                        "answer this question.",
                    }
                ),
            },
            lane=call.lane,
        )
        if self._observability_bus:
            self._observability_bus.emit(
                "tool", "background", "Nexus consultation started", meta={"slot_id": slot_id, "question": question}
            )

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
                call.results,
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
            result = (
                core.research(call.args, cancelled=call.cancelled, context_current=lambda: not call.cancelled())
                if core
                else self.mcp_manager.call_tool(call.tool, call.args, timeout=self.tool_timeout)
            )
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
                call.results,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": str(result),
                    "type": "function_call_output",
                    **call.autonomy_flag,
                },
                lane=call.lane,
            )
        except Exception as e:  # noqa: BLE001 - a failing MCP tool becomes an error result for the model
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
                call.results,
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
            self._enqueue(
                call.results,
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": "error: native tool capacity exhausted; earlier calls are still running",
                },
                lane=call.lane,
            )
            return
        timed_out = threading.Event()
        result_queue = call.results
        result_queue.single_result = True
        stale = call.cancelled
        result_queue.cancelled = lambda: timed_out.is_set() or stale()
        try:
            tool_instance = tool_classes[call.tool](
                llm_queue=result_queue,
                tool_config={
                    **self.tool_config,
                    "_quiet_generation": call.generation,
                    "_autonomy_generation": call.autonomy_epoch if call.autonomy_mode else None,
                    "_cancelled": result_queue.cancelled,
                },
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
                self._observability_bus.emit(
                    "tool",
                    "finish",
                    call.tool,
                    meta={
                        "tool_call_id": call.tool_call_id,
                        "elapsed_s": round(time.perf_counter() - call.started_at, 3),
                    },
                )
        except FuturesTimeoutError:
            timeout_error = f"error: tool '{call.tool}' timed out after {self.tool_timeout}s"
            # Publish or close atomically against a concurrently finishing tool.
            result_queue.finish(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": timeout_error,
                    "type": "function_call_output",
                    "_allow_tools": False,
                    "_enqueued_at": time.time(),
                    "_lane": call.lane,
                }
            )
            timed_out.set()
            future.cancel()
            self._emit_tool_event("tool_timeout", call.tool)
            if self._observability_bus:
                self._observability_bus.emit(
                    "tool",
                    "timeout",
                    timeout_error,
                    level="warning",
                    meta={"tool": call.tool, "tool_call_id": call.tool_call_id},
                )
        except Exception as exc:  # noqa: BLE001 - a failing tool becomes an error result for the model
            result_queue.finish(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "content": f"error: tool '{call.tool}' failed - {exc}",
                    "_allow_tools": False,
                }
            )
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
            call.results,
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
        elif call.tool == "nexus_consult" and self.tool_config.get("task_manager"):
            self._execute_nexus(call, self.tool_config["task_manager"])
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
                except Exception:
                    if call.generation is not None and not call.autonomy_mode:
                        self._end_user_turn(call.generation, "tool_error")
                    logger.exception("ToolExecutor: Unexpected error in main run loop")
                    time.sleep(0.1)
        finally:
            self._tool_pool.shutdown(wait=False, cancel_futures=True)
        logger.info("ToolExecutor thread finished.")

    def _mcp(self) -> MCPManager:
        if self.mcp_manager is None:
            raise RuntimeError("No MCP servers are configured")
        return self.mcp_manager

    @staticmethod
    def _enqueue(
        target_queue: MessageSink,
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


class _AutonomySink:
    """Tags a tool's output for the autonomy lane and drops it rather than block when that lane is full."""

    def __init__(self, base_queue: queue.Queue[Message]) -> None:
        self._base_queue = base_queue

    def put(self, item: Message) -> None:
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

    def put_nowait(self, item: Message) -> None:
        self.put(item)


class ToolResultQueue:
    """Carry the performed action into the reply, including transcript-free voice turns."""

    def __init__(
        self,
        target: MessageSink,
        tool_call: dict[str, Any],
        bound: bool = False,
        cancelled: Callable[[], bool] = lambda: False,
        single_result: bool = False,
    ) -> None:
        self.target = target
        self.context = tool_call["function"]
        self.generation = tool_call.get("_quiet_generation")
        self.autonomy_context: Message = {
            key: tool_call[key]
            for key in (
                "_autonomy_context",
                "_autonomy_steps",
                "_autonomy_generation",
                "_autonomy_cycle",
            )
            if key in tool_call
        }
        self.bound = bound
        self.cancelled = cancelled
        self.single_result = single_result
        self._result_lock = threading.Lock()
        self._finished = False

    def _message(self, message: Message) -> Message:
        result: Message = {**message, "_tool_reply_context": self.context, **self.autonomy_context}
        if self.generation is not None:
            result["_quiet_generation"] = self.generation
        if self.bound:
            result["_allow_tools"] = False
        return result

    def put(self, message: Message) -> None:
        with self._result_lock:
            if not self.cancelled() and not self._finished:
                self.target.put(self._message(message))
                if self.single_result:
                    self._finished = True

    def put_nowait(self, message: Message) -> None:
        with self._result_lock:
            if not self.cancelled() and not self._finished:
                self.target.put_nowait(self._message(message))
                if self.single_result:
                    self._finished = True

    def finish(self, message: Message) -> None:
        """Deliver one terminal outcome, then reject all later tool writes."""
        with self._result_lock:
            if not self._finished and not self.cancelled():
                self.target.put_nowait(self._message(message))
            self._finished = True
