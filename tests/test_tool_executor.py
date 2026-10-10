"""Tool execution, cancellation, argument parsing, and result routing."""

import json
import queue
import threading
from unittest.mock import Mock

from loguru import logger
import pytest

from glados.core.tool_executor import ToolExecutor


@pytest.fixture(autouse=True)
def capture_loguru(caplog):
    sink = logger.add(caplog.handler, format="{message}")
    caplog.set_level("INFO")
    try:
        yield
    finally:
        logger.remove(sink)


def make_executor():
    return ToolExecutor(queue.Queue(), queue.Queue(), queue.Queue(), threading.Event(), threading.Event())


def run_call(executor, call, completed):
    executor.tool_calls_queue.put(call)
    thread = threading.Thread(target=executor.run)
    thread.start()
    try:
        assert completed.wait(2), "Tool execution did not finish"
    finally:
        executor.shutdown_event.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_run_shutdown_event(caplog):
    executor = make_executor()
    executor.shutdown_event.set()
    executor.run()
    assert "ToolExecutor thread started." in caplog.text
    assert "ToolExecutor thread finished." in caplog.text


def test_tool_call_discarded_if_processing_inactive(mocker, caplog):
    tool = Mock()
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {"test tool": tool})
    executor = make_executor()
    completed = threading.Event()
    sink = logger.add(lambda message: completed.set() if "discarding tool call" in str(message) else None)
    try:
        run_call(executor, {"function": {"name": "test tool", "arguments": "{}"}, "id": "123"}, completed)
    finally:
        logger.remove(sink)
    assert "Interruption signal active, discarding tool call" in caplog.text
    tool.assert_not_called()


@pytest.mark.parametrize("arguments, expected", [(json.dumps({"key": "value"}), {"key": "value"}), ("invalid_json", {})])
def test_process_tool_arguments(mocker, caplog, arguments, expected):
    completed = threading.Event()
    instance = Mock()
    instance.run.side_effect = lambda *_: completed.set()
    tool = Mock(return_value=instance)
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {"test tool": tool})
    executor = make_executor()
    executor.processing_active_event.set()
    run_call(executor, {"function": {"name": "test tool", "arguments": arguments}, "id": "123"}, completed)
    instance.run.assert_called_once_with("123", expected)
    tool.assert_called_once()
    kwargs = tool.call_args.kwargs
    assert kwargs["llm_queue"].target is executor.llm_queue_priority
    assert kwargs["tool_config"]["_quiet_generation"] == 0
    assert callable(kwargs["tool_config"]["_cancelled"])


def test_unknown_tool(mocker, caplog):
    tool = Mock()
    mocker.patch("glados.core.tool_executor.all_tools", ["test tool"])
    mocker.patch("glados.core.tool_executor.tool_classes", {"test tool": tool})
    executor = make_executor()
    executor.processing_active_event.set()
    completed = threading.Event()
    sink = logger.add(lambda message: completed.set() if "no tool named unknown tool" in str(message) else None)
    call = {"function": {"name": "unknown tool", "arguments": "{}"}, "id": "123"}
    try:
        run_call(executor, call, completed)
    finally:
        logger.remove(sink)
    result = executor.llm_queue_priority.get(timeout=2)
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "123"
    assert result["content"] == "error: no tool named unknown tool is available"
    assert result["_tool_reply_context"] == call["function"]
    assert executor.llm_queue_autonomy.empty()
    tool.assert_not_called()


def test_timed_out_tool_does_not_block_next_call_or_publish_late_result(monkeypatch):
    import glados.core.tool_executor as module
    released = threading.Event()
    finished = threading.Event()

    class Tool:
        def __init__(self, llm_queue, tool_config):
            self.queue = llm_queue
        def run(self, call_id, args):
            if call_id == "slow":
                released.wait(2)
            self.queue.put({"role": "tool", "tool_call_id": call_id, "content": call_id})
            if call_id == "slow":
                finished.set()

    monkeypatch.setattr(module, "all_tools", ["test"])
    monkeypatch.setattr(module, "tool_classes", {"test": Tool})
    executor = make_executor()
    executor.tool_timeout = .03
    executor.processing_active_event.set()
    worker = threading.Thread(target=executor.run)
    worker.start()
    def submit(call_id):
        executor.tool_calls_queue.put({"id": call_id, "function": {"name": "test", "arguments": {}}})
    try:
        submit("slow")
        timeout = executor.llm_queue_priority.get(timeout=1)
        assert "timed out" in timeout["content"]
        submit("fast")
        assert executor.llm_queue_priority.get(timeout=1)["content"] == "fast"
        released.set()
        assert finished.wait(1)
        assert executor.llm_queue_priority.empty()
    finally:
        released.set()
        executor.shutdown_event.set()
        worker.join(1)
    assert not worker.is_alive()


def test_hung_tools_have_bounded_admission(monkeypatch):
    import glados.core.tool_executor as module
    released = threading.Event()
    started = []
    class Tool:
        def __init__(self, **kwargs):
            pass
        def run(self, call_id, args):
            started.append(call_id)
            released.wait(2)
    monkeypatch.setattr(module, "all_tools", ["test"])
    monkeypatch.setattr(module, "tool_classes", {"test": Tool})
    executor = make_executor()
    executor.tool_timeout = .02
    executor.processing_active_event.set()
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        for i in range(5):
            executor.tool_calls_queue.put({"id": str(i), "function": {"name": "test", "arguments": {}}})
            output = executor.llm_queue_priority.get(timeout=1)
            assert ("timed out" if i < 4 else "capacity exhausted") in output["content"]
        assert len(started) == 4
    finally:
        released.set()
        executor.shutdown_event.set()
        worker.join(1)
    assert not worker.is_alive()


def test_failed_and_timed_out_tools_carry_lane_metadata(monkeypatch):
    import glados.core.tool_executor as module
    released = threading.Event()

    class Tool:
        def __init__(self, llm_queue, tool_config):
            pass
        def run(self, call_id, args):
            if call_id == "slow":
                released.wait(2)
            else:
                raise RuntimeError("boom")

    monkeypatch.setattr(module, "all_tools", ["test"])
    monkeypatch.setattr(module, "tool_classes", {"test": Tool})
    executor = make_executor()
    executor.tool_timeout = .03
    executor.processing_active_event.set()
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        for call_id, expected in (("broken", "failed"), ("slow", "timed out")):
            executor.tool_calls_queue.put({"id": call_id, "function": {"name": "test", "arguments": {}}})
            result = executor.llm_queue_priority.get(timeout=1)
            assert expected in result["content"]
            assert result["_lane"] == "priority" and "_enqueued_at" in result
            assert result["_allow_tools"] is False and result["type"] == "function_call_output"
    finally:
        released.set()
        executor.shutdown_event.set()
        worker.join(1)
    assert not worker.is_alive()
