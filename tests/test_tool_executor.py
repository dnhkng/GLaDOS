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
