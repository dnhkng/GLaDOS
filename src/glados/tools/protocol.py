"""The two interfaces between the tool executor and the tools it runs."""

from __future__ import annotations

from typing import Any, Protocol

Message = dict[str, Any]


class MessageSink(Protocol):
    """Where a tool hands back its result: a model queue, or a wrapper that tags and filters it."""

    def put(self, item: Message) -> None: ...

    def put_nowait(self, item: Message) -> None: ...


class Tool(Protocol):
    """A tool runs one call and reports to its sink; it never returns the result directly."""

    def __init__(self, llm_queue: MessageSink, tool_config: dict[str, Any] | None = None) -> None: ...

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None: ...
