"""Draft completion must not lose deltas queued at the end of the producer."""

import queue
import threading
from unittest.mock import Mock

import pytest

from glados.core.speculative import SpeculativeStream


def make_stream() -> SpeculativeStream:
    processing = threading.Event()
    processing.set()
    return SpeculativeStream(Mock(), "http://example.test", {}, {"model": "test"}, threading.Event(), processing)


@pytest.mark.parametrize("producer_error", [None, RuntimeError("producer failed")])
def test_completion_after_timeout_drains_last_deltas(
    monkeypatch: pytest.MonkeyPatch, producer_error: RuntimeError | None
) -> None:
    stream = make_stream()
    original_get = stream.chunks.get

    def timeout_then_finish(block: bool = True, timeout: float | None = None) -> bytes:
        if not block:
            return original_get(block=False)
        stream.chunks.put(b"final content")
        stream.chunks.put(b"[DONE]")
        stream.error = producer_error
        stream.finished.set()
        raise queue.Empty

    monkeypatch.setattr(stream.chunks, "get", timeout_then_finish)
    lines = stream.iter_lines()
    assert next(lines) == b"final content"
    assert next(lines) == b"[DONE]"
    if producer_error:
        with pytest.raises(RuntimeError, match="producer failed"):
            next(lines)
    else:
        assert list(lines) == []


def test_cancelled_draft_does_not_release_queued_text() -> None:
    stream = make_stream()
    stream.chunks.put(b"private draft")
    stream.finished.set()
    stream.cancel()
    assert list(stream.iter_lines()) == []
