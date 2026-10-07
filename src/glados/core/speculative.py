"""A private, cancellable stream. Nothing reaches speech/history until the router releases it."""

from collections.abc import Callable, Iterator
import queue
import threading

import requests

from .inference import InferenceScheduler


class SpeculativeStream:
    def __init__(
        self,
        scheduler: InferenceScheduler,
        url: str,
        headers: dict,
        data: dict,
        shutdown: threading.Event,
        processing: threading.Event,
        cancelled_if: Callable[[], bool] = lambda: False,
    ) -> None:
        self.scheduler, self.url, self.headers, self.data = scheduler, url, headers, data
        self.shutdown, self.processing = shutdown, processing
        self.cancelled_if = cancelled_if
        self.cancelled = threading.Event()
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.chunks: queue.Queue = queue.Queue(maxsize=2048)
        self.response: requests.Response | None = None
        self.error: Exception | None = None
        self.started = False
        self.status_code = 200
        self.text, self.reason = "", ""

    def stopped(self) -> bool:
        return (self.cancelled.is_set() or self.shutdown.is_set() or not self.processing.is_set()
                or self.cancelled_if())

    def start(self) -> None:
        # Called only after the router holds its slot. Speculation never queues ahead of it.
        lease = self.scheduler.try_acquire("GLaDOS draft", "speculative", self.data["model"])
        if lease is None:
            return
        self.started = True

        def produce() -> None:
            try:
                if self.stopped():
                    return
                with requests.post(self.url, headers=self.headers, json=self.data, stream=True, timeout=30) as response:
                    self.response = response
                    response.raise_for_status()
                    self.ready.set()
                    for chunk in response.iter_lines(chunk_size=1):
                        if self.stopped():
                            break
                        while not self.stopped():
                            try:
                                self.chunks.put(chunk, timeout=0.05)
                                break
                            except queue.Full:
                                pass
            except Exception as exc:
                self.error = exc
            finally:
                self.scheduler.release(lease)
                self.ready.set()
                self.finished.set()

        threading.Thread(target=produce, name="glados-draft", daemon=True).start()

    def cancel(self) -> None:
        self.cancelled.set()
        # The producer owns/cleans up the HTTP response; no shared engine event is cleared.

    def __enter__(self) -> "SpeculativeStream":
        while not self.ready.wait(0.05):
            if self.stopped():
                raise requests.RequestException("Draft cancelled")
        self.raise_for_status()
        return self

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error

    def iter_lines(self, chunk_size: int = 1) -> Iterator[bytes]:
        while not self.stopped():
            try:
                yield self.chunks.get(timeout=0.05)
            except queue.Empty:
                if self.finished.is_set():
                    # The producer can queue its last delta after get() times
                    # out but before we observe finished. It cannot add more
                    # chunks once finished is set, so draining is sufficient.
                    while not self.stopped():
                        try:
                            yield self.chunks.get_nowait()
                        except queue.Empty:
                            break
                    self.raise_for_status()
                    break

    def __exit__(self, *args: object) -> None:
        self.cancel()
