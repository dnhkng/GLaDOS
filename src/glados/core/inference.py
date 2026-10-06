"""Shared inference admission and observable slots for every language-model mind."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading
import time
from typing import ClassVar
import uuid

from pydantic import BaseModel, Field, model_validator


class InferenceConfig(BaseModel):
    slots: int = Field(default=2, ge=1, le=32)
    reserved_interactive: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def valid_reservation(self) -> "InferenceConfig":
        if self.reserved_interactive >= self.slots:
            raise ValueError("reserved_interactive must be smaller than slots (use 0 for one slot)")
        return self


class InferenceCancelledError(Exception):
    """A request was cancelled while waiting for inference capacity."""


@dataclass
class InferenceRequest:
    id: str
    owner: str
    lane: str
    model: str
    queued_at: float
    started_at: float = 0
    slot: int = -1


class InferenceScheduler:
    PRIORITY: ClassVar[dict[str, int]] = {"router": 0, "priority": 1, "speculative": 2, "autonomy": 3}

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._condition = threading.Condition()
        self._active: dict[int, InferenceRequest] = {}
        self._waiting: list[InferenceRequest] = []
        self._interaction: tuple[int, float] | None = None
        self._last_release: dict | None = None

    def begin_interaction(self, generation: int) -> None:
        with self._condition:
            self._interaction = (generation, time.monotonic())
            self._condition.notify_all()

    def end_interaction(self, generation: int, reason: str) -> None:
        with self._condition:
            if self._interaction and self._interaction[0] == generation:
                self._last_release = {"generation": generation, "reason": reason,
                                      "duration_s": time.monotonic() - self._interaction[1]}
                self._interaction = None
                self._condition.notify_all()

    def _expire_interaction(self) -> None:
        if self._interaction and time.monotonic() - self._interaction[1] >= 120:
            self.end_interaction(self._interaction[0], "safety_timeout")

    def _eligible(self, lane: str) -> bool:
        return self._interaction is None or lane in {"router", "priority", "speculative"}

    def _free_slot(self, lane: str) -> int | None:
        first = 0 if lane in {"router", "priority", "speculative"} else self.config.reserved_interactive
        return next((i for i in range(first, self.config.slots) if i not in self._active), None)


    def acquire(
        self, owner: str, lane: str, model: str, cancelled: Callable[[], bool] = lambda: False
    ) -> InferenceRequest:
        request = InferenceRequest(uuid.uuid4().hex, owner, lane, model, time.time())
        with self._condition:
            self._waiting.append(request)
            try:
                while True:
                    if cancelled():
                        raise InferenceCancelledError()
                    self._expire_interaction()
                    eligible = [r for r in self._waiting if self._eligible(r.lane)]
                    candidate = min(eligible, key=lambda r: self.PRIORITY.get(r.lane, 3), default=None)
                    free = self._free_slot(lane)
                    if candidate is request and free is not None:
                        request.slot, request.started_at = free, time.time()
                        self._active[free] = request
                        return request
                    self._condition.wait(0.05)
            finally:
                self._waiting.remove(request)
                self._condition.notify_all()

    def try_acquire(self, owner: str, lane: str, model: str) -> InferenceRequest | None:
        """Use spare capacity only; speculative work must never delay queued requests."""
        with self._condition:
            self._expire_interaction()
            if not self._eligible(lane) or any(self._eligible(r.lane) for r in self._waiting):
                return None
            free = self._free_slot(lane)
            if free is None:
                return None
            request = InferenceRequest(uuid.uuid4().hex, owner, lane, model, time.time(), time.time(), free)
            self._active[free] = request
            return request

    def release(self, request: InferenceRequest) -> None:
        with self._condition:
            if self._active.get(request.slot) is request:
                del self._active[request.slot]
            self._condition.notify_all()

    @contextmanager
    def lease(
        self, owner: str, lane: str, model: str, cancelled: Callable[[], bool] = lambda: False
    ) -> Iterator[InferenceRequest]:
        request = self.acquire(owner, lane, model, cancelled)
        try:
            yield request
        finally:
            self.release(request)

    def snapshot(self) -> dict:
        with self._condition:
            self._expire_interaction()
            return {
                "interaction_hold": ({"generation": self._interaction[0],
                                      "age_s": time.monotonic() - self._interaction[1]}
                                     if self._interaction else None),
                "last_interaction_release": self._last_release,
                "capacity": self.config.slots,
                "reserved_interactive": self.config.reserved_interactive,
                "active": [asdict(r) for r in self._active.values()],
                "waiting": [{**asdict(r), "wait_reason": "capacity" if self._eligible(r.lane)
                             else "user_response"} for r in self._waiting],
            }
