from dataclasses import dataclass


@dataclass(frozen=True)
class VisionUpdateEvent:
    description: str
    prev_description: str | None
    change_score: float
    captured_at: float


@dataclass(frozen=True)
class TimeTickEvent:
    ticked_at: float


@dataclass(frozen=True)
class TaskUpdateEvent:
    slot_id: str
    title: str
    status: str
    summary: str
    notify_user: bool
    updated_at: float
    importance: float | None = None
    confidence: float | None = None
    next_run: float | None = None
