from __future__ import annotations

from dataclasses import dataclass
import threading
import time


@dataclass
class MindStatus:
    mind_id: str
    title: str
    status: str
    summary: str
    updated_at: float
    role: str | None = None


class MindRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minds: dict[str, MindStatus] = {}

    def register(
        self,
        mind_id: str,
        title: str,
        status: str = "idle",
        summary: str = "",
        role: str | None = None,
        updated_at: float | None = None,
    ) -> MindStatus:
        if updated_at is None:
            updated_at = time.time()
        status_obj = MindStatus(
            mind_id=mind_id,
            title=title,
            status=status,
            summary=summary,
            updated_at=updated_at,
            role=role,
        )
        with self._lock:
            self._minds[mind_id] = status_obj
        return status_obj

    def update(
        self,
        mind_id: str,
        status: str,
        summary: str = "",
        updated_at: float | None = None,
    ) -> MindStatus:
        if updated_at is None:
            updated_at = time.time()
        with self._lock:
            existing = self._minds.get(mind_id)
            title = existing.title if existing else mind_id
            role = existing.role if existing else None
            status_obj = MindStatus(
                mind_id=mind_id,
                title=title,
                status=status,
                summary=summary,
                updated_at=updated_at,
                role=role,
            )
            self._minds[mind_id] = status_obj
        return status_obj

    def snapshot(self) -> list[MindStatus]:
        with self._lock:
            return list(self._minds.values())
