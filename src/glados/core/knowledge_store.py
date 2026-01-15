from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

from loguru import logger


@dataclass(frozen=True)
class KnowledgeEntry:
    entry_id: int
    text: str
    created_at: float
    updated_at: float


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def list_entries(self) -> list[KnowledgeEntry]:
        with self._lock:
            return self._load_entries()

    def add(self, text: str) -> KnowledgeEntry:
        with self._lock:
            entries = self._load_entries()
            next_id = max((entry.entry_id for entry in entries), default=0) + 1
            now = time.time()
            entry = KnowledgeEntry(entry_id=next_id, text=text, created_at=now, updated_at=now)
            entries.append(entry)
            self._write_entries(entries)
            return entry

    def update(self, entry_id: int, text: str) -> KnowledgeEntry | None:
        with self._lock:
            entries = self._load_entries()
            updated = None
            now = time.time()
            new_entries: list[KnowledgeEntry] = []
            for entry in entries:
                if entry.entry_id == entry_id:
                    updated = KnowledgeEntry(
                        entry_id=entry.entry_id,
                        text=text,
                        created_at=entry.created_at,
                        updated_at=now,
                    )
                    new_entries.append(updated)
                else:
                    new_entries.append(entry)
            if updated:
                self._write_entries(new_entries)
            return updated

    def delete(self, entry_id: int) -> bool:
        with self._lock:
            entries = self._load_entries()
            new_entries = [entry for entry in entries if entry.entry_id != entry_id]
            if len(new_entries) == len(entries):
                return False
            self._write_entries(new_entries)
            return True

    def clear(self) -> int:
        with self._lock:
            entries = self._load_entries()
            if not entries:
                return 0
            self._write_entries([])
            return len(entries)

    def _load_entries(self) -> list[KnowledgeEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("KnowledgeStore: failed to read %s: %s", self._path, exc)
            return []

        if isinstance(data, dict):
            items = data.get("entries", [])
        else:
            items = data

        entries: list[KnowledgeEntry] = []
        for item in items:
            try:
                entries.append(
                    KnowledgeEntry(
                        entry_id=int(item["entry_id"]),
                        text=str(item["text"]),
                        created_at=float(item.get("created_at", 0.0)),
                        updated_at=float(item.get("updated_at", 0.0)),
                    )
                )
            except Exception:
                continue
        return entries

    def _write_entries(self, entries: list[KnowledgeEntry]) -> None:
        payload = {
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "text": entry.text,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                }
                for entry in entries
            ]
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
