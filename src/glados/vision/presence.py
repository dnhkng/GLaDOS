"""Observed presence transitions; camera downtime is never evidence of absence."""

from datetime import date, datetime
import json
from pathlib import Path

from loguru import logger


class DailyGreetingHistory:
    """Remember one first sighting per local day, independently of camera resets."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._seen_date: str | None = None
        if path and path.exists():
            try:
                saved = json.loads(path.read_text())
                self._seen_date = date.fromisoformat(saved["last_seen_date"]).isoformat()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Could not load daily greeting history: {}", exc)
        self.reset()

    def reset(self) -> None:
        """Reset live evidence without forgetting today's sighting."""
        self._last_at: float | None = None
        self._last_presence: str | None = None
        self._event: dict | None = None

    def observe(self, presence: str, captured_at: float, max_gap_s: float = 30) -> dict | None:
        if self._last_at is not None and captured_at <= self._last_at:
            return self._event
        if self._last_at is not None and captured_at - self._last_at > max_gap_s:
            self.reset()
        continuously_present = self._last_presence == "present"
        local = datetime.fromtimestamp(captured_at).astimezone()
        day = local.date().isoformat()
        if presence == "present" and self._seen_date != day:
            # Record afternoon sightings too: a clock/camera reset must not create a later "first" sighting.
            saved = {"last_seen_date": day, "first_seen_at": captured_at}
            if self._path:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self._path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(saved) + "\n")
                    temporary.replace(self._path)
                except OSError as exc:
                    logger.warning("Could not save daily greeting history: {}", exc)
            self._seen_date = day
            if local.hour < 10 and not continuously_present:
                self._event = {
                    "kind": "first_seen_today",
                    "captured_at": captured_at,
                    "local_date": day,
                    "local_time": local.strftime("%H:%M:%S"),
                    "timezone": local.tzname(),
                    "attention_key": f"morning_greeting@{day}",
                }
        if presence != "present" or (self._event and captured_at - self._event["captured_at"] > 60):
            self._event = None
        self._last_at, self._last_presence = captured_at, presence
        return self._event


class PresenceHistory:
    def __init__(self, absence_s: float = 60, arrival_fresh_s: float = 60) -> None:
        self.absence_s = absence_s
        self.arrival_fresh_s = arrival_fresh_s
        self.reset()

    def reset(self) -> None:
        self.last_at: float | None = None
        self.absent_since: float | None = None
        self.arrival: dict | None = None

    def observe(self, presence: str, captured_at: float, max_gap_s: float = 30) -> dict | None:
        if self.last_at is not None and captured_at <= self.last_at:
            return self.arrival
        if self.last_at is not None and captured_at - self.last_at > max_gap_s:
            self.reset()
        self.last_at = captured_at
        if presence == "absent":
            if self.absent_since is None:
                self.absent_since = captured_at
            self.arrival = None
        elif presence == "present":
            if self.absent_since is not None and captured_at - self.absent_since >= self.absence_s:
                self.arrival = {
                    "kind": "person_arrived",
                    "captured_at": captured_at,
                    "observed_absence_s": round(captured_at - self.absent_since, 1),
                    "attention_key": f"person_arrived@{captured_at}",
                }
            self.absent_since = None
            if self.arrival and captured_at - self.arrival["captured_at"] > self.arrival_fresh_s:
                self.arrival = None
        else:
            self.absent_since = None
            self.arrival = None
        return self.arrival
