from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Callable, Iterable

import httpx
from loguru import logger

from .config import AutonomyJobsConfig, HackerNewsJobConfig, WeatherJobConfig
from .task_manager import TaskManager, TaskResult
from ..observability import ObservabilityBus, trim_message

WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}

SEVERE_WEATHER_CODES = {82, 95, 96, 99}


@dataclass(frozen=True)
class JobDefinition:
    slot_id: str
    title: str
    interval_s: float
    runner: Callable[[], TaskResult]
    run_on_start: bool = True


class BackgroundJobScheduler:
    def __init__(
        self,
        jobs: Iterable[JobDefinition],
        task_manager: TaskManager,
        shutdown_event: threading.Event,
        observability_bus: ObservabilityBus | None = None,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._jobs = list(jobs)
        self._task_manager = task_manager
        self._shutdown_event = shutdown_event
        self._observability_bus = observability_bus
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._next_run: dict[str, float] = {}

    def run(self) -> None:
        if not self._jobs:
            logger.info("BackgroundJobScheduler: no jobs configured.")
            return
        logger.info("BackgroundJobScheduler: starting with %d job(s).", len(self._jobs))

        now = time.time()
        for job in self._jobs:
            initial = now if job.run_on_start else now + job.interval_s
            self._next_run[job.slot_id] = initial

        while not self._shutdown_event.is_set():
            now = time.time()
            for job in self._jobs:
                next_run = self._next_run.get(job.slot_id, now + job.interval_s)
                if now < next_run:
                    continue
                if self._is_running(job.slot_id):
                    continue
                scheduled = now + job.interval_s
                self._next_run[job.slot_id] = scheduled
                self._submit(job, scheduled)
            self._shutdown_event.wait(timeout=self._poll_interval_s)

        logger.info("BackgroundJobScheduler: stopped.")

    def _is_running(self, slot_id: str) -> bool:
        with self._lock:
            return slot_id in self._running

    def _set_running(self, slot_id: str, running: bool) -> None:
        with self._lock:
            if running:
                self._running.add(slot_id)
            else:
                self._running.discard(slot_id)

    def _submit(self, job: JobDefinition, next_run: float) -> None:
        self._set_running(job.slot_id, True)
        if self._observability_bus:
            self._observability_bus.emit(
                source="jobs",
                kind="schedule",
                message=job.title,
                meta={"slot_id": job.slot_id, "next_run": round(next_run)},
            )

        def runner() -> TaskResult:
            try:
                result = job.runner()
                if result.next_run is None:
                    return replace(result, next_run=next_run)
                return result
            finally:
                self._set_running(job.slot_id, False)

        self._task_manager.submit(job.slot_id, job.title, runner, notify_user=True)


class HackerNewsJob:
    def __init__(self, config: HackerNewsJobConfig) -> None:
        self._config = config
        self._last_ids: list[int] = []

    def run(self) -> TaskResult:
        top_ids = self._fetch_top_ids()
        if not top_ids:
            return TaskResult(status="error", summary="HN fetch failed", notify_user=False)
        top_ids = top_ids[: self._config.top_n]

        items = [self._fetch_item(story_id) for story_id in top_ids]
        items = [item for item in items if item]
        if not items:
            return TaskResult(status="error", summary="HN items unavailable", notify_user=False)

        new_ids = [story_id for story_id in top_ids if story_id not in self._last_ids]
        new_items = [item for item in items if item["id"] in new_ids]
        eligible_items = [item for item in new_items if item.get("score", 0) >= self._config.min_score]
        top_item = items[0]

        if not self._last_ids:
            summary = f"Top HN: {top_item['title']} ({top_item.get('score', 0)} points)"
            notify_user = False
            importance = 0.3
        elif eligible_items:
            titles = ", ".join(item["title"] for item in eligible_items[:3])
            summary = f"HN update: new in top {self._config.top_n}: {titles}"
            notify_user = True
            importance = min(1.0, 0.4 + 0.1 * len(eligible_items))
        else:
            summary = f"HN steady: {top_item['title']} stays on top"
            notify_user = False
            importance = 0.2

        self._last_ids = top_ids
        return TaskResult(
            status="done",
            summary=summary,
            notify_user=notify_user,
            importance=importance,
            confidence=0.7,
        )

    def _fetch_top_ids(self) -> list[int]:
        try:
            response = httpx.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=8.0,
            )
            response.raise_for_status()
            return list(response.json())
        except Exception as exc:
            logger.warning("HackerNewsJob: failed to fetch top stories: %s", exc)
            return []

    def _fetch_item(self, story_id: int) -> dict[str, object] | None:
        try:
            response = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=8.0,
            )
            response.raise_for_status()
            data = response.json()
            if not data or "title" not in data:
                return None
            return {
                "id": data.get("id"),
                "title": data.get("title", "Unknown"),
                "score": data.get("score", 0),
            }
        except Exception as exc:
            logger.warning("HackerNewsJob: failed to fetch item %s: %s", story_id, exc)
            return None


class WeatherJob:
    def __init__(self, config: WeatherJobConfig) -> None:
        self._config = config
        self._last_temp: float | None = None
        self._last_code: int | None = None

    def run(self) -> TaskResult:
        if self._config.latitude is None or self._config.longitude is None:
            return TaskResult(
                status="error",
                summary="Weather disabled: location not set",
                notify_user=False,
            )

        data = self._fetch_weather()
        if not data:
            return TaskResult(status="error", summary="Weather fetch failed", notify_user=False)

        current = data.get("current", {})
        temp = float(current.get("temperature_2m", 0.0))
        wind = float(current.get("wind_speed_10m", 0.0))
        code = int(current.get("weather_code", -1))
        condition = WEATHER_CODES.get(code, f"code {code}")
        summary = f"Weather: {condition}, {temp:.1f} C, wind {wind:.0f} km/h"

        notify_user = False
        importance = 0.2
        if code in SEVERE_WEATHER_CODES:
            notify_user = True
            importance = max(importance, 0.8)
        if wind >= self._config.wind_alert_kmh:
            notify_user = True
            importance = max(importance, 0.7)
        if self._last_temp is not None and abs(temp - self._last_temp) >= self._config.temp_change_c:
            notify_user = True
            importance = max(importance, 0.6)

        self._last_temp = temp
        self._last_code = code
        return TaskResult(
            status="done",
            summary=summary,
            notify_user=notify_user,
            importance=importance,
            confidence=0.7,
        )

    def _fetch_weather(self) -> dict[str, object] | None:
        params = {
            "latitude": self._config.latitude,
            "longitude": self._config.longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "timezone": self._config.timezone,
        }
        try:
            response = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("WeatherJob: failed to fetch weather: %s", exc)
            return None


def build_jobs(
    config: AutonomyJobsConfig,
    observability_bus: ObservabilityBus | None = None,
) -> list[JobDefinition]:
    jobs: list[JobDefinition] = []

    if config.hacker_news.enabled:
        job = HackerNewsJob(config.hacker_news)
        jobs.append(
            JobDefinition(
                slot_id="hn_top",
                title="Hacker News",
                interval_s=config.hacker_news.interval_s,
                runner=job.run,
                run_on_start=True,
            )
        )

    if config.weather.enabled:
        if config.weather.latitude is None or config.weather.longitude is None:
            message = "Weather job enabled but latitude/longitude are missing."
            logger.warning(message)
            if observability_bus:
                observability_bus.emit(
                    source="jobs",
                    kind="error",
                    message=trim_message(message),
                    level="warning",
                )
        else:
            job = WeatherJob(config.weather)
            jobs.append(
                JobDefinition(
                    slot_id="weather",
                    title="Weather",
                    interval_s=config.weather.interval_s,
                    runner=job.run,
                    run_on_start=True,
                )
            )

    return jobs
