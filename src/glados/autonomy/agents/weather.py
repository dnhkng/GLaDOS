"""
Weather subagent for GLaDOS autonomy system.

Periodically fetches weather data and alerts on significant changes.
"""

from __future__ import annotations

import httpx
from loguru import logger

from ..subagent import Subagent, SubagentConfig, SubagentOutput

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


class WeatherSubagent(Subagent):
    """
    Subagent that monitors weather conditions.

    Fetches current weather at configured intervals and notifies on
    significant temperature changes, severe weather, or high winds.
    """

    def __init__(
        self,
        config: SubagentConfig,
        latitude: float | None = None,
        longitude: float | None = None,
        timezone: str = "auto",
        temp_change_c: float = 4.0,
        wind_alert_kmh: float = 40.0,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._latitude = latitude
        self._longitude = longitude
        self._timezone = timezone
        self._temp_change_c = temp_change_c
        self._wind_alert_kmh = wind_alert_kmh
        self._last_temp: float | None = None
        self._last_code: int | None = None

    def tick(self) -> SubagentOutput | None:
        """Fetch and analyze current weather."""
        if self._latitude is None or self._longitude is None:
            return SubagentOutput(
                status="error",
                summary="Weather disabled: location not set",
                notify_user=False,
            )

        data = self._fetch_weather()
        if not data:
            return SubagentOutput(
                status="error",
                summary="Weather fetch failed",
                notify_user=False,
            )

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

        if wind >= self._wind_alert_kmh:
            notify_user = True
            importance = max(importance, 0.7)

        if self._last_temp is not None and abs(temp - self._last_temp) >= self._temp_change_c:
            notify_user = True
            importance = max(importance, 0.6)

        self._last_temp = temp
        self._last_code = code

        return SubagentOutput(
            status="done",
            summary=summary,
            notify_user=notify_user,
            importance=importance,
            confidence=0.7,
            next_run=self._config.loop_interval_s,
        )

    def _fetch_weather(self) -> dict[str, object] | None:
        """Fetch current weather from Open-Meteo API."""
        params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "timezone": self._timezone,
        }
        try:
            response = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
                timeout=8.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("WeatherSubagent: failed to fetch weather: %s", exc)
            return None
