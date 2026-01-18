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
        humidity = float(current.get("relative_humidity_2m", 0.0))
        code = int(current.get("weather_code", -1))
        condition = WEATHER_CODES.get(code, f"code {code}")
        summary = f"{condition}, {temp:.1f}C, wind {wind:.0f}km/h"

        notify_user = False
        importance = 0.2
        alerts = []

        if code in SEVERE_WEATHER_CODES:
            notify_user = True
            importance = max(importance, 0.8)
            alerts.append(f"Severe weather: {condition}")

        if wind >= self._wind_alert_kmh:
            notify_user = True
            importance = max(importance, 0.7)
            alerts.append(f"High winds: {wind:.0f} km/h")

        if self._last_temp is not None and abs(temp - self._last_temp) >= self._temp_change_c:
            notify_user = True
            importance = max(importance, 0.6)
            change = temp - self._last_temp
            direction = "risen" if change > 0 else "dropped"
            alerts.append(f"Temperature {direction} {abs(change):.1f}C")

        self._last_temp = temp
        self._last_code = code

        # Generate detailed report when there's something notable
        report = None
        if alerts or importance >= 0.5:
            report = self._generate_report(data, current, temp, wind, humidity, condition, alerts)

        return SubagentOutput(
            status="done",
            summary=summary,
            report=report,
            notify_user=notify_user,
            importance=importance,
            confidence=0.7,
            next_run=self._config.loop_interval_s,
        )

    def _generate_report(
        self,
        data: dict,
        current: dict,
        temp: float,
        wind: float,
        humidity: float,
        condition: str,
        alerts: list[str],
    ) -> str:
        """Generate detailed weather report."""
        lines = ["## Weather Report", ""]

        # Alerts section
        if alerts:
            lines.append("### Alerts")
            for alert in alerts:
                lines.append(f"- {alert}")
            lines.append("")

        # Current conditions
        lines.append("### Current Conditions")
        lines.append(f"- **Condition:** {condition}")
        lines.append(f"- **Temperature:** {temp:.1f}C")
        lines.append(f"- **Wind:** {wind:.0f} km/h")
        lines.append(f"- **Humidity:** {humidity:.0f}%")
        lines.append("")

        # Hourly forecast (next 6 hours)
        hourly = data.get("hourly", {})
        hourly_temps = hourly.get("temperature_2m", [])
        hourly_codes = hourly.get("weather_code", [])
        hourly_times = hourly.get("time", [])

        if hourly_temps and hourly_codes and len(hourly_temps) >= 6:
            lines.append("### Next 6 Hours")
            for i in range(6):
                time_str = hourly_times[i].split("T")[1] if i < len(hourly_times) else f"+{i}h"
                h_temp = hourly_temps[i]
                h_code = hourly_codes[i]
                h_cond = WEATHER_CODES.get(h_code, f"code {h_code}")
                lines.append(f"- {time_str}: {h_temp:.1f}C, {h_cond}")
            lines.append("")

        # Daily forecast (if available)
        daily = data.get("daily", {})
        daily_max = daily.get("temperature_2m_max", [])
        daily_min = daily.get("temperature_2m_min", [])
        daily_codes = daily.get("weather_code", [])
        daily_dates = daily.get("time", [])

        if daily_max and daily_min and len(daily_max) >= 3:
            lines.append("### 3-Day Outlook")
            for i in range(min(3, len(daily_max))):
                date_str = daily_dates[i] if i < len(daily_dates) else f"Day {i+1}"
                d_max = daily_max[i]
                d_min = daily_min[i]
                d_code = daily_codes[i] if i < len(daily_codes) else -1
                d_cond = WEATHER_CODES.get(d_code, "")
                lines.append(f"- {date_str}: {d_min:.0f}-{d_max:.0f}C, {d_cond}")

        return "\n".join(lines)

    def _fetch_weather(self) -> dict[str, object] | None:
        """Fetch current weather from Open-Meteo API."""
        params = {
            "latitude": self._latitude,
            "longitude": self._longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "hourly": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": 3,
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
