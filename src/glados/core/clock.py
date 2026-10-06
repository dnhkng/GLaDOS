"""Fresh host-clock context, independent of tool execution."""

from datetime import date, datetime, timedelta
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def current_time(timezone: str | None = None) -> dict[str, str]:
    if timezone is not None and (not isinstance(timezone, str) or len(timezone) > 100):
        raise ValueError("timezone must be an IANA timezone name")
    try:
        now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("Unknown timezone; use an IANA name such as Europe/Berlin") from exc
    return {
        "datetime": now.isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "timezone": timezone or now.tzname() or "local",
        "utc_offset": now.strftime("%z"),
        "source": "system clock",
    }


def clock_context(reading: dict[str, str]) -> str:
    """Make the authoritative local reading explicit near the current input."""
    offset = reading.get("utc_offset", "")
    if len(offset) == 5:
        offset = offset[:3] + ":" + offset[3:]
    lines = ["[Live system clock for this request · authoritative local reading]", "Local time: " + reading["time"]]
    if reading.get("date"):
        lines.append("Local date: " + reading["date"] + " · " + reading.get("weekday", ""))
        today = date.fromisoformat(reading["date"])
        tomorrow = today + timedelta(days=1)
        lines.append("Tomorrow: " + tomorrow.isoformat() + " · " + tomorrow.strftime("%A"))
        lines.append("Resolve relative dates from this clock; do not ask the user to supply today's date.")
    if reading.get("timezone"):
        lines.append("Time zone: " + reading["timezone"] + (" (UTC" + offset + ")" if offset else ""))
    if reading.get("datetime"):
        lines.append("Read from the host at: " + reading["datetime"])
    return "\n".join(lines)


def resolve_relative_dates(text: str, reading: dict[str, str]) -> tuple[str, list[str]]:
    """Pin ordinary relative days to one request's local clock, including across midnight."""
    offsets = {"day after tomorrow": 2, "tomorrow": 1, "next day": 1, "today": 0, "tonight": 0, "yesterday": -1}
    today = date.fromisoformat(reading["date"])
    dates: list[str] = []

    def replace_day(match: re.Match[str]) -> str:
        day = today + timedelta(days=offsets[match.group().lower()])
        if day.isoformat() not in dates:
            dates.append(day.isoformat())
        return day.strftime("%A") + " " + day.isoformat()

    resolved = re.sub(
        r"\b(?:day after tomorrow|tomorrow|next day|today|tonight|yesterday)\b", replace_day, text, flags=re.I
    )
    return resolved, dates
