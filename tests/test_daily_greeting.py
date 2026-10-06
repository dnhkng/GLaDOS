"""Daily greetings use the local day and stay quiet across refreshes and restarts."""

from datetime import datetime
import json
from pathlib import Path

import pytest

from glados.vision.presence import DailyGreetingHistory, PresenceHistory


def stamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


@pytest.mark.parametrize("time,greet", [("09:59:59", True), ("10:00:00", False), ("16:30:00", False)])
def test_first_sighting_obeys_local_10am_boundary(time: str, greet: bool) -> None:
    history = DailyGreetingHistory()
    event = history.observe("present", stamp("2026-10-07T" + time))
    assert bool(event) == greet
    if event:
        assert event["local_date"] == "2026-10-07" and event["local_time"] == time
        assert event["kind"] == "first_seen_today" and event["attention_key"] == "morning_greeting@2026-10-07"


def test_sighting_survives_restart_and_later_day_can_greet(tmp_path: Path) -> None:
    path = tmp_path / "greetings.json"
    now = stamp("2026-10-07T09:00:00")
    history = DailyGreetingHistory(path)
    first = history.observe("present", now)
    assert first and history.observe("present", now + 5) == first
    assert json.loads(path.read_text()) == {"last_seen_date": "2026-10-07", "first_seen_at": now}
    assert path.stat().st_size < 120
    history.reset()
    assert history.observe("present", now + 10) is None
    restarted = DailyGreetingHistory(path)
    assert restarted.observe("present", now + 20) is None
    assert restarted.observe("present", stamp("2026-10-08T09:00:00"))["local_date"] == "2026-10-08"


def test_presence_uncertainty_and_camera_gaps_do_not_repeat_morning() -> None:
    history = DailyGreetingHistory()
    now = stamp("2026-10-07T09:00:00")
    assert history.observe("uncertain", now) is None
    assert history.observe("present", now + 5)
    assert history.observe("uncertain", now + 10) is None
    assert history.observe("present", now + 15) is None
    assert history.observe("absent", now + 20) is None
    assert history.observe("present", now + 200) is None


def test_continuous_presence_across_midnight_does_not_trigger_morning() -> None:
    history = DailyGreetingHistory()
    assert history.observe("present", stamp("2026-10-06T23:59:45")) is None
    assert history.observe("present", stamp("2026-10-07T00:00:10")) is None
    assert history.observe("present", stamp("2026-10-07T09:00:00")) is None


def test_pending_morning_expires_even_with_continuing_observations() -> None:
    history = DailyGreetingHistory()
    now = stamp("2026-10-07T09:00:00")
    assert history.observe("present", now)
    for elapsed in (20, 40, 60):
        assert history.observe("present", now + elapsed)
    assert history.observe("present", now + 65) is None


def test_turning_away_while_present_does_not_create_a_return() -> None:
    history = PresenceHistory()
    for second in range(0, 301, 5):
        assert history.observe("present", second) is None
    assert history.observe("uncertain", 305) is None
    assert history.observe("present", 310) is None
