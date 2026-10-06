"""Local clock replies stay fresh and do not execute a clock tool."""

from collections.abc import Iterator
import json
from pathlib import Path
import queue
import threading
from unittest.mock import MagicMock, Mock

import pytest

from glados.core.clock import clock_context, resolve_relative_dates
from glados.core.decision_lists import DecisionListStore
from glados.core.inference import InferenceScheduler
from glados.tools import tool_definitions
from glados.tools.get_time import GetTime, tool_definition
from tests.test_speech_markup import make_processor


def test_clock_display_includes_date_weekday_and_timezone() -> None:
    text = clock_context(
        {
            "time": "08:37:20",
            "date": "2026-10-06",
            "weekday": "Tuesday",
            "timezone": "CEST",
            "utc_offset": "+0200",
            "datetime": "2026-10-06T08:37:20+02:00",
        }
    )
    assert "authoritative local reading" in text
    assert "Local date: 2026-10-06 · Tuesday" in text
    assert "Tomorrow: 2026-10-07 · Wednesday" in text
    assert "Time zone: CEST (UTC+02:00)" in text
    assert "Read from the host at: 2026-10-06T08:37:20+02:00" in text


@pytest.mark.parametrize("today,expected", [("2026-12-31", "2027-01-01"), ("2028-02-28", "2028-02-29")])
def test_relative_dates_cross_calendar_boundaries(today: str, expected: str) -> None:
    text, dates = resolve_relative_dates("Munich weather tomorrow", {"date": today})
    assert dates == [expected] and expected in text and "tomorrow" not in text


def test_saved_local_clock_tools_migrate_but_foreign_timezone_stays(tmp_path: Path) -> None:
    path = tmp_path / "choices.json"
    original = DecisionListStore(path, lambda: tool_definitions).snapshot()
    row = original["lists"][0]
    row["options"].insert(
        2,
        {
            "id": "time",
            "description": "Local clock",
            "enabled": True,
            "category": None,
            "action": "tool",
            "tool": "get_time",
            "arguments": {},
            "context_source": None,
        },
    )
    row["options"].extend(
        [
            {
                **row["options"][2],
                "id": "custom_clock",
                "description": "My clock command",
                "tool": "run_safe_command",
                "arguments": {"task": "time"},
            },
            {**row["options"][2], "id": "utc", "arguments": {"timezone": "UTC"}},
        ]
    )
    path.write_text(json.dumps(original))
    store = DecisionListStore(path, lambda: tool_definitions)
    migrated = store.snapshot()
    options = {o["id"]: o for o in migrated["lists"][0]["options"]}
    assert "time" not in options and "custom_clock" not in options
    assert options["utc"]["action"] == "tool" and options["utc"]["arguments"] == {"timezone": "UTC"}
    assert migrated["lists"][0]["revision"] == row["revision"] + 1
    assert not store.authorize({"list_id": row["id"], "revision": row["revision"], "option_id": "time"})
    assert json.loads(path.read_text()) == migrated


def test_timezone_tool_requires_named_zone() -> None:
    assert tool_definition["function"]["parameters"]["required"] == ["timezone"]
    output = queue.Queue()
    GetTime(output).run("local", {})
    assert "requires a named timezone" in output.get_nowait()["content"]


def test_small_saved_list_stays_valid_after_clock_removal(tmp_path: Path) -> None:
    path = tmp_path / "choices.json"
    original = DecisionListStore(path, lambda: tool_definitions).snapshot()
    row = original["lists"][0]
    row["options"] = [
        {"id": "legacy_clock", "description": "Local clock", "action": "reply", "context_source": "clock"},
        {"id": "fallback", "description": "Ask for missing information", "action": "clarify"},
    ]
    path.write_text(json.dumps(original))
    store = DecisionListStore(path, lambda: tool_definitions)
    migrated = store.get()
    assert migrated.revision == row["revision"] + 1
    assert len([o for o in migrated.options if o.enabled]) == 2
    assert any(o.action == "reply" for o in migrated.options)
    assert all(o.context_source is None for o in migrated.options)


def test_saved_foreign_clock_category_migrates_and_revokes_old_permit(tmp_path: Path) -> None:
    path = tmp_path / "choices.json"
    original = DecisionListStore(path, lambda: tool_definitions).snapshot()
    row = original["lists"][0]
    row["options"].append(
        {
            "id": "tokyo",
            "description": "Tokyo time",
            "action": "tool",
            "tool": "get_time",
            "category": "clock",
            "arguments": {"timezone": "Asia/Tokyo"},
        }
    )
    path.write_text(json.dumps(original))
    store = DecisionListStore(path, lambda: tool_definitions)
    migrated = store.get()
    assert migrated.revision == row["revision"] + 1
    assert next(o for o in migrated.options if o.id == "tokyo").category == "system"
    assert not store.authorize({"list_id": row["id"], "revision": row["revision"], "option_id": "tokyo"})


def test_clock_only_saved_list_gets_distinct_conversation_choices(tmp_path: Path) -> None:
    path = tmp_path / "choices.json"
    original = DecisionListStore(path, lambda: tool_definitions).snapshot()
    row = original["lists"][0]
    row["options"] = [
        {"id": identity, "description": "Local clock", "action": "reply", "context_source": "clock"}
        for identity in ("time", "date")
    ]
    path.write_text(json.dumps(original))
    migrated = DecisionListStore(path, lambda: tool_definitions).get()
    assert {o.action for o in migrated.options} == {"reply", "clarify"}
    assert len({o.description for o in migrated.options}) == 2


@pytest.mark.parametrize("native_audio", [False, True])
def test_clock_reply_reads_after_admission_and_never_offers_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_audio: bool,
) -> None:
    processor = make_processor()
    processor._inference_scheduler = InferenceScheduler()
    store = DecisionListStore(tmp_path / "choices.json", lambda: tool_definitions)
    reading = ["10:01:00"]
    monkeypatch.setattr("glados.core.llm_processor.current_time", lambda: {"time": reading[0]})
    processor._reply_tools = lambda: [tool_definition]
    generated = threading.Event()
    payloads = []

    def post(*args: object, **kwargs: object) -> MagicMock:
        payloads.append(kwargs["json"])
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        fresh = "10:02:00" in json.dumps(payloads[-1])

        def lines(chunk_size: int = 1) -> Iterator[bytes]:
            generated.set()
            text = "Fresh clock answer." if fresh else "Stale draft."
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode()
            yield b"data: [DONE]"

        response.iter_lines.side_effect = lines
        return response

    def score(*args: object, **kwargs: object) -> dict:
        if kwargs["on_admitted"]:
            kwargs["on_admitted"]()
            assert generated.wait(2)
        reading[0] = "10:02:00"
        return {"action": "reply", "context_source": "clock"}

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.router = Mock(store=store, score=score)
    message = {"role": "user", "content": "What time is it?"}
    if native_audio:
        message["_native_audio"] = [{"type": "input_audio", "input_audio": {"data": "recording", "format": "wav"}}]
    processor.llm_input_queue.put(message)
    worker = threading.Thread(target=processor.run)
    worker.start()
    try:
        spoken = processor.tts_input_queue.get(timeout=3)
        assert spoken.text.strip() == "Fresh clock answer."
        assert "tools" not in payloads[-1]
        assert processor.tool_calls_queue.empty()
        assert len(payloads) == (1 if native_audio else 2)
        assert not any("tool_calls" in call.args[0] for call in processor._conversation_store.append.call_args_list)
    finally:
        processor.shutdown_event.set()
        worker.join(2)
    assert not worker.is_alive()
