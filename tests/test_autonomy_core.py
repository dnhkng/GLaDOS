from copy import deepcopy
"""Autonomy yields to users, preserves notifications and terminates private checks."""

from collections.abc import Callable, Iterator
import json
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.decision import parse_decision
from glados.autonomy.event_bus import EventBus
from glados.autonomy.events import TaskUpdateEvent
from glados.autonomy.interaction_state import InteractionState
from glados.autonomy.loop import AutonomyLoop
from glados.autonomy.slots import TaskSlotStore
from glados.core.audio_data import AudioMessage
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados
from glados.core.inference import InferenceScheduler
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.speech_markup import SpeechText
from glados.core.speech_player import SpeechPlayer
from glados.core.tts_synthesizer import TextToSpeechSynthesizer


def wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.005)


def make_loop(busy: Callable[[], bool] = lambda: False, cooldown: float = 0) -> AutonomyLoop:
    return AutonomyLoop(AutonomyConfig(enabled=True, cooldown_s=cooldown), EventBus(), InteractionState(),
                        None, TaskSlotStore(), queue.Queue(), threading.Event(), threading.Event(),
                        threading.Event(), pause_time=.005, user_busy=busy, quiet_generation=lambda: 7,
                        autonomy_generation=lambda: 3)


def update(summary: str, timestamp: float = 10) -> TaskUpdateEvent:
    return TaskUpdateEvent("task_test", "Requested task", "done", summary, True, timestamp)


def test_task_updates_survive_user_activity_and_only_newest_is_considered() -> None:
    busy = threading.Event()
    busy.set()
    loop = make_loop(busy.is_set)
    thread = threading.Thread(target=loop.run)
    thread.start()
    try:
        loop._event_bus.publish(update("Old result"))
        loop._event_bus.publish(update("New result", 11))
        wait_until(lambda: loop.snapshot()["pending_updates"] == 1)
        assert loop._llm_queue.empty() and not loop._processing_active_event.is_set()
        busy.clear()
        payload = loop._llm_queue.get(timeout=2)
        assert "New result" in payload["content"] and "Old result" not in payload["content"]
        assert payload["_quiet_generation"] == 7 and payload["_autonomy_generation"] == 3
        loop.finish_cycle(payload["_autonomy_cycle"], "silent", "Already discussed")
        loop._event_bus.publish(update("New result", 12))  # Timestamp-only refresh is not new information.
        wait_until(lambda: loop._event_bus._queue.empty())
        assert loop._llm_queue.empty() and loop.snapshot()["pending_updates"] == 0
    finally:
        loop._shutdown_event.set()
        thread.join(2)
    assert not thread.is_alive()


def test_cooldown_defers_notification_until_idle_without_another_event() -> None:
    loop = make_loop(cooldown=.1)
    loop._last_prompt_ts = time.monotonic()
    loop._event_bus.publish(update("Completed requested work"))
    thread = threading.Thread(target=loop.run)
    thread.start()
    try:
        payload = loop._llm_queue.get(timeout=2)
        assert "Completed requested work" in payload["content"]
    finally:
        loop._shutdown_event.set()
        thread.join(2)


def test_silence_backoff_and_cancelled_notifications_are_recoverable() -> None:
    loop = make_loop(cooldown=20)
    loop._remember(update("Task done"))
    assert loop._dispatch("Check a task")
    first = loop._llm_queue.get_nowait()
    loop._active_updates = dict(loop._pending_updates)
    loop._pending_updates.clear()
    loop.reset()
    assert loop.snapshot()["pending_updates"] == 1
    loop._last_prompt_ts = 0
    assert loop._dispatch("Another check")
    second = loop._llm_queue.get_nowait()
    loop.finish_cycle(first["_autonomy_cycle"], "speak", "A stale completion")
    assert loop.snapshot()["active_cycle"] == second["_autonomy_cycle"]
    loop.finish_cycle(second["_autonomy_cycle"], "silent", "No fresh information")
    state = loop.snapshot()
    assert state["silent_checks"] == 1 and 39 <= state["next_check_s"] <= 40
    assert state["last_decision"]["reason"] == "No fresh information"


def test_reviewer_sees_all_cores_but_very_recent_user_input_blocks_dispatch() -> None:
    loop = make_loop()
    for slot in ("vision", "emotion", "compaction"):
        loop._slot_store.update_slot(slot, slot, "idle", "Core heartbeat", notify_user=False)
    assert {s["slot_id"] for s in json.loads(loop._task_summary())} == {"vision", "emotion", "compaction"}
    loop._interaction_state.mark_user()
    assert not loop._dispatch("Idle check")
    assert loop._llm_queue.empty()


def run_check(monkeypatch: pytest.MonkeyPatch, choose: Callable[[dict, int], tuple[str, dict] | str],
              interrupt_stream: bool = False) -> tuple:
    history = ConversationStore([{"role": "user", "content": "Check on my task later."}])
    original = deepcopy(history.snapshot())
    shutdown, active = threading.Event(), threading.Event()
    active.set()
    decisions, requests = [], []
    epoch = [3]
    def done(*args: str) -> None:
        decisions.append(args)
        shutdown.set()
    def prompt(decision: dict, meta: dict) -> bool:
        done(meta["_autonomy_cycle"], "prompt", decision["reason"])
        return True
    slots = TaskSlotStore()
    slots.update_slot("task_test", "Requested task", "done", "Completed", notify_user=False)
    processor = LanguageModelProcessor(
        queue.Queue(), queue.Queue(), queue.Queue(), history, "http://localhost/v1/chat/completions", "test",
        None, active, shutdown, lane="autonomy", inference_scheduler=InferenceScheduler(), on_autonomy_done=done,
        autonomy_generation=lambda: epoch[0],
        on_autonomy_prompt=prompt, slot_store=slots,
    )
    processor.llm_input_queue.put({"role": "user", "content": "Periodic task update.", "autonomy": True,
                                   "_autonomy_cycle": "check", "_autonomy_generation": 3})
    def post(*args: object, **kwargs: object) -> MagicMock:
        data = kwargs["json"]
        requests.append(data)
        choice = choose(data, len(requests))
        assert "tools" not in data and data["max_tokens"] == 256
        if isinstance(choice, str):
            content = choice
        else:
            name, arguments = choice
            if name == "do_nothing":
                decision = None
            elif name == "speak":
                decision = {"action": "prompt", "instruction": "Report the completed requested task",
                            "slot_ids": ["task_test"], "reason": "Task complete"}
            else:
                decision = {"action": "tool", "tool": name, "arguments": arguments, "reason": "Check requested task"}
            content = json.dumps(decision)
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        def lines(chunk_size: int = 1) -> Iterator[bytes]:
            midpoint = len(content) // 2
            yield b'data: ' + json.dumps({"choices": [{"delta": {"content": content[:midpoint]}}]}).encode()
            if interrupt_stream:
                epoch[0] += 1
            yield b'data: ' + json.dumps({"choices": [{"delta": {"content": content[midpoint:]}}]}).encode()
            yield b'data: [DONE]'
        response.iter_lines.side_effect = lines
        return response
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    threads = [threading.Thread(target=processor.run)]
    for thread in threads:
        thread.start()
    try:
        wait_until(lambda: bool(decisions))
    finally:
        shutdown.set()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
    assert history.snapshot() == original
    assert processor._inference_scheduler.snapshot()["active"] == []
    return processor, decisions, requests


@pytest.mark.parametrize("tool", ["speak", "do_nothing"])
def test_reviewer_prompts_or_stays_silent_without_tools_speech_or_history_changes(
    monkeypatch: pytest.MonkeyPatch, tool: str,
) -> None:
    processor, decisions, requests = run_check(monkeypatch, lambda data, count: (tool, {
        "text": "[emotion:smug]Your task is finished.", "reason": "Nothing useful to add",
    }))
    assert len(requests) == 1 and processor.llm_input_queue.empty()
    assert processor.tts_input_queue.empty() and processor.tool_calls_queue.empty()
    assert decisions[0][1] == ("prompt" if tool == "speak" else "silent")


def test_reviewer_cannot_use_tools_or_write_a_spoken_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    processor, decisions, requests = run_check(monkeypatch, lambda data, count: (
        "get_report", {"agent_id": "task_test"}))
    assert len(requests) == 1 and decisions[0][1] == "error"
    assert processor.llm_input_queue.empty() and processor.tool_calls_queue.empty()


def test_disabling_and_reenabling_autonomy_discards_old_speech_and_preserves_user_reply() -> None:
    pending, output = queue.Queue(), queue.Queue()
    pending.put(SpeechText("Stale autonomous reply", generation=0, autonomy_generation=3))
    pending.put(SpeechText("Current user reply", generation=0))
    pending.put(SpeechText("<EOS>", generation=0))
    shutdown = threading.Event()
    put = output.put
    def emit(item: object) -> None:
        put(item)
        if item.is_eos:
            shutdown.set()
    output.put = emit
    model = Mock(sample_rate=16000, generate_speech_audio=Mock(return_value=np.ones(160, dtype=np.float32)))
    TextToSpeechSynthesizer(pending, output, model, SimpleNamespace(text_to_spoken=lambda t: t), shutdown, .001,
                           autonomy_generation=lambda: 5).run()
    model.generate_speech_audio.assert_called_once_with("Current user reply")
    assert [m.text for m in list(output.queue)] == ["Current user reply", ""]


def test_autonomy_toggle_invalidates_its_own_work_without_cancelling_user_generation() -> None:
    engine = SimpleNamespace(
        _quiet_lock=threading.RLock(), _quiet_generation=7, _autonomy_generation=3,
        autonomy_config=AutonomyConfig(enabled=True), autonomy_loop=Mock(), llm_queue_autonomy=queue.Queue(),
        speech_player=SimpleNamespace(autonomy_speaking=False), audio_io=Mock(), observability_bus=Mock(),
    )
    engine.llm_queue_autonomy.put({"content": "Old check"})
    Glados.set_autonomy_enabled(engine, False)
    Glados.set_autonomy_enabled(engine, True)
    assert engine._autonomy_generation == 5 and engine._quiet_generation == 7
    assert engine.llm_queue_autonomy.empty()
    engine.audio_io.stop_speaking.assert_not_called()


@pytest.mark.parametrize("values", [{"tick_interval_s": 0}, {"tick_interval_s": -1}, {"cooldown_s": -1}])
def test_invalid_timers_are_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        AutonomyConfig(**values)


@pytest.mark.parametrize("text", [
    'do_nothing(reason="Idle")', '{"action":"speak","text":"Partial',
    '{"action":"tool","tool":"unavailable","arguments":{},"reason":"Task"}',
    '{"action":"speak","text":" ","reason":"Empty"}',
    '{"action":"silent","reason":"Idle","extra":"unexpected"}',
    '{"action":"tool","tool":"get_report","arguments":{"slot_id":7},"reason":"Task"}',
    '{"action":"prompt","instruction":"Warn","slot_ids":["missing"],"reason":"Alert"}',
    '{"action":"prompt","instruction":" ","slot_ids":["health"],"reason":"Alert"}',
])
def test_incomplete_or_invalid_decisions_never_authorize_actions(text: str) -> None:
    with pytest.raises(ValueError):
        parse_decision(text, {"health"})


@pytest.mark.parametrize("text", ['{"action":"null"}', 'null'])
def test_valid_no_action_never_requests_a_handoff(text: str) -> None:
    assert parse_decision(text, {"health"}) is None


def test_failed_decision_retains_notification_and_backs_off(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("glados.autonomy.loop.time.monotonic", lambda: now[0])
    loop = make_loop()
    loop._remember(update("Task ready"))
    assert loop._dispatch("Check a task")
    payload = loop._llm_queue.get_nowait()
    loop._active_updates = dict(loop._pending_updates)
    loop._pending_updates.clear()
    loop.finish_cycle(payload["_autonomy_cycle"], "error", "Invalid decision")
    assert loop.snapshot()["pending_updates"] == 1 and loop._should_skip()
    now[0] += 11
    assert not loop._should_skip()
    assert loop.snapshot()["last_decision"]["outcome"] == "error"


def test_malformed_stream_is_an_error_rather_than_a_silent_success(monkeypatch: pytest.MonkeyPatch) -> None:
    processor, decisions, requests = run_check(monkeypatch, lambda data, count: 'speak("Unvalidated prose")')
    assert decisions[0][1] == "error" and len(requests) == 1
    assert processor.tts_input_queue.empty() and processor.tool_calls_queue.empty()


def test_switching_autonomy_off_during_a_decision_prevents_tool_and_speech_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, decisions, _ = run_check(monkeypatch, lambda data, count: ("speak", {"text": "Stale speech"}),
                                      interrupt_stream=True)
    assert decisions[0][1] == "cancelled"
    assert processor.tts_input_queue.empty() and processor.tool_calls_queue.empty()


def test_user_routing_is_busy_before_main_inference_is_admitted() -> None:
    request = threading.Event()
    request.set()
    engine = SimpleNamespace(speech_listener=None, llm_processor=SimpleNamespace(_request_active=request),
                             _priority_inflight=SimpleNamespace(value=lambda: 0),
                             llm_queue_priority=queue.Queue(), tts_queue=queue.Queue(), audio_queue=queue.Queue())
    assert Glados._autonomy_user_busy(engine)
    request.clear()
    assert not Glados._autonomy_user_busy(engine)


def test_cancelled_playback_drops_autonomy_tail_and_keeps_user_reply_history() -> None:
    audio, history = Mock(), ConversationStore([])
    epoch, shutdown = [3], threading.Event()
    pending = queue.Queue()
    for text in ("Autonomous speech", "Stale continuation"):
        pending.put(AudioMessage(np.ones(100, dtype=np.float32), text, autonomy_generation=3))
    pending.put(AudioMessage(np.array([], dtype=np.float32), "", is_eos=True, autonomy_generation=3))
    pending.put(AudioMessage(np.ones(100, dtype=np.float32), "Current user reply"))
    pending.put(AudioMessage(np.array([], dtype=np.float32), "", is_eos=True))
    def measure(*args: object) -> tuple[bool, float]:
        epoch[0] = 5  # OFF/ON occurs during the old autonomous utterance.
        return False, 100
    audio.measure_percentage_spoken.side_effect = measure
    append = history.append
    def record(message: dict) -> None:
        append(message)
        shutdown.set()
    history.append = record
    SpeechPlayer(audio, pending, history, 16000, shutdown, threading.Event(), threading.Event(), .001,
                 autonomy_generation=lambda: epoch[0]).run()
    assert audio.start_speaking.call_count == 2
    assert history.snapshot() == [{"role": "assistant", "content": "Current user reply"}]
