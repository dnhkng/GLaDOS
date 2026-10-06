"""Independent slot producers wake a reviewer; Central Core owns the delivered reply."""

from collections.abc import Iterator
import json
import queue
import threading
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from glados.autonomy.context import chat_evidence
from glados.autonomy.task_manager import TaskManager
from glados.core.conversation_store import ConversationStore
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.speech_player import SpeechPlayer
from glados.core.tool_executor import ToolExecutor
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from glados.vision.presence import PresenceHistory
from tests.test_autonomy_core import make_loop, wait_until


def decision(slot: str = "health") -> dict:
    return {
        "action": "prompt",
        "instruction": "Let the user know there is an issue with the GPU; cite its temperature",
        "slot_ids": [slot],
        "reason": "A fresh important alert",
    }


def test_health_refresh_retains_pending_alert_and_recovers_without_a_second_bus_event() -> None:
    busy = threading.Event()
    busy.set()
    loop = make_loop(busy.is_set)
    slots = loop._slot_store
    slots.update_slot(
        "health",
        "Health Core",
        "active",
        "GPU 97C",
        notify_user=True,
        importance=0.9,
        attention_key="gpu_temperature@100",
    )
    loop._scan_slots()
    slots.update_slot(
        "health",
        "Health Core",
        "active",
        "GPU 98C",
        notify_user=False,
        importance=0.9,
        attention_key="gpu_temperature@100",
    )
    loop._scan_slots()
    assert loop.snapshot()["pending_updates"] == 1
    assert loop._pending_updates["health"].summary == "GPU 98C"
    busy.clear()
    thread = threading.Thread(target=loop.run)
    thread.start()
    try:
        request = loop._llm_queue.get(timeout=2)
        assert "GPU 98C" in request["content"]
        loop.finish_cycle(request["_autonomy_cycle"], "silent")
        loop._scan_slots()
        assert loop.snapshot()["pending_updates"] == 0
        slots.update_slot("health", "Health Core", "active", "GPU normal", notify_user=False, attention_key=None)
        loop._scan_slots()
        slots.update_slot(
            "health",
            "Health Core",
            "active",
            "GPU 99C",
            notify_user=False,
            importance=0.9,
            attention_key="gpu_temperature@200",
        )
        loop._scan_slots()
        assert "GPU 99C" in loop._pending_updates["health"].summary
    finally:
        loop._shutdown_event.set()
        thread.join(2)


def test_source_recovery_rejects_old_handoff_and_a_queued_old_central_request() -> None:
    loop = make_loop()
    slots, main = loop._slot_store, queue.Queue()

    def alert(key: str | None) -> None:
        slots.update_slot(
            "health", "Health Core", "active", "GPU hot" if key else "GPU healthy", notify_user=False, attention_key=key
        )

    alert("gpu@100")
    assert loop._dispatch("Review the GPU alert")
    cycle = loop._llm_queue.get_nowait()["_autonomy_cycle"]
    alert(None)
    assert not loop.prompt_main(cycle, decision(), main)
    alert("gpu@100")
    assert loop.prompt_main(cycle, decision(), main)
    assert loop.request_current(cycle)
    alert(None)
    assert not loop.request_current(cycle)
    assert main.qsize() == 1


@pytest.mark.parametrize("kind", ["person_arrived@100", "morning_greeting@2026-10-07"])
@pytest.mark.parametrize("same_event", [True, False])
def test_caption_refresh_during_delivery_remembers_the_greeted_event(kind: str, same_event: bool) -> None:
    loop = make_loop()
    slots, main = loop._slot_store, queue.Queue()
    slots.update_slot("vision", "Vision Core", "active", "You returned", attention_key=kind,
                      report="Recorded greeting event", notify_user=False)
    assert loop._dispatch("Review this greeting")
    cycle = loop._llm_queue.get_nowait()["_autonomy_cycle"]
    assert loop.prompt_main(cycle, {**decision("vision"), "instruction": "Greet the user"}, main)
    slots.update_slot("vision", "Vision Core", "active", "A reworded description while speech plays",
                      attention_key=kind if same_event else "person_arrived@200",
                      report="Fresh capture; same scene", notify_user=False)
    loop.finish_cycle(cycle, "response", "Greeting delivered")
    if same_event:
        assert loop._announced["vision"] == (kind,)
        assert loop._dispatch("Review another capture")
        payload = loop._llm_queue.get_nowait()
        assert "vision" in payload["_autonomy_announced_slots"]
        assert not loop.prompt_main(payload["_autonomy_cycle"], decision("vision"), main)
    else:
        assert "vision" not in loop._announced, "A distinct future event must remain eligible"


def test_summary_and_recent_chat_are_available_without_another_summary_inference() -> None:
    store = ConversationStore([{"role": "system", "content": "UNRELATED PERSONA"}])
    store.append({"role": "user", "content": "Let me know when the web search finishes."})
    records = store.records()
    assert store.compact([records[-1]], "[summary] The user requested background search results.", 0)
    for i in range(20):
        store.append({"role": "assistant", "content": f"Recent reply {i}"})
    evidence = json.loads(chat_evidence(store).split("\n", 1)[1])
    assert "background search" in evidence["summaries"][0]["text"]
    assert len(evidence["recent_turns"]) == 8 and evidence["recent_turns"][0]["text"] == "Recent reply 12"
    assert "UNRELATED PERSONA" not in json.dumps(evidence)


def test_reviewer_to_central_to_delivery_keeps_personality_and_private_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = make_loop()
    shutdown, active = loop._shutdown_event, loop._processing_active_event
    active.set()
    history = ConversationStore(
        [
            {"role": "system", "content": "You are GLaDOS. Be dry and concise."},
            {"role": "user", "content": "Tell me if the GPU is overheating."},
        ]
    )
    original = history.deep_snapshot()
    slots, main_queue, tools, speech, audio = (
        loop._slot_store,
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
    )
    slots.update_slot(
        "health",
        "Health Core",
        "active",
        "GPU temperature is 97C",
        notify_user=False,
        report="Measured GPU 0 temperature: 97C; warning threshold: 90C",
        attention_key="gpu@100",
    )
    requests, outcomes = [], []

    def done(cycle: str, outcome: str, reason: str) -> None:
        if loop.snapshot()["active_cycle"] == cycle:
            outcomes.append(outcome)
            loop.finish_cycle(cycle, outcome, reason)
            shutdown.set()

    def prompt(plan: dict, meta: dict) -> bool:
        assert speech.empty() and tools.empty()
        return loop.prompt_main(meta["_autonomy_cycle"], plan, main_queue)

    args = {
        "tool_calls_queue": tools,
        "tts_input_queue": speech,
        "conversation_store": history,
        "completion_url": "http://test/v1/chat/completions",
        "model_name": "test",
        "api_key": None,
        "processing_active_event": active,
        "shutdown_event": shutdown,
        "slot_store": slots,
        "quiet_generation": lambda: 7,
        "autonomy_generation": lambda: 3,
        "on_autonomy_done": done,
    }
    reviewer = LanguageModelProcessor(
        llm_input_queue=loop._llm_queue, lane="autonomy", on_autonomy_prompt=prompt, **args
    )
    central = LanguageModelProcessor(
        llm_input_queue=main_queue,
        lane="priority",
        **args,
        autonomy_request_current=lambda meta: loop.request_current(meta["_autonomy_cycle"]),
    )
    central._before_reply = Mock(side_effect=AssertionError("Internal alerts are not user emotion events"))
    central._before_context = Mock(side_effect=AssertionError("Internal alerts are not recall requests"))

    def post(*args: object, **kwargs: object) -> MagicMock:
        body = kwargs["json"]
        requests.append(body)
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        content = (
            json.dumps(decision())
            if "response_format" in body
            else ("[emotion:neutral]Your GPU is at 97 degrees Celsius. Please check its cooling.")
        )

        def stream(chunk_size: int = 1) -> Iterator[bytes]:
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": content}}]}).encode()
            yield b"data: [DONE]"

        response.iter_lines.side_effect = stream
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    muted = threading.Event()
    muted.set()
    model = Mock(sample_rate=16000, generate_speech_audio=Mock(return_value=np.ones(100, dtype=np.float32)))
    synth = TextToSpeechSynthesizer(
        speech,
        audio,
        model,
        Mock(text_to_spoken=lambda text: text),
        shutdown,
        0.001,
        tts_muted_event=muted,
        quiet_generation=lambda: 7,
        autonomy_generation=lambda: 3,
    )
    player = SpeechPlayer(
        Mock(),
        audio,
        history,
        16000,
        shutdown,
        threading.Event(),
        active,
        0.001,
        tts_muted_event=muted,
        quiet_generation=lambda: 7,
        autonomy_generation=lambda: 3,
        on_autonomy_done=done,
    )
    threads = [threading.Thread(target=f) for f in (reviewer.run, central.run, synth.run, player.run)]
    for thread in threads:
        thread.start()
    try:
        loop._scan_slots()
        assert loop._dispatch("Review this fresh GPU alert")
        wait_until(shutdown.is_set)
    finally:
        shutdown.set()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
    assert outcomes == ["response"] and len(requests) == 2
    assert "tools" not in requests[0] and "tools" not in requests[1]
    assert "GLaDOS" not in requests[0]["messages"][0]["content"]  # Reviewer has its own instructions.
    assert "You are GLaDOS" in requests[1]["messages"][0]["content"]
    assert history.snapshot()[: len(original)] == original
    delivered = history.snapshot()[len(original):]
    assert len(delivered) == 1 and delivered[0]["role"] == "assistant"
    assert " ".join(delivered[0]["content"].split()) == "Your GPU is at 97 degrees Celsius. Please check its cooling."
    assert "Internal request" not in json.dumps(history.snapshot())
    assert "health" in loop._announced
    model.generate_speech_audio.assert_not_called()
    # Even after the conversation window changes, the same condition cannot speak twice.
    shutdown.clear()
    for i in range(20):
        history.append({"role": "assistant", "content": f"Unrelated conversation {i}"})
    loop._config.cooldown_s = 0
    assert loop._dispatch(loop._build_prompt(None))
    request = loop._llm_queue.get_nowait()
    assert request["_autonomy_announced_slots"] == ["health"]
    assert not loop.prompt_main(request["_autonomy_cycle"], decision(), main_queue)


def test_search_saves_result_without_blocking_tool_executor() -> None:
    loop = make_loop()
    slots, bus = loop._slot_store, loop._event_bus
    tasks = TaskManager(slots, bus)
    calls, replies, active, shutdown, release, entered = (
        queue.Queue(),
        queue.Queue(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    active.set()

    def search(*args: object, **kwargs: object) -> str:
        entered.set()
        assert release.wait(2)
        return "Verified search result; source https://example.test/news"

    manager = Mock(call_tool=Mock(side_effect=search))
    executor = ToolExecutor(
        replies, queue.Queue(), calls, active, shutdown, mcp_manager=manager, tool_config={"task_manager": tasks}
    )
    calls.put(
        {
            "id": "search",
            "function": {"name": "mcp.internet_search.web_search_exa", "arguments": {"query": "latest GPU news"}},
        }
    )
    thread = threading.Thread(target=executor.run)
    thread.start()
    try:
        ack = replies.get(timeout=2)
        assert entered.wait(1)
        data = json.loads(ack["content"])
        slot = slots.get_slot(data["task_id"])
        assert slot.status == "running" and not slot.notify_user
        release.set()
        wait_until(lambda: slots.get_slot(data["task_id"]).status == "done")
        slot = slots.get_slot(data["task_id"])
        assert slot.notify_user and "example.test/news" in slot.report
        loop._scan_slots()
        assert data["task_id"] in loop._pending_updates
        assert replies.empty(), "Completion belongs in a slot, not a fake user/tool follow-up"
    finally:
        release.set()
        shutdown.set()
        thread.join(2)
        tasks.shutdown(wait=True)


def test_arrival_requires_observed_absence_and_does_not_repeat_or_use_camera_downtime() -> None:
    history = PresenceHistory(absence_s=60)
    assert history.observe("present", 0) is None
    for stamp in (10, 30, 50, 70):
        assert history.observe("absent", stamp) is None
    arrival = history.observe("present", 75)
    assert arrival["observed_absence_s"] == 65
    assert history.observe("present", 80) == arrival
    assert history.observe("uncertain", 81) is None
    assert history.observe("present", 82) is None
    history.observe("absent", 90)
    assert history.observe("present", 200) is None  # Gap in observations resets baseline.
    history.observe("absent", 210)
    history.reset()
    assert history.observe("present", 220) is None


def test_delayed_arrival_expires_while_person_remains_visible() -> None:
    history = PresenceHistory(absence_s=60)
    for stamp in (0, 20, 40, 60):
        history.observe("absent", stamp)
    assert history.observe("present", 65)
    for stamp in (85, 105, 125):
        assert history.observe("present", stamp)
    assert history.observe("present", 130) is None


def test_search_keeps_the_synchronous_result_path_when_autonomy_is_off() -> None:
    loop = make_loop()
    tasks = TaskManager(loop._slot_store, loop._event_bus)
    calls, replies = queue.Queue(), queue.Queue()
    active, shutdown = threading.Event(), threading.Event()
    active.set()
    executor = ToolExecutor(
        replies,
        queue.Queue(),
        calls,
        active,
        shutdown,
        mcp_manager=Mock(call_tool=Mock(return_value="Actual search result")),
        tool_config={"task_manager": tasks},
        autonomy_enabled=lambda: False,
    )
    calls.put(
        {"id": "search", "function": {"name": "mcp.internet_search.web_search_exa", "arguments": {"query": "news"}}}
    )
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        assert replies.get(timeout=2)["content"] == "Actual search result"
        jobs = [s for s in loop._slot_store.list_slots() if s.owner_id]
        assert len(jobs) == 1 and jobs[0].handled
        assert jobs[0].report == "Actual search result"
    finally:
        shutdown.set()
        worker.join(2)
        tasks.shutdown(wait=True)
