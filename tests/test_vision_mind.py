"""Freshness, focus, E4B isolation, image comparisons and camera controls."""

import base64
from collections.abc import Iterator
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.event_bus import EventBus
from glados.autonomy.events import TimeTickEvent
from glados.autonomy.interaction_state import InteractionState
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.loop import AutonomyLoop
from glados.autonomy.slots import TaskSlotStore
from glados.core.engine import Glados
from glados.core.inference import InferenceConfig, InferenceScheduler
from glados.tools.vision_look import VisionLook
from glados.vision.frame_picker import (
    CameraFrame,
    CameraSampler,
    MotionTracker,
    RecentFrames,
    focus_score,
    laplacian_variance,
)
from glados.vision.vision_config import VisionConfig
from glados.vision.vision_mind import VisionMind
from glados.vision.vision_state import VisionState
from glados.webapp.serializers import build_minds, build_state
from glados.webapp.server import WebappServer
from tests.test_speech_markup import make_processor
from tests.test_webapp import _FakeEngine


def frame(sequence: int, sharpness: float = 1, captured_at: float | None = None) -> CameraFrame:
    return CameraFrame(
        np.full((16, 16, 3), sequence, dtype=np.uint8),
        time.time() if captured_at is None else captured_at,
        sharpness,
        0.1,
        sequence,
    )


@pytest.fixture
def mind(monkeypatch: pytest.MonkeyPatch) -> VisionMind:
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock())
    agent = VisionMind(
        VisionConfig(face_backend="e4b", interval_s=1, morning_greeting_enabled=False),
        LLMConfig(url="http://e4b.test/v1/chat/completions", model="gemma-4-E4B"),
        VisionState(),
        TaskSlotStore(),
    )
    monkeypatch.setattr(agent.camera, "wait_ready", lambda: True)
    return agent


def reply(
    scene: str = "A desk.",
    changes: str = "A cup appeared.",
    face: dict | None = None,
    expressions: str | None = None,
) -> Mock:
    response = Mock()
    content = {"scene": scene, "changes": changes, "recent_events": "A cup appeared across recent observations.",
               "face": face or {"presence": "uncertain", "box": None}}
    if expressions is not None:
        content["expressions"] = expressions
    response.json.return_value = {"choices": [{"message": {"content": json.dumps(content)}}]}
    return response


def test_focus_matches_signed_laplacian_and_rejects_blur() -> None:
    image = np.random.default_rng(10).integers(0, 256, (240, 320), dtype=np.uint8)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    reference = cv2.filter2D(image.astype(np.float64), -1, kernel)[1:-1, 1:-1].var()
    assert laplacian_variance(image) == pytest.approx(reference, rel=1e-10)
    assert focus_score(image) > 10 * focus_score(cv2.GaussianBlur(image, (9, 9), 2))
    assert laplacian_variance(np.zeros((2, 2), dtype=np.uint8)) == 0


def test_observed_return_is_published_once_and_pause_clears_it(
    mind: VisionMind, monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter([(0, "present"), (10, "absent"), (30, "absent"), (50, "absent"),
                    (70, "absent"), (75, "present"), (80, "present")])
    sequence = [0]
    def observe() -> object:
        stamp, presence = next(samples)
        response = reply()
        content = json.loads(response.json.return_value["choices"][0]["message"]["content"])
        content["presence"] = presence
        response.json.return_value["choices"][0]["message"]["content"] = json.dumps(content)
        monkeypatch.setattr(mind, "_infer", Mock(return_value=response))
        sequence[0] += 1
        return mind._observe(frame(sequence[0], captured_at=stamp), 1, False)
    monkeypatch.setattr(mind, "run", lambda runtime: observe())
    for _ in range(5):
        mind.runtime.publish(mind.run(mind.runtime))
        assert mind._slot_store.get_slot("vision").attention_key is None
    mind.runtime.publish(mind.run(mind.runtime))
    arrived = mind._slot_store.get_slot("vision")
    assert arrived.attention_key and "person_arrived" in arrived.report
    assert mind.snapshot()["arrival"]["observed_absence_s"] == 65
    mind.runtime.publish(mind.run(mind.runtime))
    assert mind._slot_store.get_slot("vision").attention_key == arrived.attention_key
    mind.set_paused(True)
    assert mind._slot_store.get_slot("vision").attention_key is None
    assert mind._slot_store.get_slot("vision").status == "suspended"


def test_morning_greeting_is_published_once_and_not_recreated_after_pause(
    mind: VisionMind, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    mind.settings.morning_greeting_enabled = True
    stamp = datetime(2026, 10, 7, 9, 0).timestamp()
    response = reply(face={"presence": "present", "box": None})
    content = json.loads(response.json.return_value["choices"][0]["message"]["content"])
    content["presence"] = "present"
    response.json.return_value["choices"][0]["message"]["content"] = json.dumps(content)
    monkeypatch.setattr(mind, "_infer", Mock(return_value=response))
    first = mind._observe(frame(1, captured_at=stamp), 1, False)
    assert first.update_priority == "important" and first.attention_key == "morning_greeting@2026-10-07"
    assert "Observed morning greeting evidence" in first.report and '"local_time":"09:00:00"' in first.report
    second = mind._observe(frame(2, captured_at=stamp + 5), 1, False)
    assert second.attention_key == first.attention_key
    mind.set_paused(True)
    mind.set_paused(False)
    after_pause = mind._observe(frame(3, captured_at=stamp + 10), 1, False)
    assert after_pause.attention_key is None and after_pause.update_priority == "regular"


def test_interval_changes_schedule_and_survives_restart(
    mind: VisionMind, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "vision_settings.json"
    mind._settings_path = path
    mind.set_paused(True)
    mind.set_interval_range(6, 8)
    assert mind.snapshot()["interval_min_s"] == 6 and mind.snapshot()["interval_max_s"] == 8
    assert mind.snapshot()["target_hz"] == pytest.approx(1 / 7, abs=0.005)
    assert mind.paused
    assert json.loads(path.read_text()) == {"interval_min_s": 6, "interval_max_s": 8}
    restarted = VisionMind(VisionConfig(), mind.llm, VisionState(), TaskSlotStore(), settings_path=path)
    assert restarted.settings.interval_s == 7
    assert restarted.settings.face_interval_s == VisionConfig().face_interval_s
    for bounds in ((True, 8), ("4", 8), (0, 8), (6, 61), (9, 8), (2.5, 8), (float("nan"), 8)):
        with pytest.raises(ValueError):
            mind.set_interval_range(*bounds)
        assert json.loads(path.read_text())["interval_min_s"] == 6



def test_cpu_motion_ignores_exposure_and_decays_after_movement(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = MotionTracker()
    now = [100.0]
    monkeypatch.setattr("glados.vision.frame_picker.time.time", lambda: now[0])
    image = np.full((72, 96, 3), 60, dtype=np.uint8)
    tracker.update(image, now[0])
    for value in (80, 100, 120):
        now[0] += 0.1
        tracker.update(np.full_like(image, value), now[0])
    assert tracker.snapshot()["activity"] == 0
    for index in range(6):
        now[0] += 0.1
        moved = np.full_like(image, 120)
        left = 10 if index % 2 else 50
        moved[10:45, left:left + 30] = 240
        tracker.update(moved, now[0])
    assert tracker.snapshot()["activity"] > 0.9
    for _ in range(35):
        now[0] += 0.1
        tracker.update(moved, now[0])
    assert tracker.snapshot()["activity"] < 0.02
    now[0] += 2
    assert tracker.snapshot()["activity"] == 0
    tracker.reset()
    assert tracker.snapshot()["activity"] == 0


def test_legacy_fixed_interval_can_still_be_loaded(mind: VisionMind, tmp_path: Path) -> None:
    path = tmp_path / "vision_settings.json"
    path.write_text('{"interval_s":4}')
    restored = VisionMind(VisionConfig(), mind.llm, VisionState(), TaskSlotStore(), settings_path=path)
    assert restored.settings.interval_min_s == restored.settings.interval_max_s == 4


def test_http_vision_interval_updates_real_schedule(mind: VisionMind) -> None:
    engine = _FakeEngine()
    engine.vision_agent = mind
    server = WebappServer(engine, port=0)
    server.start()

    def post(body: dict, origin: str | None = None) -> tuple[int, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        try:
            connection.request("POST", "/api/vision/settings", json.dumps(body), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        status, result = post({"interval_min_s": 2, "interval_max_s": 5})
        assert status == 200 and result["interval_min_s"] == 2 and result["interval_max_s"] == 5
        assert mind.settings.interval_s == 3.5
        for body in ({"interval_min_s": 0, "interval_max_s": 5},
                     {"interval_min_s": True, "interval_max_s": 5},
                     {"interval_min_s": 6, "interval_max_s": 5},
                     {"interval_min_s": 2.5, "interval_max_s": 5},
                     {"interval_min_s": 2, "interval_max_s": 5, "enabled": False}):
            assert post(body)[0] == 400
        assert post({"interval_min_s": 3, "interval_max_s": 6}, "https://other.test")[0] == 403
        assert mind.settings.interval_min_s == 2 and mind.settings.interval_max_s == 5
        engine.vision_agent = None
        assert post({"interval_min_s": 2, "interval_max_s": 5})[0] == 404
    finally:
        engine.shutdown_event.set()
        server.shutdown()


def test_picker_bounds_memory_and_rejects_old_or_already_observed_frames() -> None:
    frames = RecentFrames(capacity=3)
    for item in [frame(1, 10000, 99), frame(2, 4, 99.8), frame(3, 6, 99.9), frame(4, 5, 100)]:
        frames.add(item)
    selected, count = frames.pick(now=100)
    assert (selected.sequence, count) == (3, 3)
    assert frames.pick(now=100, after=3)[0].sequence == 4
    assert frames.pick(now=102) == (None, 0)
    assert frames.pick(now=100, after=4) == (None, 0)
    frames.clear()
    assert frames.pick(now=100) == (None, 0)


def test_comparisons_advance_only_after_success(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(return_value=reply())
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    mind.camera.frames.add(frame(1))
    assert not mind.run(mind.runtime).notify_user
    first = [c for c in post.call_args.kwargs["json"]["messages"][-1]["content"] if c["type"] == "image_url"][-1]
    assert mind.snapshot()["changes"] == "First observation; no previous image."
    post.side_effect = requests.Timeout()
    mind.camera.frames.add(frame(2))
    assert mind.run(mind.runtime).status == "error"
    assert mind.snapshot()["revision"] == 1
    post.side_effect = None
    mind.camera.frames.add(frame(3, 10))
    assert mind.run(mind.runtime).status == "active"
    content = post.call_args.kwargs["json"]["messages"][-1]["content"]
    assert "HISTORICAL image" in content[0]["text"] and content[1] == first
    assert "CURRENT image" in content[2]["text"] and content[3] != first
    assert mind.snapshot()["changes"] == "A cup appeared."
    assert mind.snapshot()["revision"] == 2
    assert mind.preview().startswith(b"\xff\xd8")
    assert "seconds old" in mind.vision_state.as_message()["content"]


def test_four_image_window_interleaves_true_timestamps_and_skips_failed_frames(
    mind: VisionMind, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr("glados.vision.vision_mind.time.time", lambda: now[0])
    post = Mock(return_value=reply())
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    for sequence, captured in enumerate((100.0, 102.0, 104.0, 107.0, 108.0, 113.0), start=1):
        now[0] = captured
        mind.camera.frames.add(frame(sequence, captured_at=captured))
        post.side_effect = requests.Timeout() if captured == 104 else None
        output = mind.run(mind.runtime)
        assert output.status == ("error" if captured == 104 else "active")
    content = post.call_args.kwargs["json"]["messages"][-1]["content"]
    assert len(content) == 9
    assert [entry["type"] for entry in content] == ["text", "image_url"] * 4 + ["text"]
    labels = [entry["text"] for entry in content[:-1] if entry["type"] == "text"]
    assert "total first-to-CURRENT span = 11.000 seconds" in content[-1]["text"]
    assert "[5.000, 1.000, 5.000]" in content[-1]["text"]
    assert "1970-01-01T00:01:42.000+00:00" in labels[0]
    assert "11.000 s before CURRENT" in labels[0]
    assert "5.000 s since preceding observation" in labels[1]
    assert "1.000 s since preceding observation" in labels[2]
    assert "CURRENT image" in labels[3] and "5.000 s since preceding observation" in labels[3]
    assert all("HISTORICAL image" in label for label in labels[:3])
    assert len(mind._recent_images) == 4
    assert mind.snapshot()["window_capture_times"] == [102.0, 107.0, 108.0, 113.0]
    assert mind.snapshot()["window_span_s"] == 11.0
    assert "Recent events:" in mind.vision_state.as_message()["content"]
    assert json.loads(mind.look())["recent_events"] == "A cup appeared across recent observations."
    mind.select_camera(1)
    assert not mind._recent_images and mind.snapshot()["window_capture_times"] == []


def test_incomplete_events_report_does_not_advance_window(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply()
    content = json.loads(response.json.return_value["choices"][0]["message"]["content"])
    content.pop("recent_events")
    response.json.return_value["choices"][0]["message"]["content"] = json.dumps(content)
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(return_value=response))
    mind.camera.frames.add(frame(1))
    assert mind.run(mind.runtime).status == "error" and not mind._recent_images


def test_pick_happens_after_scheduler_admission(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def admission(*args: object) -> Iterator[None]:
        # A newer frame arrives while inference capacity is occupied.
        mind.camera.frames.add(frame(2, 100))
        yield

    mind.llm.scheduler = SimpleNamespace(lease=admission)
    mind.camera.frames.add(frame(1))
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(return_value=reply()))
    mind.run(mind.runtime)
    assert mind._previous_sequence == 2


def test_preview_overlay_matches_selected_frame_without_annotating_model_input(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), time.time(), 10, 0.1, 1)
    face = {
        "available": True,
        "present": True,
        "bbox": [0.25, 0.25, 0.25, 0.25],
        "x": -0.25,
        "y": -0.25,
        "observed_at": selected.captured_at,
    }
    locate = Mock(side_effect=AssertionError("The same E4B observation must supply the face"))
    monkeypatch.setattr(mind.camera.faces, "locate", locate)
    # The live tracker can already be observing a different location.
    mind.camera.faces._state = {"available": True, "present": True, "x": 0.8, "y": 0.8}
    expression = "Closed mouth, lowered eyes, head turned."
    post = Mock(return_value=reply(face={"presence": "present", "box": [250, 250, 500, 500]}, expressions=expression))
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    mind.camera.frames.add(selected)
    mind.run(mind.runtime)
    locate.assert_not_called()
    for key, value in face.items():
        assert mind.snapshot()["preview_face"][key] == value
    assert mind.camera.faces._state["x"] == -0.25
    assert mind.camera.faces._state["observed_at"] == selected.captured_at
    assert expression in mind.snapshot()["scene"]
    assert expression in mind.vision_state.as_message()["content"]
    assert expression in json.loads(mind.look())["scene"]
    assert not selected.image.any()
    image_url = [c for c in post.call_args.kwargs["json"]["messages"][-1]["content"] if c["type"] == "image_url"][-1][
        "image_url"
    ]["url"]
    raw = cv2.imdecode(np.frombuffer(base64.b64decode(image_url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    preview = cv2.imdecode(np.frombuffer(mind.preview(), np.uint8), cv2.IMREAD_COLOR)
    assert not raw.any(), "E4B must see the unannotated image"
    assert preview[144, 192].max() > 100, "The preview crosshair must mark this frame's face center"
    assert preview[120, 128].max() > 100, "The face box must appear on the selected frame"
    # A new displayed frame without a face must not retain an old box.
    post.return_value = reply(face={"presence": "absent", "box": None})
    mind.camera.frames.add(CameraFrame(selected.image, time.time(), 10, 0.1, 2))
    mind.run(mind.runtime)
    assert not cv2.imdecode(np.frombuffer(mind.preview(), np.uint8), cv2.IMREAD_COLOR).any()
    post.return_value = reply(face={"presence": "present", "box": [800, 200, 600, 400]})
    mind.camera.frames.add(CameraFrame(selected.image, time.time(), 10, 0.1, 3))
    assert mind.run(mind.runtime).status == "active", "Invalid face geometry must not discard a valid scene"
    assert mind.camera.faces._state["present"] is True and "x" not in mind.camera.faces._state


def test_pause_discards_inflight_result_and_manual_tick_releases_camera(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pause_during_request(*args: object, **kwargs: object) -> Mock:
        mind.set_paused(True)
        return reply()

    monkeypatch.setattr("glados.vision.vision_mind.requests.post", pause_during_request)
    mind.camera.frames.add(frame(1))
    assert mind.run(mind.runtime) is None
    assert mind.vision_state.snapshot() is None and mind.snapshot()["revision"] == 0
    mind.runtime.manual = True
    mind.camera.frames.add(frame(2))
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(return_value=reply()))
    assert mind.run(mind.runtime).status == "active"
    assert mind.paused and not mind.camera.snapshot()["enabled"]
    assert mind.snapshot()["revision"] == 1
    mind.set_paused(False)
    assert mind.camera.snapshot()["enabled"]


def test_malformed_observation_does_not_replace_scene(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(return_value=reply()))
    mind.camera.frames.add(frame(1))
    mind.run(mind.runtime)
    response = reply()
    response.json.return_value["choices"][0]["message"]["content"] = '{"scene": 123}'
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(return_value=response))
    mind.camera.frames.add(frame(2, 10))
    assert mind.run(mind.runtime).status == "error"
    assert mind.snapshot()["scene"] == "A desk." and mind.snapshot()["revision"] == 1


def test_api_conversation_keeps_vision_on_e4b(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(return_value=mind)
    monkeypatch.setattr("glados.vision.vision_mind.VisionMind", factory)
    autonomy = AutonomyConfig(enabled=False)
    autonomy.emotion.enabled = False
    autonomy.tokens.enabled = False  # This test isolates the Vision endpoint.
    autonomy.tokens.recall.enabled = False
    engine = SimpleNamespace(
        subagent_manager=Mock(),
        autonomy_config=autonomy,
        inference_scheduler=Mock(),
        completion_url="https://conversation-api.test/v1/chat/completions",
        api_key="conversation-key",
        llm_model="api-speaking-model",
        llm_request_options={"reasoning_effort": "minimal"},
        vision_config=VisionConfig(),
        vision_state=mind.vision_state,
        vision_agent=None,
        autonomy_slots=mind._slot_store,
        mind_registry=Mock(),
        observability_bus=Mock(),
        shutdown_event=threading.Event(),
        quiet_event=threading.Event(),
        autonomy_loop=None,
    )
    Glados._register_subagents(engine)
    config = factory.call_args.kwargs["llm_config"]
    assert config.model == "gemma-4-E4B"
    assert config.url == "http://127.0.0.1:18080/v1/chat/completions"
    assert config.api_key is None and config.request_options == {}
    assert config.scheduler is engine.inference_scheduler
    assert engine.vision_agent is mind
    engine.subagent_manager.list_agents.return_value = [
        SimpleNamespace(
            agent_id="vision",
            title="Vision",
            running=True,
            tick_count=0,
            last_tick=0,
        )
    ]
    engine.subagent_manager.get.return_value = mind
    row = next(row for row in build_minds(engine) if row["id"] == "vision")
    assert row["model"] == "gemma-4-E4B"


def test_vision_tool_is_available_to_primary_and_reads_cached_observation(mind: VisionMind) -> None:
    processor = make_processor()
    processor.vision_state = mind.vision_state
    assert "vision_look" in {t["function"]["name"] for t in processor._build_tools(False)}
    processor.vision_state = None
    assert "vision_look" not in {t["function"]["name"] for t in processor._build_tools(False)}
    messages = queue.Queue()
    tool = VisionLook(messages, {"vision_agent": mind})
    tool.run("look", {})
    assert "No webcam observation" in messages.get_nowait()["content"]
    with mind._state_lock:
        mind._state.update(scene="A desk.", changes="No clear visual change.", captured_at=time.time() - 5)
    tool.run("look", {})
    assert json.loads(messages.get_nowait()["content"])["age_seconds"] >= 5


def test_camera_pause_and_shutdown_release_device(monkeypatch: pytest.MonkeyPatch) -> None:
    released = threading.Event()
    camera = Mock()
    camera.read.return_value = (True, np.zeros((32, 32, 3), dtype=np.uint8))
    camera.release.side_effect = released.set
    monkeypatch.setattr("glados.vision.frame_picker.cv2.VideoCapture", Mock(return_value=camera))
    sampler = CameraSampler()
    sampler.start()
    try:
        assert sampler.wait_ready(2)
        sampler.set_enabled(False)
        assert released.wait(1)
        assert sampler.frames.pick()[0] is None
        released.clear()
        sampler.set_enabled(True)
        assert sampler.wait_ready(2)
    finally:
        sampler.stop()
    assert released.is_set() and not sampler.snapshot()["connected"]


def test_paused_vision_does_not_leak_old_description_into_task_context(mind: VisionMind) -> None:
    store = mind._slot_store
    store.update_slot("vision", "Vision", "active", "Old room view", notify_user=False)
    assert Glados._format_slots(SimpleNamespace(autonomy_slots=store)) is None
    store.update_slot("task_test", "Test", "open", "Test the new camera mind", notify_user=False)
    prompt = Glados._format_slots(SimpleNamespace(autonomy_slots=store))
    assert "Test the new camera mind" in prompt and "Old room view" not in prompt


def test_autonomy_ticks_use_observation_age_and_respect_camera_pause(mind: VisionMind) -> None:
    loop = AutonomyLoop(
        config=AutonomyConfig(),
        event_bus=EventBus(),
        interaction_state=InteractionState(),
        vision_state=mind.vision_state,
        slot_store=mind._slot_store,
        llm_queue=queue.Queue(),
        processing_active_event=threading.Event(),
        currently_speaking_event=threading.Event(),
        shutdown_event=threading.Event(),
    )
    mind.vision_state.update("A blue cup.", captured_at=time.time() - 5)
    mind._slot_store.update_slot("vision", "Vision", "active", "Old camera view", notify_user=False)
    prompt = loop._build_prompt(TimeTickEvent(time.time()))
    assert "seconds old" in prompt and "A blue cup." in prompt
    assert "Old camera view" not in prompt
    mind.set_paused(True)
    assert "A blue cup." not in loop._build_prompt(TimeTickEvent(time.time()))


def test_preview_and_live_state_are_real_and_same_origin(mind: VisionMind) -> None:
    engine = _FakeEngine()
    engine.vision_agent = mind
    engine.vision_state = mind.vision_state
    server = WebappServer(engine, port=0)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
    try:
        connection.request("GET", "/api/vision/frame")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        mind._preview = b"\xff\xd8test-jpeg"
        connection.request("GET", "/api/vision/frame?revision=1")
        response = connection.getresponse()
        assert response.status == 200 and response.read() == mind._preview
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("Cross-Origin-Resource-Policy") == "same-origin"
        connection.request("GET", "/api/vision/frame", headers={"Origin": "https://other.test"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        assert build_state(engine)["vision_mind"]["target_hz"] == 1
    finally:
        connection.close()
        engine.shutdown_event.set()
        server.shutdown()


def test_live_preview_is_independent_of_caption_and_matches_its_frame(mind: VisionMind) -> None:
    camera = mind.camera
    selected = CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), time.time(), 10, 0.1, 12)
    face = {"available": True, "present": True, "bbox": [0.25, 0.25, 0.25, 0.25], "x": -0.25, "y": -0.25}
    camera._status["connected"] = True
    camera._live_frame = (selected, face)
    # Live presence can refer to another frame; it must not supply this overlay.
    camera.faces._state = {"available": True, "present": True, "x": 0.8, "y": 0.8}
    jpeg, sequence = mind.live_preview()
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image[180, 240].max() > 100
    assert image[120, 160].max() > 100
    assert not selected.image.any()
    assert sequence == 12 and mind.snapshot()["revision"] == 0
    assert mind.live_preview()[0] is jpeg, "Viewers share the encoded JPEG for this frame"
    raw, raw_sequence = mind.live_preview(overlay=False)
    assert raw_sequence == sequence
    assert not cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).any()
    assert mind.live_preview(overlay=False)[0] is raw
    assert mind.live_preview()[0] is jpeg, "Raw viewers must not overwrite the annotated cache"
    camera._live_frame = (CameraFrame(selected.image, time.time(), 10, 0.1, 13), {"present": False})
    assert not cv2.imdecode(np.frombuffer(mind.live_preview()[0], np.uint8), cv2.IMREAD_COLOR).any()
    mind.set_paused(True)
    assert mind.live_preview() is None


def test_live_stream_has_independent_frames_and_blocks_cross_origin(mind: VisionMind) -> None:
    engine = _FakeEngine()
    engine.vision_agent = mind
    engine.vision_state = mind.vision_state
    server = WebappServer(engine, port=0)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=3)
    try:
        connection.request("GET", "/api/vision/live")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        mind.live_preview = Mock(side_effect=[(b"first-jpeg", 1), (b"first-jpeg", 1), (b"second-jpeg", 2), None])
        connection.request("GET", "/api/vision/live")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=glados-frame"
        data = response.read()
        assert data.count(b"first-jpeg") == 1 and data.count(b"second-jpeg") == 1
        assert data.endswith(b"--glados-frame--\r\n")
        assert all(call.kwargs == {"overlay": True} for call in mind.live_preview.call_args_list)
        mind.live_preview = Mock(side_effect=[(b"raw-jpeg", 3), (b"raw-jpeg", 3), None])
        connection.request("GET", "/api/vision/live?overlay=0")
        response = connection.getresponse()
        assert response.status == 200 and b"raw-jpeg" in response.read()
        assert all(call.kwargs == {"overlay": False} for call in mind.live_preview.call_args_list)
        assert mind.snapshot()["revision"] == 0, "Streaming never invokes E4B or advances its captions"
        for headers in ({"Origin": "https://other.test"}, {"Sec-Fetch-Site": "cross-site"}):
            connection.request("GET", "/api/vision/live", headers=headers)
            response = connection.getresponse()
            assert response.status == 403
            response.read()
    finally:
        connection.close()
        engine.shutdown_event.set()
        server.shutdown()


def test_cpu_faces_do_not_request_e4b_boxes_or_overwrite_live_tracking(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mind.settings.face_backend = "yunet"
    locate = Mock(return_value={"available": True, "present": False})
    monkeypatch.setattr(mind.camera.faces, "locate", locate)
    post = Mock(return_value=reply())
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    mind.camera.faces._state = {"available": True, "present": True, "x": 0.8, "y": 0.8}
    mind.camera.frames.add(frame(1))
    assert mind.run(mind.runtime).status == "active"
    request = post.call_args.kwargs["json"]
    assert "box_2d" not in request["messages"][0]["content"]
    assert mind.camera.faces._state["x"] == 0.8
    assert locate.call_count == 1
    assert VisionConfig().interval_min_s == 2 and VisionConfig().interval_max_s == 5
    assert VisionConfig().face_backend == "yunet"


def question_reply(answer: str = "Yes, a zipper is visible.") -> Mock:
    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": answer,
                            "evidence": "A zipper pull and metal teeth are visible on the jacket.",
                        }
                    )
                }
            }
        ]
    }
    return response


def test_question_inspects_fresh_image_without_replacing_overview(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mind._state.update(scene="A person.", changes="No change.", revision=9)
    previous = {"type": "image_url", "image_url": {"url": "previous-overview-image"}}
    mind._recent_images.append((previous, time.time() - 5))
    mind.camera.frames.add(frame(1))

    @contextmanager
    def admission(owner: str, lane: str, model: str, cancelled: object) -> Iterator[None]:
        assert owner == "Vision question" and lane == "priority" and model == "gemma-4-E4B"
        mind.camera.frames.clear()
        mind.camera.frames.add(frame(2, 100))
        yield

    mind.llm.scheduler = SimpleNamespace(lease=admission)
    post = Mock(return_value=question_reply())
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    messages = queue.Queue()
    VisionLook(messages, {"vision_agent": mind}).run("zipper", {"question": "Does my jacket have a zipper?"})
    result = json.loads(messages.get_nowait()["content"])
    assert result["answer"] == "Yes, a zipper is visible."
    assert result["question"] == "Does my jacket have a zipper?"
    assert result["age_seconds"] < 1
    request = post.call_args.kwargs["json"]
    content = request["messages"][-1]["content"]
    assert "Does my jacket have a zipper?" in content[-1]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "previous-overview-image" not in json.dumps(request)
    assert request["model"] == "gemma-4-E4B"
    assert mind._recent_images[0][0] is previous and mind.snapshot()["revision"] == 9
    assert mind.snapshot()["scene"] == "A person."
    assert mind.snapshot()["last_question"]["answer"] == result["answer"]


def test_question_can_run_while_background_observation_is_busy(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = mind.llm.scheduler = InferenceScheduler()
    background = scheduler.acquire("Vision", "autonomy", "gemma-4-E4B")
    mind.camera.frames.add(frame(1))
    received = queue.Queue()

    def post(*args: object, **kwargs: object) -> Mock:
        active = scheduler.snapshot()["active"]
        assert {item["owner"] for item in active} == {"Vision", "Vision question"}
        return question_reply()

    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    # A background request holds this lock throughout its inference.
    mind._observation_lock.acquire()
    thread = threading.Thread(target=lambda: received.put(mind.ask("Does my jacket have a zipper?")), daemon=True)
    thread.start()
    try:
        assert json.loads(received.get(timeout=2))["answer"].startswith("Yes")
    finally:
        mind._observation_lock.release()
        scheduler.release(background)
        thread.join(2)
    assert not scheduler.snapshot()["active"] and not scheduler.snapshot()["waiting"]


def test_question_has_bounded_admission_wait(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = mind.llm.scheduler = InferenceScheduler(InferenceConfig(slots=1, reserved_interactive=0))
    active = scheduler.acquire("Busy", "priority", "gemma-4-E4B")
    mind.llm.timeout = 0.05
    post = Mock()
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    try:
        assert "timed out" in mind.ask("Is there a zipper?")
        assert not scheduler.snapshot()["waiting"]
        post.assert_not_called()
    finally:
        scheduler.release(active)


def test_question_respects_pause_and_stale_camera(mind: VisionMind, monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock()
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", post)
    mind.set_paused(True)
    assert "Vision Core is suspended" in mind.ask("Is there a zipper?")
    assert not mind.camera.snapshot()["enabled"]
    mind.set_paused(False)
    mind.camera.frames.add(frame(1, captured_at=time.time() - 5))
    assert "No fresh webcam frame" in mind.ask("Is there a zipper?")
    assert mind.ask("").startswith("error:")
    assert mind.ask(None).startswith("error:")
    post.assert_not_called()


@pytest.mark.parametrize("failure", [requests.Timeout(), ValueError("malformed JSON")])
def test_question_failure_does_not_fabricate_an_answer_or_change_overview(
    mind: VisionMind,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    mind._state.update(scene="A person.", revision=7)
    mind.camera.frames.add(frame(1))
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(side_effect=failure))
    assert "no visual answer available" in mind.ask("Is there a zipper?")
    assert mind.snapshot()["scene"] == "A person." and mind.snapshot()["revision"] == 7
    assert "last_question" not in mind.snapshot()


def test_inference_blink_cue_starts_at_http_request_and_survives_failure(
    mind: VisionMind, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def infer(*args: object, **kwargs: object) -> Mock:
        tracking = mind.tracking_snapshot()
        assert tracking["inference_sequence"] == 1 and tracking["inference_active"] == 1
        assert mind.snapshot()["revision"] == 0
        return reply()

    monkeypatch.setattr("glados.vision.vision_mind.requests.post", infer)
    mind.run(mind.runtime)  # No fresh frame: no inference and no blink.
    assert mind.tracking_snapshot()["inference_sequence"] == 0
    mind.camera.frames.add(frame(1))
    mind.run(mind.runtime)
    assert mind.tracking_snapshot()["inference_active"] == 0
    monkeypatch.setattr("glados.vision.vision_mind.requests.post", Mock(side_effect=requests.Timeout()))
    mind.camera.frames.add(frame(2))
    assert mind.run(mind.runtime).status == "error"
    assert mind.tracking_snapshot()["inference_sequence"] == 2
    assert mind.tracking_snapshot()["inference_active"] == 0
