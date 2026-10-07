"""Input gating, affect ordering and cancellation across sleep/wake."""

from collections.abc import Iterator
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from glados.autonomy.agents.emotion_agent import EmotionAgent
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.subagent import SubagentConfig
from glados.core.audio_data import AudioMessage
from glados.core.engine import Glados
from glados.core.speech_markup import SpeechText
from glados.core.tool_executor import _ToolResultQueue
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from tests.test_speech_markup import make_processor


def emotion_agent(monkeypatch: pytest.MonkeyPatch) -> EmotionAgent:
    memory = Mock()
    memory.get.return_value = None
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock(return_value=memory))
    return EmotionAgent(SubagentConfig("emotion", "Emotion"), LLMConfig("http://test"), slot_store=Mock())


def test_reaction_uses_current_input_and_idle_does_not_reset_anger(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = emotion_agent(monkeypatch)
    from dataclasses import replace
    model = Mock(side_effect=lambda events, audio=None, **kw: replace(
        kw["state"], pleasure=-.8, arousal=.8, dominance=.9))
    monkeypatch.setattr(agent, "_ask_llm", model)
    audio = [{"type": "input_audio", "input_audio": {"data": "audio", "format": "wav"}}]
    agent.react("You are useless", audio)
    model.assert_not_called()
    agent.tick()
    assert agent.state.pleasure == -.8
    assert "angry glare" in agent.state.response_instructions()
    assert model.call_args.args[1] == audio
    assert "You are useless" in model.call_args.args[0][0].description
    agent.tick()
    assert model.call_count == 1
    agent.on_stop()


def test_failed_affect_retains_previous_state(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = emotion_agent(monkeypatch)
    previous = agent.state.to_dict()
    monkeypatch.setattr(agent, "_ask_llm", Mock(return_value=None))
    agent.react("Hello")
    agent.tick()
    current = agent.state.to_dict()
    previous.pop("last_update")
    current.pop("last_update")
    assert current == previous
    agent.on_stop()


def test_sleep_pauses_minds_drains_work_and_restores_previous_pauses() -> None:
    agents = {"camera": SimpleNamespace(paused=False), "emotion": SimpleNamespace(paused=True)}
    for agent in agents.values():
        agent.set_paused = lambda paused, a=agent: setattr(a, "paused", paused)
    manager = SimpleNamespace(list_agents=lambda: [SimpleNamespace(agent_id=i) for i in agents], get=agents.get)
    engine = SimpleNamespace(
        quiet_event=threading.Event(),
        _quiet_lock=threading.RLock(),
        _quiet_generation=0,
        inference_scheduler=Mock(),
        _quiet_saved_pauses={},
        processing_active_event=threading.Event(),
        currently_speaking_event=threading.Event(),
        audio_io=Mock(),
        speech_animation=Mock(),
        subagent_manager=manager,
        observability_bus=Mock(),
    )
    for name in ("tts_queue", "audio_queue", "llm_queue_autonomy", "tool_calls_queue"):
        pending = queue.Queue()
        pending.put("stale")
        setattr(engine, name, pending)
    Glados.set_quiet_mode(engine, True)
    assert engine.quiet_event.is_set() and all(a.paused for a in agents.values())
    assert engine.audio_io.stop_speaking.call_count == 1
    assert all(
        getattr(engine, n).empty() for n in ("tts_queue", "audio_queue", "llm_queue_autonomy", "tool_calls_queue")
    )
    Glados.set_quiet_mode(engine, False)
    assert not engine.quiet_event.is_set() and engine._quiet_generation == 2
    assert not agents["camera"].paused and agents["emotion"].paused


@pytest.mark.parametrize("quiet", [True, False])
def test_quiet_input_and_noise_do_not_update_affect_or_generate_reply(
    monkeypatch: pytest.MonkeyPatch, quiet: bool
) -> None:
    route = {"action": "ignore", "accepted": True}
    processor = make_processor()
    processor._quiet_mode = lambda: quiet
    processor._set_quiet_mode = Mock()
    processor._before_reply = Mock()
    processor.router = Mock()
    processor.router.store.get.return_value = SimpleNamespace(strategy="hierarchical")

    def score(*args: object, **kwargs: object) -> dict:
        processor.shutdown_event.set()
        return route

    processor.router.score.side_effect = score
    processor.router.quiet_score.side_effect = score
    processor.llm_input_queue.put(
        {"role": "user", "content": "[Voice input]", "_native_audio": [{"type": "input_audio"}]}
    )
    post = Mock()
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    processor._before_reply.assert_not_called()
    post.assert_not_called()
    assert processor.tts_input_queue.empty()


def test_affect_updates_before_reply_request(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    order = []
    processor._before_reply = lambda message: order.append("affect")
    processor.llm_input_queue.put({"role": "user", "content": "You are useless", "_allow_tools": False})

    def post(*args: object, **kwargs: object) -> MagicMock:
        order.append("reply")
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response

        def lines(chunk_size: int = 1) -> Iterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"[emotion:angry glare]How original."}}]}'
            processor.shutdown_event.set()

        response.iter_lines.side_effect = lines
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert order == ["affect", "reply"]


def test_speech_and_tool_results_from_old_generation_are_dropped() -> None:
    processor = make_processor()
    processor._quiet_generation = lambda: 2
    processor._reply_generation = 0
    processor._process_sentence_for_tts(["Old reply"])
    assert processor.tts_input_queue.empty()
    target = queue.Queue()
    wrapper = _ToolResultQueue(target, {"function": {"name": "get_time"}}, cancelled=lambda: True)
    wrapper.put({"role": "tool", "content": "old"})
    assert target.empty()
    pending = queue.Queue()
    pending.put(SpeechText("Old", generation=0))
    pending.put(SpeechText("<EOS>", generation=2))
    output = queue.Queue()
    shutdown = threading.Event()
    original = output.put

    def put(item: AudioMessage) -> None:
        original(item)
        shutdown.set()

    output.put = put
    model = Mock(sample_rate=16000)
    TextToSpeechSynthesizer(
        pending, output, model, SimpleNamespace(text_to_spoken=lambda t: t), shutdown, 0.001, quiet_generation=lambda: 2
    ).run()
    model.generate_speech_audio.assert_not_called()
    assert output.get_nowait().is_eos


@pytest.mark.parametrize("enabled,quiet", [(False, False), (True, True)])
def test_autonomy_does_not_dispatch_when_disabled_or_quiet(enabled: bool, quiet: bool) -> None:
    from glados.autonomy.loop import AutonomyLoop

    loop = SimpleNamespace(_config=SimpleNamespace(enabled=enabled), _quiet_mode=lambda: quiet, _llm_queue=Mock())
    AutonomyLoop._dispatch(loop, "A scheduled response")
    assert AutonomyLoop._should_skip(loop)
    loop._llm_queue.put_nowait.assert_not_called()


def test_late_connection_error_after_sleep_and_wake_cannot_speak(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    processor = make_processor()
    epoch = [0]
    processor._quiet_generation = lambda: epoch[0]
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": False})

    def fail(*args: object, **kwargs: object) -> None:
        epoch[0] = 2
        processor.shutdown_event.set()
        raise requests.ConnectionError("Disconnected after sleep and wake")

    monkeypatch.setattr("glados.core.llm_processor.requests.post", fail)
    processor.run()
    assert all(item.text == "<EOS>" and item.generation == 0 for item in list(processor.tts_input_queue.queue))


def test_new_user_turn_invalidates_work_without_entering_quiet_mode() -> None:
    engine = SimpleNamespace(_quiet_lock=threading.RLock(), _quiet_generation=4, inference_scheduler=Mock(),
                             processing_active_event=threading.Event(), audio_io=Mock(),
                             quiet_event=threading.Event())
    engine.processing_active_event.set()
    for name in ("llm_queue_priority", "tts_queue", "audio_queue", "tool_calls_queue"):
        pending = queue.Queue()
        pending.put("old")
        setattr(engine, name, pending)
    assert Glados._begin_user_turn(engine) == 5
    engine.inference_scheduler.begin_interaction.assert_called_once_with(5)
    assert not engine.quiet_event.is_set() and not engine.processing_active_event.is_set()
    assert all(getattr(engine, name).empty() for name in
               ("llm_queue_priority", "tts_queue", "audio_queue", "tool_calls_queue"))
    engine.audio_io.stop_speaking.assert_called_once()


def test_cancelled_stream_cannot_resume_after_processing_flag_is_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    epoch = [1]
    processor._quiet_generation = lambda: epoch[0]
    processor.llm_input_queue.put({"role": "user", "content": "Old question", "_quiet_generation": 1})

    def post(*args: object, **kwargs: object) -> MagicMock:
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.__exit__.side_effect = lambda *args: processor.shutdown_event.set()
        def lines(chunk_size: int = 1):
            epoch[0] = 2
            processor.processing_active_event.clear()
            processor.processing_active_event.set()
            yield b'data: {"choices":[{"delta":{"content":"Old reply."}}]}'
        response.iter_lines.side_effect = lines
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert all(item.text == "<EOS>" and item.generation == 1 for item in list(processor.tts_input_queue.queue))
    assert processor.tool_calls_queue.empty()


def test_queued_old_turn_is_discarded_and_tool_result_keeps_its_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor._quiet_generation = lambda: 2
    processor.llm_input_queue.put({"role": "user", "content": "stale", "_quiet_generation": 1})
    processor.llm_input_queue.put({"role": "user", "content": "current", "_quiet_generation": 2, "_allow_tools": False})
    requests_seen = []
    def post(*args, **kwargs):
        requests_seen.append(kwargs["json"])
        processor.shutdown_event.set()
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = iter(())
        return response
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert len(requests_seen) == 1
    assert "stale" not in str(requests_seen)
    target = queue.Queue()
    _ToolResultQueue(target, {"function": {"name": "get_time"}, "_quiet_generation": 2}).put(
        {"role": "tool", "content": "result"})
    assert target.get_nowait()["_quiet_generation"] == 2


def test_synthesis_finishing_after_new_turn_is_discarded() -> None:
    import numpy as np
    epoch = [1]
    shutdown = threading.Event()
    pending, output = queue.Queue(), queue.Queue()
    pending.put(SpeechText("Old sentence", generation=1))
    def generate(text):
        epoch[0] = 2
        shutdown.set()
        return np.ones(100, dtype=np.float32)
    model = SimpleNamespace(sample_rate=16000, generate_speech_audio=generate)
    TextToSpeechSynthesizer(pending, output, model, SimpleNamespace(text_to_spoken=lambda t: t),
                            shutdown, .001, quiet_generation=lambda: epoch[0]).run()
    assert output.empty()
