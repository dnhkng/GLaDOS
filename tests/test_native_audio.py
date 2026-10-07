"""Native audio bypasses Parakeet and keeps encoded recordings out of shared history."""

import base64
from collections.abc import Iterator
import io
import json
from pathlib import Path
import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
import soundfile as sf

from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados, GladosConfig
from glados.core.inference import InferenceCancelledError, InferenceScheduler
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.native_audio import NativeAudioConfig, NativeAudioInput
from glados.core.speech_listener import SpeechListener
from glados.core.speech_markup import SpeechText


def test_native_audio_encodes_valid_wav_and_rejects_silence_or_oversized_clips() -> None:
    native = NativeAudioInput(NativeAudioConfig(enabled=True))
    samples = np.full(16000, 0.1, dtype=np.float32)
    message = native.message([samples])
    assert message is not None
    content = message["_native_audio"]
    assert "English" in content[0]["text"]
    decoded, rate = sf.read(io.BytesIO(base64.b64decode(content[1]["input_audio"]["data"])))
    assert rate == 16000
    np.testing.assert_allclose(decoded, samples, atol=1 / 32768)
    assert native.message([]) is None
    assert native.message([np.zeros(10)]) is None
    with pytest.raises(ValueError, match="clip limit"):
        native.message([np.ones(480001)])


def test_transcript_stream_cancel_closes_request_and_discards_partial_text(monkeypatch: pytest.MonkeyPatch) -> None:
    native = NativeAudioInput(NativeAudioConfig(enabled=True))
    response = MagicMock()
    response.__enter__.return_value = response
    cancel = threading.Event()
    def lines(**kwargs: object) -> Iterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"Partial"}}]}'
        cancel.set()
        yield b'data: {"choices":[{"delta":{"content":" transcript"}}]}'
    response.iter_lines.side_effect = lines
    monkeypatch.setattr("glados.core.native_audio.requests.post", MagicMock(return_value=response))
    with pytest.raises(InferenceCancelledError):
        native.transcribe([{"type": "text"}, {"type": "input_audio"}], "http://localhost/v1/chat/completions",
                          "E4B", {}, cancelled=cancel.is_set)
    assert response.__exit__.call_args.args[0] is InferenceCancelledError


def test_listener_queues_audio_without_asr_and_caps_turns() -> None:
    native = NativeAudioInput(NativeAudioConfig(enabled=True, max_duration_s=1))
    pending = queue.Queue()
    active = threading.Event()
    listener = SpeechListener(
        MagicMock(), pending, threading.Event(), threading.Event(), active, None, None, 0.001, native_audio=native
    )
    for _ in range(32):
        listener._handle_audio_sample(np.full(512, 0.1, dtype=np.float32), True)
    assert pending.empty()
    assert listener._samples == []
    for _ in range(20):
        listener._handle_audio_sample(np.zeros(512, dtype=np.float32), False)
    assert not listener._recording_started
    for _ in range(5):
        listener._handle_audio_sample(np.full(512, 0.1, dtype=np.float32), True)
    for _ in range(20):
        listener._handle_audio_sample(np.zeros(512, dtype=np.float32), False)
    message = pending.get_nowait()
    assert "_native_audio" in message
    assert active.is_set()
    audio, rate = sf.read(io.BytesIO(base64.b64decode(message["_native_audio"][1]["input_audio"]["data"])))
    assert len(audio) <= rate


def test_listener_recovers_capture_even_when_no_samples_arrive() -> None:
    from types import SimpleNamespace
    from glados.observability import ObservabilityBus
    shutdown = threading.Event()
    samples = queue.Queue()
    recovered = []
    def ensure():
        recovered.append(True)
        shutdown.set()
        return True
    audio = SimpleNamespace(get_sample_queue=lambda: samples, ensure_listening=ensure, stop_listening=lambda: None)
    bus = ObservabilityBus()
    listener = SpeechListener(audio, queue.Queue(), shutdown, threading.Event(), threading.Event(), None, None, .001,
                              observability_bus=bus)
    listener._samples = [np.ones(512, dtype=np.float32)]
    listener._recording_started = True
    listener.run()
    assert recovered == [True] and not listener._recording_started and listener._samples == []
    assert any(event.kind == "recovered" for event in bus.drain(20))


def test_listener_confirms_speech_before_barge_in_and_tags_the_turn() -> None:
    native = NativeAudioInput(NativeAudioConfig(enabled=True))
    pending = queue.Queue()
    speaking, active = threading.Event(), threading.Event()
    speaking.set()
    active.set()
    begin = MagicMock(return_value=7)
    audio = MagicMock()
    listener = SpeechListener(audio, pending, threading.Event(), speaking, active, None, None, .001,
                              native_audio=native, begin_user_turn=begin)
    sample = np.full(512, .1, dtype=np.float32)
    listener._handle_audio_sample(sample, True)
    listener._handle_audio_sample(sample, False)
    audio.stop_speaking.assert_not_called()
    assert active.is_set()
    for _ in range(5):
        listener._handle_audio_sample(sample, True)
    begin.assert_called_once()
    assert not active.is_set()
    for _ in range(20):
        listener._handle_audio_sample(np.zeros(512, dtype=np.float32), False)
    assert pending.get_nowait()["_quiet_generation"] == 7
    assert active.is_set()


@pytest.mark.parametrize("profile,loads_asr", [("glados_config.yaml", False), ("glados_extended_config.yaml", True)])
def test_profile_loads_only_required_asr_model(monkeypatch: pytest.MonkeyPatch, profile: str, loads_asr: bool) -> None:
    config = GladosConfig.from_yaml(Path("configs") / profile)
    asr = MagicMock()
    captured = {}

    def initialize(self: Glados, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("glados.core.engine.get_audio_transcriber", asr)
    monkeypatch.setattr("glados.core.engine.get_speech_synthesizer", MagicMock())
    monkeypatch.setattr("glados.core.engine.get_audio_system", MagicMock())
    monkeypatch.setattr(Glados, "__init__", initialize)
    Glados.from_config(config)
    assert asr.called is loads_asr
    if not loads_asr:
        assert captured["asr_model"] is None
        assert config.native_audio.language == "English"
        assert config.native_audio.user_transcripts is False


@pytest.mark.parametrize("transcripts", [False, True, "failure"])
def test_llm_receives_audio_but_history_keeps_only_optional_transcript(
    monkeypatch: pytest.MonkeyPatch, transcripts: bool | str
) -> None:
    native = NativeAudioInput(NativeAudioConfig(enabled=True, user_transcripts=bool(transcripts)))
    message = native.message([np.full(16000, 0.1, dtype=np.float32)])
    pending = queue.Queue()
    pending.put({**message, "_allow_tools": False})
    history = ConversationStore()
    active, shutdown = threading.Event(), threading.Event()
    active.set()
    output = queue.Queue()
    calls = []

    transcript_done = threading.Event()

    def post(url: str, **kwargs: object) -> MagicMock:
        payload = kwargs["json"]
        calls.append(payload)
        content = payload["messages"][-1]["content"]
        if transcripts is True:
            assert content == "Hello GLaDOS."
        else:
            assert content[1]["type"] == "input_audio"
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"[emotion:smug]Hello human."}}]}',
            b'data: [DONE]',
        ]
        return response

    def transcribe(*args: object, **kwargs: object) -> str:
        assert output.empty()
        transcript_done.set()
        if transcripts == "failure":
            raise requests.ConnectionError("transcription unavailable")
        return "Hello GLaDOS."

    native.transcribe = transcribe
    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor = LanguageModelProcessor(
        pending,
        queue.Queue(),
        output,
        history,
        "http://localhost:18080/v1/chat/completions",
        "gemma-4-E4B",
        None,
        active,
        shutdown,
        native_audio=native,
        inference_scheduler=InferenceScheduler(),
    )
    if transcripts is True:
        from types import SimpleNamespace
        processor.router = MagicMock()
        processor.router.store.get.return_value = SimpleNamespace(strategy="hierarchical")
        processor.router.score.return_value = {"action": "reply"}
        processor._before_reply = MagicMock()
    worker = threading.Thread(target=processor.run)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            saved = history.snapshot()
            if any(item.text == "<EOS>" for item in list(output.queue)) and (
                not transcripts or (saved and "pending" not in saved[0]["content"])
            ):
                break
            time.sleep(0.005)
        else:
            raise AssertionError("Reply or deferred transcript did not complete")
        assert transcript_done.is_set() is bool(transcripts)
    finally:
        shutdown.set()
        worker.join(2)
    assert processor._inference_scheduler.snapshot()["active"] == []
    assert len(calls) == 1
    assert list(output.queue) == [SpeechText("Hello human.", "smug", 0), SpeechText("<EOS>", generation=0)]
    saved = history.snapshot()
    assert "input_audio" not in json.dumps(saved)
    assert "_native_audio" not in json.dumps(saved)
    if transcripts is True:
        assert saved[0]["content"] == "Hello GLaDOS."
        assert processor.router.score.call_args.args[2] is None
        assert "_native_audio" not in processor._before_reply.call_args.args[0]
    elif transcripts == "failure":
        assert "unavailable" in saved[0]["content"]
    else:
        assert "transcript disabled" in saved[0]["content"]
