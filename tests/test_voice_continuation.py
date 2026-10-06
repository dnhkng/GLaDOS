"""Speech resumed before delivery replaces the pending turn with combined audio."""

import base64
import io
import queue
import threading
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from glados.core.native_audio import NativeAudioConfig, NativeAudioInput
from glados.core.speech_listener import SpeechListener
from glados.core.conversation_store import ConversationStore


def listener(native=True, max_duration=30):
    generation = [0]
    pending = queue.Queue()

    def begin():
        generation[0] += 1
        while not pending.empty():
            pending.get_nowait()
        return generation[0]

    core = SpeechListener(
        MagicMock(),
        pending,
        threading.Event(),
        threading.Event(),
        threading.Event(),
        MagicMock(),
        None,
        0.001,
        native_audio=NativeAudioInput(NativeAudioConfig(enabled=True, max_duration_s=max_duration)) if native else None,
        begin_user_turn=begin,
        turn_is_current=lambda value: value == generation[0],
    )
    return core, pending, generation


def say(core, value, frames=6):
    for _ in range(frames):
        core._handle_audio_sample(np.full(512, value, dtype=np.float32), True)
    for _ in range(core.PAUSE_LIMIT // core.VAD_SIZE):
        core._handle_audio_sample(np.zeros(512, dtype=np.float32), False)


def decode(message):
    return sf.read(io.BytesIO(base64.b64decode(message["_native_audio"][1]["input_audio"]["data"])))[0]


def test_resumed_audio_retains_each_segment_once_and_reuses_turn_id():
    core, pending, generation = listener()
    say(core, 0.1)
    first = pending.get_nowait()
    first_pcm = decode(first)
    say(core, 0.2)
    second = pending.get_nowait()
    second_pcm = decode(second)
    assert second["_quiet_generation"] == generation[0] == 2
    assert first["_voice_turn_id"] == second["_voice_turn_id"]
    assert second["_voice_continuation"]
    np.testing.assert_array_equal(second_pcm[: len(first_pcm)], first_pcm)
    assert np.count_nonzero(np.isclose(second_pcm, 0.1, atol=1 / 32768)) == 6 * 512
    assert np.count_nonzero(np.isclose(second_pcm, 0.2, atol=1 / 32768)) == 6 * 512
    say(core, 0.3)
    third = pending.get_nowait()
    assert third["_voice_turn_id"] == first["_voice_turn_id"]
    np.testing.assert_array_equal(decode(third)[: len(second_pcm)], second_pcm)


def test_new_onset_cancels_queued_turn_before_merged_submission():
    core, pending, generation = listener()
    say(core, 0.1)
    assert pending.qsize() == 1
    for _ in range(3):
        core._handle_audio_sample(np.full(512, 0.2, dtype=np.float32), True)
    assert pending.empty() and not core.processing_active_event.is_set()
    assert generation[0] == 2 and core._voice_continuation


@pytest.mark.parametrize("boundary", ["playback", "reset", "different_turn", "expired", "speaking"])
def test_delivered_or_invalidated_turn_is_not_merged(boundary, monkeypatch):
    core, pending, generation = listener()
    say(core, 0.1)
    first = pending.get_nowait()
    if boundary == "playback":
        core.response_started(first["_quiet_generation"])
    elif boundary == "reset":
        core.reset()
    elif boundary == "different_turn":
        generation[0] += 1
    elif boundary == "expired":
        submitted = core._pending_voice[3]
        monkeypatch.setattr("glados.core.speech_listener.time.monotonic", lambda: submitted + 31)
    else:
        core.currently_speaking_event.set()
    say(core, 0.2)
    second = pending.get_nowait()
    assert second["_voice_turn_id"] != first["_voice_turn_id"]
    assert not second["_voice_continuation"]
    assert not np.any(np.isclose(decode(second), 0.1, atol=1 / 32768))


def test_stale_playback_acknowledgement_cannot_clear_new_pending_audio():
    core, pending, _ = listener()
    say(core, 0.1)
    first = pending.get_nowait()
    say(core, 0.2)
    second = pending.get_nowait()
    core.response_started(first["_quiet_generation"])
    assert core._pending_voice[1] == second["_quiet_generation"]


def test_combined_native_clip_limit_never_sends_truncated_request():
    core, pending, _ = listener(max_duration=1)
    say(core, 0.1)  # 192ms speech + 416ms silence.
    pending.get_nowait()
    say(core, 0.2)
    assert pending.empty()
    assert core._pending_voice is None and not core._recording_started


def test_parakeet_retranscribes_combined_samples():
    core, pending, _ = listener(native=False)
    clips = []
    core.asr = lambda samples: clips.append(np.concatenate(samples)) or "Transcript"
    say(core, 0.1)
    first = pending.get_nowait()
    say(core, 0.2)
    second = pending.get_nowait()
    assert second["_voice_turn_id"] == first["_voice_turn_id"]
    np.testing.assert_array_equal(clips[1][: len(clips[0])], clips[0])


def test_voice_history_replaces_partial_text_without_persisting_audio(tmp_path):
    path = tmp_path / "history.json"
    store = ConversationStore(path=path)
    store.append_voice_input({"role": "user", "content": "What should I"}, "utterance")
    store.remove_voice_input("utterance")
    assert not store.snapshot()
    store.append_voice_input({"role": "user", "content": "What should I have for dinner?"}, "utterance")
    store.append_voice_input({"role": "user", "content": "What should I have for dinner tomorrow?"}, "utterance")
    assert len(store.snapshot()) == 1
    assert store.snapshot()[0]["content"].endswith("tomorrow?")
    restored = ConversationStore(path=path)
    assert restored.records()[0].voice_turn_id == "utterance"
    assert "_native_audio" not in path.read_text()


def test_processor_withdraws_partial_voice_input_before_building_request(monkeypatch):
    from tests.test_speech_markup import make_processor

    processor = make_processor()
    processor._conversation_store = ConversationStore()
    while not processor.llm_input_queue.empty():
        processor.llm_input_queue.get_nowait()
    processor._conversation_store.append_voice_input({"role": "user", "content": "PARTIAL EARLIER SEGMENT"}, "same")
    processor.llm_input_queue.put(
        {
            "role": "user",
            "content": "Complete continued question",
            "_voice_turn_id": "same",
            "_voice_continuation": True,
            "_allow_tools": False,
        }
    )
    payloads = []

    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        processor.shutdown_event.set()
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        response.iter_lines.return_value = iter(())
        return response

    monkeypatch.setattr("glados.core.llm_processor.requests.post", post)
    processor.run()
    assert payloads
    assert "PARTIAL EARLIER SEGMENT" not in str(payloads)
    users = [m for m in processor._conversation_store.snapshot() if m["role"] == "user"]
    assert users[-1] == {"role": "user", "content": "Complete continued question"}
    assert sum(m["content"] == "Complete continued question" for m in users) == 1
    assert all(m["content"] != "PARTIAL EARLIER SEGMENT" for m in users)


@pytest.mark.parametrize("cancelled", [False, True])
def test_playback_boundary_acknowledges_only_current_generation(cancelled):
    from glados.core.speech_player import SpeechPlayer
    from glados.core.audio_data import AudioMessage

    entered, shutdown = threading.Event(), threading.Event()
    lock = threading.RLock()

    class Gate:
        def __enter__(self):
            entered.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    generation = [1]
    audio = MagicMock()
    acknowledged = []

    def play(*args):
        assert acknowledged == [1]
        shutdown.set()

    audio.start_speaking.side_effect = play
    audio.measure_percentage_spoken.return_value = (False, 100)
    messages = queue.Queue()
    messages.put(AudioMessage(audio=np.ones(512, dtype=np.float32), text="Answer", generation=1))
    player = SpeechPlayer(
        audio,
        messages,
        ConversationStore(),
        16000,
        shutdown,
        threading.Event(),
        threading.Event(),
        0.001,
        quiet_generation=lambda: generation[0],
        on_response_started=acknowledged.append,
        playback_lock=Gate(),
    )
    with lock:
        worker = threading.Thread(target=player.run)
        worker.start()
        assert entered.wait(2)
        if cancelled:
            generation[0] = 2
            shutdown.set()
    worker.join(2)
    assert not worker.is_alive()
    assert acknowledged == ([] if cancelled else [1])
    assert audio.start_speaking.call_count == (0 if cancelled else 1)
