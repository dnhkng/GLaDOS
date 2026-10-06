"""A conversation blocks new background inference, not routing or live monitoring."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
import queue

import numpy as np
import pytest

from glados.core.inference import InferenceConfig, InferenceScheduler, InferenceCancelledError
from glados.core.speech_markup import SpeechText
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from tests.test_voice_continuation import listener, say


def test_hold_admission_and_reserved_slots():
    s = InferenceScheduler(InferenceConfig(slots=4, reserved_interactive=2))
    running = s.acquire('Vision', 'autonomy', 'test')
    s.begin_interaction(1)
    assert s.try_acquire('Memory', 'autonomy', 'test') is None
    assert s.try_acquire('Future core', 'unknown', 'test') is None
    leases = [s.acquire(lane, lane, 'test') for lane in ['router', 'priority', 'speculative']]
    assert running in s._active.values()  # Active inference is allowed to finish.
    for lease in leases + [running]:
        s.release(lease)
    s.end_interaction(1, 'first_response_ready')
    assert s.try_acquire('Memory', 'autonomy', 'test').slot == 2


def test_waiter_does_not_block_routing_or_draft_and_resumes():
    s = InferenceScheduler()
    s.begin_interaction(1)
    admitted = threading.Event()
    cancelled = threading.Event()
    def background():
        try:
            with s.lease('Vision', 'autonomy', 'test', cancelled.is_set):
                admitted.set()
        except InferenceCancelledError:
            pass
    thread = threading.Thread(target=background)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while not s.snapshot()['waiting'] and time.monotonic() < deadline:
            time.sleep(.001)
        assert s.snapshot()['waiting'][0]['wait_reason'] == 'user_response'
        with s.lease('Routing', 'router', 'test'):
            draft = s.try_acquire('Draft', 'speculative', 'test')
            assert draft is not None
            s.release(draft)
        assert not admitted.is_set()
        s.end_interaction(1, 'first_response_ready')
        assert admitted.wait(1)
    finally:
        cancelled.set()
        thread.join(1)
    assert not thread.is_alive()


def test_stale_release_and_timeout(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('glados.core.inference.time.monotonic', lambda: clock[0])
    s = InferenceScheduler()
    s.begin_interaction(1)
    s.begin_interaction(2)
    s.end_interaction(1, 'late_audio')
    assert s.snapshot()['interaction_hold']['generation'] == 2
    clock[0] = 120
    assert s.try_acquire('Memory', 'autonomy', 'test') is not None
    assert s.snapshot()['last_interaction_release']['reason'] == 'safety_timeout'


@pytest.mark.parametrize('native', [True, False])
def test_voice_hold_survives_submission_and_continuation(native):
    core, pending, generation = listener(native=native)
    s = InferenceScheduler()
    old_begin = core._begin_user_turn
    def begin():
        g = old_begin()
        s.begin_interaction(g)
        return g
    core._begin_user_turn = begin
    core._end_user_turn = s.end_interaction
    core.asr = lambda samples: 'Hello'
    say(core, .1)
    assert not pending.empty()
    assert s.snapshot()['interaction_hold']['generation'] == 1
    say(core, .2)
    s.end_interaction(1, 'late_audio')
    assert s.snapshot()['interaction_hold']['generation'] == 2


@pytest.mark.parametrize('outcome', ['empty', 'failure', 'reset'])
def test_abandoned_recording_releases(outcome):
    core, _, _ = listener(native=False)
    s = InferenceScheduler()
    def begin():
        s.begin_interaction(1)
        return 1
    core._begin_user_turn = begin
    core._end_user_turn = s.end_interaction
    if outcome == 'failure':
        core.asr = MagicMock(side_effect=RuntimeError('ASR failed'))
        with pytest.raises(RuntimeError):
            say(core, .1)
    elif outcome == 'empty':
        core.asr = lambda samples: ''
        say(core, .1)
    else:
        core._turn_generation = begin()
        core.reset()
    assert s.snapshot()['interaction_hold'] is None


@pytest.mark.parametrize('muted', [True, False])
def test_first_audio_ready_releases_without_waiting_for_playback(muted):
    s = InferenceScheduler()
    s.begin_interaction(1)
    shutdown = threading.Event()
    pending, output = queue.Queue(), queue.Queue()
    pending.put(SpeechText('Hello.', generation=1))
    mute = threading.Event()
    if muted:
        mute.set()
    model = MagicMock(sample_rate=16000)
    def synthesize(text):
        assert s.snapshot()['interaction_hold'] is not None
        return np.ones(100, dtype=np.float32)
    model.generate_speech_audio.side_effect = synthesize
    def ready(generation, reason):
        assert not output.empty()
        s.end_interaction(generation, reason)
        shutdown.set()
    TextToSpeechSynthesizer(pending, output, model, SimpleNamespace(text_to_spoken=lambda s: s),
        shutdown, .001, tts_muted_event=mute, quiet_generation=lambda: 1,
        on_response_ready=ready).run()
    assert s.snapshot()['interaction_hold'] is None
    assert model.generate_speech_audio.call_count == (0 if muted else 1)


def test_synthesis_failure_releases_hold():
    s = InferenceScheduler()
    s.begin_interaction(1)
    shutdown = threading.Event()
    pending = queue.Queue()
    pending.put(SpeechText('Hello.', generation=1))
    model = MagicMock(sample_rate=16000)
    model.generate_speech_audio.side_effect = RuntimeError('synthesis failed')
    def ready(generation, reason):
        s.end_interaction(generation, reason)
        shutdown.set()
    TextToSpeechSynthesizer(pending, queue.Queue(), model, SimpleNamespace(text_to_spoken=lambda s: s),
        shutdown, .001, quiet_generation=lambda: 1, on_response_ready=ready).run()
    assert s.snapshot()['last_interaction_release']['reason'] == 'synthesis_error'


@pytest.mark.parametrize('tool_handoff', [False, True])
def test_processor_terminal_response_vs_tool_handoff(monkeypatch, tool_handoff):
    import json
    from tests.test_speech_markup import make_processor
    processor = make_processor()
    s = processor._inference_scheduler = InferenceScheduler()
    s.begin_interaction(0)
    processor.llm_input_queue.put({'role': 'user', 'content': 'Hello', '_quiet_generation': 0})
    processor._build_tools = lambda autonomy: [{'type': 'function', 'function': {'name': 'test_tool'}}]
    def lines(**kwargs):
        if tool_handoff:
            delta = {'tool_calls': [{'index': 0, 'id': 'call_test', 'type': 'function',
                'function': {'name': 'test_tool', 'arguments': '{}'}}]}
            yield b'data: ' + json.dumps({'choices': [{'delta': delta}]}).encode()
        processor.shutdown_event.set()
        yield b'data: [DONE]'
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.side_effect = lines
    monkeypatch.setattr('glados.core.llm_processor.requests.post', lambda *args, **kwargs: response)
    processor.run()
    assert bool(s.snapshot()['interaction_hold']) == tool_handoff
    assert (not processor.tool_calls_queue.empty()) == tool_handoff


def test_held_request_can_be_cancelled_without_releasing_new_turn():
    s = InferenceScheduler()
    s.begin_interaction(1)
    with pytest.raises(InferenceCancelledError):
        s.acquire('Memory', 'autonomy', 'test', lambda: True)
    assert not s.snapshot()['waiting']
    assert s.snapshot()['interaction_hold']['generation'] == 1
