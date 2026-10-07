"""Directions survive streaming and synthesis but are only performed at playback."""

from collections.abc import Iterator
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from glados.core.audio_data import AudioMessage
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.speech_animation import SpeechAnimationState
from glados.core.speech_markup import SPEECH_DIRECTION_PROMPT, SpeechMarkupParser, SpeechText
from glados.core.speech_player import SpeechPlayer
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from glados.observability import ObservabilityBus


def grouped(segments: list[SpeechText]) -> list[SpeechText]:
    result = []
    for segment in segments:
        if result and result[-1].emotion == segment.emotion:
            result[-1] = SpeechText(result[-1].text + segment.text, segment.emotion)
        else:
            result.append(segment)
    return result


def test_every_stream_split_preserves_speech_and_directions() -> None:
    text = "Hello [test]. [emotion:smug]Well done. [emotion:disappointed]For a human."
    expected = [
        SpeechText("Hello [test]. "),
        SpeechText("Well done. ", "smug"),
        SpeechText("For a human.", "disappointed"),
    ]
    for boundary in range(len(text) + 1):
        parser = SpeechMarkupParser()
        assert grouped(parser.feed(text[:boundary]) + parser.feed(text[boundary:], final=True)) == expected
    parser = SpeechMarkupParser()
    assert (
        grouped([segment for char in text for segment in parser.feed(char)] + parser.feed("", final=True)) == expected
    )


def test_unknown_and_incomplete_tags_are_silent_and_plain_text_survives() -> None:
    parser = SpeechMarkupParser()
    assert grouped(parser.feed("[emotion:SMUG]One.[emotion:invalid]Two.[emotion:disapp", final=True)) == [
        SpeechText("One.Two.", "smug")
    ]
    assert SpeechMarkupParser().feed("Read [chapter one] and [", final=True) == [SpeechText("Read [chapter one] and [")]
    assert SpeechMarkupParser().feed("plain response", final=True) == [SpeechText("plain response")]


def make_processor() -> LanguageModelProcessor:
    active = threading.Event()
    active.set()
    return LanguageModelProcessor(
        llm_input_queue=queue.Queue(),
        tool_calls_queue=queue.Queue(),
        tts_input_queue=queue.Queue(),
        conversation_store=MagicMock(snapshot=lambda: [{"role": "user", "content": "Hello"}]),
        completion_url="http://localhost/v1/chat/completions",
        model_name="test",
        api_key=None,
        processing_active_event=active,
        shutdown_event=threading.Event(),
    )


def test_prompt_is_only_added_to_user_facing_requests() -> None:
    processor = make_processor()
    assert any(message["content"] == SPEECH_DIRECTION_PROMPT for message in processor._build_messages(False))
    assert all(message["content"] != SPEECH_DIRECTION_PROMPT for message in processor._build_messages(True))


def test_llm_stream_to_tts_keeps_directions_out_of_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": False})

    def lines(chunk_size: int = 1) -> Iterator[bytes]:
        for char in "[emotion:smug]Well done. [emotion:disappointed]For a human.":
            yield b"data: " + json.dumps({"choices": [{"delta": {"content": char}}]}).encode()
        processor.shutdown_event.set()

    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.side_effect = lines
    monkeypatch.setattr("glados.core.llm_processor.requests.post", lambda *args, **kwargs: response)
    processor.run()
    items = list(processor.tts_input_queue.queue)
    assert items[-1] == SpeechText("<EOS>", generation=0)
    speech = [item for item in items if isinstance(item, SpeechText) and item.text.strip() and item.text != "<EOS>"]
    assert [(item.text.strip(), item.emotion) for item in speech] == [
        ("Well done.", "smug"),
        ("For a human.", "disappointed"),
    ]


def test_tts_preserves_metadata_and_legacy_strings() -> None:
    shutdown = threading.Event()
    pending = queue.Queue()
    output = queue.Queue()
    for item in (SpeechText("Excellent.", "smug"), "[emotion:disappointed]For a human.", "<EOS>"):
        pending.put(item)
    original_put = output.put

    def put(item: AudioMessage) -> None:
        original_put(item)
        if item.is_eos:
            shutdown.set()

    output.put = put
    model = MagicMock(sample_rate=16000)
    model.generate_speech_audio.return_value = np.ones(100, dtype=np.float32)
    converter = SimpleNamespace(text_to_spoken=lambda text: text)
    TextToSpeechSynthesizer(pending, output, model, converter, shutdown, 0.001).run()
    messages = list(output.queue)
    assert [(item.text, item.emotion) for item in messages[:2]] == [
        ("Excellent.", "smug"),
        ("For a human.", "disappointed"),
    ]
    assert messages[2].is_eos
    assert [call.args[0] for call in model.generate_speech_audio.call_args_list] == ["Excellent.", "For a human."]


@pytest.mark.parametrize("outcome", ["complete", "interrupt", "error"])
def test_playback_controls_start_and_clear_on_every_exit(outcome: str) -> None:
    bus = ObservabilityBus()
    animation = SpeechAnimationState(bus)
    subscriber = bus.subscribe()
    shutdown = threading.Event()
    speaking = threading.Event()
    pending = queue.Queue()
    pending.put(AudioMessage(np.ones(100, dtype=np.float32), "Excellent work.", emotion="smug"))
    audio = MagicMock()

    def measure(*args: object) -> tuple[bool, float]:
        assert speaking.is_set()
        assert animation.snapshot()["emotion"] == "smug"
        assert animation.snapshot()["active"] is True
        shutdown.set()
        if outcome == "error":
            raise RuntimeError("device failed")
        return outcome == "interrupt", 50 if outcome == "interrupt" else 100

    audio.measure_percentage_spoken.side_effect = measure
    player = SpeechPlayer(
        audio, pending, MagicMock(), 16000, shutdown, speaking, threading.Event(), 0.001, speech_animation=animation
    )
    player.run()
    assert not speaking.is_set()
    assert animation.snapshot()["active"] is False
    assert animation.snapshot()["emotion"] is None
    events = list(subscriber.queue)
    assert [(event.meta["active"], event.meta["emotion"]) for event in events] == [(True, "smug"), (False, None)]
    assert [event.meta["revision"] for event in events] == [1, 2]
    animation.set(False)
    assert subscriber.qsize() == 2


def test_bundled_sentence_reaches_tts_before_stream_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = make_processor()
    processor.llm_input_queue.put({"role": "user", "content": "Hello", "_allow_tools": False})

    def lines(*, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size == 1
        yield b'data: ' + json.dumps({"choices": [{"delta": {"content": "[emotion:smug]It works. More"}}]}).encode()
        spoken = list(processor.tts_input_queue.queue)
        assert any(isinstance(item, SpeechText) and item.text.strip() == "It works." for item in spoken)
        assert not any(isinstance(item, SpeechText) and "More" in item.text for item in spoken)
        yield b'data: ' + json.dumps({"choices": [{"delta": {"content": " words."}}]}).encode()
        processor.shutdown_event.set()

    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.side_effect = lines
    monkeypatch.setattr("glados.core.llm_processor.requests.post", lambda *args, **kwargs: response)
    processor.run()
    speech = [item.text.strip() for item in processor.tts_input_queue.queue
              if isinstance(item, SpeechText) and item.text.strip() and item.text != "<EOS>"]
    assert speech == ["It works.", "More words."]


@pytest.mark.parametrize("text, clauses, remainder", [
    ("Yes. Next", ["Yes."], " Next"),
    ("Done! Really?", ["Done!", " Really?"], ""),
    ("It is 3.", [], "It is 3."),
    ("It is 3.14 volts. Next", ["It is 3.14 volts."], " Next"),
    ("At 12:30 today.", ["At 12:30 today."], ""),
    ("See https://example.com/path. Later.", ["See https://example.com/path. Later."], ""),
    ('He said “done.” Next', ['He said “done.”'], ' Next'),
])
def test_speech_clauses(text: str, clauses: list[str], remainder: str) -> None:
    from glados.core.speech_chunking import split_speech_clauses
    assert split_speech_clauses(text) == (clauses, remainder)
