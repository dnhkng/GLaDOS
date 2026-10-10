import io
import queue
import threading

from glados.autonomy.interaction_state import InteractionState
from glados.core.text_listener import TextListener
from glados.observability import ObservabilityBus


def test_text_listener_enqueues_lines() -> None:
    llm_queue: queue.Queue[dict[str, str]] = queue.Queue()
    processing_active_event = threading.Event()
    shutdown_event = threading.Event()
    interaction_state = InteractionState()
    input_stream = io.StringIO("hello\n\nworld\n")

    listener = TextListener(
        llm_queue=llm_queue,
        processing_active_event=processing_active_event,
        shutdown_event=shutdown_event,
        pause_time=0.01,
        interaction_state=interaction_state,
        input_stream=input_stream,
    )

    listener.run()

    assert llm_queue.qsize() == 2
    first = llm_queue.get_nowait()
    second = llm_queue.get_nowait()
    assert first["content"] == "hello"
    assert second["content"] == "world"
    assert processing_active_event.is_set()
    assert interaction_state.seconds_since_user() is not None


def test_text_listener_handles_commands() -> None:
    llm_queue: queue.Queue[dict[str, str]] = queue.Queue()
    processing_active_event = threading.Event()
    shutdown_event = threading.Event()
    input_stream = io.StringIO("/help\nhello\n")
    called: list[str] = []

    def handler(command: str) -> str:
        called.append(command)
        return "ok"

    listener = TextListener(
        llm_queue=llm_queue,
        processing_active_event=processing_active_event,
        shutdown_event=shutdown_event,
        pause_time=0.01,
        input_stream=input_stream,
        command_handler=handler,
    )

    listener.run()

    assert called == ["/help"]
    assert llm_queue.qsize() == 1
    message = llm_queue.get_nowait()
    assert message["content"] == "hello"


def test_input_events_identify_the_queued_turn() -> None:
    bus = ObservabilityBus()
    pending: queue.Queue = queue.Queue()
    generations = iter((41, 42))
    listener = TextListener(
        llm_queue=pending,
        processing_active_event=threading.Event(),
        shutdown_event=threading.Event(),
        pause_time=0.01,
        input_stream=io.StringIO("hello\nworld\n"),
        observability_bus=bus,
        begin_user_turn=lambda: next(generations),
    )
    listener.run()
    events = bus.drain()
    assert [event.meta["generation"] for event in events] == [
        pending.get_nowait()["_quiet_generation"], pending.get_nowait()["_quiet_generation"],
    ] == [41, 42]


def test_engine_constructs_text_and_both_input_modes(tmp_path, monkeypatch) -> None:
    """Exercise the engine's real listener kwargs without models or worker startup."""
    from unittest.mock import Mock
    from glados.autonomy.config import AutonomyConfig
    from glados.core.engine import Glados
    from glados.autonomy.agents.health_agent import HealthConfig
    from glados.core.routing import RoutingConfig
    from glados.autonomy.agents.search_agent import SearchConfig

    monkeypatch.setattr("glados.core.engine.resource_path", lambda name: tmp_path / name)
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    for mode in ("text", "both"):
        config = AutonomyConfig(enabled=False)
        config.emotion.enabled = False
        config.tokens.enabled = False
        config.tokens.recall.enabled = False
        config.tokens.state_path = None
        engine = Glados(asr_model=None, tts_model=Mock(sample_rate=16000), audio_io=Mock(),
            completion_url="http://test/v1/chat/completions", llm_model="test", input_mode=mode,
            autonomy_config=config, health_config=HealthConfig(enabled=False),
            search_config=SearchConfig(enabled=False), routing_config=RoutingConfig(enabled=False))
        assert isinstance(engine.text_listener, TextListener)
        assert (engine.speech_listener is not None) == (mode == "both")
        engine.shutdown_event.set()
        engine.autonomy_tasks.shutdown()
        engine.tool_executor._tool_pool.shutdown(wait=False)
