import io
import queue
import threading

from glados.autonomy.interaction_state import InteractionState
from glados.core.text_listener import TextListener


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
