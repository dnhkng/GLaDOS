"""Time-based recovery is independent of tick frequency and model reactions."""

from unittest.mock import Mock

import pytest

from glados.autonomy.agents.emotion_agent import EmotionAgent
from glados.autonomy.emotion_state import EmotionEvent
from tests.test_quiet_emotion import emotion_agent


def set_extreme(agent: EmotionAgent) -> None:
    agent._state.last_update = 1000.0
    for name, value in [("pleasure", -1.0), ("arousal", 1.0), ("dominance", 1.0)]:
        setattr(agent._state, name, value)
        setattr(agent._state, "mood_" + name, value)


def test_extreme_is_95_percent_neutral_after_six_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = emotion_agent(monkeypatch)
    set_extreme(agent)
    agent._apply_baseline_drift(1360.0)
    assert agent.state.pleasure == pytest.approx(-0.05)
    assert agent.state.arousal == pytest.approx(0.05)
    assert agent.state.dominance == pytest.approx(0.05)
    assert agent.state.mood_pleasure == pytest.approx(-0.05)
    assert agent.state.mood_arousal == pytest.approx(0.05)
    assert agent.state.mood_dominance == pytest.approx(0.05)


@pytest.mark.parametrize("times", [list(range(1001, 1361)), [1013, 1111, 1260, 1360], [1360]])
def test_regular_and_delayed_ticks_follow_same_curve(monkeypatch: pytest.MonkeyPatch, times: list[int]) -> None:
    agent = emotion_agent(monkeypatch)
    set_extreme(agent)
    for now in times:
        agent._apply_baseline_drift(now)
    assert agent.state.pleasure == pytest.approx(-0.05)
    assert agent.state.dominance == pytest.approx(0.05)


def test_backward_clock_never_amplifies_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = emotion_agent(monkeypatch)
    set_extreme(agent)
    previous = agent.state.to_dict()
    agent._apply_baseline_drift(900)
    assert agent.state.to_dict() == previous


def test_decay_is_applied_before_new_input_after_pause(monkeypatch):
    agent = emotion_agent(monkeypatch)
    set_extreme(agent)
    agent.set_paused(True)
    model = Mock(return_value=None)
    monkeypatch.setattr(agent, "_ask_llm", model)
    monkeypatch.setattr("glados.autonomy.agents.emotion_agent.time.time", lambda: 1360.)
    agent.set_paused(False)
    agent.react("Hello again")
    model.assert_not_called()
    agent.run(agent.runtime)
    assert model.call_args.kwargs["state"].pleasure == pytest.approx(-.05)


def test_idle_ticks_decay_without_inference(monkeypatch):
    agent = emotion_agent(monkeypatch)
    set_extreme(agent)
    model = Mock(return_value=None)
    monkeypatch.setattr(agent, "_ask_llm", model)
    clock = [1000.]
    monkeypatch.setattr("glados.autonomy.agents.emotion_agent.time.time", lambda: clock[0])
    monkeypatch.setattr("glados.autonomy.agents.emotion_agent.time.monotonic", lambda: clock[0])
    for now in range(1000, 1361):
        clock[0] = now
        agent.run(agent.runtime)
    model.assert_not_called()
    assert agent.state.pleasure == pytest.approx(-.05)


def test_interaction_work_is_separate_from_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = emotion_agent(monkeypatch)
    model = Mock(side_effect=lambda *args, **kwargs: kwargs["state"])
    monkeypatch.setattr(agent, "_ask_llm", model)
    trigger = Mock()
    agent.runtime.request_run = trigger
    agent.react("Hello")
    model.assert_not_called()
    trigger.assert_called_once()
    agent.run(agent.runtime)
    assert not agent._events
    agent.run(agent.runtime)  # An idle run only applies decay.
    model.assert_called_once()


def test_failed_scores_retain_events_for_retry(monkeypatch):
    agent = emotion_agent(monkeypatch)
    model = Mock(return_value=None)
    monkeypatch.setattr(agent, "_ask_llm", model)
    agent.react("Hello")
    agent.run(agent.runtime)
    assert len(agent._events) == 1
    agent.run(agent.runtime)
    assert model.call_count == 2  # Scheduler decides when this retry happens.


def test_parallel_axis_scores_and_superseded_results(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    module = "glados.autonomy.agents.emotion_agent."
    agent = emotion_agent(monkeypatch)
    barrier = threading.Barrier(4)
    release = threading.Event()
    lanes, requests = [], []
    class Scheduler:
        @contextmanager
        def lease(self, owner, lane, model, cancelled):
            lanes.append(lane)
            yield
    agent._llm_config.scheduler = Scheduler()
    monkeypatch.setattr(module + "token_ids", lambda *a: dict(zip("ABCDE", range(5))))
    def scores(url, headers, data, ids, timeout):
        requests.append(data)
        barrier.wait(timeout=2)
        assert release.wait(2)
        return [.0, .2, .6, .2, .0]
    monkeypatch.setattr(module + "request_scores", scores)
    agent.react("First input")
    with ThreadPoolExecutor(1) as worker:
        tick = worker.submit(agent.run, agent.runtime)
        barrier.wait(timeout=2)
        # This would deadlock if inference held the update lock.
        agent.react("Newer input")
        release.set()
        assert tick.result(timeout=2) is None
    assert lanes == ["autonomy"] * 3
    assert all(r["max_tokens"] == 1 for r in requests)
    assert all("First input" not in r["messages"][0]["content"] for r in requests)
    assert agent.state.pleasure == 0
    assert len(agent._events) == 2  # Latest input remains available for the pending run.
    agent.on_stop()


def test_probability_weighted_axes_and_pause_rejects_inflight(monkeypatch):
    module = "glados.autonomy.agents.emotion_agent."
    agent = emotion_agent(monkeypatch)
    monkeypatch.setattr(module + "token_ids", lambda *a: dict(zip("ABCDE", range(5))))
    monkeypatch.setattr(module + "request_scores", lambda *a: [.5, .2, .1, .1, .1])
    agent.react("An event")
    agent.run(agent.runtime)
    assert agent.state.pleasure == pytest.approx(-.45)
    assert agent.state.arousal == pytest.approx(-.45)
    assert not agent._events
    def pause(*args):
        agent.set_paused(True)
        return [.0, .0, .0, .0, 1.]
    monkeypatch.setattr(module + "request_scores", pause)
    agent.react("Second event")
    agent.run(agent.runtime)
    assert agent.state.pleasure < 0
    agent.on_stop()
