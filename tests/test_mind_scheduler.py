"""Timing, bounded concurrency and controls are independent of mind domain work."""

from collections.abc import Callable, Iterator
import threading
from unittest.mock import Mock

import pytest

from glados.autonomy.mind_runtime import MindRuntime
from glados.autonomy.mind_schedule import AdaptiveInterval, FixedInterval, OnDemand, RandomAdaptive
from glados.autonomy.mind_scheduler import MindScheduler, SubagentStatus
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.subagent import Subagent, SubagentConfig, SubagentOutput
from glados.observability import MindRegistry
from tests.test_autonomy_core import wait_until


class Worker(Subagent):
    def __init__(
        self,
        name: str,
        store: TaskSlotStore,
        work: Callable[[MindRuntime], SubagentOutput | None] = lambda runtime: None,
    ) -> None:
        super().__init__(SubagentConfig(name, name), store)
        self.work = work
        self.started = Mock()
        self.stopped = Mock()

    def run(self, runtime: MindRuntime) -> SubagentOutput | None:
        return self.work(runtime)

    def on_start(self) -> None:
        self.started()

    def on_stop(self) -> None:
        self.stopped()


SchedulerFixture = tuple[MindScheduler, TaskSlotStore, MindRegistry]


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> Iterator[SchedulerFixture]:
    monkeypatch.setattr("glados.autonomy.subagent.SubagentMemory", Mock())
    store, registry = TaskSlotStore(), MindRegistry()
    runtime = MindScheduler(store, registry, max_workers=2)
    yield runtime, store, registry
    runtime.shutdown(timeout=2)


def status(scheduler: MindScheduler, name: str) -> SubagentStatus:
    return next(s for s in scheduler.list_agents() if s.agent_id == name)


def test_fixed_timer_resets_after_completion_and_long_missed_time_coalesces(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    clock = [100.0]
    scheduler._clock = lambda: clock[0]
    finished = threading.Event()

    def work(runtime: MindRuntime) -> SubagentOutput | None:
        clock[0] += 2  # Work/inference time is not part of the five-second idle timer.
        finished.set()

    mind = Worker("emotion", store, work)
    scheduler.register(mind, FixedInterval(5), run_on_start=False)
    scheduler.start_all()
    scheduler.trigger("emotion", manual=False)
    assert finished.wait(2)
    wait_until(lambda: status(scheduler, "emotion").status == "waiting")
    assert status(scheduler, "emotion").next_due_in_s == 5
    clock[0] += 1000  # No catch-up runs for all the missed intervals.
    finished.clear()
    scheduler.reschedule("emotion")
    assert finished.wait(2)
    wait_until(lambda: status(scheduler, "emotion").status == "waiting")
    assert status(scheduler, "emotion").tick_count == 2
    assert status(scheduler, "emotion").next_due_in_s == 5


def test_random_adaptive_draws_once_and_motion_shortens_pending_deadline() -> None:
    activity = [0.0]
    draw = Mock(return_value=0.5)
    policy = RandomAdaptive(lambda: (2, 5), lambda: activity[0], draw)
    policy.reset()
    quiet = policy.delay()
    assert quiet > 4
    assert policy.delay() == quiet
    activity[0] = 1
    assert 2 <= policy.delay() < 3
    assert draw.call_count == 1
    policy.reset()
    assert draw.call_count == 2


def test_random_adaptive_bias_and_configurable_bounds() -> None:
    bounds, activity = [2, 5], [0.0]
    policy = RandomAdaptive(lambda: tuple(bounds), lambda: activity[0])
    quiet, active = [], []
    for index in range(100):
        policy.quantile = (index + 0.5) / 100
        activity[0] = 0
        quiet.append(policy.delay())
        activity[0] = 1
        active.append(policy.delay())
    assert sum(quiet) / 100 > 4
    assert sum(active) / 100 < 3
    bounds[:] = [6, 8]
    assert 6 <= policy.delay() <= 8


def test_adaptive_presence_changes_pending_schedule(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    clock, present = [100.0], [False]
    scheduler._clock = lambda: clock[0]
    scheduler.register(Worker("presence", store), AdaptiveInterval(lambda: 2 if present[0] else 30), run_on_start=False)
    scheduler.start_all()
    assert status(scheduler, "presence").next_due_in_s == 30
    clock[0] += 1
    present[0] = True
    assert status(scheduler, "presence").next_due_in_s == 1
    assert status(scheduler, "presence").tick_count == 0


def test_slow_mind_does_not_block_other_minds_and_triggers_coalesce(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def slow(runtime: MindRuntime) -> SubagentOutput:
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        else:
            second.set()
        return SubagentOutput("done", "Finished")

    scheduler.register(Worker("slow", store, slow), OnDemand(), run_on_start=False)
    quick = threading.Event()
    scheduler.register(Worker("quick", store, lambda runtime: quick.set()), OnDemand(), run_on_start=False)
    scheduler.start_all()
    scheduler.trigger("slow")
    assert entered.wait(2)
    for _ in range(100):
        scheduler.trigger("slow")
    scheduler.trigger("quick")
    assert quick.wait(2), "Timer dispatch must not execute slow work inline"
    release.set()
    assert second.wait(2)
    wait_until(lambda: status(scheduler, "slow").status == "waiting")
    assert len(calls) == 2
    assert status(scheduler, "slow").next_due_in_s is None


def test_pause_cancels_old_result_even_if_resumed_before_completion(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    entered, release = threading.Event(), threading.Event()

    def work(runtime: MindRuntime) -> SubagentOutput | None:
        entered.set()
        assert release.wait(2)
        return SubagentOutput("done", "Obsolete result")

    scheduler.register(Worker("test", store, work), OnDemand(), run_on_start=False)
    scheduler.start_all()
    scheduler.trigger("test")
    assert entered.wait(2)
    scheduler.pause("test")
    scheduler.resume("test")
    release.set()
    wait_until(lambda: status(scheduler, "test").status == "waiting")
    assert store.get_slot("test") is None


def test_stop_is_local_and_does_not_set_shared_engine_shutdown(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    shutdown = threading.Event()
    first = Worker("first", store)
    first.runtime.shutdown = shutdown
    scheduler.register(first, OnDemand())
    ran = threading.Event()
    scheduler.register(Worker("second", store, lambda runtime: ran.set()), OnDemand(), run_on_start=False)
    scheduler.start_all()
    scheduler.stop("first")
    assert not shutdown.is_set()
    first.stopped.assert_called_once()
    scheduler.trigger("second")
    assert ran.wait(2)
    with pytest.raises(ValueError):
        scheduler.trigger("first")


def test_worker_capacity_queue_is_visible_and_no_overlap(scheduler: SchedulerFixture) -> None:
    scheduler, store, registry = scheduler
    release = threading.Event()
    for name in ("first", "second", "third"):
        scheduler.register(Worker(name, store, lambda runtime: release.wait(2)), OnDemand())
    scheduler.start_all()
    try:
        wait_until(lambda: status(scheduler, "third").status == "queued")
        assert sum(s.status == "running" for s in scheduler.list_agents()) == 2
        assert next(s for s in registry.snapshot() if s.mind_id == "third").status == "queued"
    finally:
        release.set()
    wait_until(lambda: status(scheduler, "third").tick_count == 1)


def test_error_is_isolated_and_progress_and_result_share_slot(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler

    def broken(runtime: MindRuntime) -> None:
        raise ValueError("probe failed")

    scheduler.register(Worker("bad", store, broken), OnDemand())

    def working(runtime: MindRuntime) -> SubagentOutput:
        runtime.publish(SubagentOutput("running", "Progress", context="Useful fact"))
        assert store.get_slot("good").context == "Useful fact"
        return SubagentOutput("done", "Result", report="Details", update_priority="important")

    scheduler.register(Worker("good", store, working), OnDemand())
    scheduler.start_all()
    wait_until(
        lambda: store.get_slot("bad") is not None
        and store.get_slot("good") is not None
        and store.get_slot("good").status == "done"
    )
    assert store.get_slot("bad").status == "error"
    assert store.get_slot("good").report == "Details"


def test_shutdown_discards_late_work_and_finishes_cleanup(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    entered, release = threading.Event(), threading.Event()

    def work(runtime: MindRuntime) -> SubagentOutput | None:
        entered.set()
        release.wait(2)
        return SubagentOutput("done", "Too late")

    mind = Worker("test", store, work)
    scheduler.register(mind, OnDemand())
    scheduler.start_all()
    assert entered.wait(2)
    scheduler.shutdown(timeout=0.01)
    release.set()
    wait_until(lambda: not mind.runtime.executing)
    wait_until(lambda: mind.stopped.call_count == 1)
    assert store.get_slot("test") is None


@pytest.mark.parametrize("delay", [0, -1, float("inf"), float("nan")])
def test_invalid_intervals_rejected(delay: float) -> None:
    with pytest.raises(ValueError):
        FixedInterval(delay)
    with pytest.raises(ValueError):
        AdaptiveInterval(lambda: delay).delay()


def test_stop_during_startup_cannot_reactivate_a_mind(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    entered, release = threading.Event(), threading.Event()
    mind = Worker("test", store)

    def initialize() -> None:
        entered.set()
        assert release.wait(2)

    mind.on_start = initialize
    scheduler.register(mind, OnDemand())
    starter = threading.Thread(target=scheduler.start_all)
    starter.start()
    assert entered.wait(2)
    scheduler.stop("test", timeout=0.01)
    release.set()
    starter.join(2)
    assert not starter.is_alive() and not mind.is_running
    mind.stopped.assert_called_once()
    assert status(scheduler, "test").tick_count == 0


def test_failed_live_signal_does_not_kill_other_minds(scheduler: SchedulerFixture) -> None:
    scheduler, store, _ = scheduler
    delay = [10.0]
    scheduler.register(Worker("bad", store), AdaptiveInterval(lambda: delay[0]), run_on_start=False)
    ran = threading.Event()
    scheduler.register(Worker("good", store, lambda runtime: ran.set()), OnDemand(), run_on_start=False)
    scheduler.start_all()
    delay[0] = float("nan")
    scheduler.reschedule("bad")
    wait_until(lambda: status(scheduler, "bad").status == "error")
    scheduler.trigger("good")
    assert ran.wait(2)
