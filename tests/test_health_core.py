"""Health context remains fresh, bounded and truthful without per-question probes."""

from copy import deepcopy
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from glados.autonomy.agents.health_agent import HealthAgent, HealthConfig, collect_host_status
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.slots import TaskSlotStore
from glados.core.context import ContextBuilder
from glados.core.decision_lists import default_list
from glados.core.routing_tree import RoutingTree
from glados.observability import ObservabilityBus
from glados.tools import tool_definitions

HealthFixture = tuple[HealthAgent, list[float], dict, dict, ObservabilityBus]


@pytest.fixture
def health() -> HealthFixture:
    time = [100.0]
    host = {
        "os": "Linux",
        "kernel": "test",
        "cpu_count": 4,
        "load": [1, 2, 3],
        "uptime_s": 7200,
        "ram": {"total_bytes": 16 * 1024**3, "available_bytes": 8 * 1024**3, "used_percent": 50},
        "disks": [{"path": "/", "free_bytes": 20 * 1024**3, "total_bytes": 40 * 1024**3, "free_percent": 50}],
        "gpus": [{"name": "test", "total_mib": 10240, "used_mib": 8500, "free_mib": 1300, "temperature_c": 60}],
        "logs": [],
        "max_temperature_c": 60,
    }
    runtime = {
        "inference": {"capacity": 4, "active": 0, "waiting": 0, "oldest_wait_s": 0},
        "mcp": [{"name": "search", "connected": True}],
        "audio": {"enabled": False, "connected": False},
    }
    bus = ObservabilityBus()
    core = HealthAgent(
        health_config=HealthConfig(startup_grace_s=0),
        completion_url="http://test/v1/chat/completions",
        runtime_status=lambda: deepcopy(runtime),
        host_status=lambda: deepcopy(host),
        clock=lambda: time[0],
        slot_store=TaskSlotStore(),
        observability_bus=bus,
    )
    return core, time, host, runtime, bus


def test_cached_context_has_freshness_and_never_runs_new_probe(health: HealthFixture) -> None:
    core, clock, _, _, _ = health
    probe = core._host_status = Mock(wraps=core._host_status)
    core.run(core.runtime)
    builder = ContextBuilder()
    builder.register("health", core.as_prompt, volatile=True)
    for _ in range(10):
        entry = builder.build_system_entries()[0]
        assert entry["source"] == "health" and entry["volatile"]
        assert "available_gib" in entry["message"]["content"]
    assert probe.call_count == 1
    assert core.covered_metrics() >= {"uptime", "cpu_load", "memory_usage", "disk_usage", "system_info"}
    clock[0] += 31
    assert core.snapshot()["stale"]
    assert core.covered_metrics() == set()
    assert "STALE" in core.as_prompt()
    clock[0] -= 31
    core.set_paused(True)
    assert core.covered_metrics() == set()


def test_alerts_only_emit_on_condition_transitions(health: HealthFixture) -> None:
    core, _, host, runtime, bus = health
    core.run(core.runtime)
    assert not bus.snapshot()
    host["ram"]["used_percent"] = 96
    runtime["audio"] = {"enabled": True, "connected": False}
    assert core.run(core.runtime).notify_user
    assert len(bus.snapshot()) == 2
    host["ram"]["used_percent"] = 97
    assert not core.run(core.runtime).notify_user
    assert len(bus.snapshot()) == 2
    host["ram"]["used_percent"] = 50
    runtime["audio"]["enabled"] = False  # Muted capture is not a failed microphone.
    core.run(core.runtime)
    assert [e.kind for e in bus.snapshot()] == ["alert", "alert", "recovered", "recovered"]
    assert core.snapshot()["alerts"] == []


def test_gpu_headroom_and_log_threshold_alerts(health: HealthFixture) -> None:
    core, _, host, _, bus = health
    host["gpus"][0]["free_mib"] = 200
    host["logs"] = [{"name": "capture.log", "bytes": 21 * 1024**2, "files": 3}]
    core.run(core.runtime)
    assert {e.kind for e in bus.snapshot()} == {"alert"}
    assert any("GPU" in a for a in core.snapshot()["alerts"])
    assert any("Log family" in a for a in core.snapshot()["alerts"])


def test_alert_identity_survives_resampling_but_changes_after_recovery(health: HealthFixture) -> None:
    core, clock, host, _, _ = health
    core.settings.summary_enabled = False
    host["gpus"][0]["temperature_c"] = 97
    core.runtime.publish(core.run(core.runtime))
    first = core._slot_store.get_slot("health")
    assert first.attention_key and first.notify_user
    clock[0] += 10
    host["gpus"][0]["temperature_c"] = 98
    core.runtime.publish(core.run(core.runtime))
    second = core._slot_store.get_slot("health")
    assert second.attention_key == first.attention_key and not second.notify_user
    clock[0] += 10
    host["gpus"][0]["temperature_c"] = 60
    core.runtime.publish(core.run(core.runtime))
    assert core._slot_store.get_slot("health").attention_key is None
    clock[0] += 10
    host["gpus"][0]["temperature_c"] = 99
    core.runtime.publish(core.run(core.runtime))
    assert core._slot_store.get_slot("health").attention_key != first.attention_key


def test_requested_capture_is_monitored_even_if_stream_never_started(health: HealthFixture) -> None:
    core, _, _, runtime, bus = health
    runtime["audio"] = {"enabled": False, "expected": True, "connected": False}
    core.run(core.runtime)
    assert len(bus.snapshot()) == 1
    runtime["audio"]["expected"] = False  # A muted microphone is intentionally unused.
    runtime["audio"]["enabled"] = True
    core.run(core.runtime)
    assert core.snapshot()["alerts"] == []
    assert bus.snapshot()[-1].kind == "recovered"


def test_periodic_summary_is_cached_and_yields_to_interaction(
    health: HealthFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, clock, _, _, _ = health
    core.llm = LLMConfig(url="http://test", model="E4B")
    summarize = Mock(return_value="Load is steady; no monitored alerts.")
    monkeypatch.setattr("glados.autonomy.agents.health_agent.llm_call", summarize)
    core.run(core.runtime)
    core._summary_thread.join(2)
    assert summarize.call_count == 1
    assert core.snapshot()["comment"] == summarize.return_value
    assert summarize.call_args.args[0].lane == "autonomy"
    assert summarize.call_args.args[0].request_options["max_tokens"] == 128
    for _ in range(5):
        clock[0] += 10
        core.run(core.runtime)
        core.as_prompt()
    assert summarize.call_count == 1
    clock[0] += 10
    core._interactive_busy = lambda: True
    core.run(core.runtime)
    assert summarize.call_count == 1
    core._interactive_busy = lambda: False
    core.run(core.runtime)
    core._summary_thread.join(2)
    assert summarize.call_count == 2
    for _ in range(20):
        clock[0] += 10
        core.run(core.runtime)
        core._summary_thread.join(2)
    assert len(core._samples) == 12
    assert len(json.loads(summarize.call_args.args[2])) <= 12


def test_unavailable_readings_are_not_removed_from_routing(health: HealthFixture) -> None:
    core, _, host, _, _ = health
    host["ram"] = None
    host["max_temperature_c"] = None
    core.run(core.runtime)
    assert "memory_usage" not in core.covered_metrics()
    assert "temperatures" not in core.covered_metrics()
    assert "system_overview" not in core.covered_metrics()


def test_health_context_replaces_metric_branches_but_keeps_other_tools(health: HealthFixture) -> None:
    core, clock, _, _, _ = health
    core.run(core.runtime)
    tool = {
        "type": "function",
        "function": {
            "name": "mcp.system_info.memory_usage",
            "description": "Read RAM",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    tools = [*tool_definitions, tool]
    decision = default_list(include_commands=True)
    tree = RoutingTree(decision, tools, [], core.covered_metrics())
    options = [option for node in tree.nodes.values() for option in node.decision.options]
    assert not any(o.tool == "run_safe_command" and o.arguments.get("task") == "memory_usage" for o in options)
    assert tool["function"]["name"] not in tree.tools
    assert "vision_look" in tree.tools and "get_time" in tree.tools and "manage_slot" in tree.tools
    assert "quiet_control" in tree.nodes
    assert "ordinary reply" in tree.nodes["area"].decision.instructions
    clock[0] += 31
    tree = RoutingTree(decision, tools, [], core.covered_metrics())
    options = [option for node in tree.nodes.values() for option in node.decision.options]
    assert any(o.tool == "run_safe_command" and o.arguments.get("task") == "memory_usage" for o in options)
    assert tool["function"]["name"] in tree.tools


def test_log_probe_is_bounded_to_known_file_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.autonomy.agents.health_agent.shutil.which", lambda _: None)
    log = tmp_path / "capture.log"
    log.write_text("abcd")
    (tmp_path / "capture.log.1").write_text("123")
    (tmp_path / "unrelated.log").write_text("x" * 1000)
    host = collect_host_status([str(log)])
    assert host["logs"] == [{"name": "capture.log", "bytes": 7, "files": 2}]


def test_slow_commentary_does_not_block_sampling_or_start_duplicate_calls(
    health: HealthFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, clock, host, _, _ = health
    core.llm = LLMConfig(url="http://test", model="E4B")
    entered, release = threading.Event(), threading.Event()

    def slow(*_args: object) -> str:
        entered.set()
        assert release.wait(2)
        return "Earlier samples were stable."

    summarize = Mock(side_effect=slow)
    monkeypatch.setattr("glados.autonomy.agents.health_agent.llm_call", summarize)
    try:
        core.run(core.runtime)
        assert entered.wait(1)
        clock[0] += 100
        host["ram"]["used_percent"] = 55
        core.run(core.runtime)
        assert core.snapshot()["host"]["ram"]["used_percent"] == 55
        assert core.snapshot()["age_s"] == 0
        assert summarize.call_count == 1
    finally:
        release.set()
        core._summary_thread.join(2)
    assert core.snapshot()["comment_source_at"] == 100
    assert core.snapshot()["comment_at"] == 200
