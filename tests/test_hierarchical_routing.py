"""Capability selection, live MCP injection, and scoped tool execution."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from glados.core.decision_lists import DecisionListStore, DecisionOption
from glados.core.inference import InferenceScheduler
from glados.core.routing import DecisionRouter, RoutingConfig
from glados.core.routing_tree import identity
from glados.mcp.config import MCPServerConfig
from glados.mcp.manager import MCPManager, MCPToolEntry
from glados.tools import tool_definitions


def definition(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]},
        },
    }


@pytest.fixture
def router(tmp_path: Path) -> DecisionRouter:
    tools = [
        *tool_definitions,
        definition("mcp.lights.turn_on", "Turn on a named light"),
        definition("mcp.weather.forecast", "Get the weather for a location"),
    ]
    store = DecisionListStore(tmp_path / "choices.json", lambda: tools, enabled=True)
    catalog = [
        {"name": name, "description": description, "tools": [{"name": tool, "description": description}]}
        for name, tool, description in [
            ("lights", "mcp.lights.turn_on", "Control named household lights"),
            ("weather", "mcp.weather.forecast", "Read weather forecasts"),
        ]
    ]
    result = DecisionRouter(
        store,
        InferenceScheduler(),
        "http://test/v1/chat/completions",
        "E4B",
        {},
        RoutingConfig(),
        mcp_catalog=lambda: catalog,
    )
    result.token_ids = lambda labels: {label: i for i, label in enumerate(labels)}
    return result


def scripted_scores(
    router: DecisionRouter,
    monkeypatch: pytest.MonkeyPatch,
    choices: list[tuple[str, str]],
    uncertain: int | None = None,
) -> Mock:
    tree = router.tree(router.store.get())
    count = 0

    def respond(*args: object, **kwargs: object) -> Mock:
        nonlocal count
        node_id, option_id = choices[count]
        node = tree.nodes[node_id]
        options = node.decision.options
        index = next(i for i, option in enumerate(options) if option.id == option_id)
        data = kwargs["json"]
        assert data["max_tokens"] == 1
        assert data["top_logprobs"] == len(options)
        assert node.decision.name in data["messages"][0]["content"]
        assert options[index].description in data["messages"][0]["content"]
        probabilities = [0.001] * len(options)
        probabilities[index] = 0.99 if count != uncertain else 0.5
        if count == uncertain:
            probabilities[(index + 1) % len(options)] = 0.49
        count += 1
        return Mock(
            json=lambda: {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_probs": [
                                        {"id": i, "prob": probability} for i, probability in enumerate(probabilities)
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        )

    post = Mock(side_effect=respond)
    monkeypatch.setattr("glados.core.routing.requests.post", post)
    return post


def test_local_time_reply_uses_clock_without_tool_permit(router: DecisionRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    post = scripted_scores(router, monkeypatch, [("area", "reply")])
    result = router.score(router.store.get(), "What time is it?")
    assert result["action"] == "reply" and result["tool"] is None
    assert result["context_source"] is None and len(result["stages"]) == 1
    assert result["category"] is None and result["tool_scope"] == []
    assert not router.store.authorize(result)
    assert post.call_count == 1 and router.latest is result
    assert router.scheduler.snapshot()["active"] == []


def test_router_rules_stay_identical_when_history_changes(
    router: DecisionRouter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = scripted_scores(router, monkeypatch, [("area", "reply")]*3)
    decision = router.store.get()
    router.score(decision, "What time is it?", context=[{"role": "assistant", "content": "Earlier answer"}])
    audio = [{"type": "input_audio", "input_audio": {"data": "recording", "format": "wav"}}]
    router.score(decision, "What is the date?", context=[{"role": "assistant", "content": "New answer"}])
    router.score(decision, "", audio, context=[{"role": "assistant", "content": "New answer"}])
    payloads = [call.kwargs["json"] for call in post.call_args_list]
    assert payloads[0]["messages"][0] == payloads[1]["messages"][0]
    assert "Typed input" in payloads[0]["messages"][0]["content"]
    assert "Spoken input" in payloads[2]["messages"][0]["content"]
    assert "Earlier answer" in payloads[0]["messages"][-1]["content"]
    assert "New answer" in payloads[1]["messages"][-1]["content"]
    assert "New answer" in payloads[2]["messages"][-1]["content"][0]["text"]
    assert payloads[2]["messages"][-1]["content"][-1] == audio[0]


def test_visual_question_keeps_original_audio_and_scopes_to_vision(
    router: DecisionRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = scripted_scores(router, monkeypatch, [("area", "area_vision")])
    audio = [{"type": "input_audio", "input_audio": {"format": "wav", "data": "test-audio"}}]
    result = router.score(router.store.get(), "[Speech]", audio=audio, dry_run=True)
    assert result["action"] == "plan" and result["tool_scope"] == ["vision_look"]
    assert result["tool"] is None and result["arguments"] == {}
    assert result["category"] == "vision" and post.call_count == 1
    assert router.store.authorize_scope(result, "vision_look")
    assert not router.store.authorize_scope(result, "set_preference")
    assert all(call.kwargs["json"]["messages"][-1]["content"][-1] == audio[0] for call in post.call_args_list)
    assert not router.store.path.exists(), "A preview must not execute or change settings"


def test_single_tool_mcp_selects_server_and_injects_live_capabilities(
    router: DecisionRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = scripted_scores(
        router,
        monkeypatch,
        [
            ("area", "area_mcp"),
            ("mcp", identity("choose_", "lights")),
        ],
    )
    result = router.score(router.store.get(), "Turn on the kitchen light", dry_run=True)
    assert result["server"] == "lights" and len(result["stages"]) == 2
    assert result["tool_scope"] == ["mcp.lights.turn_on"] and result["action"] == "plan"
    root_prompt = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
    assert "Control named household lights" in root_prompt and "Read weather forecasts" in root_prompt
    assert not router.store.authorize_scope(result, "mcp.weather.forecast")
    assert "Kitchen" not in str(result["arguments"]), "Single-token routing never invents arguments"


@pytest.mark.parametrize("uncertain", [0, 1])
def test_uncertainty_stops_the_pipeline_without_tool_authority(
    router: DecisionRouter,
    monkeypatch: pytest.MonkeyPatch,
    uncertain: int,
) -> None:
    post = scripted_scores(router, monkeypatch, [("area", "area_system"), ("system", "command_cpu_load")], uncertain=uncertain)
    result = router.score(router.store.get(), "Maybe do it")
    assert result["action"] == "assist" and not result["accepted"]
    assert result["tool"] is None and result["tool_scope"] == []
    assert post.call_count == uncertain + 1
    assert not router.store.authorize_scope(result, "get_time")


def test_saved_edits_revoke_scoped_calls(router: DecisionRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_scores(router, monkeypatch, [("area", "area_memory"), ("memory", identity("tool_", "set_preference"))])
    result = router.score(router.store.get(), "Remember that I prefer English")
    assert router.store.authorize_scope(result, "set_preference")
    settings = router.store.snapshot()
    router.store.mutate({"action": "save", "revision": settings["revision"], "list": settings["lists"][0]})
    assert not router.store.authorize_scope(result, "set_preference")


def test_unavailable_tools_and_large_registries_keep_short_choices(router: DecisionRouter) -> None:
    tools = [definition(f"mcp.large.action_{i}", f"Perform action number {i}") for i in range(100)]
    router.store.tools = lambda: tools
    tree = router.tree(router.store.get())
    assert all(2 <= len(node.decision.options) <= 19 for node in tree.nodes.values())
    assert "vision" not in tree.nodes and "system" not in tree.nodes
    assert len(tree.nodes) > 5
    router.store.tools = lambda: []
    assert not router.tree(router.store.get()).catalog


def test_mcp_manager_catalog_uses_registered_tools_and_excludes_configuration_secrets() -> None:
    manager = MCPManager(
        [
            MCPServerConfig(name="lights", description="Household lamps", token="secret"),
            MCPServerConfig(name="offline", command="secret-command"),
        ]
    )
    manager._tool_registry = {"mcp.lights.turn_on": MCPToolEntry("lights", "turn_on", "Turn on a light", {})}
    catalog = manager.get_routing_catalog()
    assert catalog == [
        {
            "name": "lights",
            "description": "Household lamps",
            "category": "mcp",
            "tools": [{"name": "mcp.lights.turn_on", "description": "Turn on a light"}],
        }
    ]
    manager._remove_tools_for_server("lights")
    assert manager.get_routing_catalog() == []


def test_executor_rejects_a_call_when_its_scoped_settings_changed(
    router: DecisionRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import queue
    import threading

    from glados.core.tool_executor import ToolExecutor

    scripted_scores(
        router,
        monkeypatch,
        [
            ("area", "area_mcp"),
            ("mcp", identity("choose_", "lights")),
        ],
    )
    result = router.score(router.store.get(), "Turn on the kitchen light")
    permit = {k: result[k] for k in ("list_id", "revision", "settings_revision", "tool_scope")}
    settings = router.store.snapshot()
    router.store.mutate({"action": "save", "revision": settings["revision"], "list": settings["lists"][0]})
    priority, autonomy, calls = queue.Queue(), queue.Queue(), queue.Queue()
    active, shutdown = threading.Event(), threading.Event()
    active.set()
    manager = Mock()
    executor = ToolExecutor(
        priority, autonomy, calls, active, shutdown, mcp_manager=manager, decision_store=router.store
    )
    calls.put(
        {
            "id": "pending",
            "function": {"name": "mcp.lights.turn_on", "arguments": '{"target":"kitchen"}'},
            "_routing_permit": permit,
        }
    )
    worker = threading.Thread(target=executor.run)
    worker.start()
    try:
        output = priority.get(timeout=2)
        assert "cancelled" in output["content"] and not output["_allow_tools"]
        manager.call_tool.assert_not_called()
    finally:
        shutdown.set()
        worker.join(2)


def test_sleep_branch_checks_affirmation_before_changing_mode(router: DecisionRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_scores(router, monkeypatch, [("area", "quiet_mode_enter"), ("quiet_control", "proceed")])
    result = router.score(router.store.get(), "Don't go to sleep")
    assert result["action"] == "reply" and len(result["stages"]) == 2


def test_named_timezone_keeps_timezone_tool_scope(router: DecisionRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_scores(router, monkeypatch, [("area", "area_system"), ("system", identity("tool_", "get_time"))])
    result = router.score(router.store.get(), "What time is it in Tokyo?")
    assert result["action"] == "plan" and result["tool_scope"] == ["get_time"]
    assert result["context_source"] is None
    assert router.store.authorize_scope(result, "get_time")
    assert result["category"] == "system"


def test_clock_choice_is_absent_and_legacy_context_aliases_are_ignored(router: DecisionRouter) -> None:
    decision = router.store.get()
    decision.options.append(DecisionOption(id="legacy_clock", action="reply", context_source="clock",
                                           description="Local clock from context"))
    tree = router.tree(decision)
    assert "clock" not in tree.nodes and "area_clock" not in tree.nodes["area"].children
    assert not any(o.context_source == "clock" for o in tree.nodes["area"].decision.options)
    assert any(o.id == "reply" for o in tree.nodes["area"].decision.options)
    assert tree.nodes["system"].scopes[identity("tool_", "get_time")] == ["get_time"]


def test_saved_timezone_binding_moves_from_system_and_retains_permit(
    router: DecisionRouter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = router.store.get()
    decision.options.append(DecisionOption(
        id="tokyo_clock", action="tool", tool="get_time", category="system",
        arguments={"timezone": "Asia/Tokyo"}, description="Read the current Tokyo time",
    ))
    settings = router.store.snapshot()
    router.store.mutate({"action": "save", "revision": settings["revision"], "list": decision.model_dump()})
    scripted_scores(router, monkeypatch, [("area", "area_system"), ("system", "tokyo_clock")])
    result = router.score(router.store.get(), "What time is it in Tokyo?")
    assert result["category"] == "system" and result["action"] == "tool"
    assert result["arguments"] == {"timezone": "Asia/Tokyo"} and router.store.authorize(result)


@pytest.mark.parametrize("branch", ["area", "system", "mcp"])
@pytest.mark.parametrize("fallback", ["assist", "clarify"])
def test_explicit_no_match_uses_fallback_without_tool_authority(
    router: DecisionRouter, monkeypatch: pytest.MonkeyPatch, branch: str, fallback: str,
) -> None:
    decision = router.store.get().model_copy(update={"fallback": fallback})
    choices = [] if branch == "area" else [("area", "area_" + branch)]
    scripted_scores(router, monkeypatch, [*choices, (branch, "back_to_assistant")])
    result = router.score(decision, "This does not match these capabilities")
    assert result["action"] == fallback and not result["accepted"]
    assert result["fallback_selected"] and result["tool_scope"] == []
    assert result["tool"] is None and result["arguments"] == {}
    assert not router.store.authorize_scope(result, "run_safe_command")
    node = next(n for n in router.tree(decision).snapshot()["nodes"] if n["id"] == branch)
    assert next(o for o in node["options"] if o["fallback"])["display_name"] == "No matching choice"


def test_multi_tool_mcp_retains_tool_selection(router: DecisionRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [*router.store.tools(), definition("mcp.lights.turn_off", "Turn off a named light")]
    router.store.tools = lambda: tools
    post = scripted_scores(router, monkeypatch, [
        ("area", "area_mcp"), ("mcp", identity("choose_", "lights")),
        (identity("server_", "lights"), identity("tool_", "mcp.lights.turn_off")),
    ])
    result = router.score(router.store.get(), "Turn off the kitchen light")
    assert result["server"] == "lights" and post.call_count == 3
    assert result["tool_scope"] == ["mcp.lights.turn_off"]
    assert not router.store.authorize_scope(result, "mcp.lights.turn_on")


def test_slow_clap_prefers_mcp_and_retains_offline_or_explicit_local_binding(router: DecisionRouter) -> None:
    local_tools = router.store.tools()
    tools = [*local_tools, definition("mcp.slow_clap_demo.slow_clap", "Perform a slow clap")]
    router.store.tools = lambda: tools
    decision = router.store.get()
    tree = router.tree(decision)
    assert "slow clap" not in tree.tools and "other" not in tree.nodes
    mcp_choice = identity("choose_", "slow_clap_demo")
    assert tree.nodes["mcp"].scopes[mcp_choice] == ["mcp.slow_clap_demo.slow_clap"]
    assert tree.server_choices[mcp_choice] == "slow_clap_demo"
    explicit = decision.model_copy(update={"options": [*decision.options, DecisionOption(
        id="local_clap", action="tool", tool="slow clap", arguments={"claps": 2},
        description="Play the local slow-clap audio twice",
    )]})
    assert "local_clap" in router.tree(explicit).nodes["other"].bindings
    router.store.tools = lambda: local_tools
    offline = router.tree(decision)
    assert offline.nodes["area"].scopes["area_other"] == ["slow clap"]
    assert all(name not in offline.tools for name in ("speak", "do_nothing"))


def test_fixed_binding_to_no_argument_tool_has_no_duplicate_dynamic_choice(router: DecisionRouter) -> None:
    tools = [{"type": "function", "function": {
        "name": "get_preferences", "description": "Read saved preferences",
        "parameters": {"type": "object", "properties": {}},
    }}]
    router.store.tools = lambda: tools
    decision = router.store.get()
    decision.options.append(DecisionOption(
        id="read_preferences", action="tool", tool="get_preferences", arguments={},
        description="Read saved preferences",
    ))
    tree = router.tree(decision)
    node = tree.nodes["memory"]
    assert [o.id for o in node.decision.options] == ["read_preferences", node.fallback_id]
    assert "read_preferences" in node.bindings and not node.scopes


def test_fixed_system_choice_retains_saved_execution_permit(
    router: DecisionRouter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_scores(router, monkeypatch, [("area", "area_system"), ("system", "command_uptime")])
    result = router.score(router.store.get(), "How long has this machine been running?")
    assert result["action"] == "tool" and result["tool"] == "run_safe_command"
    assert result["arguments"] == {"task": "uptime"}
    assert result["option_id"] == "command_uptime" and router.store.authorize(result)


def test_collapsed_vision_scope_is_revoked_after_saved_edit(
    router: DecisionRouter, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_scores(router, monkeypatch, [("area", "area_vision")])
    result = router.score(router.store.get(), "What can you see?")
    assert router.store.authorize_scope(result, "vision_look")
    settings = router.store.snapshot()
    router.store.mutate({"action": "save", "revision": settings["revision"], "list": settings["lists"][0]})
    assert not router.store.authorize_scope(result, "vision_look")
