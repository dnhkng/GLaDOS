"""Building the background cores (subagents) and registering them with their schedules.

This is wiring only, kept out of the engine: each builder reads the engine's configuration
and shared services, constructs one core and registers it. Adding a core means writing one
builder and listing it in ``_CORES`` or ``_JOBS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..autonomy import SubagentConfig
from ..autonomy.agents import CompactionAgent, EmotionAgent, HackerNewsSubagent, ObserverAgent, WeatherSubagent
from ..autonomy.agents.health_agent import HealthAgent
from ..autonomy.agents.nexus_agent import NexusAgent
from ..autonomy.agents.search_agent import SearchAgent
from ..autonomy.llm_client import LLMConfig
from ..autonomy.mind_schedule import FixedInterval, OnDemand, RandomAdaptive
from ..utils.resources import resource_path

if TYPE_CHECKING:
    from ..autonomy import MindScheduler
    from ..mcp import MCPManager
    from ..vision import VisionConfig
    from .engine import Glados

# Each builder constructs one core from the engine's configuration and registers it.
Builder = Callable[["Glados", "MindScheduler", LLMConfig], None]


def register_background_cores(engine: Glados) -> None:
    """Register every configured core; jobs only run when autonomy and jobs are both enabled."""
    manager = engine.subagent_manager
    if not manager:
        return
    llm = _conversation_llm(engine)
    for build in _CORES:
        build(engine, manager, llm)
    if engine.autonomy_config.enabled and engine.autonomy_config.jobs.enabled:
        for build in _JOBS:
            build(engine, manager, llm)


# ---------------------------------------------------------------- shared wiring
def _conversation_llm(engine: Glados) -> LLMConfig:
    """The conversation model, shared by the cores that do not need vision."""
    return LLMConfig(
        url=str(engine.completion_url),
        api_key=engine.api_key,
        model=engine.llm_model,
        timeout=getattr(engine, "llm_timeout", None) or LLMConfig.timeout,
        request_options=engine.llm_request_options,
        scheduler=engine.inference_scheduler,
        shutdown_event=engine.shutdown_event,
        cancelled=engine.quiet_event.is_set,
    )


def _vision_llm(engine: Glados, vision: VisionConfig, owner: str) -> LLMConfig:
    return LLMConfig(
        url=vision.completion_url,
        model=vision.model,
        api_key=vision.api_key,
        timeout=vision.timeout_s,
        owner=owner,
        scheduler=engine.inference_scheduler,
        shutdown_event=engine.shutdown_event,
        cancelled=engine.quiet_event.is_set,
    )


def _services(engine: Glados) -> dict[str, Any]:
    """What every core shares: where it reports, and when to stop."""
    return {
        "slot_store": engine.autonomy_slots,
        "mind_registry": engine.mind_registry,
        "observability_bus": engine.observability_bus,
        "shutdown_event": engine.shutdown_event,
    }


def _interactive_busy(engine: Glados) -> Callable[[], bool]:
    """Whether someone is talking to GLaDOS, so background work should leave the model alone."""

    def busy() -> bool:
        inflight = getattr(engine, "_priority_inflight", None)
        queue = getattr(engine, "llm_queue_priority", None)
        return (
            engine.quiet_event.is_set()
            or bool(inflight and inflight.value())
            or bool(queue is not None and not queue.empty())
        )

    return busy


def _mcp(engine: Glados) -> MCPManager:
    if engine.mcp_manager is None:
        raise RuntimeError("Search Core needs MCP, and no MCP servers are configured")
    return engine.mcp_manager


# ---------------------------------------------------------------------- cores
def _health(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    health = getattr(engine, "health_config", None)
    if not (health and health.enabled):
        return
    engine.health_agent = HealthAgent(
        health_config=health,
        completion_url=str(engine.completion_url),
        runtime_status=engine._health_runtime_status,
        llm_config=llm,
        interactive_busy=_interactive_busy(engine),
        **_services(engine),
    )
    manager.register(engine.health_agent, FixedInterval(health.interval_s))


def _search(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    search = getattr(engine, "search_config", None)
    if not (search and search.enabled):
        return
    engine.search_agent = SearchAgent(
        settings=search,
        llm_config=llm,
        settings_path=resource_path("data/search_settings.yaml"),
        search=lambda arguments, timeout: _mcp(engine).call_tool(
            "mcp.internet_search.web_search_exa", arguments, timeout=timeout
        ),
        **_services(engine),
    )
    manager.register(engine.search_agent, OnDemand())


def _nexus(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    nexus = getattr(engine, "nexus_config", None)
    if not (nexus and nexus.enabled):
        return
    local_headers = {"Content-Type": "application/json", **(getattr(engine, "_llm_headers", None) or {})}
    if engine.api_key:
        local_headers["Authorization"] = f"Bearer {engine.api_key}"
    engine.nexus_agent = NexusAgent(
        settings=nexus,
        local_backend={
            "url": str(engine.completion_url),
            "model": engine.llm_model,
            "headers": local_headers,
            "request_options": dict(engine.llm_request_options),
        },
        set_backend=lambda backend: engine.llm_processor.set_backend(backend),
        idle=lambda: not engine._autonomy_user_busy(),
        settings_path=resource_path("data/nexus_settings.yaml"),
        # Native audio needs the local multimodal endpoint, so Central cannot move to the API.
        base_switching=engine.native_audio is None,
        **_services(engine),
    )
    engine.context_builder.register("nexus", engine.nexus_agent.as_prompt, priority=7, volatile=True)
    manager.register(engine.nexus_agent, FixedInterval(30))


def _vision(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    # Vision stays on E4B even when conversation uses a remote API model.
    vision = engine.vision_config
    if engine.vision_state is None or vision is None:
        return
    from ..vision.vision_mind import VisionMind

    agent = engine.vision_agent = VisionMind(
        vision_config=vision,
        llm_config=_vision_llm(engine, vision, "Vision"),
        vision_state=engine.vision_state,
        settings_path=resource_path("data/vision_settings.yaml"),
        greetings_path=resource_path("data/vision_greetings.json"),
        **_services(engine),
    )
    manager.register(
        agent,
        RandomAdaptive(
            bounds=lambda: (agent.settings.interval_min_s, agent.settings.interval_max_s),
            activity=lambda: agent.camera.motion.snapshot()["activity"],
        ),
    )


def _emotion(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    # Emotional regulation is independent of background job scheduling.
    emotion = engine.autonomy_config.emotion
    if not emotion.enabled:
        return
    agent = engine._emotion_agent = EmotionAgent(
        config=SubagentConfig(agent_id="emotion", title="Emotion Core", role="emotional_regulation"),
        llm_config=replace(llm, owner="emotion"),
        emotion_config=emotion,
        **_services(engine),
    )
    manager.register(agent, FixedInterval(emotion.tick_interval_s))


def _memory(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    # Context maintenance is a background Mind even with autonomous speech OFF.
    tokens = engine.autonomy_config.tokens
    if not (tokens.enabled or tokens.recall.enabled):
        return
    threshold = tokens.token_threshold
    if tokens.model_context_window:
        threshold = min(threshold, int(tokens.model_context_window * tokens.target_utilization))
    # Memory work runs on the vision model when there is one, keeping the conversation model free.
    vision = engine.vision_config
    engine.compaction_agent = CompactionAgent(
        config=SubagentConfig(agent_id="compaction", title="Memory Core", role="Recall and context management"),
        llm_config=_vision_llm(engine, vision, "Compaction")
        if vision is not None and vision.enabled
        else replace(llm, owner="Compaction"),
        conversation_store=engine._conversation_store,
        token_threshold=threshold,
        preserve_recent=tokens.preserve_recent_messages,
        summary_max_tokens=tokens.summary_max_tokens,
        summary_input_tokens=tokens.summary_input_tokens,
        recall_config=tokens.recall,
        compaction_enabled=tokens.enabled,
        interactive_busy=_interactive_busy(engine),
        **_services(engine),
    )
    manager.register(engine.compaction_agent, FixedInterval(tokens.tick_interval_s))


# ----------------------------------------------------------------------- jobs
def _hacker_news(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    news = engine.autonomy_config.jobs.hacker_news
    if not news.enabled:
        return
    agent = HackerNewsSubagent(
        config=SubagentConfig(agent_id="hn_top", title="Hacker News", role="news_monitor"),
        top_n=news.top_n,
        min_score=news.min_score,
        llm_config=replace(llm, owner="hn_top"),
        **_services(engine),
    )
    manager.register(agent, FixedInterval(news.interval_s))


def _weather(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    weather = engine.autonomy_config.jobs.weather
    if not weather.enabled:
        return
    if weather.latitude is None or weather.longitude is None:
        logger.warning("Weather subagent enabled but latitude/longitude are missing.")
        return
    agent = WeatherSubagent(
        config=SubagentConfig(agent_id="weather", title="Weather", role="weather_monitor"),
        latitude=weather.latitude,
        longitude=weather.longitude,
        timezone=weather.timezone,
        temp_change_c=weather.temp_change_c,
        wind_alert_kmh=weather.wind_alert_kmh,
        llm_config=replace(llm, owner="weather"),
        **_services(engine),
    )
    manager.register(agent, FixedInterval(weather.interval_s))


def _observer(engine: Glados, manager: MindScheduler, llm: LLMConfig) -> None:
    # Monitors behaviour and proposes adjustments.
    agent = ObserverAgent(
        config=SubagentConfig(agent_id="observer", title="Behavior Observer", role="meta_supervision"),
        llm_config=replace(llm, owner="observer"),
        conversation_store=engine._conversation_store,
        constitutional_state=engine.constitutional_state,
        sample_count=10,
        min_samples_for_analysis=5,
        **_services(engine),
    )
    manager.register(agent, FixedInterval(120), run_on_start=False)


_CORES: tuple[Builder, ...] = (_health, _search, _nexus, _vision, _emotion, _memory)
_JOBS: tuple[Builder, ...] = (_hacker_news, _weather, _observer)
