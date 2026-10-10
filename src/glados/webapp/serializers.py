"""Map live engine state objects to plain JSON-serializable dicts.

These helpers mirror the TUI's panels: they read the exact same thread-safe
accessors (MindRegistry, TaskSlotStore, SubagentManager, MCPManager, AudioState,
InteractionState, queue sizes, ...) and turn them into plain dicts for the
webapp. No state is duplicated - the console just reads what the engine tracks.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _plain(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays and dataclasses to JSON-safe values."""
    if _HAS_NUMPY and isinstance(value, np.generic):
        return value.item()
    if _HAS_NUMPY and isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def to_jsonable(obj: Any) -> Any:
    return _plain(obj)


def dumps(obj: Any) -> str:
    """Serialize to JSON without choking on dataclasses, numpy, or timestamps."""
    return json.dumps(obj, default=_json_default)


def _json_default(value: Any) -> Any:
    try:
        return _plain(value)
    except Exception:  # pragma: no cover
        return str(value)


def serialize_event(event: Any) -> dict[str, Any]:
    """ObservabilityEvent -> {timestamp, source, kind, level, message, meta}."""
    return {
        "timestamp": getattr(event, "timestamp", time.time()),
        "source": getattr(event, "source", ""),
        "kind": getattr(event, "kind", ""),
        "level": getattr(event, "level", "info"),
        "message": getattr(event, "message", ""),
        "meta": to_jsonable(getattr(event, "meta", {})),
    }


def serialize_mind(mind: Any) -> dict[str, Any]:
    return {
        "mind_id": mind.mind_id,
        "title": mind.title,
        "status": mind.status,
        "summary": mind.summary,
        "role": mind.role,
        "updated_at": mind.updated_at,
    }


def serialize_slot(slot: Any) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "title": slot.title,
        "status": slot.status,
        "summary": slot.summary,
        "notify_user": slot.notify_user,
        "update_priority": slot.update_priority,
        "turn_id": slot.turn_id,
        "revision": slot.revision,
        "owner_id": slot.owner_id,
        "handled": slot.handled,
        "queue_position": slot.queue_position,
        "attention_key": getattr(slot, "attention_key", None),
        "importance": slot.importance,
        "confidence": slot.confidence,
        "next_run": slot.next_run,
        "updated_at": slot.updated_at,
        "has_report": bool(slot.report),
    }


def serialize_slot_full(slot: Any) -> dict[str, Any]:
    data = serialize_slot(slot)
    data["report"] = slot.report
    data["context"] = getattr(slot, "context", None)
    return data


def serialize_memory_entry(entry: Any) -> dict[str, Any]:
    return {
        "key": entry.key,
        "value": to_jsonable(entry.value),
        "created_at": entry.created_at,
        "shown_at": entry.shown_at,
    }


def serialize_agent(agent_status: Any) -> dict[str, Any]:
    return {
        "agent_id": agent_status.agent_id,
        "title": agent_status.title,
        "running": agent_status.running,
        "tick_count": agent_status.tick_count,
        "last_tick": agent_status.last_tick,
    }


def build_audio_state(engine: Any) -> dict[str, Any]:
    audio_state = getattr(engine, "audio_state", None)
    if audio_state is None:
        return {"rms": 0.0, "vad_active": False, "updated_at": 0.0}
    snap = audio_state.snapshot()
    return {
        "rms": float(snap.rms), "vad_active": bool(snap.vad_active),
        "updated_at": float(getattr(snap, "updated_at", 0.0)),
    }


def _emotion_state(engine: Any) -> dict[str, Any] | None:
    agent = getattr(engine, "_emotion_agent", None)
    if agent is None:
        return None
    try:
        return {**to_jsonable(agent.state), "response_instructions": agent.state.response_instructions()}
    except Exception:  # pragma: no cover
        return None


def _mcp_summary(engine: Any) -> dict[str, Any]:
    manager = getattr(engine, "mcp_manager", None)
    if manager is None:
        return {"enabled": False, "servers": []}
    try:
        servers = manager.status_snapshot()
    except Exception:  # pragma: no cover
        servers = []
    return {"enabled": True, "servers": servers}


def build_lanes(engine: Any) -> dict[str, Any]:
    scheduler = getattr(engine, "inference_scheduler", None)
    state = scheduler.snapshot() if scheduler else None
    active = state["active"] if state else []
    waiting = state["waiting"] if state else []
    return {
        "enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "priority": {
            "queue": int(engine.llm_queue_priority.qsize()) + sum(r["lane"] != "autonomy" for r in waiting),
            "inflight": sum(r["lane"] != "autonomy" for r in active) if state else int(engine._priority_inflight.value()),
        },
        "autonomy": {
            "queue": int(engine.llm_queue_autonomy.qsize()) + sum(r["lane"] == "autonomy" for r in waiting),
            "inflight": sum(r["lane"] == "autonomy" for r in active) if state else int(engine._autonomy_inflight.value()),
            "workers": len(getattr(engine, "autonomy_llm_processors", ())),
        },
    }


def build_state(engine: Any) -> dict[str, Any]:
    """Lightweight payload streamed periodically to keep gauges/clock live."""
    return {
        "t": time.time(),
        "controls": build_controls(engine),
        "interaction": {
            "seconds_since_user": engine.interaction_state.seconds_since_user(),
            "seconds_since_assistant": engine.interaction_state.seconds_since_assistant(),
        },
        "lanes": build_lanes(engine),
        "inference": engine.inference_scheduler.snapshot() if getattr(engine, "inference_scheduler", None) else None,
        "routing": engine.router.snapshot() if getattr(engine, "router", None) else None,
        "command_slot": engine.command_runner.snapshot() if getattr(engine, "command_runner", None) else None,
        "autonomy_core": engine.autonomy_loop.snapshot() if getattr(engine, "autonomy_loop", None) else None,
        "vision_mind": engine.vision_agent.snapshot() if getattr(engine, "vision_agent", None) else None,
        "search_settings": (engine.search_agent.preferences.snapshot()
                            if getattr(engine, "search_agent", None) else None),
        "vision": engine.vision_state.snapshot() if getattr(engine, "vision_state", None) else None,
        "audio": build_audio_state(engine),
        "emotion": _emotion_state(engine),
        "mcp": _mcp_summary(engine),
        "speaking": bool(engine.currently_speaking_event.is_set()),
        "performance": engine.speech_animation.snapshot() if getattr(engine, "speech_animation", None) else None,
    }


def build_snapshot(engine: Any) -> dict[str, Any]:
    """Aggregate snapshot for the console's initial paint (GET /api/snapshot)."""
    snapshot: dict[str, Any] = {
        "t": time.time(),
        "controls": build_controls(engine),
        "operator": engine.operator_state.snapshot() if getattr(engine, "operator_state", None) else None,
        "version": "0.1",
        "autonomy_enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "lanes": build_lanes(engine),
        "inference": engine.inference_scheduler.snapshot() if getattr(engine, "inference_scheduler", None) else None,
        "routing": engine.router.snapshot() if getattr(engine, "router", None) else None,
        "command_slot": engine.command_runner.snapshot() if getattr(engine, "command_runner", None) else None,
        "autonomy_core": engine.autonomy_loop.snapshot() if getattr(engine, "autonomy_loop", None) else None,
        "vision_mind": engine.vision_agent.snapshot() if getattr(engine, "vision_agent", None) else None,
        "search_settings": (engine.search_agent.preferences.snapshot()
                            if getattr(engine, "search_agent", None) else None),
        "audio": build_audio_state(engine),
        "emotion": _emotion_state(engine),
        "mcp": _mcp_summary(engine),
        "interaction": {
            "seconds_since_user": engine.interaction_state.seconds_since_user(),
            "seconds_since_assistant": engine.interaction_state.seconds_since_assistant(),
        },
        "speaking": bool(engine.currently_speaking_event.is_set()),
        "performance": engine.speech_animation.snapshot() if getattr(engine, "speech_animation", None) else None,
        "minds": [_safe(serialize_mind, m) for m in engine.mind_registry.snapshot()],
        "slots": [_safe(serialize_slot, s) for s in getattr(engine, "autonomy_slots", None).list_slots()] if getattr(engine, "autonomy_slots", None) else [],
        "agents": (_safe_agents(engine) if getattr(engine, "subagent_manager", None) else []),
        "vision": getattr(engine, "vision_state", None).snapshot() if getattr(engine, "vision_state", None) else None,
        "commands": sorted(k for k in getattr(engine, "_command_order", ())),
    }
    snapshot["agent_minds"] = build_minds(engine)
    snapshot["tools"] = build_tools(engine)
    snapshot["decisions"] = engine.decision_lists.snapshot() if getattr(engine, "decision_lists", None) else None
    return snapshot


def build_tools(engine: Any) -> list[dict[str, Any]]:
    processor = getattr(engine, "llm_processor", None)
    if processor is None:
        return []
    return [tool["function"] for tool in processor._build_tools(False) if "function" in tool]


def build_context(engine: Any, mode: str = "user", view: str = "live") -> dict[str, Any]:
    """On-demand context inspection, outside the high-frequency telemetry stream."""
    if mode not in {"user", "autonomy"} or view not in {"live", "request"}:
        raise ValueError("Choose user/autonomy mode and live/request view")
    processor = getattr(engine, "llm_processor", None)
    if processor is None:
        return {"available": False, "reason": "Inference context is unavailable."}
    if view == "request":
        processors = [processor] if mode == "user" else getattr(engine, "autonomy_llm_processors", [])
        requests = [p.last_context() for p in processors if p and hasattr(p, "last_context")]
        matching = [r for r in requests if r and r["mode"] == mode]
        if not matching:
            return {"available": False, "reason": "No request has been submitted in this mode since startup."}
        return max(matching, key=lambda r: r["captured_at"])
    return processor.context_preview(mode == "autonomy")


def build_minds(engine: Any) -> list[dict[str, Any]]:
    """Agents that perform work, distinct from infrastructure threads and outputs."""
    result = [{"id": "glados", "title": "Central Core · GLaDOS", "role": "Conversation and facility control",
               "running": not engine.shutdown_event.is_set(), "kind": "primary",
               "summary": "Answers your requests and uses tools. Works on tracked tasks when asked.",
               "model": getattr(engine, "llm_model", "Unknown")}]
    router = getattr(engine, "router", None)
    autonomy = getattr(engine, "autonomy_config", None)
    if hasattr(engine, "set_autonomy_enabled"):
        result.append({"id": "autonomy", "title": "Autonomy Core", "kind": "autonomy",
                       "role": "Independent attention review and Central Core handoffs",
                       "running": bool(autonomy and autonomy.enabled),
                       "model": getattr(engine, "llm_model", "Unknown"),
                       "summary": (engine.autonomy_loop.snapshot()["status"]
                                   if getattr(engine, "autonomy_loop", None) else
                                   "Proactive responses enabled." if autonomy and autonomy.enabled else
                                   "Off. Enable in Settings; other cores run independently.")})
    if router:
        enabled = router.store.snapshot()["enabled"]
        result.append({"id": "router", "title": "Routing Core", "role": "One-token intent classification",
                       "kind": "router", "running": enabled, "model": engine.llm_model,
                       "summary": "Selects a capability, then a tool or action before speech." if enabled else "Routing disabled in Settings."})
    manager = getattr(engine, "subagent_manager", None)
    if manager:
        for status in manager.list_agents():
            agent = manager.get(status.agent_id)
            if agent is None:
                continue
            row = serialize_agent(status)
            row.update(id=status.agent_id, kind="background", role=agent.config.role,
                       paused=agent.paused, interval_s=getattr(status, "interval_s", None),
                       execution_status=getattr(status, "status", "waiting"),
                       next_due_in_s=getattr(status, "next_due_in_s", None),
                       model=agent.llm.model if status.agent_id == "vision" else getattr(engine, "llm_model", "Unknown"))
            if status.agent_id == "vision":
                row.update(interval_min_s=agent.settings.interval_min_s, interval_max_s=agent.settings.interval_max_s)
            store = getattr(engine, "autonomy_slots", None)
            slot = store.get_slot(status.agent_id) if store else None
            row["summary"] = slot.summary if slot else "Waiting for first result"
            if status.agent_id == "compaction":
                row.update(model=agent.model, compaction=agent.snapshot())
            if status.agent_id == "health":
                row.update(model=agent.llm.model if agent.llm and agent.settings.summary_enabled else 'System probes',
                           health=agent.snapshot(), summary_interval_s=agent.settings.summary_interval_s)
            if status.agent_id == "search":
                row.update(model=agent.llm.model, research=agent.snapshot())
            result.append(row)
    return result


def build_controls(engine: Any) -> dict[str, Any]:
    native = getattr(engine, "native_audio", None)
    emotion = getattr(engine, "_emotion_agent", None)
    return {
        "available": hasattr(engine, "set_asr_muted"),
        "quiet_available": hasattr(engine, "set_quiet_mode"),
        "quiet_mode": bool(getattr(engine, "quiet_event", None) and engine.quiet_event.is_set()),
        "autonomy_available": hasattr(engine, "set_autonomy_enabled"),
        "autonomy_enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "microphone_muted": bool(getattr(engine, "asr_muted_event", None) and engine.asr_muted_event.is_set()),
        "voice_muted": bool(getattr(engine, "tts_muted_event", None) and engine.tts_muted_event.is_set()),
        "audio_mode": "Direct audio" if native else "Transcribed audio",
        "input_mode": getattr(engine, "input_mode", "audio"),
        "model": getattr(engine, "llm_model", "Unknown"),
        "native_audio": native is not None,
        "user_transcripts": native.config.user_transcripts if native else True,
        "emotion_available": emotion is not None,
        "emotion_running": bool(emotion and getattr(emotion, "is_running", False)),
        "emotion_paused": bool(emotion and getattr(emotion, "paused", False)),
        "started_at": getattr(engine, "started_at", None),
    }


def _safe(serializer: Any, item: Any) -> dict[str, Any]:
    try:
        return serializer(item)
    except Exception:  # pragma: no cover
        return {}


def _safe_agents(engine: Any) -> list[dict[str, Any]]:
    try:
        return [_safe(serialize_agent, a) for a in engine.subagent_manager.list_agents()]
    except Exception:  # pragma: no cover
        return []


__all__ = [
    "build_snapshot",
    "build_state",
    "dumps",
    "serialize_agent",
    "serialize_event",
    "serialize_memory_entry",
    "serialize_mind",
    "serialize_slot",
    "serialize_slot_full",
    "to_jsonable",
]
