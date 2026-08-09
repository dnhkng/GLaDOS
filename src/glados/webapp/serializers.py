"""Map live engine state objects to plain JSON-serializable dicts.

These helpers mirror the TUI's panels: they read the exact same thread-safe
accessors (MindRegistry, TaskSlotStore, SubagentManager, MCPManager, AudioState,
InteractionState, queue sizes, ...) and turn them into plain dicts for the
webapp. No state is duplicated - the console just reads what the engine tracks.
"""
# The console intentionally adapts several optional engine components through a
# runtime-shaped boundary. Concrete protocols for every combination would add
# coupling without improving safety at this read-only telemetry edge.
# ruff: noqa: ANN401

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any, cast

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
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def to_jsonable(obj: Any) -> Any:
    """Return a recursively JSON-compatible representation of ``obj``."""
    return _plain(obj)


def dumps(obj: Any) -> str:
    """Serialize to JSON without choking on dataclasses, numpy, or timestamps."""
    return json.dumps(obj, default=_json_default)


def _json_default(value: Any) -> Any:
    """Coerce otherwise unsupported values for :func:`json.dumps`."""
    try:
        plain = _plain(value)
        return str(value) if plain is value else plain
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
    """Serialize a registered mind's public status."""
    return {
        "mind_id": mind.mind_id,
        "title": mind.title,
        "status": mind.status,
        "summary": mind.summary,
        "role": mind.role,
        "updated_at": mind.updated_at,
    }


def serialize_slot(slot: Any) -> dict[str, Any]:
    """Serialize task-slot summary fields."""
    return {
        "slot_id": slot.slot_id,
        "title": slot.title,
        "status": slot.status,
        "summary": slot.summary,
        "notify_user": slot.notify_user,
        "importance": slot.importance,
        "confidence": slot.confidence,
        "next_run": slot.next_run,
        "updated_at": slot.updated_at,
        "has_report": bool(slot.report),
    }


def serialize_slot_full(slot: Any) -> dict[str, Any]:
    """Serialize a task slot including its report."""
    data = serialize_slot(slot)
    data["report"] = slot.report
    return data


def serialize_memory_entry(entry: Any) -> dict[str, Any]:
    """Serialize a subagent memory entry."""
    return {
        "key": entry.key,
        "value": to_jsonable(entry.value),
        "created_at": entry.created_at,
        "shown_at": entry.shown_at,
    }


def serialize_agent(agent_status: Any) -> dict[str, Any]:
    """Serialize a subagent's runtime status."""
    return {
        "agent_id": agent_status.agent_id,
        "title": agent_status.title,
        "running": agent_status.running,
        "tick_count": agent_status.tick_count,
        "last_tick": agent_status.last_tick,
    }


def _audio_state(engine: Any) -> dict[str, Any]:
    """Read the current audio meter state from the engine."""
    audio_state = getattr(engine, "audio_state", None)
    if audio_state is None:
        return {"rms": 0.0, "vad_active": False}
    snap = audio_state.snapshot()
    return {"rms": float(snap.rms), "vad_active": bool(snap.vad_active)}


def _emotion_state(engine: Any) -> dict[str, Any] | None:
    """Return the optional emotion-agent state."""
    agent = getattr(engine, "_emotion_agent", None)
    if agent is None:
        return None
    try:
        return cast(dict[str, Any], to_jsonable(agent.state))
    except Exception:  # pragma: no cover
        return None


def _mcp_summary(engine: Any) -> dict[str, Any]:
    """Return a failure-tolerant MCP status summary."""
    manager = getattr(engine, "mcp_manager", None)
    if manager is None:
        return {"enabled": False, "servers": []}
    try:
        servers = manager.status_snapshot()
    except Exception:  # pragma: no cover
        servers = []
    return {"enabled": True, "servers": servers}


def build_lanes(engine: Any) -> dict[str, Any]:
    """Summarize priority and autonomy inference lanes."""
    return {
        "enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "priority": {"queue": int(engine.llm_queue_priority.qsize())},
        "autonomy": {
            "queue": int(engine.llm_queue_autonomy.qsize()),
            "inflight": int(engine._autonomy_inflight.value()),
            "workers": len(getattr(engine, "autonomy_llm_processors", ())),
        },
    }


def build_state(engine: Any) -> dict[str, Any]:
    """Lightweight payload streamed periodically to keep gauges/clock live."""
    return {
        "t": time.time(),
        "interaction": {
            "seconds_since_user": engine.interaction_state.seconds_since_user(),
            "seconds_since_assistant": engine.interaction_state.seconds_since_assistant(),
        },
        "lanes": build_lanes(engine),
        "audio": _audio_state(engine),
        "emotion": _emotion_state(engine),
        "mcp": _mcp_summary(engine),
        "speaking": bool(engine.currently_speaking_event.is_set()),
    }


def build_snapshot(engine: Any) -> dict[str, Any]:
    """Aggregate snapshot for the console's initial paint (GET /api/snapshot)."""
    vision_state = getattr(engine, "vision_state", None)
    snapshot: dict[str, Any] = {
        "t": time.time(),
        "version": "0.1",
        "autonomy_enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "lanes": build_lanes(engine),
        "audio": _audio_state(engine),
        "emotion": _emotion_state(engine),
        "mcp": _mcp_summary(engine),
        "interaction": {
            "seconds_since_user": engine.interaction_state.seconds_since_user(),
            "seconds_since_assistant": engine.interaction_state.seconds_since_assistant(),
        },
        "speaking": bool(engine.currently_speaking_event.is_set()),
        "minds": [_safe(serialize_mind, m) for m in engine.mind_registry.snapshot()],
        "slots": (
            [_safe(serialize_slot, s) for s in engine.autonomy_slots.list_slots()]
            if getattr(engine, "autonomy_slots", None)
            else []
        ),
        "agents": (_safe_agents(engine) if getattr(engine, "subagent_manager", None) else []),
        "vision": vision_state.snapshot() if vision_state else None,
    }
    return snapshot


def _safe(serializer: Any, item: Any) -> dict[str, Any]:
    """Serialize one item without allowing telemetry to break snapshots."""
    try:
        return cast(dict[str, Any], serializer(item))
    except Exception:  # pragma: no cover
        return {}


def _safe_agents(engine: Any) -> list[dict[str, Any]]:
    """Return all subagent statuses, or an empty list if unavailable."""
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
