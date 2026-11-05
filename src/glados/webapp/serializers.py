"""Map live engine state objects to plain JSON-serializable dicts.

These helpers mirror the TUI's panels: they read the exact same thread-safe
accessors (MindRegistry, TaskSlotStore, SubagentManager, MCPManager, AudioState,
InteractionState, queue sizes, ...) and turn them into plain dicts for the
webapp. No state is duplicated - the console just reads what the engine tracks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..autonomy.mind_scheduler import SubagentStatus
    from ..autonomy.slots import TaskSlot
    from ..autonomy.subagent_memory import MemoryEntry
    from ..core.engine import Glados
    from ..observability.events import ObservabilityEvent
    from ..observability.minds import MindStatus


try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


def _plain(value: object) -> object:
    """Recursively coerce numpy scalars/arrays and dataclasses to JSON-safe values."""
    if _HAS_NUMPY and isinstance(value, np.generic):
        return value.item()
    if _HAS_NUMPY and isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def to_jsonable(obj: object) -> Any:  # noqa: ANN401 - JSON of any shape; callers index into it
    return _plain(obj)


def dumps(obj: object) -> str:
    """Serialize to JSON without choking on dataclasses, numpy, or timestamps."""
    return json.dumps(obj, default=_json_default)


def _json_default(value: object) -> object:
    try:
        return _plain(value)
    except Exception:  # noqa: BLE001 - anything unserializable is shown as text
        return str(value)


def serialize_event(event: ObservabilityEvent) -> dict[str, Any]:
    """ObservabilityEvent -> {timestamp, source, kind, level, message, meta}."""
    return {
        "timestamp": event.timestamp,
        "source": event.source,
        "kind": event.kind,
        "level": event.level,
        "message": event.message,
        "meta": to_jsonable(event.meta),
    }


def serialize_mind(mind: MindStatus) -> dict[str, Any]:
    return {
        "mind_id": mind.mind_id,
        "title": mind.title,
        "status": mind.status,
        "summary": mind.summary,
        "role": mind.role,
        "updated_at": mind.updated_at,
    }


def serialize_slot(slot: TaskSlot) -> dict[str, Any]:
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
        "attention_key": slot.attention_key,
        "importance": slot.importance,
        "confidence": slot.confidence,
        "next_run": slot.next_run,
        "updated_at": slot.updated_at,
        "has_report": bool(slot.report),
    }


def serialize_slot_full(slot: TaskSlot) -> dict[str, Any]:
    data = serialize_slot(slot)
    data["report"] = slot.report
    data["context"] = slot.context
    return data


def serialize_memory_entry(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "value": to_jsonable(entry.value),
        "created_at": entry.created_at,
        "shown_at": entry.shown_at,
    }


def serialize_agent(agent_status: SubagentStatus) -> dict[str, Any]:
    return {
        "agent_id": agent_status.agent_id,
        "title": agent_status.title,
        "running": agent_status.running,
        "tick_count": agent_status.tick_count,
        "last_tick": agent_status.last_tick,
    }


def build_audio_state(engine: Glados) -> dict[str, Any]:
    audio_state = getattr(engine, "audio_state", None)
    if audio_state is None:
        return {"rms": 0.0, "vad_active": False, "updated_at": 0.0}
    snap = audio_state.snapshot()
    return {
        "rms": float(snap.rms),
        "vad_active": bool(snap.vad_active),
        "updated_at": float(getattr(snap, "updated_at", 0.0)),
    }


def _remote_audio(engine: Glados) -> dict[str, Any] | None:
    status = getattr(getattr(engine, "audio_io", None), "remote_status", None)
    return status() if callable(status) else None


def _emotion_state(engine: Glados) -> dict[str, Any] | None:
    agent = getattr(engine, "_emotion_agent", None)
    if agent is None:
        return None
    try:
        return {**to_jsonable(agent.state), "response_instructions": agent.state.response_instructions()}
    except Exception:  # noqa: BLE001 - the console shows what it can
        return None


def _mcp_summary(engine: Glados) -> dict[str, Any]:
    manager = getattr(engine, "mcp_manager", None)
    if manager is None:
        return {"enabled": False, "servers": []}
    try:
        servers = manager.status_snapshot()
    except Exception:  # noqa: BLE001 - the console shows what it can
        servers = []
    return {"enabled": True, "servers": servers}


def build_lanes(engine: Glados) -> dict[str, Any]:
    scheduler = getattr(engine, "inference_scheduler", None)
    state = scheduler.snapshot() if scheduler else None
    active = state["active"] if state else []
    waiting = state["waiting"] if state else []
    return {
        "enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "priority": {
            "queue": int(engine.llm_queue_priority.qsize()) + sum(r["lane"] != "autonomy" for r in waiting),
            "inflight": sum(r["lane"] != "autonomy" for r in active)
            if state
            else int(engine._priority_inflight.value()),
        },
        "autonomy": {
            "queue": int(engine.llm_queue_autonomy.qsize()) + sum(r["lane"] == "autonomy" for r in waiting),
            "inflight": sum(r["lane"] == "autonomy" for r in active)
            if state
            else int(engine._autonomy_inflight.value()),
            "workers": len(getattr(engine, "autonomy_llm_processors", ())),
        },
    }


def _snapshot[T](engine: Glados, name: str, read: Callable[[Any], T] = lambda part: part.snapshot()) -> T | None:
    """Read one optional engine part; lightweight profiles and test doubles may lack it."""
    part = getattr(engine, name, None)
    return read(part) if part else None


def _live_parts(engine: Glados) -> dict[str, Any]:
    """State shared by the initial snapshot and the periodic stream."""
    return {
        "t": time.time(),
        "controls": build_controls(engine),
        "interaction": {
            "seconds_since_user": engine.interaction_state.seconds_since_user(),
            "seconds_since_assistant": engine.interaction_state.seconds_since_assistant(),
        },
        "lanes": build_lanes(engine),
        "inference": _snapshot(engine, "inference_scheduler"),
        "routing": _snapshot(engine, "router"),
        "command_slot": _snapshot(engine, "command_runner"),
        "autonomy_core": _snapshot(engine, "autonomy_loop"),
        "vision_mind": _snapshot(engine, "vision_agent"),
        "search_settings": _snapshot(engine, "search_agent", lambda agent: agent.preferences.snapshot()),
        "nexus": _snapshot(engine, "nexus_agent", lambda agent: agent.status()),
        "vision": _snapshot(engine, "vision_state"),
        "audio": build_audio_state(engine),
        "emotion": _emotion_state(engine),
        "mcp": _mcp_summary(engine),
        "speaking": bool(engine.currently_speaking_event.is_set()),
        "performance": _snapshot(engine, "speech_animation"),
        "hardware": getattr(engine, "hardware_status", None),
    }


def build_state(engine: Glados) -> dict[str, Any]:
    """Lightweight payload streamed periodically to keep gauges/clock live."""
    return {**_live_parts(engine), "remote_audio": _remote_audio(engine)}


def build_snapshot(engine: Glados) -> dict[str, Any]:
    """Aggregate snapshot for the console's initial paint (GET /api/snapshot)."""
    return {
        **_live_parts(engine),
        "operator": _snapshot(engine, "operator_state"),
        "version": "0.1",
        "autonomy_enabled": bool(getattr(engine, "autonomy_config", None) and engine.autonomy_config.enabled),
        "latency": _snapshot(engine, "latency_tracker") or [],
        "end_of_speech_ms": getattr(getattr(engine, "speech_listener", None), "PAUSE_LIMIT", None),
        "minds": [_safe(serialize_mind, m) for m in engine.mind_registry.snapshot()],
        "slots": _snapshot(
            engine, "autonomy_slots", lambda store: [_safe(serialize_slot, s) for s in store.list_slots()]
        )
        or [],
        "agents": _safe_agents(engine),
        "commands": sorted(getattr(engine, "_command_order", ())),
        "agent_minds": build_minds(engine),
        "tools": build_tools(engine),
        "decisions": _snapshot(engine, "decision_lists"),
    }


def build_tools(engine: Glados) -> list[dict[str, Any]]:
    processor = getattr(engine, "llm_processor", None)
    if processor is None:
        return []
    return [tool["function"] for tool in processor._build_tools(False) if "function" in tool]


def build_context(engine: Glados, mode: str = "user", view: str = "live") -> dict[str, Any]:
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
    preview: dict[str, Any] = processor.context_preview(mode == "autonomy")
    return preview


def build_minds(engine: Glados) -> list[dict[str, Any]]:
    """Agents that perform work, distinct from infrastructure threads and outputs."""
    result = [
        {
            "id": "glados",
            "title": "Central Core · GLaDOS",
            "role": "Conversation and facility control",
            "running": not engine.shutdown_event.is_set(),
            "kind": "primary",
            "summary": "Answers your requests and uses tools. Works on tracked tasks when asked.",
            "model": getattr(engine, "llm_model", "Unknown"),
        }
    ]
    router = getattr(engine, "router", None)
    autonomy = getattr(engine, "autonomy_config", None)
    if hasattr(engine, "set_autonomy_enabled"):
        result.append(
            {
                "id": "autonomy",
                "title": "Autonomy Core",
                "kind": "autonomy",
                "role": "Independent attention review and Central Core handoffs",
                "running": bool(autonomy and autonomy.enabled),
                "model": getattr(engine, "llm_model", "Unknown"),
                "summary": (
                    loop.snapshot()["status"]
                    if (loop := getattr(engine, "autonomy_loop", None))
                    else "Proactive responses enabled."
                    if autonomy and autonomy.enabled
                    else "Off. Enable in Settings; other cores run independently."
                ),
            }
        )
    if router:
        enabled = router.store.snapshot()["enabled"]
        result.append(
            {
                "id": "router",
                "title": "Routing Core",
                "role": "One-token intent classification",
                "kind": "router",
                "running": enabled,
                "model": engine.llm_model,
                "summary": "Selects a capability, then a tool or action before speech."
                if enabled
                else "Routing disabled in Settings.",
            }
        )
    manager = getattr(engine, "subagent_manager", None)
    if manager:
        for status in manager.list_agents():
            agent = manager.get(status.agent_id)
            if agent is None:
                continue
            row = serialize_agent(status)
            row.update(
                id=status.agent_id,
                kind="background",
                role=agent.config.role,
                paused=agent.paused,
                interval_s=getattr(status, "interval_s", None),
                execution_status=getattr(status, "status", "waiting"),
                next_due_in_s=getattr(status, "next_due_in_s", None),
                model=agent.llm.model if status.agent_id == "vision" else getattr(engine, "llm_model", "Unknown"),
            )
            if status.agent_id == "vision":
                row.update(interval_min_s=agent.settings.interval_min_s, interval_max_s=agent.settings.interval_max_s)
            store = getattr(engine, "autonomy_slots", None)
            slot = store.get_slot(status.agent_id) if store else None
            row["summary"] = slot.summary if slot else "Waiting for first result"
            if status.agent_id == "compaction":
                row.update(model=agent.model, compaction=agent.snapshot())
            if status.agent_id == "health":
                row.update(
                    model=agent.llm.model if agent.llm and agent.settings.summary_enabled else "System probes",
                    health=agent.snapshot(),
                    summary_interval_s=agent.settings.summary_interval_s,
                )
            if status.agent_id == "search":
                row.update(model=agent.llm.model, research=agent.snapshot())
            result.append(row)
    return result


def build_controls(engine: Glados) -> dict[str, Any]:
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


def _safe[T](serializer: Callable[[T], dict[str, Any]], item: T) -> dict[str, Any]:
    try:
        return serializer(item)
    except Exception:  # noqa: BLE001 - the console shows what it can
        return {}


def _safe_agents(engine: Glados) -> list[dict[str, Any]]:
    manager = getattr(engine, "subagent_manager", None)
    if manager is None:
        return []
    try:
        return [_safe(serialize_agent, a) for a in manager.list_agents()]
    except Exception:  # noqa: BLE001 - the console shows what it can
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
