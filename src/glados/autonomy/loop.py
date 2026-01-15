import queue
import threading
import time
from datetime import datetime
from typing import Any

from loguru import logger

from .config import AutonomyConfig
from .event_bus import EventBus
from .events import TaskUpdateEvent, TimeTickEvent, VisionUpdateEvent
from .interaction_state import InteractionState
from .slots import TaskSlotStore
from ..observability import ObservabilityBus, trim_message
from ..vision.vision_state import VisionState


class AutonomyLoop:
    def __init__(
        self,
        config: AutonomyConfig,
        event_bus: EventBus,
        interaction_state: InteractionState,
        vision_state: VisionState | None,
        slot_store: TaskSlotStore,
        llm_queue: queue.Queue[dict[str, Any]],
        processing_active_event: threading.Event,
        currently_speaking_event: threading.Event,
        shutdown_event: threading.Event,
        observability_bus: ObservabilityBus | None = None,
        pause_time: float = 0.1,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._interaction_state = interaction_state
        self._vision_state = vision_state
        self._slot_store = slot_store
        self._llm_queue = llm_queue
        self._processing_active_event = processing_active_event
        self._currently_speaking_event = currently_speaking_event
        self._shutdown_event = shutdown_event
        self._observability_bus = observability_bus
        self._pause_time = pause_time
        self._last_prompt_ts = 0.0
        self._last_scene: str | None = None

    def run(self) -> None:
        logger.info("AutonomyLoop thread started.")
        while not self._shutdown_event.is_set():
            try:
                event = self._event_bus.get(timeout=self._pause_time)
            except queue.Empty:
                continue

            if isinstance(event, TaskUpdateEvent) and not event.notify_user:
                continue

            if self._should_skip():
                continue

            prompt = self._build_prompt(event)
            if not prompt:
                continue
            self._dispatch(prompt)
        logger.info("AutonomyLoop thread finished.")

    def _should_skip(self) -> bool:
        if self._currently_speaking_event.is_set():
            return True
        if self._config.cooldown_s <= 0:
            return False
        return (time.time() - self._last_prompt_ts) < self._config.cooldown_s

    def _dispatch(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        if self._observability_bus:
            self._observability_bus.emit(
                source="autonomy",
                kind="dispatch",
                message=trim_message(prompt),
            )
        self._llm_queue.put({"role": "user", "content": prompt, "autonomy": True})
        self._processing_active_event.set()
        self._last_prompt_ts = time.time()

    def _build_prompt(self, event: object) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        since_user = self._interaction_state.seconds_since_user()
        since_assistant = self._interaction_state.seconds_since_assistant()
        since_user_text = f"{since_user:.1f}" if since_user is not None else "unknown"
        since_assistant_text = f"{since_assistant:.1f}" if since_assistant is not None else "unknown"

        scene = self._current_scene()
        prev_scene = self._last_scene or "unknown"
        change_score = "unknown"

        if isinstance(event, VisionUpdateEvent):
            prev_scene = event.prev_description or "unknown"
            scene = event.description
            change_score = f"{event.change_score:.4f}"
            self._last_scene = event.description
        elif isinstance(event, TimeTickEvent):
            if scene:
                self._last_scene = scene
        elif isinstance(event, TaskUpdateEvent):
            if scene:
                self._last_scene = scene

        tasks = self._task_summary()
        try:
            return self._config.tick_prompt.format(
                now=now,
                since_user=since_user_text,
                since_assistant=since_assistant_text,
                prev_scene=prev_scene,
                scene=scene or "unknown",
                change_score=change_score,
                tasks=tasks,
            )
        except KeyError:
            return self._config.tick_prompt

    def _current_scene(self) -> str | None:
        if self._vision_state is None:
            return "camera disabled"
        return self._vision_state.snapshot()

    def _task_summary(self) -> str:
        slots = self._slot_store.list_slots()
        if not slots:
            return "none"
        lines = []
        for slot in slots:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            meta_parts = []
            if slot.importance is not None:
                meta_parts.append(f"importance={slot.importance:.2f}")
            if slot.confidence is not None:
                meta_parts.append(f"confidence={slot.confidence:.2f}")
            if slot.next_run is not None:
                meta_parts.append(f"next_run={slot.next_run:.0f}")
            meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"{slot.title}: {slot.status}{summary_text}{meta_text}")
        return "\n".join(lines)

    def update_slot(
        self,
        slot_id: str,
        title: str,
        status: str,
        summary: str,
        notify_user: bool = True,
        updated_at: float | None = None,
    ) -> None:
        self._slot_store.update_slot(
            slot_id=slot_id,
            title=title,
            status=status,
            summary=summary,
            notify_user=notify_user,
            updated_at=updated_at,
        )
