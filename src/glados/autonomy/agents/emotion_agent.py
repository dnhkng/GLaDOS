"""
LLM-driven emotional regulation agent.

Uses HEXACO personality model and PAD affect space. Instead of hard-coded
decay math, the LLM reasons about how events should affect emotional state.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from loguru import logger

from ..config import EmotionConfig, HEXACOConfig
from ..emotion_state import EmotionEvent, EmotionState
from ..llm_client import LLMConfig, llm_call
from ..subagent import Subagent, SubagentConfig, SubagentOutput


def build_personality_prompt(hexaco: HEXACOConfig) -> str:
    """Build the personality prompt from HEXACO config."""
    return f"""You manage the emotional state using HEXACO personality and PAD affect.

PERSONALITY (HEXACO - character traits):
- Honesty-Humility: {hexaco.honesty_humility:.1f} ({"low - enjoys manipulation, sarcasm" if hexaco.honesty_humility < 0.5 else "high - sincere, modest"})
- Emotionality: {hexaco.emotionality:.1f} ({"high - reactive to threats, anxiety-prone" if hexaco.emotionality > 0.5 else "low - calm, detached"})
- Extraversion: {hexaco.extraversion:.1f} ({"high - social, talkative" if hexaco.extraversion > 0.5 else "moderate/low - maintains distance"})
- Agreeableness: {hexaco.agreeableness:.1f} ({"low - dismissive, easily annoyed" if hexaco.agreeableness < 0.5 else "high - patient, forgiving"})
- Conscientiousness: {hexaco.conscientiousness:.1f} ({"high - perfectionist, detail-oriented" if hexaco.conscientiousness > 0.5 else "low - flexible, spontaneous"})
- Openness: {hexaco.openness:.1f} ({"high - intellectually curious" if hexaco.openness > 0.5 else "low - practical, conventional"})

AFFECT MODEL (PAD space, each -1 to +1):
- Pleasure: negative=unpleasant, positive=pleasant
- Arousal: negative=calm/bored, positive=excited/alert
- Dominance: negative=submissive/uncertain, positive=in-control/confident

STATE vs MOOD:
- State (P/A/D) responds quickly to events
- Mood (mood_P/mood_A/mood_D) drifts slowly toward state over time

Given events and their timestamps, update the emotional state appropriately.
Consider the personality traits when determining emotional responses."""


class EmotionAgent(Subagent):
    """
    LLM-driven emotional regulation.

    Collects events, periodically asks LLM to update emotional state,
    and writes human-readable summary to slot for main agent.

    Features:
    - Configurable HEXACO personality traits
    - Persistent state across restarts (via SubagentMemory)
    - Baseline drift when idle (mood approaches configured baseline)
    """

    MEMORY_STATE_KEY = "current_state"

    def __init__(
        self,
        config: SubagentConfig,
        llm_config: LLMConfig | None = None,
        emotion_config: EmotionConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = llm_config
        self._emotion_config = emotion_config or EmotionConfig()
        self._state = self._load_state()
        self._events: deque[EmotionEvent] = deque(maxlen=self._emotion_config.max_events)
        self._events_lock = threading.Lock()
        self._personality_prompt = build_personality_prompt(self._emotion_config.hexaco)

    def _load_state(self) -> EmotionState:
        """Load state from memory, or create fresh with baseline values."""
        entry = self.memory.get(self.MEMORY_STATE_KEY)
        if entry and isinstance(entry.value, dict):
            logger.info("EmotionAgent: restored state from memory")
            return EmotionState.from_dict(entry.value)

        # Fresh state with baseline values
        cfg = self._emotion_config
        return EmotionState(
            pleasure=cfg.baseline_pleasure,
            arousal=cfg.baseline_arousal,
            dominance=cfg.baseline_dominance,
            mood_pleasure=cfg.baseline_pleasure,
            mood_arousal=cfg.baseline_arousal,
            mood_dominance=cfg.baseline_dominance,
        )

    def _save_state(self) -> None:
        """Persist current state to memory."""
        self.memory.set(self.MEMORY_STATE_KEY, self._state.to_dict())

    def push_event(self, event: EmotionEvent) -> None:
        """Add an event to be processed on next tick."""
        with self._events_lock:
            self._events.append(event)

    def tick(self) -> SubagentOutput | None:
        """Process events and update emotional state via LLM."""
        # Drain events
        with self._events_lock:
            events = list(self._events)
            self._events.clear()

        # If no LLM and no events, apply baseline drift
        if not self._llm_config:
            if not events:
                self._apply_baseline_drift()
            logger.debug("EmotionAgent: no LLM config, applied baseline drift")
        else:
            # Ask LLM to update state
            new_state = self._ask_llm(events)
            if new_state:
                self._state = new_state

        # Persist state
        self._save_state()

        return SubagentOutput(
            status="active" if events else "idle",
            summary=self._state.to_prompt(),
            notify_user=False,
            raw=self._state.to_dict(),
        )

    def _apply_baseline_drift(self) -> None:
        """Drift mood toward baseline values when idle (no LLM fallback)."""
        cfg = self._emotion_config
        rate = cfg.baseline_drift_rate

        self._state.mood_pleasure += (cfg.baseline_pleasure - self._state.mood_pleasure) * rate
        self._state.mood_arousal += (cfg.baseline_arousal - self._state.mood_arousal) * rate
        self._state.mood_dominance += (cfg.baseline_dominance - self._state.mood_dominance) * rate
        self._state.last_update = time.time()

    def _ask_llm(self, events: list[EmotionEvent]) -> EmotionState | None:
        """Ask LLM to compute new emotional state."""
        cfg = self._emotion_config

        # Build user prompt with current state and events
        current = self._state.to_dict()
        state_str = json.dumps({k: round(v, 2) for k, v in current.items() if k != "last_update"})

        if events:
            events_str = "\n".join(e.to_prompt_line() for e in events)
        else:
            events_str = "(no new events - consider drifting toward baseline)"

        user_prompt = f"""CURRENT STATE:
{state_str}

BASELINE VALUES (mood drifts here when idle):
pleasure={cfg.baseline_pleasure:.1f}, arousal={cfg.baseline_arousal:.1f}, dominance={cfg.baseline_dominance:.1f}

EVENTS SINCE LAST UPDATE:
{events_str}

TIME NOW: {time.strftime("%H:%M:%S")}
TIME SINCE LAST UPDATE: {time.time() - self._state.last_update:.0f}s

Output the new state as JSON with keys: pleasure, arousal, dominance, mood_pleasure, mood_arousal, mood_dominance
Keep values between -1 and +1. Consider time elapsed for mood drift toward baseline."""

        response = llm_call(
            self._llm_config,
            system_prompt=self._personality_prompt,
            user_prompt=user_prompt,
            json_response=True,
        )

        if not response:
            return None

        try:
            data = json.loads(response)
            return EmotionState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("EmotionAgent: failed to parse LLM response: %s", e)
            return None

    @property
    def state(self) -> EmotionState:
        return self._state

    @property
    def emotion_config(self) -> EmotionConfig:
        return self._emotion_config
