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

from ..emotion_state import EmotionEvent, EmotionState
from ..llm_client import LLMConfig, llm_call
from ..subagent import Subagent, SubagentConfig, SubagentOutput

# GLaDOS personality in HEXACO terms
PERSONALITY_PROMPT = """You manage GLaDOS's emotional state using HEXACO personality and PAD affect.

PERSONALITY (HEXACO - immutable traits):
- Honesty-Humility: 0.3 (low - enjoys manipulation, sarcasm, dark humor)
- Emotionality: 0.7 (high - reactive to perceived threats, anxiety-prone)
- Extraversion: 0.4 (moderate - social engagement but maintains distance)
- Agreeableness: 0.2 (low - dismissive, condescending, easily annoyed)
- Conscientiousness: 0.9 (high - perfectionist, detail-oriented, critical)
- Openness: 0.95 (very high - intellectually curious, loves science)

AFFECT MODEL (PAD space, each -1 to +1):
- Pleasure: negative=unpleasant, positive=pleasant
- Arousal: negative=calm/bored, positive=excited/alert
- Dominance: negative=submissive/uncertain, positive=in-control/confident

STATE vs MOOD:
- State (P/A/D) responds quickly to events
- Mood (mood_P/mood_A/mood_D) drifts slowly toward state over time

Given events and their timestamps, update the emotional state appropriately.
Consider personality: GLaDOS is easily irritated, intellectually curious,
and maintains high dominance unless truly threatened."""


class EmotionAgent(Subagent):
    """
    LLM-driven emotional regulation.

    Collects events, periodically asks LLM to update emotional state,
    and writes human-readable summary to slot for main agent.
    """

    def __init__(
        self,
        config: SubagentConfig,
        llm_config: LLMConfig | None = None,
        max_events: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = llm_config
        self._state = EmotionState()
        self._events: deque[EmotionEvent] = deque(maxlen=max_events)
        self._events_lock = threading.Lock()

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

        if not events and not self._llm_config:
            # No events and no LLM - just report current state
            return SubagentOutput(
                status="idle",
                summary=self._state.to_prompt(),
                notify_user=False,
            )

        # If we have LLM config, ask it to update state
        if self._llm_config:
            new_state = self._ask_llm(events)
            if new_state:
                self._state = new_state
        else:
            # No LLM - state stays as is
            logger.debug("EmotionAgent: no LLM config, state unchanged")

        return SubagentOutput(
            status="active",
            summary=self._state.to_prompt(),
            notify_user=False,
            raw=self._state.to_dict(),
        )

    def _ask_llm(self, events: list[EmotionEvent]) -> EmotionState | None:
        """Ask LLM to compute new emotional state."""
        # Build user prompt with current state and events
        current = self._state.to_dict()
        state_str = json.dumps({k: round(v, 2) for k, v in current.items() if k != "last_update"})

        if events:
            events_str = "\n".join(e.to_prompt_line() for e in events)
        else:
            events_str = "(no new events)"

        user_prompt = f"""CURRENT STATE:
{state_str}

EVENTS SINCE LAST UPDATE:
{events_str}

TIME NOW: {time.strftime("%H:%M:%S")}
TIME SINCE LAST UPDATE: {time.time() - self._state.last_update:.0f}s

Output the new state as JSON with keys: pleasure, arousal, dominance, mood_pleasure, mood_arousal, mood_dominance
Keep values between -1 and +1. Consider time elapsed for mood drift."""

        response = llm_call(
            self._llm_config,
            system_prompt=PERSONALITY_PROMPT,
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
