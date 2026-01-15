# Autonomy System Plan

## Overview
This document describes the autonomy system, background jobs/slots, and a prompt governor that can safely update sub-agent prompts. The goal is a responsive, low-noise system that can decide when to speak and when to stay silent.

## Goals
- Provide proactive updates when they are important and timely.
- Avoid spam and interruptions during active conversation.
- Enable safe prompt evolution for sub-agents with fast rollback and user feedback.
- Keep core personality stable while allowing contextual flexibility.

## Core Components
1) Background jobs + slots
2) Importance scoring + notification policy
3) Prompt governor + observer agent (sub-agent prompt updates + rollback)
4) Memory scoping + decay
5) Safety boundaries for external tools (Home Assistant)
6) Observability (latency, tool failure, interruption metrics)

## Background Jobs + Slots
Slots are durable state objects that store updates from background jobs.

### Slot Model
- id, title, type, status (idle, running, updated, error)
- summary (1-3 lines), details (optional payload)
- last_updated, next_run
- importance (0.0-1.0), confidence (0.0-1.0)
- cooldown_s, repeatable, expires_at
- tags (weather, news, system, vision, personal, etc.)
- source (job name), trigger (schedule/event/manual)

### Slot Types to Track
- Weather: hourly forecast + alerts (storms, wind, temperature swings)
- News: top Hacker News items + changes in top 10
- Calendar: upcoming events + time-to-leave reminders
- System health: CPU/GPU temp, disk space, service uptime
- Model status: VAD/ASR/TTS failures, LLM latency spikes
- Personal reminders: hydration, breaks, posture
- Social: unread messages or emails (summary only)
- Vision: new face recognized, room occupancy changes
- Ideas: internal thoughts stream with top insights
- Emotional state: tone/affect estimate for the session

### Scheduling + Triggers
- Periodic jobs (weather hourly, HN every 30-60 min)
- Event-based jobs (vision change, user question, system error)
- Deferred jobs (run after current TTS or when idle)

### Importance Scoring
Importance is a composite of severity, novelty, urgency, and context.

Example formula:
```
importance = base * novelty * urgency * context_boost * (1 - cooldown_penalty)
```

### Decision Policy (When to Speak)
- importance >= 0.8: speak immediately (unless user speaking)
- 0.5-0.8: queue for next natural pause
- < 0.5: update silently, include in context only

### Context Injection
Provide a compact system message such as:
```
[background slots]
- Weather: storm warning 18:00-23:00, 90% rain
- HN: "X" climbed to #2, 1200 points
- Mood: neutral, dry wit
```

### Emotional State Slot
- Inputs: recent conversation tone + task load + context
- Config: baseline Big Five emotion weights (joy, sadness, anger, fear, disgust)
  plus horniness, boredom, and arousal scalars, with min/max clamps
- Output: short descriptor ("dryly amused", "neutral", "impatient")
  plus the numeric state vector
- Used to guide tone, not necessarily spoken
- Example state vector:
  - emotions: {joy: 0.10, sadness: 0.70, anger: 0.20, fear: 0.15, disgust: 0.30}
  - drives: {horniness: 0.85, boredom: 0.75, arousal: 0.25}

### Personality Config (Stable)
Personality should live in config as OCEAN traits (stable across sessions).
This informs baseline tone and influences how emotion/drive vectors affect speech.

## Prompt Governor (Meta-Agent)
A higher-level agent that updates sub-agent prompts without changing the core system prompt.

### Prompt Layers
1) Immutable constitution (never edited)
2) Personality profile (OCEAN)
3) Mode prompt (autonomy on/off, quiet mode)
4) Sub-agent prompt pack (weather, news, memory, etc.)
5) Dynamic context (slots, vision, memory summaries)

The governor edits only layers 3-4 via versioned prompt packs.

## Observer Agent (Feedback-Driven Updates)
An observer agent reads recent outputs, the current prompt packs, and explicit user feedback.
It proposes small prompt deltas for sub-agents and applies them with rollback support.

### Observer Slot
- Stores the latest feedback, proposed delta, confidence, and applied version.
- Example user feedback: \"That report was too detailed. Summaries only.\"
  -> Observer updates brevity and max_sentences in the sub-agent pack.

### Prompt Pack Format (Example)
```
subagent: weather
version: 3
identity: "Weather Curator"
style:
  brevity: "short"
  tone: "dry"
  max_sentences: 2
rules:
  - "Speak only if importance >= 0.7"
  - "Lead with the most severe change"
allowed_tools:
  - "mcp.home_assistant.call_service"
```

### Governor Loop (Observer-Driven)
1) Collect signals (user feedback, interruptions, output review)
2) Propose prompt delta (short, bounded change)
3) Apply change (versioned prompt pack)
4) Monitor feedback and output quality in real usage
5) Roll back on explicit user request or repeated complaints

### Rollback Mechanism
- Keep the previous prompt pack version for each sub-agent.
- Provide a manual override: \"undo last prompt change\".
- Auto-rollback after repeated negative feedback within a cooldown window.

## Memory Scoping + Decay
- Short-term: per session
- Medium-term: daily summaries
- Long-term: optional MCP memory server
- Decay rules to prevent drift

## Safety + Permissions
- Domain-level allowlist for Home Assistant
- Confirmation for risky actions (locks, alarms)
- Cooldowns for repetitive actions

## Observability
- Metrics: LLM latency, tool failure rates, interruptions accepted/ignored
- Logs: slot updates, prompt changes, rollbacks, feedback events

## MVP Sequence
1) Slot schema + storage
2) Job runner with 2 jobs (weather + HN)
3) Importance scoring + cooldowns
4) Autonomy integration (speak vs. silence)
5) Context injection (latest updates only)
6) Prompt pack format + basic governor
