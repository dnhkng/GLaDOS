# Roadmap

Future feature ideas for GLaDOS.

## Audio Setup Wizard

Interactive setup to configure audio devices:
- Select input/output devices
- Calibrate VAD threshold
- Run loopback test
- Webcam check (describe what it sees)

## Home Assistant Mapping Editor

Map HA entities to MCP tool calls with safe, discoverable aliases:
- Data model: id, label, description, server, tool, args_template, confirm, cooldown_s, tags, examples
- Discovery: pull entity list from server or manual entry
- Editor UX: select server → search entities → pick action → edit args → test → save
- Safety: required arg validation, cooldowns, optional confirmation for risky domains

## Emotional State System

LLM-driven emotional regulation using HEXACO personality and PAD (Pleasure-Arousal-Dominance) affect:
- Personality traits defined in system prompt (immutable)
- PAD state updated by LLM based on events
- Mood drifts slowly toward current state
- Informs response tone without explicit mention

## Additional Slot Types

Potential background jobs to add:
- Calendar: upcoming events + time-to-leave reminders
- System health: GPU/CPU temp, disk space, service status
- Personal reminders: hydration, breaks, posture
- Social: unread messages or emails (summary only)
