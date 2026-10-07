# Roadmap

Product direction and future feature ideas for GLaDOS.

## Delivery Sequence

Bring the current GLaDOS codebase close to feature complete, then migrate it to
Rust. The working application provides the behavior and user experience against
which the port is checked. Small feasibility probes can continue during feature
development to resolve migration risks.

The [Rust migration plan](rust-migration.md) records the desktop and model
direction, native inference approach, and the tested Sonora AEC results. Native
AEC is fast enough in the Linux probe and removes playback echo effectively;
overlapping speech remains intelligible with noticeable degradation and needs
further validation before production adoption.

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
