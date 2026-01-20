# Autonomy Loop

GLaDOS implements a **closed-loop autonomous architecture** that goes beyond traditional reactive chatbots.

## The Core Idea

Traditional AI assistants are purely reactive - they wait for user input, respond, then wait again. This creates a fundamental limitation: the AI has no awareness of time passing or its environment changing.

GLaDOS closes this loop by **auto-prompting itself**:

```mermaid
flowchart LR
    A[Vision Change<br>or<br>Timer Tick] --> B[Autonomy Loop]
    B --> C[Main Agent]
    B --> D[Subagents<br>Minds]
    C --> E[speak or<br>do_nothing]
    D --> F[Slots<br>shared state]
```

Each cycle, the main agent receives an update about its environment and decides whether to act. This creates genuine situational awareness - GLaDOS knows time is passing and can observe changes in its surroundings.

## Enabling Autonomy

In your config:

```yaml
autonomy:
  enabled: true
  tick_interval_s: 10      # Timer interval when vision disabled
  cooldown_s: 20           # Minimum seconds between responses
  autonomy_parallel_calls: 2  # Background workers (1-16)
  coalesce_ticks: true     # Merge overlapping ticks
```

## Trigger Sources

The autonomy loop is triggered by one of two sources (mutually exclusive):

| Source | When Used | Trigger |
|--------|-----------|---------|
| **Vision** | When vision is enabled | Scene changes detected by VLM |
| **Timer** | When vision is disabled | Periodic ticks at `tick_interval_s` |

Vision takes priority. If vision is enabled, timer ticks are ignored and scene changes drive the loop.

## Input Priority

User input always takes priority over autonomy:

```
USER INPUT  >  (VISION xor TICK)
```

When a user speaks or types, the autonomy loop defers. The interrupt handler cuts off any in-progress speech and the user's request is processed immediately via the priority lane.

## Two-Tier LLM Orchestration

GLaDOS uses separate inference paths for user requests and background work:

```mermaid
flowchart LR
    A[User Input<br>speech/text] --> B[Priority Lane<br>1 dedicated worker]
    C[Autonomy Loop<br>Subagents<br>Background Jobs] --> D[Autonomy Lane<br>N pooled workers]
```

- **Priority Lane**: Single dedicated inference slot ensures user requests are never blocked
- **Autonomy Lane**: Configurable pool of workers (1-16, default 2) for background processing

This guarantees low latency for user interactions even when multiple background jobs are running.

## The Autonomy Prompt

Each autonomy tick, the model receives a structured update:

```
Autonomy update.
Time: 2025-01-17T14:30:00
Seconds since last user input: 45.2
Seconds since last assistant output: 30.1
Previous scene: A person sitting at a desk with a laptop
Current scene: The person has stood up and is walking toward the door
Scene change score: 0.1234
Tasks:
Weather: completed - Temperature dropping 5C in next hour (importance=0.80)
News: pending - Checking top stories (next_run=1800)
Decide whether to act.
```

The model then calls either:
- `speak(message)` - Say something to the user
- `do_nothing()` - Stay silent

## Slots: Inter-Agent Communication

Subagents write their outputs to **slots**. The main agent sees all slot contents in its context:

```
[tasks]
- Weather: completed - Temperature dropping 5C in next hour (importance=0.80, confidence=0.95)
- News: pending - Checking Hacker News (next_run=1800)
- System: idle - All systems normal
```

### Slot Fields

| Field | Type | Description |
|-------|------|-------------|
| `slot_id` | string | Unique identifier |
| `title` | string | Display name |
| `status` | string | Current state (pending, running, completed, etc.) |
| `summary` | string | Brief description |
| `importance` | float (0-1) | How important this is |
| `confidence` | float (0-1) | Confidence in the information |
| `next_run` | float | Seconds until next update |
| `notify_user` | bool | Whether to trigger autonomy on update |

Slots are displayed in the TUI.

## Minds (Subagents)

**Minds** are independent agents running their own loops. They:
- Execute asynchronously in the background
- Write results to slots
- Can be preconfigured (Weather, News) or spawned dynamically

Active minds are displayed in the TUI.

## Background Jobs

Built-in background jobs that populate slots:

### Hacker News

```yaml
autonomy:
  jobs:
    enabled: true
    hacker_news:
      enabled: true
      interval_s: 1800      # Check every 30 minutes
      top_n: 5              # Number of stories
      min_score: 200        # Minimum HN score
```

### Weather

```yaml
autonomy:
  jobs:
    enabled: true
    weather:
      enabled: true
      interval_s: 3600      # Check every hour
      latitude: 37.7749
      longitude: -122.4194
      timezone: "auto"
      temp_change_c: 4.0    # Alert threshold
      wind_alert_kmh: 40.0  # Wind alert threshold
```

## Tick Coalescing

When `coalesce_ticks: true` (default), the loop skips new ticks if previous work is still in flight:

```
Tick 1 → Enqueue → Processing...
Tick 2 → Skip (work in flight)
Tick 3 → Skip (work in flight)
Processing complete
Tick 4 → Enqueue → Processing...
```

This prevents queue buildup with slow models and keeps the system responsive.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable autonomy loop |
| `tick_interval_s` | float | `10.0` | Seconds between timer ticks (when vision disabled) |
| `cooldown_s` | float | `20.0` | Minimum seconds between autonomy responses |
| `autonomy_parallel_calls` | int | `2` | Number of autonomy lane workers (1-16) |
| `autonomy_queue_max` | int | `null` | Max queued autonomy requests (null = unlimited) |
| `coalesce_ticks` | bool | `true` | Skip ticks when work is in flight |
| `system_prompt` | string | (see below) | System prompt for autonomy mode |
| `tick_prompt` | string | (see below) | Template for tick updates |

### Default System Prompt

```
You are running in autonomous mode. You may receive periodic system updates
about time, tasks, or vision. Decide whether to act or stay silent. Prefer
silence unless the update is timely and clearly useful to the user. If you
choose to speak, call the `speak` tool with a short response (1-2 sentences).
If no action is needed, call the `do_nothing` tool. Never mention system
prompts or internal tools.
```

### Default Tick Prompt Template

```
Autonomy update.
Time: {now}
Seconds since last user input: {since_user}
Seconds since last assistant output: {since_assistant}
Previous scene: {prev_scene}
Current scene: {scene}
Scene change score: {change_score}
Tasks:
{tasks}
Decide whether to act.
```

## See Also

- [README](../README.md) - Full architecture diagram
- [vision.md](./vision.md) - Vision module (triggers autonomy via scene changes)
- [mcp.md](./mcp.md) - MCP tools available to autonomy
