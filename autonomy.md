# Autonomy Loop (Experimental)

GLaDOS can run a lightweight autonomy loop driven by vision updates or time ticks.

When enabled, the model receives periodic state updates and decides whether to act.
It should call the `speak` tool to say something, or `do_nothing` to stay silent.

## Enable
In your config:

```
autonomy:
  enabled: true
  tick_interval_s: 10   # Used when vision is disabled
  cooldown_s: 20
  jobs:
    enabled: false
    hacker_news:
      enabled: false
      interval_s: 1800
      top_n: 5
    weather:
      enabled: false
      interval_s: 3600
      latitude: 0.0
      longitude: 0.0
```

If vision is enabled, the loop is triggered by scene changes.
Jobs are optional and run in the background to populate task slots.

## Coalescing ticks
When `autonomy.coalesce_ticks` is enabled, the loop will skip or merge
new ticks if the previous autonomy request is still in flight. This prevents
the autonomy queue from growing under slow models and keeps latency low.

- `true`: collapse overlapping ticks into one (recommended).
- `false`: enqueue every tick (can build backlog if the model is slow).
