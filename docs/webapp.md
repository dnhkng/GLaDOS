# Webapp Console

An in-process "mission control" for the running GLaDOS engine. It streams the
live parallel state — the two autonomy lanes, subagent contexts, tool state,
PAD emotion, audio/MCP health — to a browser over HTTP + Server-Sent Events.
No separate service to run; no new runtime dependencies.

## Decoupled launcher (key design)

The webapp console is **not** part of the core engine. It is started by a
dedicated CLI command — `glados webapp` — that mirrors how `glados tui` works:
it loads the config, builds a `Glados` engine, starts the in-process
`WebappServer` on its own port, then runs the engine loop and shuts the server
down when the loop exits. The engine itself holds no webapp knowledge.

The observable state (`ObservabilityBus`, `MindRegistry`, `TaskSlotStore`,
subagent memory, interaction/emotion state) only exists inside the running
`Glados` object, so the console server runs in the same process and reads those
objects directly — exactly the pattern the WebSocket audio backend uses. This
keeps it dependency-light and side-effect free for every engine entry point
(`start`, `tui`, `say`).

The webapp and the TUI are **mutually exclusive UI options** — you run either
`glados webapp` or `glados tui`, never both; the core engine stays agnostic to
which one is attached.

## Running it

The console is **off by default**. Enable it, then start it with the webapp
launcher:

1) YAML config:

   ```yaml
   webapp:
     enabled: true
     host: 127.0.0.1
     port: 8050
   ```

2) Environment variables (no config edit):

   ```bash
   GLADOS_WEBAPP_ENABLED=1 GLADOS_WEBAPP_PORT=8050 uv run glados webapp
   ```

Both can be combined with `--config`, `--input-mode`, `--tts-enabled`/
`--tts-disabled`, and `--asr-muted`/`--asr-unmuted`:

```bash
uv run glados webapp --config ./configs/glados_webapp_config.yaml
```

Then open `http://127.0.0.1:8050/`.

> **Dummy fallback.** If the page is opened without a live engine — e.g.
> `examples/webapp/index.html` or a directly-opened static file — the page
> detects the missing API and falls back to simulated data so it can be
> previewed and styled.

## Endpoints

| Method | Path                  | Purpose |
| ------ | --------------------- | ------- |
| GET    | `/`                   | Static console (`static/index.html`). |
| GET    | `/api/snapshot`       | Aggregate JSON snapshot (minds, agents, slots, lanes, audio, emotion, MCP, interaction, vision). |
| GET    | `/api/state`          | Lightweight state JSON for the live gauges. |
| GET    | `/api/stream`         | SSE stream (see contract below). |
| GET    | `/api/minds`          | Registered mind statuses. |
| GET    | `/api/minds/{id}`     | Single mind status. |
| GET    | `/api/minds/{id}/memory` | That agent's private jsonlines memory entries. |
| GET    | `/api/slots`          | Task slots (summary fields). |
| GET    | `/api/slots/{id}`     | Full slot including the on-demand report. |
| GET    | `/api/agents`         | Subagent statuses (`agent_id, title, running, tick_count, last_tick`). |

The API is deliberately read-only and loopback-only. It does not enable CORS,
rejects cross-origin browser requests, and accepts only `127.0.0.1` or
`localhost` as its configured host. Remote control needs a separately designed
authentication and authorization boundary; this demo does not pretend to
provide one.

## SSE contract — `/api/stream`

Each connection first replays the last 100 events from the bus, then receives:

- `obs` events — the same shape the TUI `ObsScreen` renders:

  ```json
  {"timestamp": 1750000000.0, "source": "autonomy", "kind": "slot.update",
   "level": "info", "message": "weather brief -> done", "meta": {"slot": "s_weather"}}
  ```

  Real `source`/`kind` combos include `llm.request`, `llm.queue`,
  `llm.tool_calls`, `autonomy.dispatch`, `autonomy.slot.update`,
  `subagent.start/stop`, `tool.start/finish/error/timeout`, `tts.*`,
  `mcp.*`, `vision.update`, `text.user_input`.

- `state` events — every ~0.5 s, mirroring `/api/state`, so gauges, clock, and
  lane chips stay live without re-sending the whole snapshot.

### Multi-consumer (the important bit)

The TUI's `ObservabilityScreen` consumes the bus via `drain()`, which is a
*single-consumer* FIFO. The webapp never calls `drain()`. Instead every SSE
connection registers its own private `subscribe()` queue on
`ObservabilityBus`, so multiple browsers each get their own copy and never
steal events from the TUI or from each other. This is fully backward-compatible:
`drain()` and `snapshot()` keep their existing behavior.

## Failure behavior

Because the webapp is the point of the `glados webapp` launcher, a disabled
console or a bind failure (port in use) is **fatal for that command**: the
launcher logs an error and exits rather than running the engine without the
console. This does not affect other entry points — `tui`, `start`, and `say`
are completely independent of the webapp and never bind a port.

- Request exceptions return JSON error bodies and never crash a worker thread.
- Client disconnects free their subscription.

## Lifecycle

The `glados webapp` command orchestrates the whole lifecycle:

1. Load `GladosConfig.from_yaml` and apply CLI overrides.
2. Refuse to start if `webapp.enabled` is false.
3. Build `Glados.from_config`, then `WebappServer(engine, host, port).start()`.
4. Run `engine.run()`; on any exit (`try/finally`), shut the server down.

The core engine stays decoupled and side-effect free. This mirrors the TUI's
launcher pattern: the same `GLADOS_WEBAPP_*` environment variables live on
`GladosConfig.webapp`, so the launcher reads one merged config.
`GladosConfig.webapp` stays a field on the shared config model, but the engine
never reads it at runtime — the decoupling is at the launcher boundary.

## Development

- Console page: `src/glados/webapp/static/index.html` — single self-contained
  file (Aperture/GLaDOS theme, no build step). `examples/webapp/index.html` is
  the standalone mockup it derives from.
- Server: `src/glados/webapp/server.py`; serializers: `serializers.py`;
  config: `config.py`.
- Tests: `tests/test_webapp.py` — bus fan-out, serializers, and an in-process
  HTTP smoke test against a stub engine.
- Example config: `configs/glados_webapp_config.yaml`.
