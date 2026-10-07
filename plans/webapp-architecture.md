# Webapp Console — Architecture Plan

**Status:** Implemented (option 1 — decoupled launcher)
**Scope:** How the in-process observability web server and the engine `main`
interact — without changing or breaking the existing CLI (`glados start` /
`glados tui`).

---

## 1. Design goal

A browser "mission control" that streams the engine's real live state
(`ObservabilityBus` events, `MindRegistry`, task slots, emotion, queues, MCP,
audio/vision). It must run **in-process** (the live state only exists inside the
running `Glados` object), on its own port, behind a config flag that is
**off by default**.

The hard constraint is the user's own wording: *do not mess up the CLI version.*
The cleanest way to guarantee that is **option 1 — keep the core engine
decoupled.** The webapp is not part of the engine at all. A dedicated
`glados webapp` command — mirroring how `glados tui` is a launcher — builds the
engine, starts the web server alongside it, and owns its lifecycle. The engine
holds no webapp knowledge.

- `glados start`, `glados tui`, and `glados say` are byte-identical in behavior
  and never bind a port.
- The webui and the TUI are mutually exclusive UI options: you run either
  `glados webapp` or `glados tui`, never both. Because that isolation is at the
  launcher boundary (separate CLI commands), it needs no coordination inside the
  engine.
- Because the console is the whole point of the `glados webapp` command, a
  disabled config or a bind failure **aborts that command** rather than silently
  degrading — a behavior difference that is confined to the launcher.

---

## 2. Components

```
┌────────────────────── glados webapp (launcher in cli.py) ──────────────┐
│  1. GladosConfig.from_yaml(...)  (reads webapp: + GLADOS_WEBAPP_*)      │
│  2. refuse to run if webapp.enabled is false                            │
│  3. Glados.from_config(config)                                          │
│  4. WebappServer(engine, host, port).start()   ── bind-or-abort         │
│  5. engine.run()   try/finally ⇒ server.shutdown()                      │
│                                                                         │
│     ┌─────────────────────── core Glados (decoupled) ───────────────┐  │
│     │ ObservabilityBus ──► subscribe() fan-out                       │  │
│     │ MindRegistry / slots / emotion / audio ...                     │  │
│     └───────────────────────────▲───────────────────────────────────┘  │
│                                  │ reads live state only               │
│   browser ◄──HTTP/SSE────────────┘                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

- The server never holds engine locks; it only *reads* snapshots of state.
- It runs on a daemon thread inside the launcher's process (i.e. the engine's
  process), so it can touch the thread-safe live objects directly.

---

## 3. The critical change: bus fan-out

The webui is **multi-consumer**, but `ObservabilityBus` is natively
single-consumer (`drain()`). So we added a private per-subscriber path:

- `subscribe()` → returns a per-consumer `queue.Queue`; `unsubscribe(queue)`
  releases it.
- `publish()`/`emit()` still append to `_history` and push to the built-in FIFO
  (the TUI's `ObsScreen`), and now **also** push to every live subscriber queue
  (bounded; oldest dropped on lag).
- The SSE handler calls `subscribe()` per connection and reads its own queue, so
  browsers never steal events from the TUI or each other. `drain()` and
  `snapshot()` are untouched.

This is additive and does not change any `drain()` caller, so the TUI is safe.

---

## 4. Webapp server lifecycle (server ↔ main interaction)

**Decision: tied to the launcher's flow, not the engine.**

- `run_webapp()` in `cli.py` builds the engine, starts `WebappServer`, then runs
  `engine.run()`.
- `try/finally` guarantees `server.shutdown()` on every exit (including
  KeyboardInterrupt, which `run()` already converts into a graceful shutdown).
- Because the server is *only* started by the launcher, no other code path —
  `start`, `tui`, `say`, tests — ever binds a port.
- `GladosConfig.webapp` stays `WebappConfig | None = None` with
  `WebappConfig.enabled = False` by default, so a default config is unchanged.

---

## 5. Failure-handling rules

- Config has no `webapp` or `enabled: false` → `run_webapp()` logs an error and
  exits 1 (a console-less webui-launch is useless).
- Port already in use → `WebappServer.start()` returns not-running; the launcher
  logs an error and exits 1.
- Unhandled exception in a request → JSON error body; never crashes a worker
  thread, never takes down the engine.
- SSE client drops → `unsubscribe()` and free resources (bounded backlog).

---

## 6. API surface

| Endpoint | Purpose |
|---|---|
| `GET /` | serves `static/index.html` |
| `GET /api/snapshot` | aggregate state (minds, slots, agents, lanes, audio, emotion, mcp, interaction, vision, commands) |
| `GET /api/state` | lightweight state (`build_state`) |
| `GET /api/stream` | SSE: replays `bus.snapshot()` then live `obs` events + periodic `state` pings (0.5 s) |
| `GET /api/minds[/{id}[/memory]]` | per-mind detail + jsonlines memory |
| `GET /api/slots[/{id}]` | slot card + full report |
| `GET /api/agents` | agent status list |
| `POST /api/command` | drive engine commands (mirrors the TUI command palette) |

**Event contract:** `{timestamp, source, kind, level, message, meta}` — exactly
the TUI renders.

---

## 7. CLI interaction (start vs tui vs webapp)

| Command | engine built by | webui server | shutdown |
|---|---|---|---|
| `glados start` | `start()` | never | `_graceful_shutdown` from `run()` |
| `glados tui` | `instantiate_glados()` worker | never | TUI sets `shutdown_event` → `run()` exits |
| `glados webapp` | `run_webapp()` | started (bind-or-abort) | `finally: server.shutdown()` |

The first two never touch a port; only the dedicated `webapp` command starts the
server. no cross-coupling in the engine.

---

## 8. Implementation checklist

1. `ObservabilityBus.subscribe()/unsubscribe()` fan-out (keep `drain()`/`snapshot`).
2. Remove the earlier webapp coupling from `Glados` (constructor param,
   `disable_webapp()`, `_start_webapp()`, `run()`/`_graceful_shutdown` hooks).
   Drop the TUI's `disable_webapp()` call.
3. Add `run_webapp()` and the `webapp` subcommand (+ shared CLI args) in `cli.py`,
   wired into `main()`.
4. `WebappServer` remains engine-agnostic (duck-typed engine) with a
   daemon/`serve_forever` thread; `is_running` reflects bind success.
5. Ship `static/index.html` dual-mode (live fetch + dummy fallback).
6. Tests: bus fan-out, serializers, HTTP smoke test, and launcher tests
   (refuses-when-disabled, starts-then-shuts-down).
7. `docs/webapp.md` documents the dedicated command + mutual exclusivity.

---

## 9. Decision log

- **Option 1 (decoupled, adopted):** the engine holds no webapp knowledge; a
  separate `glados webapp` launcher owns the server lifecycle. Cleanest answer to
  "how should the server and `main` interact without messing up the CLI."
- **Rejected — engine-coupled best-effort (`_start_webapp()` at top of
  `run()`):** a bind failure only warns, so `glados start`/`tui` could
  unintentionally start a console, and the engine gets private webui knowledge.
