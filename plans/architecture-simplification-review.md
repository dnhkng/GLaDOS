# Architecture Review: Simplification Opportunities

Branch: `feat/websocket-audio` (12 commits ahead of `main`, about +94k/−3k lines, 27k LOC of Python in `src/`).
Date: 2026-10-07. Read-only review; nothing in `src/` was changed.

## TL;DR

The core ideas are good and worth keeping:

- the Society-of-Mind slot contract;
- one `InferenceScheduler` as the only admission point;
- generation tokens for cancellation;
- the stable/volatile context split for KV reuse;
- one-token logprob classification;
- the two-stage autonomy reviewer, then Central.

The complexity comes from **mechanisms that were added beside these ideas instead of replacing what they made redundant**:

- 4 LLM-calling paths;
- 4 pub/sub mechanisms;
- 4 "what are the minds doing" registries;
- 3 turn-cancellation mechanisms;
- 5 stacked tool filters;
- quiet/wake classification defined in 3 places;
- about 10 persistence formats.

On top of that sits one 645-line `run()` method.

Realistic reduction is **about 4,000–5,000 LOC** (about 15–18% of `src/`), plus about 1.4k more if the TUI is retired. It would also remove about 8 threads, 2 queues and 2 LLM client paths, with no loss of features.

Context for prioritising: `docs/benchmarks/voice-latency-2026-10-06.json` shows **2.3–3.0 s from input to playback**, against the README's 600 ms target. Routing costs 125–316 ms of that on the critical path. Simplifications that shorten the hot path deserve to go first.

---

## 0. Bugs found during the review (verified)

| # | Where | Problem |
|---|---|---|
| B1 | `core/engine.py:537` → `core/text_listener.py:31-41` | `TextListener(turn_is_current=...)` is passed, but `__init__` has no such parameter. `input_mode: text` or `both` raises `TypeError` at startup. |
| B2 | `core/tool_executor.py:326` | `with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` on exit, so `tool_timeout` reports a timeout but the single ToolExecutor thread still blocks on the hung tool. |
| B3 | `core/llm_processor.py:1007` | The speculative draft uses only `_build_messages(False)` and skips the `_add_request_context` injections at 1152-1220. A reply released from the draft sees different context than a normal reply. |
| B4 | `audio_io/websocket_io.py:363-391` | Dropped mic chunks never set a capture discontinuity (`sounddevice_io.py:99-110` does). After an overflow, two speech fragments can be glued into one turn. |
| B5 | `autonomy/agents/hacker_news.py:55` | `SubagentMemory.mark_shown` is never called anywhere, so every stored story is re-evaluated and re-announced on every tick. Latent only because HN is disabled in all configs. |
| B6 | `core/llm_processor.py:1004-1006` | Speculation is guarded by `not _native_audio`, so it **never runs on the default voice profile** (native audio on). The next line's `llm_input.get("_native_audio") or …` is therefore dead. The voice-latency benchmark notes say "routing overlaps speculative inference" for transcript-off voice turns. Check which is true. |

---

## 1. Fast decisions (one-token routing and option scoring)

I took "fast decisions on inputs" to mean the one-token logprob classifier: `core/routing.py`, `routing_tree.py`, `decision_lists.py` and `option_scores.py`, shared with Emotion's PAD scoring. Scoring constrained letter tokens with `logit_bias` and reading `post_sampling_probs`, then requiring threshold + margin, is a strong design:

- It is fast.
- It yields calibrated-ish probabilities.
- It degrades safely to `assist`.
- `option_scores.py` (57 LOC) is a model of compactness.

The complexity sits around it.


### F1. Quiet/wake/noise is defined three times, with two code paths

- **The three definitions:**
  - `routing.py:79-99` (`quiet_score`, the flat or quiet path);
  - the `quiet_control` node in `routing_tree.py:153-163`;
  - the root options in `routing_tree.py:165-174`.

  Each has its own copy of the "do not sleep means continue" instruction text.
- **The callsite:** `llm_processor.py:975-997` (gate) and `1040-1046` (route) both handle quiet/wake transitions, with duplicated generation-bump code.
- **Simpler:** quiet, wake and ignore are just options of the active decision list, defined once in `default_list()`. While quiet, score the same list restricted to `{wake, stay_quiet}`. One code path. About −80 LOC.

### F2. Prompt construction by `str.replace`

`routing_tree.py:229-235` and `253-256` rewrite instructions by searching for exact earlier sentences. If anyone edits the wording, the replacement silently does nothing, and the router quietly reverts to the old behaviour. Build the instructions from named parts, for example a dict of sentences or small functions keyed by condition.

### F3. Two permit systems plus five stacked tool filters

- **Router side:** routing issues two kinds of permit:
  - `authorize` checks `list_id, revision, option_id`;
  - `authorize_scope` checks `list_id, revision, settings_revision, tool_scope`.

  Both are re-checked in the processor (`1104`, `1119`) and again in the executor (`tool_executor.py:132-139`).
- **Processor side:** `llm_processor.py:1123-1140` then applies five filters:
  1. `_build_tools`
  2. `_reply_tools`
  3. the routing scope
  4. a hard-coded read-only allowlist
  5. a keyword heuristic (`_filter_tools_for_message`, `llm_processor.py:379`, which matches substrings such as "ip", "temp" and "load")

  The keyword filter contradicts the point of having a learned router.
- **Simpler:**
  - The router returns `allowed_tools: frozenset[str] | None`, and one function produces the tool list from it.
  - A permit is a single `(settings_revision, allowed_tools)` tuple.
  - Delete the keyword heuristic.
  - About −120 LOC.

### F4. Routing is coupled to Health Core and Memory Core

The `health_metrics` and `recalled_topic` callbacks rewrite tree text and option descriptions (`routing_tree.py:227-272`), so the router must know what each core supplies.

Simpler: each context source declares what it answers, for example `ContextSource.answers = ["host CPU/RAM/disk readings"]`. The router appends one generic line, "Already supplied in response context: …; choose reply for these". New cores then need no router edits.

### F5. Migration code in a pre-release store

`DecisionListStore.__init__` (`decision_lists.py:134-166`) migrates "obsolete clock" options and an old `speculative` switch. These are artefacts of this branch's own history. Delete them and regenerate `data/` settings. About −35 LOC.

### F6. Three "decision" mechanisms for background minds

Background decisions currently use three different routes:

- Emotion uses `option_scores` with a hand-rolled lease (`emotion_agent.py:207-233`).
- HN and Weather use `core/llm_decision.llm_decide_sync`, an async function bridged back to sync. It has an unused 4-worker pool (`llm_decision.py:38`) and five prebuilt schemas, two of which are used.
- Everything else uses `autonomy/llm_client.llm_call`.

Simpler: `llm_client` exposes two functions, `llm_call(cfg, messages, schema=None)` and `llm_choice(cfg, messages, options)`, which wraps `option_scores` with a scheduler lease. Boolean urgency decisions such as HN/Weather `notify_user` become `llm_choice`. Delete `core/llm_decision.py`. About −200 LOC.

### F7. Rules that could be code, inside the autonomy reviewer prompt

- **Current:** `autonomy/decision.py:26-77` holds about 50 lines of policy:
  - a dinner/steak/spaghetti example;
  - exact greeting triggers (`age_s at most 60`, `first_seen_today before 10:00`);
  - "do not attach healthy Health readings".
- **Problems:**
  - The greeting rules are deterministic, yet a 4B model is asked to compare integers.
  - The same policy is repeated in `autonomy/config.py:139-161` and `loop.py:257-274`.
- **Simpler:**
  - Vision emits a `greeting_due` slot flag when `person_arrived and age_s <= 60 and not greeted` (or first-seen before 10:00). The LLM only chooses the wording.
  - Filter healthy Health and Observer slots out of reviewer evidence in code.
  - The prompt shrinks to its real job, "is anything here worth saying?". The reviewer becomes cheaper and less flaky.

### F8. Speculative drafting: fix it or remove it

- **Current state:** `SpeculativeStream` (98 LOC plus branches in `run()`) only runs for text input, on llama.cpp, with more than one slot, and without native audio (B6). It also builds different context from the real request (B3).
- **If you keep it:** build the draft from the same `build_reply_request()` used for the final request, and allow native audio.
- **Otherwise:** remove it. Flat routing (F1) takes about 60–130 ms, which removes most of the reason to speculate.

Small items:

- `routing.py:280` `total = 1.0` is dead, because `request_scores` already normalises.
- `import json` is inside `_score_step` (`routing.py:249`).

---

## 2. Core runtime

### C1. `LanguageModelProcessor.run()` is 645 lines (`llm_processor.py:856-1501`)

It handles all of the following in one loop:

- intake;
- Gemma transcript;
- the quiet gate;
- routing and speculation;
- tool filtering;
- about 70 lines of inline prompt text;
- provider quirks (the native Ollama protocol and its fallback loop at 1269-1418);
- SSE parsing;
- two thinking formats;
- speech markup and chunking;
- tool-call accumulation;
- autonomy JSON handoff;
- history writes, observability and hold bookkeeping.

About 12 per-request fields live on `self`, which makes the class non-reentrant. That is why parallelism needs N instances with about 33 constructor arguments each.

**Split it into:**

- `Turn`, a per-request dataclass;
- `build_reply_request(turn, route) -> (messages, tools, body)`, a pure, testable function also used by the draft;
- `stream_completion(body)`, which yields text and tool-call events;
- `ResponseSink`, for TTS and history.

Move the prompts to `core/prompts.py`. Drop the native Ollama `/api/chat` protocol, because Ollama serves `/v1/chat/completions`.

Target: `run()` under 150 lines, about −250 LOC net.

### C2. Autonomy does not need `LanguageModelProcessor`

The autonomy lane shares nothing with replies: a separate message builder, no tools, JSON output and no TTS. It still uses 2 processor threads, `llm_queue_autonomy` and `_autonomy_*` branches throughout `run()`.

Simpler:

- `AutonomyLoop` calls `llm_client.llm_call(..., schema=decision_schema)` directly.
- Merge `AutonomyTicker` into the loop: a timeout on `event_bus.get` is the tick.

About −200 LOC, −3 threads, −1 queue.

### C3. One `TurnState` in place of three cancellation mechanisms

A turn is currently invalidated by three things:

- generation tokens;
- the global `processing_active_event` (49 references, shared by both lanes);
- queue draining.

The staleness check `quiet_mode() or generation != quiet_generation()` is repeated 23 times in the processor, 9 in TTS and 11 in the player. 17 lambdas are threaded through constructors. "User is busy" is computed in four places: `engine._autonomy_user_busy` (which reads a private field), Health, Compaction and the scheduler hold.

Simpler:

- Pass one `TurnState(generation, autonomy_generation, quiet, autonomy_enabled)` everywhere, with `is_current(turn)` and `bump()`.
- Drop `processing_active_event`.
- Make `InferenceScheduler.interactive_busy()` the only busy test.
- Remove `llm_tracking.py`.

About −250 LOC across core and I/O. Medium risk, but it eliminates a class of stale-flag bugs.

### C4. `engine.py` (1996 LOC)

**Responsibilities today:**

- the config schema;
- wiring about 30 objects;
- registering 8 agent types;
- the turn state machine;
- 18 slash commands (about 640 LOC, 1309-1949);
- formatters;
- `/memory` parsing (a third hard-coded copy of `~/.glados/memory`).

The init order forces `getattr(self, …)` workarounds in 7 places.

**Simpler:**

- `Glados(config)` takes the config object (removes 25 parameters and `from_config`).
- Move commands to `commands.py`.
- Make agent registration a factory.
- Always create `SubagentManager`.
- Register vision, MCP and clock as `ContextBuilder` sources so there is one context mechanism.

The engine shrinks to about 700 LOC.

### C5. Tools return values; the executor builds messages

All 10 tools receive `llm_queue` and push their own results, which needs `_ToolResultQueue` and `AutonomyQueue` adapters. Change this to `Tool.execute(args) -> str`, with one long-lived executor pool, which also fixes B2.

The `speak` and `do_nothing` tools and the whole autonomy branch of ToolExecutor are unreachable (`_build_tools` is only called with `False`). Delete them.

### C6. Shutdown (`shutdown.py`, 319 LOC)

The phased "drain" discards queue items after `shutdown_event` is already set, so it does not preserve in-flight work as documented. Setting the event, clearing the queues and joining with deadlines does the same job in about 30 LOC.

### C7. Persistence and dead code

- **Atomic writes:** there are four different atomic-write implementations. Replace them with one `atomic_write()`.
  - `ConversationStore` rewrites the whole history on every append, and keeps `_messages` and `_metadata` in sync by hand.
  - `KnowledgeStore` re-reads JSON on every request.
- **Dead modules and functions:**
  - `memory_context.py` (171 LOC, never imported);
  - `context.py`: `build_combined_prompt`, `build_system_messages` and others;
  - `store.py`: helpers, plus a misleading docstring;
  - `llm_processor.py:710-720`, a fallback that cannot run;
  - unused `shutdown.py` API;
  - `NEUROTOXIN_RELEASE_ALLOWED`.
- **Dead config:**
  - `autonomy.coalesce_ticks` is forced to True by a validator but still shown in the UI;
  - `jobs.poll_interval_s`;
  - `tokens.estimator` and `chars_per_token`;
  - the hard-coded `MINIMAX_API_KEY` fallback.

---

## 3. Autonomy subsystem

Counts:

- 8 agents.
- About 20 threads at default.
- 4 pub/sub mechanisms: `EventBus`, `ObservabilityBus` (with drain and subscribe), direct pushes, and callbacks.
- 6 scheduling mechanisms.
- 4 status registries: `TaskSlotStore`, `MindRegistry`, `SubagentManager.list_agents`, and per-agent `snapshot()`.
- About 10 stores.

### A1. Delete code that is already dead (about −750 LOC, very low risk)

- `jobs.py` (308 LOC, not imported; it duplicates the HN and Weather agents).
- Most of `summarization.py` (used only by tests).
- `TiktokenEstimator`, `create_estimator` and `set_default_estimator`.
- `EmotionConstitutionBridge`.
- The `VisionUpdateEvent` branch: it is never published, so `loop.py:381-388` and `458-474` and `VISION_EMOTION_THRESHOLD` are dead.
- The unused `_personality_prompt`.
- `AutonomyLoop.update_slot`.

### A2. Fold `MindRegistry` into slots

`Subagent.run` writes the same title, status and summary to both stores. Keep slots as the single registry and add `running`/`tick_count`. Simplify `SubagentManager` to a list plus start/stop. About −200 LOC.

### A3. Unify scheduling

Current examples of the different patterns:

- Search is a Subagent that wakes hourly to do nothing, while its real work runs in TaskManager's special `search` executor.
- Memory recall and Health commentary spawn raw threads.
- Emotion overrides tick timing.
- Each task gets its own heartbeat thread.

Simpler:

- `Subagent(interval_s: float | None)`, where `None` means event-driven.
- All on-demand work goes through `TaskManager.submit`, with per-group `max_workers`/`max_queue` and one shared progress thread.

About −150 LOC.

### A4. Split `CompactionAgent` (521 LOC)

It is really two agents, Memory recall and Compaction. It has two recall entry points, a `write_slot` override that splices recall into the compaction summary, and 13 recall-state fields. Two agents with two slots is simpler and needs less locking.

### A5. Fewer dedup ledgers in `AutonomyLoop`

`_seen_updates`, `_announced`, `_active_updates`, `_active_slot_versions`, signatures and attention keys sit on top of the slot's own `revision`/`handled`. Store `announced_revision` on the slot and compare it with `revision`. About −150 LOC. This logic is benchmarked, so do it behind the existing tests.

### A6. Slots and persistence

- **Slot fields:** `TaskSlot` has 18 fields with two priority signals, `notify_user` (labelled "compatibility") and `update_priority`. Keep one. There are also two slot formatters (`TaskSlotStore.as_message` and `engine._format_slots`).
- **`SubagentMemory`:** 207 LOC, with Windows/Unix file locks, constructed for every agent but used only by Emotion and HN. Replace it with one JSON file per agent that needs one.
- **Directory layout:** move agent caches out of `~/.glados/memory/`, which also holds the user's facts.

### A7. Observer and Constitution: decide whether to keep them

- **Current:** Observer makes an LLM call every 2 minutes to tune 5 numeric knobs. They become one prompt line each and are not persisted.
  - It has no flag of its own; it is enabled by `jobs.enabled`.
  - It marks every adjustment as notify-worthy, which the reviewer prompt then has to suppress.
  - The constitution's `immutable_rules` are displayed but never injected.
  - Emotion already modulates tone.
- **Option 1:** delete it (about −550 LOC).
- **Option 2:** give it its own flag, set `update_priority=regular`, and either inject the rules or drop them.

---

## 4. I/O, vision and UI

### I1. The `AudioIO` ABC leaks

- **Duplicate types:** a parallel `AudioProtocol` (`audio_io/__init__.py:25-60`) is what callers are actually typed against.
- **Duck-typed methods:** callers use `getattr` for five methods that only the sounddevice backend implements:
  - `ensure_listening`
  - `consume_capture_discontinuity`
  - `capture_health`
  - `device_snapshot`
  - `select_device`
- **Unused or split API:**
  - `check_if_speaking` is never called.
  - `start_speaking` and `measure_percentage_spoken` are always called as a pair.
- **Simpler:**
  - Delete the Protocol.
  - Give the ABC no-op defaults for the optional methods.
  - Merge playback into one `play() -> (interrupted, pct)`.
  - Move VAD and the bounded drop-oldest queue (which marks discontinuities) into the base class. That fixes B4.

About −120 LOC.

### I2. One `UserTurn` and two encoders

The native-audio and Parakeet branches in `speech_listener.py:391-437` build the same envelope. `_native_audio` is special-cased in about 10 places downstream. The optional Gemma transcript is really "Gemma as ASR" with an extra round trip.

Simpler:

- The listener emits a typed `UserTurn(pcm, …)`.
- An `AsrEncoder` or a `NativeEncoder` turns it into message content.
- Gemma transcription becomes another `Transcriber`.

Underscore-key metadata on message dicts (`_lane`, `_spoken`, `_quiet_generation`, …) becomes `Turn` fields, converted to a message only at the HTTP boundary. This pairs naturally with C3.

### I3. UIs

- **Two state builders that overlap:**
  - `serializers.build_state` and `build_snapshot` repeat about 15 keys.
  - `engine._health_runtime_status` is a third partial builder.
  - Every key is wrapped in `getattr(engine, …, None)`, which hides typos.
- **Proposed fix:** add one typed `engine.status()`, and make `snapshot` equal `state` plus extras.
- **Unused routes:** `/api/state`, `/api/minds` and `/api/agents` are not used by the frontend or tests.
- **The TUI** (1479 LOC) reaches into engine internals and has fallen behind: it has no quiet mode, vision mind, native audio or decisions. Either rebase it on `engine.status()` with bus `subscribe()` (which also drops the bus's legacy `drain` FIFO), or retire it in favour of the webapp.
- **`api/`** (Litestar TTS server) is isolated and fine. Drop the `reuse_tts` knob.

### I4. Vision

The scheduling is sound: an admission lease *before* frame selection, background ticks in the autonomy lane, and questions in the priority lane. Possible trims:

- Drop the Haar face backend.
- Drop the E4B face mode, which removes a coupling in `_failed`.
- Keep a single source of truth for intervals (it is currently split between config and `data/vision_settings.yaml`).
- Remove the `interval_s` migration validator.

---

## 5. Keep these

- **`InferenceScheduler`** (141 LOC): one Condition, a single admission point, and lanes plus a reserved interactive slot. Make the 120 s hold timeout configurable.
- **The one-token option-scoring primitive** (`option_scores.py`), including threshold + margin + fallback.
- **`TaskSlotStore.update_slot`**: one place to save and publish, revision bumps only on content change, and `handled` retirement.
- **The two-stage autonomy decision**: a schema-validated reviewer, then Central speaking only after the slot evidence is revalidated.
- **Context handling:** the `ContextBuilder` stable/volatile split, `context_budget.reduce_request_context` (shrinks only the request copy after a reported overflow), and `describe_context` provenance.
- **`ConversationStore.compact()`**, with record-identity optimistic concurrency.
- **Health** edge-triggered alerts using `attention_key`.
- **`ObservabilityBus`** fan-out with SSE.
- **`speech_markup` / `speech_chunking` / `speech_animation`**: small and single-purpose.
- **Small dependencies:** stdlib `http.server` for the webapp, and flock-based `memory_records`.

---

## 6. Suggested order

| Step | Work | LOC | Risk |
|---|---|---|---|
| 1 | Fix B1–B5; check B6 | ~+40 | low |
| 2 | Delete dead code: A1, C5 (speak/do_nothing/autonomy executor), C7, F6, unused routes, `reuse_tts` | −1,400 | very low |
| 3 | One LLM client with `llm_call` + `llm_choice` (F7); autonomy off `LanguageModelProcessor` (C2); merge the ticker | −400 | low-med |
| 4 | Fast-decision cleanup: single quiet/wake definition (F2), single tool filter/permit (F4), composable prompts (F3), then F1 option A *or* B, measured with the existing routing benchmarks | −300 to −700 | med |
| 5 | `TurnState` + `UserTurn` (C3, I2); AudioIO ABC defaults (I1) | −450 | med |
| 6 | Split `run()` (C1); slim the engine (C4); `Tool.execute()` (C5); shutdown (C6) | −700 | med-high |
| 7 | Autonomy structure: MindRegistry → slots (A2), scheduling (A3), Compaction split (A4), loop ledgers (A5) | −600 | med |
| 8 | Product decisions: Observer/Constitution (A7), TUI future (I3) | −550 to −2,000 | — |

Steps 2–3 are mechanical and roughly halve the number of moving parts someone has to understand. Step 4 is where the latency budget is, so measure it against `docs/benchmarks/routing-*.json` and `voice-latency-*.json` before and after.

---

## 7. Verification of the cleanup list (step 2)

Checked by reference search across `src/` and `tests/` on 2026-10-07. Not checked by running the tests.

**Confirmed safe to delete. No references in `src/`, apart from the definitions, docstrings and their own module.**

- `core/memory_context.py`.
- `autonomy/jobs.py`.
- `iter_messages`, `list_sources` and `build_combined_prompt`.
- `build_system_messages` (its only reference is a docstring).
- `NEUROTOXIN_RELEASE_ALLOWED`.
- `_personality_prompt` and `build_personality_prompt` (assigned, never read).
- `AutonomyLoop.update_slot`.
- **`VisionUpdateEvent`** is never constructed, so the `loop.py` branches at 381-388 and 458-474 and `VISION_EMOTION_THRESHOLD` are dead.
- **`speak` / `do_nothing`** are filtered out whenever `autonomy_mode` is false, and every `_build_tools` call passes `False`. Autonomy requests send `[]`. Neither tool can reach the model.
- **The `context_builder is None` fallback** (`llm_processor.py:710-720`) can't run, because the engine always passes `context_builder`.
- **Webapp routes:** GET `/api/state`, `/api/minds` (list) and `/api/agents` are not used by the static JS or by the tests. The JS uses only `/api/minds/control` and `/api/minds/<id>/memory`.
- **Dead config:**
  - `coalesce_ticks` is forced to True by a validator.
  - `jobs.poll_interval_s` only feeds `jobs.py`.
  - `tokens.chars_per_token` and `tokens.estimator` only feed `create_estimator`, which is never called.
- **`set_emotion_agent` at `engine.py:928`** is redundant, because `engine.py:723` sets the same thing after the loop exists.

**Dead in `src/`, but have tests. Delete each together with its tests.**

- `summarize_messages` / `extract_facts`.
- `TiktokenEstimator` / `create_estimator` / `set_default_estimator`.
- `EmotionConstitutionBridge`.
- In `shutdown.py`: `is_shutting_down` / `get_results`.
- `ConversationStore.deep_snapshot`.

**Corrections to the report:**

- **`check_if_speaking`** is used inside `websocket_io.py` (538, 554). Only remove it from the ABC, and keep the method on the websocket backend.
- **The shutdown "drain"** (`shutdown.py:168, 197-215`) does discard items with `get_nowait()` after `shutdown_event` is set (134). The claim holds, but simplifying it changes documented behaviour, so it belongs in step 6, not the zero-risk step.

**Bugs re-checked in the code:** B1 (TextListener kwarg), B2 (pool `shutdown(wait=True)`), B4 (no discontinuity on websocket drop) and B5 (`mark_shown` never called) are confirmed. For B3, the draft is built from `_build_messages(False)` alone (`llm_processor.py:1007`). It only runs for non-native-audio input (1004).

---

## 8. Bug details and proposed fixes

### B1. Text input crashes on startup

- **Cause:** commit `0ec0cc4` added `turn_is_current=` to both listener constructions in `engine.py` (523 and 537), but only `SpeechListener` accepts it (`speech_listener.py:65`). `TextListener.__init__` (`text_listener.py:31-41`) has no such parameter.
- **Effect:** `input_mode: text` or `both` raises `TypeError: unexpected keyword argument 'turn_is_current'` in `Glados.__init__`. The default `audio` mode is unaffected, which is why it went unnoticed.
- **Fix:** delete line 537. `TextListener` has no voice-continuation buffer, which is the only thing `turn_is_current` guards in `SpeechListener` (219), so it has nothing to check.
- **Test:** `tests/test_text_listener.py` builds the listener directly, so it never sees the engine's arguments. Add a smoke test that constructs the engine with `input_mode="text"` (stubbing models as the webapp tests do). As a cheaper alternative, test `inspect.signature(TextListener).bind(**kwargs)` against the arguments the engine uses.

### B2. A timed-out tool still blocks the executor, and its late result can still be delivered

- **Cause:** `tool_executor.py:326` runs each native tool inside `with ThreadPoolExecutor(max_workers=1) as executor:`. On `FuturesTimeoutError` the timeout message is queued, but leaving the `with` block calls `shutdown(wait=True)`. That waits for the hung tool.
- **Effects:**
  1. The single ToolExecutor thread stays blocked, so every later tool call (user or autonomy) queues behind the hung one. The `tool_timeout` setting only changes *when the error message is sent*, not when the executor is free again.
  2. Tools push their own results through `_ToolResultQueue` (450-476), which only drops them if `cancelled()` is true, meaning the turn was invalidated. If the user hasn't spoken again, the late result is delivered after the timeout error. The model then sees two tool results for one `tool_call_id` and replies twice.
- **Fix** (about 10 lines):
  ```python
  # __init__
  self._tool_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

  # per call, before building the tool instance
  timed_out = threading.Event()
  call_cancelled = lambda: cancelled() or timed_out.is_set()
  llm_queue = _ToolResultQueue(llm_queue, tool_call, bound=..., cancelled=call_cancelled)
  tool_config = {..., "_cancelled": call_cancelled}
  future = self._tool_pool.submit(tool_instance.run, tool_call_id, args)
  try:
      future.result(timeout=self.tool_timeout)
  except FuturesTimeoutError:
      timed_out.set()          # late results are now dropped
      ...                      # existing timeout message
  # shutdown: self._tool_pool.shutdown(wait=False, cancel_futures=True)
  ```
- **Limit:** Python threads can't be killed, so a truly hung tool keeps a pool thread. With `max_workers=4` that is a bounded leak. Log it.
- **Tests:**
  - A tool that sleeps longer than `tool_timeout`: a second tool call must finish before the first one returns.
  - Exactly one tool message is delivered for the timed-out call.

### B3. The speculative draft gets different context from a normal reply

- **Cause:** the draft is built at `llm_processor.py:1007` from `_build_messages(False)` alone. The normal path adds per-request system messages afterwards through `_add_request_context` (1152-1220). A draft is only kept when the route is `reply` (1034). In that case the normal request would include the **internet-search usage block** whenever search is among the reply tools: call search now, `numResults=2`, resolve dates as YYYY-MM-DD, prefer the user's News pages, no permission-asking. The draft offers the same tools (`_reply_tools()`) without those instructions.
- **Effect:** typed questions that need current facts behave differently depending on whether the draft won the race. For example, a drafted reply may answer from memory or ask permission instead of searching. It only applies to typed text, on llama.cpp, with more than one slot, because of the guard at 1004.
- **Fix:** move the request-context rules into one function, for example `_request_context(route, tools, llm_input) -> list[str]`, and call it in both places. Build the draft for the only route it can be released for: `tools = self._reply_tools()`, then insert the same blocks. Record `draft_sources` after the inserts so the context inspector shows what was actually sent. Alternatively, delete speculation (see B6).
- **Test:** with search available, assert the draft's `messages` and the non-draft request's `messages` are identical for a `reply` route.

### B4. Dropped WebSocket microphone audio isn't reported as a gap

- **Cause:**
  - `SoundDeviceAudioIO._queue_sample` (`sounddevice_io.py:99-110`) sets `_capture_discontinuity` when it drops a chunk. `SpeechListener` (149-153) consumes that flag, resets, and discards the half-captured utterance.
  - `WebsocketAudioIO._enqueue_microphone_sample` (`websocket_io.py:363-391`) also drops the oldest chunk, but only counts and logs it. It has no `consume_capture_discontinuity`, and the listener finds it through `getattr`, so the check silently does nothing.
- **Effect:** when the consumer falls behind (GPU busy, long ASR), the utterance loses audio from the middle but is still submitted as one clip. With native audio, the model hears a spliced clip. With Parakeet, the transcript can be garbled.
- **Fix, minimal:** copy the sounddevice pattern into `websocket_io.py`:
  - `self._capture_discontinuity = threading.Event()`;
  - set it when `dropped_chunks > 0`;
  - add `consume_capture_discontinuity()`, which clears the flag and the queue.
- **Fix, better (I1):** move the bounded drop-oldest enqueue and the discontinuity flag into the `AudioIO` base class, so every backend gets it and the `getattr` disappears.
- **Also check:** whether a change of microphone owner (`_clear_microphone_ownership`, ~355) should also mark a gap. Otherwise the end of one client's audio can join the start of another's.
- **Test:** fill the websocket queue past capacity and assert that `consume_capture_discontinuity()` returns True once, then False.

### B5. Hacker News re-evaluates and re-announces the same stories forever

- **Cause:** `hacker_news.py:55` iterates `self.memory.list_unshown()`, but nothing calls `mark_shown`. Every story ever stored stays "unshown". `SubagentMemory` prunes at about 100 entries.
- **Effects (if enabled):**
  - Each tick (default 30 min) makes one `llm_decide_sync` call per stored story, up to about 100 serial LLM calls on the autonomy lane.
  - The same titles are re-reported with `notify_user=True`.
  - Rejected stories are re-judged every tick too.
- **Fix:**
  ```python
  for entry in self.memory.list_unshown():
      ...                                   # evaluate as now
      if not relevant:
          self.memory.mark_shown(entry.key)  # judged once; never re-asked
  ...
  for story in to_report:
      self.memory.mark_shown(f"hn_{story['id']}")
  ```
  Relevant stories that miss the `top_n` cut stay unshown, but they would still be re-judged by the LLM. Store the verdict on the entry (`self.memory.set(key, {**story, "_relevant": True, ...})`) and skip the call when it is already there.
- **Alternative:** HN is disabled in every shipped config, and `jobs.py` holds a second, dead copy. Deleting the agent is equally valid and removes a `llm_decide_sync` caller (F7).

### B6. Speculation never runs for native audio, which the benchmark notes say it does

- **Cause:** the draft guard at `llm_processor.py:1004` requires `not llm_input.get("_native_audio")`. The next line, `content = llm_input.get("_native_audio") or …`, therefore always takes the text branch. That suggests audio drafting was once intended and later turned off.
- **Conflict:** `docs/benchmarks/voice-latency-2026-10-06.json` was measured on voice turns with transcripts disabled, which is native audio. Its notes say "Routing overlaps speculative inference". The guard and the benchmark arrived in commits two seconds apart, so git history can't say which came first.
- **To settle it:** run one voice turn and check the scheduler snapshot or bus for a `"GLaDOS draft"` lease.
- **Fix, either way:**
  - If drafting audio was turned off deliberately (two slots encoding the same clip, or slot pressure), delete the dead `_native_audio or` branch and correct the benchmark note.
  - If it should run, drop the guard, keep the audio in the draft's last user message, and re-measure. The draft and the router would each encode the clip, so check that the GPU contention doesn't cancel out the overlap.
