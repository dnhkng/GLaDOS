# Architecture simplification: decisions and acceptance criteria

Reviewed **2026-10-11** against `codex/ui-mockup-refinements` at `21dc152`.
The [review](architecture-simplification-review.md) is now the current status
inventory; it is no longer an unchanged pre-cleanup proposal. This document
records the constraints that matter when implementing its remaining work.

## Completed and verified in source

- B1–B5 fixes, B6's benchmark correction and dead-code cleanup are implemented.
- ProcessorTurn/ResponseState, a shared draft/reply request builder, pipeline
  phases, routing builders and bounded research helpers are implemented.
- MindScheduler owns mind timing/lifecycle with external fixed, adaptive,
  random-adaptive and on-demand policies. Minds need not use a model.
- Memory recall uses scheduler triggers; Search's mind has no periodic no-op tick.
- Interaction holds defer new background inference without pausing cheap probes.
- The current branch additionally preserves turn-specific tool catalogs,
  terminal tool metadata, clean post-gap audio, HN verdicts/legacy migration and
  spoken-text edge cases. The web interface adds basic/detailed/Presence surfaces.

The AST scan finds no Python method/function over 200 lines (largest: research,
185). Processor `run` is 17 lines; the processor is still one worker per instance.
The old 631-line-loop and partial converter-test results describe superseded
working trees. See [progress](architecture-refactor-progress.md#validation) for
current verification evidence.

## Boundaries to preserve

### Minds, slots and events

A mind produces evidence. MindScheduler controls when it runs; InferenceScheduler
controls whether its model request can start. A slot is neither worker nor GPU
reservation. Execution status belongs to runtime telemetry; regular/important
findings and reports belong to slots. Keep one publication operation that saves
state and emits its update. Regular updates can withdraw/revise an alert without
forcing a new announcement.

EventBus carries review-worthy publications; ObservabilityBus carries UI/log
telemetry. Domain callbacks and tool queues serve local request/result boundaries.
They are not interchangeable event frameworks that need a universal bus.

Preserve latest-trigger coalescing, single-run-per-mind, manual run while paused,
independent stop and stale-result rejection. Fixed delays restart after completion;
Vision reuses one random draw while live motion adjusts its pending deadline.
No catch-up backlog is required.

### Autonomy review and delivery

Keep reviewer → Central → acknowledged delivery. A reviewer offers no tools and
cannot speak directly. All-slot visibility includes healthy/recovered state and
waiting task items; bounded reports should not conceal queued work. Evidence
versions captured at admission must be checked again before handoff/delivery.

An attention key identifies an incident despite wording/temperature changes.
Revision identifies exact evidence. Handled state retires a task. The delivery
ledger records what actually reached speech/display. A single announced revision
cannot replace these four meanings. Greeting eligibility/freshness can be code;
whether to greet remains a conversation-aware decision.

A future dedicated reviewer worker must keep event ingestion responsive while
waiting/running inference. Use bounded work and monotonic deadlines; continuous
slot traffic must not starve periodic checks. Do not perform network I/O while
holding the loop's state lock.

### Requests, routing and cancellation

Draft and normal reply must offer equal tools/instructions for equivalent reply
requests, snapshot live providers once per prepared request, and keep raw media
out of stored history/inspection. The clock is transient live context; local
clock questions need no tool. Direct-audio speculation remains disabled pending
an explicit quality/latency experiment.

A routing permit must retain list/option/settings revision, tool scope and fixed
arguments. Recheck availability and authorization immediately before execution.
Per-turn offered tools remain the authority during delayed tool continuations.
A generic allowed-tool set must not widen fixed-action or read-only fallback
permissions. Preserve custom saved lists and migration behavior.

Admission holds and interaction busy state differ: the hold may release at first
valid audio readiness while response streaming or playback continues. Capture,
routing, responding, queued speech, playback and quiet/autonomy cancellation need
explicit ownership. Queue draining bounds memory; generation checks reject stale
work. Neither replaces the other by itself.

### Tools, shutdown and persistence

The native four-worker pool has bounded admission and atomic one-terminal-result
queues. A timeout must free dispatch and suppress all late outcomes, including
metadata/context. Four hung Python workers exhaust native capacity and may hold
interpreter exit; cancellation cannot kill them. MCP/search deadlines are
separate contracts. A future result-returning Tool interface should migrate
queue adapters at its boundary, not remove them before callers change.

Shutdown has component priorities, deadlines, joins and cleanup. Document abort
versus finish policy and report surviving workers; a 30-line replacement is not
behaviorally equivalent without tests.

Atomic replacement is not a transaction policy. Define single-writer ownership
or lock reload/update/save before sharing persistence helpers. Preserve history
record-identity compaction, external-edit invalidation, memory provenance and
user settings. The SubagentMemory pre-lock truncation bug is fixed; stale-instance
lost updates remain. The sync helper's running-event-loop deadlock also remains.

## Compatibility/product decisions

Retain direct processor calls without a ContextBuilder, public web routes,
Ollama's native transport and TTS `reuse_tts` until callers/features migrate.
TUI, Observer/Constitution and Haar/E4B face modes are supported features, not
zero-risk deletion candidates. Immutable constitution rules currently are not
injected; prompt modifiers are. Observer does not yet have its own enable flag.

Deferred transcripts, flat routing, native embedding and Rust/AEC migration are
experiments or future work. The faster tested shared-transcript path remains.
Keep historical measurements with their dates, hardware and tested revisions;
do not present old configuration or synthetic timings as current runtime facts.

## Remaining implementation sequence

1. Common inference transport plus a bounded dedicated reviewer. Preserve provider
   headers/options, schemas, candidate scores, streaming, admission and cancellation.
2. Interaction ownership and result-returning tools. Test every cancellation
   boundary, offered-tool authorization, timeout saturation and one terminal result.
3. Persistence transactions and the async/sync helper bug. Test concurrent writers,
   external edits, restart/migrations and explicit loop usage.
4. Runtime status projection and focused engine/Memory housekeeping. Preserve
   execution/evidence separation and existing controls without adding orchestration.

Mind timing consolidation is already complete and should not be scheduled again.
The [progress document](architecture-refactor-progress.md) records the same order.

## Verification scope

Regression modules cover engine startup, request parity/media isolation, routing,
voice continuation, capture discontinuities, tool timeouts/authorization, mind
scheduling, interaction holds, memory and Autonomy evidence/delivery. The latest
full run recorded before this documentation pass is **1,049 passed** on
2026-10-10; the extended-frame update passed 16 focused webapp tests on 2026-10-11.
This documentation pass runs static code/config/link checks and no GPU inference.
A passing test suite is not a fresh end-to-end voice-latency or hardware benchmark.
