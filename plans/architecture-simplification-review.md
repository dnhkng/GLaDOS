# Architecture simplification review

Reviewed against the implementation on **2026-10-11**, branch
`codex/ui-mockup-refinements` at `21dc152`. This replaces the 2026-10-07
pre-cleanup review. Its original line numbers, reduction estimates and thread
counts were historical observations, not the current architecture.

The dead-code cleanup, confirmed bugs and structural extraction are implemented.
Background minds now share a scheduler with timing policies separate from their
work. The remaining work concerns inference transport, cancellation ownership,
result-returning tools and persistence; it is not another large-method extraction.

[Validated decisions and acceptance criteria](architecture-simplification-validation.md) ·
[Implemented work and remaining sequence](architecture-refactor-progress.md) ·
[Documentation map](../docs/README.md)

## Current design

- A mind performs one unit of work with `run(runtime)` and optionally publishes a
  regular/important update. It does not require an LLM.
- `MindScheduler` owns background deadlines, pause/run controls and a bounded
  four-worker pool. Fixed, adaptive, random-adaptive and on-demand policies live
  in `autonomy/mind_schedule.py`. Vision uses motion-biased 2–5 second timing.
- Slots hold semantic evidence, reports and task state. They are distinct from
  runtime execution status and model-server capacity.
- `InferenceScheduler` admits model requests across interactive/background lanes.
  The reference server has four slots with two reserved for interaction.
  Interaction holds defer new background inference; they do not stop camera
  capture, health probes, local decay or already-admitted requests.
- Autonomy consumes slot publications, reviews them with conversation evidence,
  and hands useful notifications to Central. Source revalidation and actual
  delivery acknowledgement prevent stale or repeated speech.
- `ProcessorTurn`, `ResponseState`, one request builder and explicit pipeline
  phases replace the old monolithic reply loop. Text drafts share reply
  instructions/tools; untranslated direct audio is deliberately not drafted.
- Stable prompts precede completed history, then live state/current input.
  Media is transient; raw audio/images are not persisted in conversation history.

## Source inventory

The current AST scan covers **134 Python files, 26,490 physical lines** in `src/`
(including comments/blanks, excluding frontend assets). No Python function or
method exceeds 200 lines. This is a maintainability inventory, not a latency or
thread-count measurement.

| Boundary | Current lines |
| --- | ---: |
| `LanguageModelProcessor.run` | 17 |
| `Glados.__init__` | 90 |
| `ToolExecutor.run` | 19 |
| `RoutingTree.__init__` | 10 |
| `SearchAgent.research` | 185 |
| `Glados._register_subagents` | 179 |

The engine is still a large wiring/command module. Short coordinator methods do
not make the processor reentrant or unify the underlying network clients.

## Original bugs: current disposition

| ID | Original concern | Current state |
| --- | --- | --- |
| B1 | Text/both startup passed a speech-only argument to TextListener. | Fixed; actual engine-construction regressions cover both modes. |
| B2 | Native timeout blocked dispatch and allowed a second result. | Fixed dispatcher/result behavior: four admitted native workers, one terminal result, timeout cancellation and late-result rejection. A hung Python worker still cannot be killed; four hangs exhaust native capacity and can affect process exit. |
| B3 | Reply drafts omitted request-specific search instructions. | Fixed with shared `_build_request` and parity tests. Captured tools/instructions belong to that turn. |
| B4 | WebSocket capture loss or owner switches spliced utterances. | Fixed discontinuity signalling and old-owner buffer reset. The listener preserves clean post-gap samples while discarding incomplete speech. |
| B5 | HN stories repeatedly requested relevance decisions. | Verdicts and `_reported` persist, including rejected stories and legacy shown-entry migration. Reporting a slot is separate from audible delivery. HN remains optional and disabled in shipped profiles. |
| B6 | Benchmark claimed native-audio draft overlap despite its guard. | Note and dead branch corrected. Text/transcribed drafts remain; direct-audio overlap needs a new experiment before enabling. |

## Proposal status

IDs are retained for comparison with the original review. “Partial” means a
useful boundary exists but the proposed consolidation is not complete.

| ID | Subject | Disposition |
| --- | --- | --- |
| F1 | Quiet/wake/noise definitions | Further sharing is possible. Preserve quiet-only wake checks, custom lists and hierarchical tool selection. Flat routing is an experiment, not an approved replacement. |
| F2 | Prompt construction with `str.replace` | Implemented: routing builders compose explicit conditional fragments. |
| F3 | Permits and tool filters | Remaining. Keep versioned list/option/scope and argument authorization, per-turn offered tools and execution-time checks. An allowed-name set alone is insufficient. |
| F4 | Generic context capability metadata | Optional small improvement; must advertise only fresh evidence already supplied. Do not introduce another core registry. |
| F5 | Remove saved-setting migrations | Rejected as blanket cleanup. Preserve operator-created lists and compatible data migrations. |
| F6 | Common background inference client | Remaining. HTTP `requests`, async decision `httpx`, option scoring and processor transport still coexist. Provider support and cancellation differ. |
| F7 | Deterministic greeting eligibility | Partly implemented in Vision presence/daily state. Autonomy retains conversational judgement and all-slot evidence; healthy/recovered state can explain silence. |
| F8 | Speculative drafting | Request parity implemented; keep bounded private text drafts. No native-audio overlap or latency gain is claimed. |
| C1 | Split reply processor | Structural phase implemented with typed turn/response state and shared request construction. Compatibility dictionaries and some worker-owned state remain. |
| C2 | Separate Autonomy reviewer transport/worker | Remaining. Review still uses autonomy LanguageModelProcessor workers and a separate ticker. MindScheduler did not replace this specialized path. |
| C3 | Single interaction/cancellation owner | Partial: generation-tagged inference holds and continuation cancellation exist. Quiet/autonomy epochs, processing flags and queue draining remain distinct mechanisms. |
| C4 | Smaller engine | Setup phases extracted. Config/commands/agent registration remain in engine.py; further boundary extraction is optional housekeeping. |
| C5 | Tools return results | Dead speech/no-op tools removed; timeout pool fixed. Local tools still publish through queue adapters. Result-returning migration remains. |
| C6 | Simplify shutdown | Remaining. Preserve deadlines, component cleanup and reporting; dispatcher exit is not proof hung workers terminated. |
| C7 | Persistence and dead code | Dead code removed. Shared atomic persistence with explicit writer/transaction ownership remains. |
| A1 | Unused autonomy code | Implemented; retain live token estimation and history preservation. |
| A2 | Merge runtime registry and slots | Literal merge rejected. A shared status projection remains useful; execution ticks must not advance evidence revisions. |
| A3 | Shared background scheduling | Implemented for minds: one timer, four workers, external policies; Search is on-demand and Memory recall uses scheduler triggers. Task progress, camera and specialized inference paths remain separate. |
| A4 | Memory responsibility separation | Recall has a dedicated MemoryRecall service, but CompactionAgent still coordinates recall and compaction. No additional public core/slot is required yet. |
| A5 | Replace ledgers with one announced revision | Rejected: evidence version, semantic incident key, handled task and acknowledged delivery have different meanings. Simplify only proven derived state. |
| A6 | Slot compatibility and agent caches | Remaining. `notify_user` compatibility and eagerly created agent caches still exist. Preserve migrations and locking semantics. |
| A7 | Observer/Constitution | Product decision pending. Observer still follows jobs.enabled; modifiers are live, immutable rules are displayed rather than injected. Do not delete this as dead code. |
| I1 | Audio ABC/Protocol consolidation | Remaining. Both backends now signal capture gaps; optional device/health capabilities still need an explicit contract. |
| I2 | Typed capture/encoding boundary | Partial through ProcessorTurn. Listener messages still carry native-audio metadata; preserve media privacy and continuation semantics when migrating. |
| I3 | UI status and APIs | Shared status projection remains. Public state/minds/agents routes and API reuse_tts remain supported. TUI is retained; its command-result modal is implemented. |
| I4 | Vision trims | Timing ownership moved into scheduler. YuNet, Haar and E4B modes, saved setting precedence and interval migrations remain supported. |

## Corrections to the original deletion advice

The direct-processor fallback without a ContextBuilder is reachable. `/api/state`,
`/api/minds` and `/api/agents` are public routes. The TTS API's `reuse_tts` controls
actual loading behavior. WebSocket `check_if_speaking` has internal callers.
MiniMax's environment-key fallback is functional configuration, not a hard-coded
credential. None of these should be removed merely because the main web frontend
uses another path.

Removed modules include `core/memory_context.py` and `autonomy/jobs.py`; their
original combined size was 479 lines, not 308. Removed test-only APIs were deleted
with their tests. The earlier 4–5k line/eight-thread savings were never established
and are not a target for further changes.

## Remaining engineering risks

1. `llm_decide_sync` submits to the running event loop and then blocks that same
   thread on `future.result`. Normal synchronous HN/Weather workers avoid this
   branch, but async callers need a clear failure or a correct async API.
2. SubagentMemory locks file writes but keeps a per-instance in-memory snapshot.
   Two writers can overwrite each other's changes. The pre-lock truncation race
   was fixed; reload/update/save transaction ownership is still unresolved.
3. Native tool and mind worker cancellation is cooperative. Network/process
   deadlines and resource cleanup matter independently of generation invalidation.
4. Smaller methods and cached prefixes do not prove lower voice latency. Preserve
   dated benchmark conditions and measure real mixed workloads for runtime changes.

See the [remaining sequence](architecture-refactor-progress.md#remaining-work)
for the next reviewable steps. No source-code changes are authorized by this
review alone.
