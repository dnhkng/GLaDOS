# Architecture simplification: validated review and implementation plan

Reviewed 2026-10-07 against the current working tree. The original
`architecture-simplification-review.md` is preserved unchanged. Its proposals are
recommendations to assess, not instructions to execute wholesale.

## Outcome

The dead-code cleanup and B1–B5 fixes are implemented. B6's unreachable branch
and inaccurate benchmark note are corrected; direct-audio speculation remains
disabled. The structural extraction phase is now implemented; see
`architecture-refactor-progress.md` for changes, final validation and the
remaining runtime roadmap. The assessment below records the original audit.

The strongest next step is to extract a shared reply request builder and a typed
per-turn state from `LanguageModelProcessor`, then separate the autonomy reviewer
from speech streaming. Keep independent cores, shared slot evidence, concurrent
inference admission and the reviewer → Central handoff. Avoid adding a generic
orchestration framework.

This patch removes approximately **1,060 net Python source lines**, excluding a
concurrent, unrelated spoken-text-converter rewrite. At the first inventory the
source contained 127 Python files and 26,208 lines; that includes comments and
blank lines, excludes frontend JS/HTML, and predates that concurrent rewrite.
These are static counts, not performance measurements. After cleanup,
`LanguageModelProcessor.run()` is still 631 lines, `Glados.__init__()` 529,
`ToolExecutor.run()` 301 and `SearchAgent.research()` 292. Those are useful
boundaries to improve. The claimed 4–5k LOC and eight-thread reductions are not
validated savings; worker counts depend on configuration and lazy pool startup.

## Bugs B1–B6

| Item | Finding | Implemented resolution and limits |
|---|---|---|
| B1 | Confirmed: engine supplied a SpeechListener-only kwarg to TextListener. | Removed that kwarg. A smoke test constructs the actual engine for both text and both modes, using fake models and suppressing worker startup. |
| B2 | Confirmed: context-managed native-tool pool waited after timeout; late results could create duplicate outcomes. | A persistent four-worker pool has four admission permits, so no unbounded pool backlog develops. Native result queues publish one terminal outcome atomically. Timeout closes the queue, marks cooperative cancellation and releases the dispatcher without waiting. A fifth occupied call fails promptly. Regressions cover subsequent-tool execution and late-result suppression. |
| B3 | Confirmed specifically for search instructions on reply drafts. Tool-result and routed-action instructions are for other request types and do not all belong in a reply draft. | Both drafts and normal replies use `_add_reply_instructions`; drafts use the same offered reply tools to determine search instructions. Existing private-draft tests now cover search instructions and background mood changes. A complete shared request builder remains the next refactor. |
| B4 | Confirmed: WebSocket overflow did not signal discontinuity. | Added the flag and consumer method. Overflow, owner release and room takeover signal a gap; owner changes also clear buffered old-owner audio. Tests cover overflow, one-shot consumption, owner release and exclusive microphone capture. The speech listener already resets incomplete/pending speech on a signalled gap. |
| B5 | Confirmed repeated relevance calls/publication; unconditional repeated audible announcements are an overstatement because Autonomy also deduplicates. | Successful relevance verdicts are persisted, including rejected stories. Relevant stories outside the report limit reuse their verdict. Published stories get `_reported`, not a false claim that they were heard. Autonomy continues to own actual speech delivery. Restart tests verify three stories are judged once and the two relevant stories are published once. Retention is bounded by the existing memory capacity; failed judgments may be retried. |
| B6 | Confirmed by the executable guard and a regression: direct audio never creates a draft. A GPU run is unnecessary to prove that branch exclusion. | Removed the dead audio-content expression and corrected the voice benchmark note. Text speculation remains enabled when slots/backend permit it. Enabling audio drafts requires a separate quality/latency experiment: both router and draft would encode the clip. No new voice latency measurement was made. |

A Python thread that never returns cannot be terminated by `Future.cancel()`.
Four stuck native tools exhaust native capacity; other dispatcher branches stay
available. `shutdown(wait=False)` stops dispatcher waiting, but Python can still
join pool workers during interpreter exit. Subprocess-backed tools should retain
real subprocess timeouts; guaranteed termination of arbitrary future tools needs
process isolation or cooperative cancellation. The patch does not claim to solve
that limitation or impose the native timeout policy on every MCP/search path.

## Cleanup implemented, and corrections to the deletion list

Removed `core/memory_context.py`, `autonomy/jobs.py`, invisible `speak` and
`do_nothing` registrations/modules/terminal executor handling, unused vision
event and emotion forwarding, unused personality builder, unused context/store
helpers, test-only summarization/fact extraction, unused tiktoken factory/setter,
EmotionConstitutionBridge, unused shutdown convenience methods and obsolete
configuration/display fields. The two whole-file sizes were 171 + 308 = 479,
not 308 combined.

Preserved `estimate_tokens`, SimpleTokenEstimator and its live default getter.
Replaced deep-snapshot test/example calls with explicit deep copies and kept the
history-preservation regressions. Old obsolete YAML keys are ignored by existing
Pydantic extra-field behavior; shipped configs no longer advertise them.

Two claimed dead paths were **not deleted**:

* The optional-context-builder fallback in `llm_processor.py` is used by direct
  processor callers/tests, even though engine construction always supplies a
  builder. Make the builder mandatory only in a coordinated caller migration.
* `/api/state`, `/api/minds` and `/api/agents` are documented public routes in
  `webapp/server.py`. Lack of static-frontend usage does not make them dead APIs.

The generic legacy autonomy tool-queue adapter remains for now. Removing it
requires removing the autonomy fields/callers together with the processor-lane
refactor, rather than deleting a queue contract while callers still supply it.

## Fast-decision proposals

| Proposal | Assessment | Recommended form |
|---|---|---|
| F1: one quiet/wake/noise definition | Accept definition sharing; qualify flat routing. | Define reusable intent text/options and one transition handler. Quiet-mode wake must work with operator-created lists. Preserve hierarchical category/tool/argument selection until flat routing is benchmarked for accuracy and label capacity. |
| F2: remove prompt `str.replace` | Accept. | Compose explicitly named instruction fragments. Cover quiet negation, health context and recalled-topic cases in prompt tests. |
| F3: one permit/filter | Accept one typed authorization interface; reject the proposed `(settings_revision, allowed_tools)` as insufficient. | Preserve list/option revisions, fixed argument binding, tool availability and scope. Recheck at execution because settings can change after scoring. Remove keyword filtering only once fallback routing quality and read-only restrictions are preserved. |
| F4: generic context capability metadata | Qualified. | A small capability description can remove hard-coded core coupling, but it must describe fresh supplied evidence, not advertise stale or missing measurements as answered. Avoid a second registry of core schemas. |
| F5: delete migrations/regenerate settings | Reject blanket deletion. | Existing saved operator choices are real user data. Retain compatibility until a versioned migration/retirement plan exists; never regenerate custom lists as cleanup. |
| F6: consolidate background inference | Accept in stages. | Common config, scheduling/cancellation and transport; thin structured-response and letter-score wrappers. Remove unused pool/schemas separately. Generic providers cannot all return llama.cpp option logits. |
| F7: deterministic greeting/pre-filter evidence | Qualified. | Compute a greeting candidate/freshness in code; Autonomy still decides whether speaking is useful in this conversation. Keep all-slot visibility. Healthy/resolved state can explain that an old alert no longer matters. Deduplicate on actual delivery. |
| F8: speculation | Keep text speculation for now. | Use one request builder, retain private buffering and cancellation until routing accepts reply. Measure native-audio overlap before enabling it. A smaller routing duration alone does not prove speculation is useless. |

## Core-runtime proposals

| Proposal | Assessment | Recommended form |
|---|---|---|
| C1: split processor | Strongly accept. | Turn data, request assembly, provider stream and response sink. Preserve provenance, stable-prefix ordering, overflow retry and native audio isolation. Snapshot live context once per prepared request; do not call clock/core providers inside a supposedly pure builder. |
| C2: separate autonomy inference | Strongly accept boundary; qualify direct synchronous loop call. | A dedicated bounded reviewer worker uses common inference transport. The event loop must remain responsive while inference waits/runs and must not hold its state lock across network I/O. Replace ticker with a monotonic deadline, not only a queue timeout that continuous events can starve. Preserve schema validation, evidence revalidation and Central handoff. |
| C3: TurnState/cancellation | Accept typed generation cancellation; reject scheduler hold as the only busy state. | Admission hold ends at first audio readiness, while speaking/streaming can continue. Track recording, responding and playing phases separately from admission pressure. Queue draining bounds memory; generation checks enforce correctness. Do not drop either until its replacement covers those roles. |
| C4: smaller engine | Accept. | Extract config, commands and agent factories without changing initialization order or dynamic enable/disable. One explicit context assembly surface is useful; sources must still preserve stable/live ordering. A 700-line target is not a correctness requirement. |
| C5: tools return results | Accept as next boundary. | Typed result-returning tools let executor own envelopes, timeout and outcome arbitration. Migrate tools in a separate patch; legacy queue adapters stay only at that edge during migration. Native pool fix is already done. |
| C6: 30-line shutdown | Reject equivalence claim. | Queue drain currently discards, but priorities, deadlines, joins, exception reporting and component cleanup still have semantics. Document abort versus finish policies, then simplify against tests. Never equate dispatcher exit with termination of hung pool work. |
| C7: shared persistence | Accept helper reuse; qualify caching/append redesign. | Share atomic replacement with explicit durability/locking policy. Preserve optimistic history compaction and summary editing. Cache knowledge with edit invalidation if external edits remain supported. Do not replace concurrency control with a shared write helper alone. The MiniMax environment-key fallback is functional configuration, not a hard-coded credential. |

Dropping native Ollama transport is a compatibility change, not automatically
safe dead-code cleanup. Validate streaming tools, thinking, headers and native
media behavior on the OpenAI-compatible endpoint before retiring a supported
backend path.

## Autonomy proposals

| Proposal | Assessment | Recommended form |
|---|---|---|
| A1: dead code | Implemented. | Preserve live token-budget/history behavior and the reachable fallback. |
| A2: MindRegistry → slots | Reject literal merge. | `Subagent._do_tick` reports execution/tick state to MindRegistry and publishes semantic output to slots. Engine registers ASR/TTS/processors too. Use one status projection for UI, with distinct runtime and evidence data; runtime ticks must not change evidence revisions or trigger announcements. |
| A3: scheduling | Accept modest consolidation. | Allow event-driven agents with no no-op polling. Route on-demand work through bounded named groups and use one progress timer. Preserve latest-topic recall cancellation, emotion interaction timing and deadlines; avoid a generalized workflow engine. |
| A4: split Memory | Accept internal responsibilities; qualify two new agents/slots. | Separate compaction and recall services behind one Memory Core first. This preserves historical summaries and the late relevant-fact scenario without adding orchestration. Split public slots only if their independent lifecycles require it. |
| A5: one revision ledger | Reject direct replacement. | Semantic incident attention keys suppress revised wording for the same incident; in-flight evidence versions prevent stale announcements; handled state retires tasks. One announced revision loses these distinctions. Document invariants and remove only proven derived caches. |
| A6: slots/caches | Accept consolidation after producer migration. | New producers use regular/important consistently; remove `notify_user` after adapters migrate. One formatter per audience/purpose, not necessarily one string everywhere. Lazy agent caches and separate directories are useful; retain safe migrations and cross-writer semantics. |
| A7: Observer/Constitution | Product decision, not dead code. | If retained, give Observer its own enable flag and regular priority for tone adjustments. Immutable rules currently are displayed but not injected by the engine; explicitly choose injection or removal. Deleting the whole feature requires a product decision. |

EventBus delivers slot changes to the reviewer. ObservabilityBus publishes UI
telemetry. Direct calls and callbacks handle local requests and completion.
They are not four interchangeable pub/sub frameworks. Keep those purposes clear
rather than routing every message through a new universal event bus.

## I/O, UI and vision proposals

| Proposal | Assessment | Recommended form |
|---|---|---|
| I1: one audio contract | Accept defaults/contract cleanup; qualify moving all behavior to a base class. | Specify discontinuity and capture health explicitly. Playback begin must remain short/nonblocking so quiet/cancel locks are not held during audio playback. WebSocket `check_if_speaking` has internal uses; it cannot simply be erased. Centralize bounded capture behavior only with race/owner-switch tests. |
| I2: UserTurn/encoders | Accept. | PCM and media metadata stay out of persisted chat/logs. Typed native/ASR encoders and optional deferred transcripts feed the shared Turn representation. Preserve continuation cancellation and English default. |
| I3: status/UI | Accept a shared status projection; reject unused-route and `reuse_tts` deletion claims. | `serializers.build_state/build_snapshot` overlap, but public routes remain. `reuse_tts` affects production API loading and has documentation/tests. Retiring the TUI is a product choice; if retained, consume the same status interface without removing its bus drain until migrated. |
| I4: vision trims | Qualified. | Yunet/E4B/Haar are supported configurable modes, not proven dead branches. Resolve precedence between config defaults and saved operator settings; preserve interval migration until stored configs migrate. No face-mode deletion without quality/compatibility evidence. |

## Additional findings from the deeper scan

1. **`llm_decide_sync` blocks its own running event loop.**
   `core/llm_decision.py` submits a coroutine with
   `run_coroutine_threadsafe(..., loop)` and immediately calls `future.result`
   from that loop's thread. An isolated fake that returns immediately still
   times out. Current HN/Weather worker-thread callers generally take the
   no-running-loop branch, so this is a latent public-helper bug. Prefer direct
   synchronous inference from synchronous workers; async callers await the
   async wrapper. Until migrated, fail clearly if sync usage occurs on a running
   loop. Cancel timed-out submitted work; do not rely on the unused `_executor`.

2. **File locking does not protect SubagentMemory against stale-instance writes.**
   Two instances loaded before either writes maintain separate `_entries`.
   A writes `first`, B writes `second`, and reload contains only `second`.
   Reproduced using temporary files. Locking serializes truncation but does not
   merge state. Before consolidating persistence, define single-writer ownership
   or lock the entire reload/update/save transaction. This is not fixed in this
   patch because it affects cache/store contracts beyond the requested cleanup.

3. **Thread and latency reduction claims need operational measurement.**
   An unused ThreadPoolExecutor creates no worker threads until submission.
   Counting constructors/callbacks is not a runtime census. The inventory found
   12 direct Thread construction sites, five pool construction sites and 11
   queue construction sites; configurations and active tasks change actual
   counts. A smaller processor may improve maintainability without reducing
   GPU prefill, audio encoding or first-token latency.

4. **Review coverage must include request parity and admission lifetime.**
   Existing direct listener tests missed B1; route tests without search tools
   missed B3. Keep engine-construction and captured-request regressions alongside
   unit tests. Exercise foreground interruption during every background core,
   not only a scheduler unit test.

## Recommended implementation sequence

1. **Shared request builder and Turn envelope.** Extract prepared context,
   route/authorization, tool selection and prompts; both normal reply and draft
   consume it. Keep a compatibility adapter for message dictionaries. Acceptance:
   equal instructions/tools for equivalent reply requests; cached stable prefix
   unchanged; live clock/vision/input last; no raw audio in stored history.
2. **Common inference transport and separate reviewer worker.** Consolidate
   provider configuration/admission/cancellation and structured/choice wrappers.
   Move Autonomy away from the speech processor, with one bounded reviewer and
   monotonic scheduling. Acceptance: updates remain ingestible during inference,
   pending/queued work stays visible, stale evidence cannot speak, no duplicate
   delivery after changed slot wording, foreground holds resume each core.
3. **Explicit interaction state and tools that return results.** Centralize
   generation/epoch/quiet checks and recording/responding/playing phases. Migrate
   local tools to result-returning operations, then remove unused autonomy tool
   plumbing. Acceptance: continuation before playback cancels old response;
   timed-out work never supplies a second outcome; busy does not end merely
   because the admission hold is released; shutdown reports remaining workers.
4. **Runtime housekeeping.** Shared UI status projection, event-driven no-op
   agents, shared progress timer, Memory internals, small atomic persistence
   helper with safe ownership, commands/factories outside engine. Acceptance:
   all current controls/APIs work, custom settings survive, regular state does
   not create important events, memory summaries and recall remain available.

After these, measure whether flat routing, direct-audio speculation or retiring
Observer/TUI/face backends improves the actual product. Do not combine those
product choices with a structural refactor. Each stage should be independently
reviewable and preserve the existing foreground/background concurrency model.

## Validation

Regression coverage includes actual engine construction in text/both modes,
native timeout/late-result behavior and four-worker saturation, persistent HN
verdicts across restart, WebSocket overflow/ownership and draft search prompts.
The servers stayed stopped; no GPU inference or end-to-end latency claims are
made. Final checks: **632 passed** with the concurrent converter test module and API
subdirectory excluded; **6 passed** in the separate API/API-config run (the API
config tests also appear in the first run). The complete non-API run reported
691 passes and seven failures in the concurrent converter rewrite. Compilation
and patch whitespace checks passed. Existing deprecation warnings and harmless
HTTP connection-reset logs remain.

During validation an unrelated rewrite of `src/glados/utils/spoken_text_converter.py`
appeared in the shared workspace. It was left untouched. The full suite then
reported seven mathematical-notation failures in that file; the unchanged HEAD
version passes a failing power-conversion example. Cleanup verification therefore
also runs with that concurrently edited test module excluded. Do not attribute
that rewrite or its test failures to this patch.
