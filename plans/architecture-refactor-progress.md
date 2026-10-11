# Architecture refactor progress

Status reviewed **2026-10-11**, against `codex/ui-mockup-refinements` at `21dc152`.
[Current review](architecture-simplification-review.md) ·
[Decisions/acceptance criteria](architecture-simplification-validation.md)

## Implemented

The 2026-10-07 cleanup and structural refactor replaced large coordinator methods
with explicit phases while preserving public construction/provider behavior.

| Coordinator | Pre-refactor lines | Current lines |
| --- | ---: | ---: |
| LanguageModelProcessor.run | 631 | 17 |
| Glados.__init__ | 529 | 90 |
| ToolExecutor.run | 301 | 19 |
| RoutingTree.__init__ | 212 | 10 |
| SearchAgent.research | 292 | 185 |

ProcessorTurn holds route, draft, lease, offered tools, prepared context and body.
ResponseState holds streaming speech/thinking/tool buffers. Normal reply and draft
share `_build_request`; stable prompts are in `core/prompts.py`. Raw media is
request-local and does not replace the persisted user transcript/placeholder.
Some state remains on each processor; instances are not reentrant.

Engine setup has explicit dependency-ordered phases. ToolInvocation carries
execution authorization, cancellation, arguments and result handling. Native
execution, MCP, requested research and unknown tools have separate paths. Native
timeout queues produce one terminal result; the dispatcher does not wait for a
timed-out worker to return.

Routing builds catalogs, capability groups, conversation options and conditional
prompt fragments directly. Search separates query/evidence/review/publication
while retaining round limits, deadlines, quote validation and lock release.

Subsequent work implemented **MindScheduler**, its bounded four-worker pool,
MindRuntime and timing policies outside mind implementations. Search is on-demand,
Memory recall coalesces scheduler triggers, Emotion reacts in background, and
Vision uses motion-biased random timing. Fixed intervals start after completion;
manual paused runs and generation cancellation preserve controls. This replaces
per-mind timer threads, not the specialized Autonomy reviewer/ticker or model
admission scheduler.

Interaction holds reserve response priority without stopping cheap probes.
Later fixes retain per-turn tool definitions, terminal result metadata and clean
post-gap microphone audio; HN verdicts and legacy shown records survive restart.
The basic/detailed/Presence UI and transparent PotatOS frame are on this branch.

## Validation

The 2026-10-11 AST scan covers 134 source Python files and 26,490 physical lines.
No function/method exceeds 200 lines; SearchAgent.research is the longest at 185.
The earlier 945-test report was the structural-refactor checkpoint. The latest
full-suite checkpoint is **1,049 passed, four warnings, 34.23 seconds** on
2026-10-10. The extended-ribbon update passed **16 webapp tests** on 2026-10-11.
Counts describe specific tested revisions, not a permanent target.

Tests cover engine text/both construction, draft/reply parity, live-provider
ordering, media privacy, routing authorization, native timeout saturation/late
results, continuation, speech markup, scheduler controls/holds, Autonomy source
revalidation/delivery, memory, search evidence and web/API behavior.

This documentation review verifies links, referenced code/config paths and source
claims. It makes no new GPU, natural-speech or end-to-end latency claim. GLaDOS
and llama-server were stopped at the user's request; CPU visual fixtures are
independent of production inference.

## Remaining work

1. **Common inference transport and dedicated Autonomy reviewer.** The reviewer
   still uses LanguageModelProcessor workers and a separate ticker. Preserve
   event ingestion during inference, monotonic scheduling and validated Central
   handoff. Consolidate structured/choice/streaming transports without dropping
   supported providers.
2. **Interaction and tool ownership.** Consolidate phase/generation/quiet checks;
   preserve the distinction between an admission hold and a busy interaction.
   Migrate local tools to returned results, then remove queue adapters. Keep
   execution-time authorization, terminal metadata and late-result suppression.
3. **Persistence and helper correctness.** Fix llm_decide_sync's same-event-loop
   blocking path and SubagentMemory's stale-instance lost writes. Define writer
   ownership and external-edit invalidation before adopting a shared atomic helper.
4. **Focused housekeeping.** Shared runtime-status projection, Memory internal
   responsibility separation, and optional config/command/registration extraction.
   Mind scheduling is already consolidated. Task progress/camera capture have
   different lifetimes and need not become generic mind work.

Public API deletion, retiring TUI/Observer/face modes, flattened routing,
direct-audio speculation and Rust/native inference are separate product or
experimental changes. Acceptance criteria live in the validation document.
