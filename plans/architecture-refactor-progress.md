# Structural refactor implemented — 2026-10-07

This implements the behavior-preserving structural work identified in the
architecture validation. It is separate from the pending changes to runtime
scheduling, cancellation ownership and persistence contracts.

## Changes

| Coordinator | Before | After |
|---|---:|---:|
| LanguageModelProcessor.run | 631 lines | 17 lines |
| Glados.__init__ | 529 lines | 90 lines |
| ToolExecutor.run | 301 lines | 19 lines |
| RoutingTree.__init__ | 212 lines | 10 lines |
| SearchAgent.research | 292 lines | 185 lines |

The full Python source scan finds no function or method exceeding 200 lines.
This is a static maintainability check, not a performance result. The longest
remaining method is the bounded search coordinator. Engine agent registration,
command declaration and speech playback remain under 180 lines.

### Turn pipeline

ProcessorTurn owns the input, interpreted route, draft, admission lease,
selected tools, prepared messages and request body. ResponseState owns thinking,
speech parser and tool-call buffers. The processor now has explicit phases for
input acceptance/transcription, routing, admission, history recording, fixed
commands, request assembly, streaming, response completion and cleanup.

Both draft and normal reply call `_build_request`; the same tool policy and
request instructions are applied. Stable instructions live in `core/prompts.py`
with their text unchanged. Context providers are resolved once for each prepared
request. Direct-audio substitution affects request messages only, preserving the
stored transcript/placeholder and native-media privacy boundary.

The existing public processor constructor, context inspection and direct helper
callers stay compatible. Some cancellation/handoff fields remain on the
processor; it is still one worker per processor instance, not a newly reentrant
shared processor. Concurrent inference admission remains unchanged.

### Engine and tools

Engine setup is split into context/state, background cores, queues/MCP,
listeners, primary inference, autonomy inference, tools/speech, autonomy loop
and worker startup. Calls retain their original dependency order. The model
injection constructor and startup controls remain compatible.

ToolInvocation carries authorization, cancellation, generation, argument and
result-queue data across tool execution phases. Native execution, MCP execution,
requested search and unknown-tool handling are separate. The bounded pool and
single-terminal-result timeout behavior from the cleanup are preserved. Shutdown
of the native pool now sits in the dispatch loop's finally block.

### Routing and research

Tool catalog assembly, capability groups, conversation choices and context
instructions are separate routing builders. Conditional health/recall prompt
fragments are composed directly instead of editing earlier prompt text with
`str.replace`. Exact comparisons against the pre-refactor implementation match
all nodes, scopes and instructions for four health/recall combinations.

Search separates preferred-page retrieval, query shaping, source merging,
evidence review and publication. Its coordinator retains explicit round limits,
deadline/cancellation checks, quote verification and guaranteed research-lock
release. This is a structural extraction, not a change to search strategy.

## Validation

* Final complete test run: **945 passed**, four existing deprecation warnings.
* New direct regressions check draft/reply request equality with and without
  tools, stable/live context ordering, one provider evaluation per prepared
  request, and native audio isolation from persisted history.
* Existing tests cover text/both engine startup, private speculative streams,
  tool authorization, native timeout capacity and late-result suppression,
  speech markup/thinking, voice continuation, scheduler holds, autonomy stale
  evidence/delivery, request overflow recovery, search dates/sources and UI/API.
* Scoped undefined-name/unused-import/import-order checks, compilation and
  `git diff --check` pass.
* Servers remain stopped. No GPU or voice-latency benchmark was run; no latency
  improvement is claimed.

Other work on the spoken-text converter, phonemizer and their benchmarks/tests
was already present or changed concurrently. This refactor leaves it intact.
The full-suite result describes the tested shared working tree, including those
concurrent changes, rather than claiming authorship of them.

## Remaining roadmap

1. Common inference transport and a dedicated bounded Autonomy reviewer worker,
   with monotonic scheduling while event ingestion remains responsive.
2. Explicit interaction-phase/cancellation ownership, then result-returning
   tools to retire legacy queue adapters. Admission hold and user busy state
   must remain distinct.
3. Runtime status projection, event-driven background scheduling, Memory
   responsibility separation and persistence single-writer/transaction policy.
   The async sync-helper deadlock and stale-instance cache overwrite identified
   in the audit still need fixes in this stage.

Those changes alter runtime ownership and require their own acceptance tests.
Public API removal, product-feature deletion, flattened routing and enabling
native-audio speculation remain outside this structural refactor.
