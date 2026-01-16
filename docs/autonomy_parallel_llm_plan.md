# Autonomy Parallel LLM Orchestration Plan

## Overall Plan
Introduce a two-tier LLM orchestration layer that guarantees low-latency user responses while allowing
bounded parallel LLM calls for autonomy/subagents. The main agent uses a priority lane, while subagents
use a pooled worker queue with configurable parallelism.

## Goals
- Keep user-visible latency under ~1s when autonomy is active.
- Allow multiple parallel LLM calls for background work (subagents, tools, jobs).
- Provide a configurable concurrency cap for autonomy calls to avoid rate-limit or GPU overloads.
- Minimize architectural changes so future refactors remain straightforward.

## Selected Approach
**Two-tier orchestration (priority + pooled worker queue).**

Rationale:
- Directly addresses the blocking observed when autonomy calls share the same single LLM worker.
- Keeps user requests fast by ensuring a dedicated lane unaffected by autonomy bursts.
- Provides an explicit concurrency control knob for autonomy calls (`autonomy_parallel_calls`).

## Alternatives Considered (and Why Not Chosen)
1) **Each subagent calls LLM directly (shared code only, no orchestration).**
   - Rejected because it offers no global throttling or prioritization, so user latency can still degrade
     under load or when the backend imposes rate limits.

2) **Single queue with priority ordering only.**
   - Rejected because a slow autonomy request can still block the single worker, even if user items are
     prioritized in the queue. Priority alone does not solve single-worker serialization.

## Detailed Plan
1) **Config additions**
   - Add `autonomy_parallel_calls` (int, default 2, bounded 1-16) to the autonomy config.
   - Optionally add `autonomy_queue_max` (int) to cap queued autonomy calls and prevent unbounded buildup.

2) **Queue separation**
   - Split the current `llm_queue` into:
     - `llm_queue_priority` for user/interactive requests.
     - `llm_queue_autonomy` for autonomy/subagent/tool follow-ups.
   - Ensure `TextListener` and speech inputs push only to the priority queue.
   - Ensure AutonomyLoop and tool outputs tagged with `autonomy=True` push to the autonomy queue.

3) **LLM workers**
   - Keep **one** dedicated LLM worker for the priority queue to guarantee latency.
   - Add **N** parallel LLM workers for autonomy queue (`autonomy_parallel_calls`).
   - Each worker runs the existing `LanguageModelProcessor` logic with minimal changes.

4) **Shared code for subagents**
   - Preserve shared request formatting, tool execution, and streaming logic by keeping the LLM worker
     class unchanged; only change its input queue and observability labels.

5) **Backpressure and safety**
   - If `autonomy_queue_max` is set, drop autonomy requests when full, and log warnings.
   - Coalesce periodic tick prompts: skip TimeTick dispatch if any autonomy work is already queued or in-flight.
   - Maintain observability events to measure queue depth and per-request latency.

6) **Metrics and logs**
   - Emit queue wait time and end-to-end latency for both lanes.
   - Add a small summary in observability (e.g., `queue_depth_priority`, `queue_depth_autonomy`).

7) **Testing**
   - Baseline: autonomy off, measure user response time.
   - Autonomy on with slow LLM: verify user response stays <1s while autonomy calls continue.
   - Autonomy burst: validate `autonomy_parallel_calls` caps concurrency and avoids starvation.

## Coalescing Options (and Choice)
- **Skip only periodic ticks when autonomy is already busy.** Chosen because it prevents timer-driven spam
  without suppressing task updates that might be user-relevant.
- **Drop all autonomy prompts when any are in-flight.** Not chosen because it can hide important task updates.
- **Allow unlimited queue growth.** Not chosen because it can create long backlogs and degrade relevance.

## Risks and Mitigations
- **Higher resource usage** with multiple LLM workers: mitigate via config defaults and sane caps.
- **Rate limits** on hosted APIs: mitigate with concurrency cap and queue size limit.
- **Complexity in observability**: mitigate by tagging events with lane identifiers.

## Rollout Notes
- Default `autonomy_parallel_calls` to 2 for safety.
- Enforce minimum of 1 worker to keep autonomy responsive without a special-case path.
