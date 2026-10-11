# Roadmap

Reviewed **2026-10-11**. Implemented behavior and future work are separated here;
[architecture progress](../plans/architecture-refactor-progress.md) tracks runtime
refactoring, and [the documentation map](README.md) links the current guides.

## Current baseline

- E4B direct audio with optional Gemma transcripts; Parakeet plus a chosen text
  model remains the extended profile. English and disabled thinking are defaults.
- One-token capability routing with editable lists, versioned tool authorization
  and text-only speculative replies where capacity permits.
- Independent Emotion/PAD, Memory recall/compaction, Health and motion-aware Vision.
  Minds need not use an LLM; their timing policies belong to MindScheduler.
- Regular/important slot publication and Autonomy reviewer → Central delivery.
  Health threshold alerts, late relevant memory and requested research are live.
- Basic Portal interface, detailed operator console and tablet Presence, with
  local themes, camera treatments, device/settings controls and memory editing.
- Microphone continuation/interruption, bounded queues, background inference
  holds, MCP/native tools and tracked research/cancellation.

These features still need broader real speech, hardware and mixed-workload
validation. Their existence does not imply an achieved voice-latency target.

## Remaining product work

### Audio setup wizard

Device selection and capture-health inspection already exist. A guided workflow
for selecting devices, VAD calibration, loopback/echo checks and camera verification
is not implemented. Production acoustic echo cancellation also remains future work.

### Home Assistant mapping editor

MCP tools and manual fixed routing choices already work with a configured server.
An editor that discovers entities, assigns useful aliases/arguments, validates
bindings and tests actions is still an idea. Avoid introducing a new mapping
framework before validating its need against existing decision lists.

### Additional evidence sources

Calendar/time-to-leave reminders, personal reminders and optional message/email
summaries are potential new minds. System/GPU health is already implemented.
Each future mind should use the existing scheduler, publication and inference
contracts; a new source does not require another orchestration layer.

## Engineering priorities

Common inference transport/dedicated reviewer, explicit interaction/tool ownership,
persistence correctness and runtime status projection remain. Mind timing
consolidation and the large-method extraction are done; see the
[remaining sequence](../plans/architecture-refactor-progress.md#remaining-work).

Broader AMD audio-model validation, streaming ASR and physical animatronics remain
uncompleted. Flat routing, direct-audio speculation and recurring decisions to
interrupt the user require explicit evaluation rather than automatic activation.

## Rust delivery direction

Bring the Python application close to the agreed feature baseline, then migrate
to Rust. [Rust migration](rust-migration.md) describes the proposed Tauri/local
model direction; [native inference](native-inference.md) and the Sonora probe
provide dated feasibility evidence. Neither native embedding nor Sonora AEC is
integrated into production. Overlapping speech quality and cross-platform/device
behavior remain acceptance work; no delivery dates are assigned.
