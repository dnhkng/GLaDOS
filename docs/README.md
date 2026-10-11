# Documentation map and review status

Reviewed **2026-10-11** against `codex/ui-mockup-refinements` at `21dc152`
(the code revision preceding this documentation pass). The documents describe
that branch, including UI changes still in draft PR #226; they are not a claim
that every change has merged to main. Current guides describe implemented behavior.
Plans distinguish completed work from proposals. Benchmarks retain measurement
dates, hardware, revisions and limitations; old service state is not live status.

## Current guides

| Document | Purpose / review outcome |
| --- | --- |
| [Project README](../README.md) | Installation, profiles, current basic/detailed/Presence UI, core architecture and Jev routing. |
| [Webapp](webapp.md) | Setup, controls, settings, one endpoint inventory and SSE; obsolete synchronous Emotion and duplicate routing accounts corrected. |
| [Autonomy](autonomy.md) | Independent minds, scheduler policies, evidence publication/review/delivery, asynchronous PAD and foreground holds. |
| [Vision](vision.md) | Motion-aware timing, four timestamped observations, live preview and question inspection; Cores navigation and arrival behavior corrected. |
| [Gemma 4](gemma4.md) | Current four-slot/16K/Q8_0/full-SWA setup and dated evaluations; automatic draft behavior clarified. |
| [WebSocket audio](audio_websocket.md) | Wire protocol, single microphone ownership, room behavior and capture discontinuities. |
| [MCP](mcp.md) | Config/transports/tools and optional local servers; default context clock, native memory and semantic recall distinguished. |
| [AMD](amd.md) | Installer's supported ROCm/wheel mapping and provider checks; remains hardware-unvalidated beyond startup selection. |

## Implementation status and future work

| Document | Status |
| --- | --- |
| [Architecture review](../plans/architecture-simplification-review.md) | Current audit; original B/F/C/A/I IDs retained with completed/partial/pending/rejected dispositions. |
| [Architecture validation](../plans/architecture-simplification-validation.md) | Invariants and acceptance criteria for remaining transport, cancellation/tool and persistence work. |
| [Refactor progress](../plans/architecture-refactor-progress.md) | Extraction and mind scheduling complete; explicit remaining sequence and dated verification. |
| [Webapp architecture](../plans/webapp-architecture.md) | Implemented launcher/server/control/telemetry boundary, not an observability-only proposal. |
| [TUI command output](../plans/fluttering-skipping-deer.md) | Implemented InfoScreen fix; filename retained for compatibility. |
| [Roadmap](roadmap.md) | Implemented baseline separated from wizard, HA editor, new sources and other future work. |
| [Rust migration](rust-migration.md) | Future application/packaging direction; no production port or AEC integration yet. |
| [Native inference](native-inference.md) | Pinned 2026-10-06 feasibility evidence; production still uses HTTP. No new upstream compatibility claim. |

## Dated reports and asset/probe notes

| Document | Treatment |
| --- | --- |
| [Context capacity](benchmarks/gemma4-context-capacity.md) | Measurements retained; eight-slot alternative explicitly historical rather than default. |
| [Four-slot capacity](benchmarks/gemma4-four-slot-capacity.md) | Measurements retained; subsequent 16K default and asynchronous PAD distinguished from the old workload. |
| [Sonora probe](../examples/sonora_aec_probe/README.md) | Reproduction/limitations reviewed against the harness; isolated probe, not production audio. |
| [YuNet asset](../src/glados/vision/models/README.md) | Bundled CPU model/version, hash and license reviewed against the actual asset. |
| [ANSI assets](../src/glados/glados_ui/images/README.md) | Existing TUI asset conversion notes; wording corrected. |

The JSON reports under [benchmarks](benchmarks) are recorded experiment data, not
living plans. Their numeric results have not been regenerated. The existing
voice-latency note correctly says direct-audio drafting is disabled. No GPU
server, microphone test or new hardware benchmark was started for this review.

## Checks for this revision

- Source AST and references: 134 Python files; no function/method over 200 lines.
- All 22 Markdown documents reviewed; 116 local links/heading anchors and 26
  YAML examples pass static validation. Referenced scripts, configs,
  code boundaries, UI paths and HTTP endpoints checked against this branch.
- Current launcher/config defaults checked: four model slots, two interactive
  reservations, 16K per slot, five-second Emotion cadence and 2–5 second Vision
  delay. Delays exclude run/admission time.
- YuNet binary size/hash/license and the pinned AEC probe checked locally.
- Patch whitespace checked. This documentation-only pass does not claim a new
  application test count, GPU validation or measured voice-latency improvement.

Update this map and the relevant guide when implementation changes. Keep remaining
engineering work in the architecture progress document rather than repeating a
second conflicting roadmap in each guide.
