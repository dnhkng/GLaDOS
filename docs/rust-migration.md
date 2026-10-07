# Rust migration plan

Decision and evidence recorded on 2026-10-06.

## Delivery sequence

**Bring the current GLaDOS codebase close to feature complete, then migrate it to
Rust.** The working Python application will provide the behavior, settings and
user experience against which the Rust implementation is evaluated.

During feature development, small feasibility probes can resolve migration risks,
as the native inference and acoustic echo cancellation (AEC) probes have done.
The production rewrite follows the feature baseline. “Mostly feature complete”
allows later additions; the exact remaining feature list is still to be agreed.

Before starting the migration, document the supported workflows and their expected
behavior: voice conversation and interruption, text input, routing and tool calls,
vision, autonomy/background work, memory, settings and device controls. Capture
representative prompts, recordings and state transitions so that parity can be
checked throughout the port. Include error handling, cancellation and shutdown.

Once that baseline is ready:

1. Establish Rust interfaces for audio, inference, application state, tools and
   persistence, using the existing behavior as the reference.
2. Implement and validate the native audio and inference components, then port
   routing, conversation, tools, vision, autonomy and memory in manageable stages.
3. Connect the desktop UI and preserve existing settings and memories through
   explicit, versioned data migrations.
4. Exercise the complete application on Linux, Windows and macOS, including
   installation, model loading, device changes and recovery.
5. Make the Rust application the main distribution when the agreed feature baseline
   and performance checks pass.

The order within the port can follow dependencies and risk. These are planned
stages, with no delivery dates assigned yet.

## Target application

The goal is a great cross-platform desktop experience on Linux, Windows and macOS,
with a single application plus local model files and a folder for settings and
memories. A proposed portable layout is:

```text
GLaDOS/
  glados                 # glados.exe on Windows; application bundle on macOS
  models/                # GGUF and ONNX assets
  data/                  # settings, memories and persistent application state
```

Tauri is the preferred UI direction, reusing the current web console where
practical. The Rust application owns audio capture, playback and processing;
the UI controls those services and displays their state. This lets audio behavior
remain consistent across desktop platforms and independent of the UI framework.

A single executable is a packaging goal. Validate it against the chosen inference
backends: llama.cpp, ONNX Runtime, GPU support and platform webviews can impose
library or system-runtime requirements. Package necessary native libraries with
the application where required. The initial probes establish Linux feasibility;
the full distribution still needs platform-specific validation.

## Model and runtime direction

Use GGUF or ONNX for the model assets in the initial Rust application:

| Component | Planned direction |
| --- | --- |
| Language, native audio input and vision | Gemma-4 E4B GGUF with its multimodal projector, through llama.cpp |
| Transcription when needed | Gemma native audio; separate Parakeet support is outside the initial Rust scope |
| Voice synthesis | GLaDOS ONNX voice and phonemizer; Kokoro voices are outside the initial Rust scope |
| Voice activity detection | Keep Silero ONNX |
| Face detection | YuNet ONNX |
| Echo cancellation | Sonora native Rust AEC3, subject to speech-quality validation |

Convert Python-specific model metadata, such as pickled phoneme/token mappings,
to a documented portable format or embedded data. AEC is signal processing and
requires no additional GGUF or ONNX model asset. These choices describe the Rust
target; existing Python profiles continue to define the feature-development baseline.

For local inference, embed upstream llama.cpp's `server-context` through a small
C ABI adapter. Preserve asynchronous requests, continuous batching, priority,
capacity reservations, streaming and cancellation. Keep upstream source reuse
and a pinned revision as the maintenance strategy. See the
[native inference assessment](native-inference.md) for the tested interfaces and
the current restriction on mixing zero-token decision tasks with generation.

## Native AEC test results

Sonora commit `413f64e7404529aaae6980613527f0d6acb21ce7` was built with Rust 1.91.0
and tested on Linux. The standalone probe used native Rust AEC3 processing, with
Python/PortAudio providing test audio devices. It ran mono 48 kHz audio in 10 ms
blocks with default AEC3 settings, including high-pass filtering. Noise suppression
and automatic gain control were disabled.

Two live tests used the UGREEN webcam microphone and Jabra Speak 710 loudspeaker.
Using the webcam microphone avoided relying on the speakerphone's microphone AEC.
The second test used clear spoken cues and captured both overlapping speech and
continued speech after playback became silent:

| Measurement | Second live test |
| --- | --- |
| Echo-only microphone energy reduction, seconds 10–16 | 24.8 dB |
| Echo-only frames exceeding the Silero speech threshold, raw / cleaned | 89.4% / 0% |
| Overlapping-speech frames exceeding the threshold, raw / cleaned | 99.1% / 78.0% |
| Speech-only frames after playback became silent, raw / cleaned | 87.1% / 88.2% |
| Rust processing per 10 ms block: mean / p95 / maximum | 0.110 / 0.149 / 0.230 ms |
| Audio callback faults / recording duration | 0 / 44 seconds |

The participant's assessment of the first cleaned overlap replay was **a bit hard
to understand, but understandable**. Record this as intelligible with noticeable
degradation. The first test's final speech-only section was invalid because an
ambiguous cue led the participant to stop early; the second test corrected that.

Controlled tests at 16 and 48 kHz achieved 41.8–57.0 dB steady echo-only reduction
with simulated multi-path echo. Quiet overlapping speech was strongly suppressed.
At 48 kHz and 80 ms echo delay, speech-detection recall against a processed
speech-only reference was 18.2%, 60.3% and 89.4% for speech levels respectively
10 dB below, equal to and 6 dB above the echo. Zero-delay recovery after overlap
was also weaker than the delayed cases, requiring further investigation.

**Conclusion:** native Rust AEC has ample processing headroom in these tests and
can prevent echo-triggered speech detection. Overlapping speech quality remains
an acceptance issue. Sonora is a promising migration candidate; its tested
defaults still need validation and tuning before production adoption.

The VAD percentages describe detection at threshold 0.8, rather than transcription
accuracy or intelligibility. Live overlapping speech has no isolated clean
reference. Processing timings cover the native render/capture calls, excluding
device buffering and Python pipe transport. Windows/macOS, long-term clock drift,
device hotplug and sustained concurrent model inference remain untested.

Detailed artifacts:

- [Probe implementation, reproduction steps and limitations](../examples/sonora_aec_probe/README.md)
- [Controlled measurements and level sweep](benchmarks/sonora-aec-2026-10-06.json)
- [Second live test](benchmarks/sonora-aec-live-2026-10-06.json)
- [First live test and participant feedback](benchmarks/sonora-aec-live-first-2026-10-06.json)

The probe left production audio code unchanged, and GLaDOS was restarted with its
prior configuration after testing. Recordings remain in the temporary test folder;
the repository contains the harness and measurement reports.

## Audio integration and acceptance

The native audio pipeline should feed the exact PCM sent to the output device into
the AEC render reference, after mixing, gain and resampling. Capture processing
then runs AEC before resampling to Silero's 16 kHz input and before passing speech
to Gemma. Track capture/playback timing and buffering, and handle clock drift,
output cancellation and device changes. Keep inference and allocation outside
real-time device callbacks through preallocated buffers and bounded worker queues.

Before adopting Sonora, compare the same recordings against C++ WebRTC AEC, tune
overlap suppression, and measure transcription accuracy as well as listening
quality. The acceptance scenarios should cover speaker-only playback without false
interruptions, quiet and loud speech over playback, speech after playback stops,
playback cancellation, device reconnects and model inference running concurrently.
Run them on all three target platforms. Passing those checks is part of the Rust
migration's audio acceptance, alongside parity with the completed GLaDOS workflows.
