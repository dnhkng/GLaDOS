# Sonora AEC probe

Standalone Rust AEC3 worker and Python measurement/device harness. This does not
change GLaDOS's production audio path. Sonora is pinned to upstream commit
`413f64e7404529aaae6980613527f0d6acb21ce7` (crate version 0.2.0).

## Run

Requires Rust 1.91+, the project's Python environment, and existing GLaDOS TTS,
phonemizer and Silero models. Run from the repository root:

```bash
cargo +1.91.0 build --release --locked --manifest-path examples/sonora_aec_probe/Cargo.toml
.venv/bin/python examples/benchmark_sonora_aec.py \
  --synthesize --work-dir /tmp/glados-aec \
  --report /tmp/glados-aec/results.json
```

For a live test, stop GLaDOS first so it cannot react to playback. List audio
devices with `.venv/bin/python -m sounddevice`. Use a microphone without built-in
AEC to distinguish Sonora's contribution from hardware processing:

```bash
.venv/bin/python examples/benchmark_sonora_aec.py \
  --synthesize --work-dir /tmp/glados-aec \
  --report /tmp/glados-aec/live.json \
  --live --live-only --input-device 9 --output-device 13
```

Device indices above were valid only on the Linux machine used on 2026-10-06.
Keep quiet initially. At “Begin speaking now,” speak over playback. At “Continue
speaking,” keep talking through the silence. Finish at “You can be quiet now.”
Each live recording is 44 seconds, after spoken instructions. Recordings are
saved as `live-render.wav`, `live-capture.wav`, and `live-clean.wav` in the work
directory; subsequent live runs replace them. Use a different work directory
to retain multiple recordings. `--render existing.wav` reuses a fixture; live
mode also needs `instructions.wav`, `talk.wav`, `near.wav`, and `stop.wav` there.

The Rust binary also accepts:

```text
file RENDER.wav CAPTURE.wav OUTPUT.wav DELAY_MS [bypass]
stream RATE DELAY_MS
```

WAV mode accepts mono integer/float WAVs with matching rates and lengths, in
whole 10 ms frames. Pipe mode reads one render frame followed by one microphone
frame as little-endian f32, and emits one cleaned frame. Rates are 16/32/48 kHz.
Timing/statistics are JSON on stderr. NS and AGC are disabled; AEC3 uses its
defaults, including enforced high-pass filtering and automatic delay estimation.

## Findings on 2026-10-06

Two live runs used UGREEN webcam capture and Jabra Speak 710 playback via the
system default output. In the second run:

| Measurement | Result |
| --- | --- |
| Echo-only microphone energy reduction, seconds 10–16 | 24.8 dB |
| Echo-only Silero speech frames, raw / cleaned | 89.4% / 0% |
| Speech frames during overlap, raw / cleaned | 99.1% / 78.0% |
| Speech frames after playback became silent, raw / cleaned | 87.1% / 88.2% |
| Native processing per 10 ms frame, mean / p95 / maximum | 0.110 / 0.149 / 0.230 ms |
| Audio callback faults / recording duration | 0 / 44 s |

The participant described the first cleaned overlap replay as “a bit hard to
understand, but understandable.” Speech quality is therefore a concern even
though echo rejection and runtime cost are promising. The first run's final
speech-only section is invalid: the participant stopped early after an ambiguous
cue containing “stop.” The repeat used clearer cues and captured continued speech.

Controlled 40-second tests at 16 and 48 kHz used GLaDOS playback, the existing
`data/0.wav` as independent near speech, and three linear echo taps. With 0/40/80/
200 ms base delays, steady echo-only reduction was 41.8–57.0 dB. However, quiet
overlapping speech was strongly suppressed. At 48 kHz and 80 ms delay, an added
level sweep gave:

| Near speech relative to echo | Silero recall against processed speech-only reference |
| --- | --- |
| −10 dB | 18.2% |
| 0 dB | 60.3% |
| +6 dB | 89.4% |

Silence produced exact zeros and the disabled-AEC control was sample-identical.
Raw-reference waveform SI-SDR includes the high-pass/filter-bank phase response;
the report also compares against an AEC-processed speech-only reference to better
isolate overlap suppression. Near-only total energy changed by about −0.19 dB
in the controlled test. Zero-delay echo recovery after overlap was weaker than
the delayed cases (about 5–7 dB), another behavior requiring investigation.

Machine-readable reports are in `docs/benchmarks/sonora-aec*-2026-10-06.json`.
Private recordings stay in the temporary work directory, outside the repository.

## Interpretation and limits

Native Rust AEC is feasible on this Linux machine. The release executable links
ordinary Linux runtime libraries; `ldd` showed no WebRTC or C++ AEC shared library.
Cargo fetched an upstream C++ reference submodule, but the probe builds the Rust
processing crates, not that reference implementation.

The current defaults are not yet accepted for production speech quality. Next
validation should compare the same recordings with the C++ WebRTC implementation
and tune overlap suppression before integrating the Rust audio engine. VAD
fractions measure detection, not transcription accuracy or intelligibility. Live
overlap has no isolated clean speech reference, and reduced total energy can
include background noise and speech filtering. The tests do not establish macOS/
Windows behavior, device hotplug, long-term clock drift, nonlinear loudspeakers,
or performance under sustained model inference load. Audio devices here use
Python/PortAudio; this tests the native Rust DSP worker, not a finished Rust device
backend. No production GLaDOS audio code was modified by this probe.
