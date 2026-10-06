# Gemma 4 E4B locally

The default profile uses **E4B for direct audio and replies**, English by
default, with thinking disabled. Parakeet is not instantiated or warmed up,
which avoids its model allocation. The tested backend is llama.cpp's
`/v1/chat/completions` endpoint, alias `gemma-4-E4B`, port 18080.

The **extended profile** uses Parakeet TDT (`asr_engine: tdt`) plus any suitable
language model with thinking disabled or minimal. It starts with Ollama E4B as
an editable example; use a current Ollama release and `ollama pull gemma4:e4b`.
Set `llm_model`, `completion_url`, and backend-specific `llm_request_options`.
For Ollama use `think: false`; for APIs supporting it use `reasoning_effort:
minimal`, or disable reasoning on the inference server. Do not send Ollama's
`think` field to an API that does not support it.
[Ollama model](https://ollama.com/library/gemma4:e4b),
[thinking parameter](https://docs.ollama.com/api/chat).

```bash
uv run glados webapp --config configs/glados_webapp_config.yaml  # Default E4B
uv run glados webapp --config configs/glados_extended_config.yaml  # Parakeet + chosen LLM
```

Existing custom configs retain the legacy transcription path unless they
explicitly enable `native_audio`. Camera observations use a dedicated E4B endpoint, including when the speaking
agent uses an API model. See [Vision](vision.md) for setup and controls.
12B was not evaluated after E4B was selected.

## Optional user transcripts

```yaml
Glados:
  native_audio:
    enabled: true
    user_transcripts: false
    language: English
    max_duration_s: 30
```

With transcripts off, each VAD-delimited recording goes directly to Gemma.
With `user_transcripts: true`, the same Gemma server first generates a transcript
for history and telemetry. Routing, Emotion and the response reuse that text.
This adds a request but loads no Parakeet. If optional transcription fails, the
original audio remains available for the direct reply. Native audio uses the existing microphone/VAD
pipeline, including mute and interruption behavior. Turns exceeding the
configured 30-second maximum are discarded with an observability warning;
recording resumes after a pause. The model never acts on a silently truncated request.

Raw audio is transient and is never stored in conversation history or telemetry.
**With transcripts off, history contains a voice-input placeholder, so exact
user wording is unavailable to later turns and memory agents.** Enable transcripts
when that context matters. Transcript-based wake words require the extended
profile; native mode rejects that combination explicitly.

The actual streaming processor was checked against E4B with a synthetic spoken
question, both with transcripts off and on. See
[native-audio results](benchmarks/gemma4-native-2026-10-04.json).

A 2026-10-06 experiment deferred transcripts until after the reply stream. On this
RTX 3080, three warm paired synthetic turns had median first-sentence latency
of 0.65 s with transcription first and 0.98 s with transcription deferred.
The latter repeatedly encodes audio and cannot use parallel text drafts. These
measurements include routing and Emotion but exclude VAD and TTS; they are a
small smoke test, not a natural-speech evaluation. The faster shared-transcript
path remains in use. See [experiment results](benchmarks/deferred-transcripts-2026-10-06.json).

## RTX 3080 smoke evaluation, 2026-10-04

E4B Q4_0 with Q8_0 audio/vision projector ran on this machine's 10 GB RTX 3080
using llama.cpp commit `687e7789`, CUDA 12.8, 4,096 context, one server slot.
Observed server GPU allocation was 4,118 MiB (not a sampled peak). The model
and projector total approximately 5.15 GB on disk. Ollama's default quantization
differs, so these measurements do not describe Ollama's memory or latency.

Inputs were two **synthetic espeak-ng utterances**, English (4.46 s) and German
(4.84 s), resampled to 16 kHz, plus one generated image. Each was repeated three
times. Backends ran separately. This is a reproducible smoke test, not an
accuracy benchmark on natural speech, noise, accents, or concurrent workloads.
Timings exclude recording, VAD, model loading, and speech synthesis.

| Task | E4B total latency | Parakeet total latency | Result |
| --- | --- | --- | --- |
| English transcription | 0.23–0.29 s | 0.30–0.31 s warm | Both preserve words; write `7` for `seven`. |
| German transcription | 0.26 s | 0.26–0.29 s | E4B uses `stell` for `stelle`; Parakeet preserves it. Both write `7`. |
| Image description | 0.33 s | — | Correct red square left / blue circle right; omitted requested `TEST 42` text. Partial pass. |
| Two emotion directions | 0.30 s | — | Correct `[emotion:smug]` then `[emotion:disappointed]` with two spoken sentences. |

Parakeet's first transcription took 3.83 s, including first-use initialization;
model construction separately took 4.97 s. Gemma measurements used an already
loaded server. Prompt caching was disabled for the recorded run, although
repeated multimodal inputs may benefit from encoder caching. Raw word-error
rates count number formatting differences as errors and should not be read as
semantic accuracy. Raw outputs: [E4B](benchmarks/gemma4-2026-10-04.json) and
[Parakeet](benchmarks/parakeet-2026-10-04.json).

The selected default is E4B direct audio for its smaller combined model footprint;
Parakeet remains the extended fallback if audio quality or speed is poor.
This small sample showed an extra German word difference and does not establish
equivalent accuracy. Broader evaluation needs recorded speech, silence/noise
cases, longer clips, and contention tests with the LLM/TTS running.
Google's documented audio limit is 30 seconds;
the model produces text, so the existing TTS is still needed.
[Gemma audio documentation](https://ai.google.dev/gemma/docs/capabilities/audio).

## Reproduce with llama.cpp

The installed Ollama on this machine was 0.6.8, so evaluation used a separate
llama.cpp build. Neither the installed Ollama nor its model store was modified.

Model repository: [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF),
revision `b8093469224f83f5c38f691eb906c380e9e63114`:

| File | SHA-256 |
| --- | --- |
| `gemma-4-E4B-it-Q4_0.gguf` | `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2` |
| `mmproj-gemma-4-E4B-it-Q8_0.gguf` | `197f49a93027f9843772bd24a6a9e0be2a32a788de5a3def330e9c585d86edd1` |

Both verified files are in the ignored `models/gemma4-benchmark/` directory.
The installed build on this machine is
`/home/dnhkng/Documents/LLM/llama.cpp/build-current/bin/llama-server` (commit `687e7789`).
The earlier `/tmp/glados-gemma4-build` may disappear after reboot. Use a Gemma 4-capable
CUDA build; the older `build/bin/llama-server` on this machine does not support this model.
Start the tested server profile with:

```bash
LLAMA_SERVER=/path/to/current/llama-server bash scripts/run_llamacpp.sh
```

`LLAMA_SERVER` defaults to `llama-server` on PATH. `GLADOS_MODEL_DIR` can override
the model directory. Extra arguments are passed to llama-server. The equivalent command is:

```bash
llama-server \
  -m models/gemma4-benchmark/gemma-4-E4B-it-Q4_0.gguf \
  --mmproj models/gemma4-benchmark/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --alias gemma-4-E4B --host 127.0.0.1 --port 18080 \
  -c 65536 -ngl all --parallel 4 --cont-batching --reasoning off \
  --swa-full --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --cache-prompt --cache-ram 8192 \
  --no-cache-idle-slots -b 2048 -ub 512 --log-verbosity 2
```

Disable thinking explicitly: merely setting a zero reasoning budget leaked
reasoning into answers in the initial trial. The benchmark also sends
`chat_template_kwargs: {enable_thinking: false}`.

Run the webapp with this server using:

```bash
uv run glados webapp --config configs/glados_gemma4_llamacpp.yaml
```

Run the smoke benchmark with `espeak-ng` installed and the Python project
environment plus SciPy available:

```bash
uv run --with scipy python examples/benchmark_gemma4.py --output /tmp/gemma4.json
# Stop llama-server first to isolate the Parakeet comparison:
uv run --with scipy python examples/benchmark_gemma4.py --backend parakeet --output /tmp/parakeet.json
```

The benchmark uses local synthetic fixtures and a local endpoint. No microphone
recording or private image is needed. For speech-driven avatar controls, see
[the webapp protocol](webapp.md#spoken-emotion-directions).

## Cache and latency tuning, 2026-10-06

The default profile uses four independent 32K contexts, Q8_0 key/value cache and
full sliding-window retention. Two application slots are reserved for interactive
work; background minds share the remaining two. The mixed conversation, four-image
vision, emotion and memory test left about 1 GiB of GPU memory free on the 10 GiB
RTX 3080. Compaction starts around 19.7K estimated stored tokens, leaving room for
instructions, tools, current input and output. See the
[four-slot capacity and prompt-reuse measurements](benchmarks/gemma4-four-slot-capacity.md).

The initial two-slot latency tuning below retained full sliding-window KV state with `--swa-full`.
Gemma's rolling 512-token window otherwise causes checkpoint rollback and extra
prompt processing when contexts change. GPU KV buffers increased from 208 MiB
to 448 MiB. Prompt caches stay in GPU memory while resident and can be saved to
the RAM prompt cache when displaced. `--cache-ram 8192` is an 8 GiB limit, not an
up-front allocation. `--no-cache-idle-slots` avoids copying every idle context to
RAM on each new request; displaced contexts are still saved by the server's slot
selection logic. That initial profile used separate KV sequences for two slots.

All 43/43 model layers and the KV buffers were confirmed on CUDA. Flash Attention
was already enabled in automatic mode; the explicit flag makes that reproducible.
The batch size remains 2048 and physical batch size 512: testing 256 and 1024
gave no consistent routing improvement. That initial test used F16 KV types;
the current four-slot default uses Q8_0. The advanced
`--cache-reuse` flag is unsupported for multimodal inputs in this build.

Nine isolated runs included original settings before and after tuning, full cache,
two batch sizes, explicit Flash Attention alone, and disabling idle-context copies.
Background minds and listening were paused during these comparisons and restored.
Five repetitions per condition used current router options, synthetic English
speech with varying synthesis speeds, and two concurrent reply requests.

| Task | Original settings, final repeat | Selected settings |
| --- | ---: | ---: |
| Two-stage typed time routing | 229 ms | 128 ms |
| Fresh-audio root classification | 122 ms | 83 ms |
| Reply first token, stable prefix / changing question | 36 ms | 19 ms |
| Two concurrent replies, total pair time | 633 ms | 541 ms |

These are small synthetic timing samples, not end-to-end voice latency or an
accuracy benchmark. Some audio CPU-load questions fell back to the full assistant
in every tested profile; routing outcomes were consistent across server settings.
Selected settings peaked at 4900 MiB total GPU usage in the isolated test, including
desktop allocations. [Raw results](benchmarks/llamacpp-tuning-2026-10-06.json).

Prompt assembly also preserves stable prefixes: personality, speech and console
rules, operator instructions and preferences precede completed history. Retrieved
memory, core reports, PAD, MCP data, clock, routing/tool-result metadata and the
camera observation follow it, just before the current user/tool exchange. Tool
calls and results stay contiguous. Live state is transient, never written into
conversation history. The context inspector displays this actual order.

Each mind keeps output instructions ahead of changing data. Emotion's PAD schema
and baselines precede state and events; decision schemas precede weather/story data.
Vision's fixed instructions precede the previous image and then the current image.
Visual questions follow their image. Compaction, observer and transcription already
use stable instructions before their input. The router retains its classification
rules while placing changing recent conversation next to the current input.
Custom autonomy tick prompts remain editable; the default puts tasks before live
timing and scene updates.

With vision running, ten typed routing previews improved from a median 436 ms to
273 ms; ten audio previews improved from 410 ms to 269 ms. All twenty final previews
were accepted and selected the expected time/date or CPU-load capabilities. These
audio trials repeated the same synthetic recordings, so they measure warm reuse.
They exclude VAD, transcription, emotion updates, answer generation and TTS.
The main context preview's exact shared prefix between live updates grew from
304 to 2074 tokens, with tool declarations omitted from that comparison. This is
prefix length, not a guarantee that all those tokens stay resident in cache.
[Live comparison](benchmarks/llamacpp-live-2026-10-06.json).

The final server also passed a live-clock reply check, a synthetic previous/current
image comparison with the new ordering, and an emotion JSON check without changing
the application's conversation or PAD. [Functional checks](benchmarks/llamacpp-functional-2026-10-06.json).

To repeat the isolated probe, pause the background cores in the console and run:

```bash
uv run --with scipy python examples/benchmark_llamacpp.py --output /tmp/llamacpp.json
```

This probe does not change settings or execute tools. Restore the cores afterwards.

## Concurrent inference and routing

Health Core samples host/runtime readings every 10 seconds and puts a timestamped
snapshot directly in response context. Every 60 seconds it can generate a short
rolling commentary using the existing background pool. Ordinary questions about
supplied CPU load, RAM, disk space, uptime, OS and GPU readings select ordinary
reply rather than system-tool branches. Missing or stale metrics retain their
tool routes. Search, actions, detailed diagnostics, foreign clocks and sleep/wake
handling continue to use routing.

The current E4B configs use four inference slots, with two reserved for
interactive requests. Launch llama-server with `--parallel 4 -c 65536` and
Q8_0 key/value cache as above; this retains 16384 tokens per slot. One model
weight allocation serves all four requests. Background minds share two slots
and excess work queues. Increasing Python workers alone does not increase server capacity.
For a one-slot server configure `inference: {slots: 1, reserved_interactive: 0}`.

Routing uses one output token from the same E4B model, including direct audio;
it does not load Parakeet. Settings → Decision lists controls its options and
optional parallel drafts. The extended profile can still use any model; the
scoring adapter requires llama.cpp's tokenization and post-sampling probability
API, so leave routing disabled for other backends. Changing endpoint/model resets
routing activation to the new profile's default while retaining editable lists.

On 2026-10-04, a two-request, 128-token-per-request smoke test observed two
server slots processing simultaneously: 1.24 s total versus 2.05 s with one
slot. Both parallel first tokens arrived at about 0.14 s; with one slot the
second request waited about 1.07 s. This is one short run with caching enabled,
not a throughput or natural-speech benchmark. See
[recorded results](benchmarks/routing-concurrency-2026-10-04.json).

### Continuing an unanswered voice turn

Voice endpointing waits for 13 silent 32 ms VAD chunks (416 ms). If speech resumes
before response delivery, the pending inference/TTS generation is invalidated and
the earlier PCM is combined with the new segment. The combined request replaces
the earlier user-history entry under one utterance ID. Parakeet mode retranscribes
the combined audio; direct-audio mode sends the complete clip to E4B without requiring
transcripts.

The continuation buffer is RAM-only, limited to the configured direct-audio clip
length (30 seconds by default), and expires after 30 seconds without continuation.
Playback beginning, a delivered text-only reply, mute/reset, capture gaps, shutdown,
or another input generation ends the window. Speech after delivery is a separate
turn. Combined clips over the limit are rejected rather than silently truncated.
Playback startup and generation cancellation share a lock so an obsolete queued
reply cannot start after the new turn wins that boundary.

The [synthetic continuation check](benchmarks/voice-continuation-2026-10-06.json)
feeds two fragments through SpeechListener and verifies their combined meaning
with local E4B; it does not record the user or play audio.

The four-slot launch now reserves 16K tokens per slot (64K total), reduced from
32K per slot to leave GPU space for Parakeet. On the RTX 3080 this reduced
observed total GPU memory use from 8943 MiB to 6517 MiB with background work
paused. Model weights, four slots, and Q8 KV precision are unchanged.
