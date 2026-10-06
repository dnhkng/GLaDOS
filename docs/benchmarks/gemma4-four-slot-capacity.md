# Four-slot inference assessment

Measured 2026-10-06 on the RTX 3080 (10,240 MiB VRAM), using llama.cpp
`687e7789`, Gemma 4 E4B Q4_0 and the Q8_0 multimodal projector. Existing batch
sizes, Flash Attention, native audio support and bounded log capture were retained.

## Work distribution

The main GLaDOS request includes conversation history, instructions, live state,
tool definitions and tool results. It is the request expected to grow longest.
Routing generates one token and includes at most four recent messages clipped
to 500 characters each, plus the current input. Emotion uses a bounded queue of
20 events and generates at most 96 tokens. Memory compaction splits history into
small inputs (default nominal input budget 1,200 tokens) and generates at most
160 tokens. Vision keeps at most four images and generates at most 256 tokens
in the active profile. Image content, audio and unusually long individual events
can increase the background input sizes; output limits are not input limits.

These minds share capacity; each does not need a permanent server slot. Memory
compaction yields while the interactive queue is busy. Routing can overlap with
a speculative GLaDOS draft, which carries the main conversation history again.
The application admission slots are not pinned to particular llama.cpp slots.

For four total slots, reserve two for interactive work. This leaves two slots
for vision, emotion, compaction and other background work. Excess background
requests queue. A transient scheduler experiment verified that routing and a
GLaDOS request can enter while two background jobs are active and a third waits.
Speculative drafts only use spare capacity and are suppressed when work is queued.

## Capacity and reuse tests

Each configuration ran four concurrent direct HTTP requests: a synthetic
31,000-token main prompt generating 128 tokens, a four-image vision request,
a memory summary and an emotion update. Every request returned HTTP 200 with
nonempty output; vision and emotion returned valid JSON. Native audio was loaded
but audio inference was not exercised. This is a capacity experiment, not an
answer quality comparison or a full application latency benchmark.

GPU use includes the desktop. Approximately 380 MiB is driver-reserved.

| Cache configuration | Capacity | Peak GPU use | Free at peak | Main cold request |
| --- | --- | ---: | ---: | ---: |
| F16 rolling SWA | Four independent 32K slots | 6,679 MiB | 3,179 MiB | 10.4 seconds |
| Q8_0 full SWA | Four independent 32K slots | 8,790 MiB | 1,068 MiB | 14.6 seconds |
| F16 full SWA, unified | 64K shared across four slots | 8,148 MiB | 1,711 MiB | 13.9 seconds |

After the mixed workload, a follow-up changed the final 2,000 tokens of the
31,000-token main prompt and generated 64 tokens. F16 rolling SWA reused zero
tokens and took 7.67 seconds. Q8_0 full SWA reused 29,000 tokens, evaluated only
2,000 new tokens and took 2.04 seconds. An unchanged follow-up reused 30,999
tokens in both configurations and took 0.67 / 1.11 seconds respectively.
The full cache has a clear benefit when live state changes earlier than the
retained sliding window; the rolling cache has better memory headroom and was
faster on cold processing in this experiment.

The unified cache permits uneven context lengths, but 64K is a shared total,
not four independent 64K allocations. It does not reserve 32K for the main
conversation. A long main context and a long speculative draft can consume
almost the whole pool before background requests are included. Additional
input-budget/admission policy would be needed to guarantee that allocation.

## Recommendation

Four independent 32K slots, Q8_0 key/value cache, full SWA retention, with two
application slots reserved for interactive requests. This retains prompt reuse
for long conversations and fits the mixed workload with approximately 1 GiB free.
Use the F16 rolling profile if additional GPU memory headroom takes priority.
Q8_0 cache answer quality was not compared with F16 in these experiments.

The existing launcher can select the recommended model profile with:

```sh
scripts/run_llamacpp.sh -c 131072 --parallel 4 -ctk q8_0 -ctv q8_0
```

The launcher already enables `--swa-full`. Corresponding application settings:

```yaml
Glados:
  inference:
    slots: 4
    reserved_interactive: 2
  autonomy:
    tokens:
      model_context_window: 32768
      token_threshold: 20000
      target_utilization: 0.6
```

The current compaction threshold is about 1,843 estimated stored tokens. Raising
only server context would leave that early compaction behavior in place. These
suggested settings start compaction at about 19,660 estimated stored tokens,
leaving room for system instructions, tools, current input and output. The
threshold counts stored history, not the entire serialized inference request.

The live two-slot, 4K-per-slot server and all application controls/pause states
were restored after the assessment. The recommended four-slot Q8_0 full-SWA
profile was subsequently adopted as the default in the launcher and native
E4B configurations, and activated in the running console. Application controls
and core pause states were preserved during activation.
