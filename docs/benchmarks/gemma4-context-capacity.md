# Gemma 4 E4B context capacity

Measured on 2026-10-06 with the RTX 3080 (10,240 MiB VRAM), llama.cpp
`687e7789`, Gemma 4 E4B Q4_0 weights and the Q8_0 multimodal projector.
The existing audio/vision-capable server command was retained, changing only
context size, slot count, cache precision and full sliding-window retention.
Batch sizes remained 2,048 / 512, with Flash Attention enabled.

## Results

GPU measurements include the desktop and other existing GPU use. Free memory
excludes approximately 380 MiB reserved by the driver.

| Configuration | Total context | Peak GPU use | Free at peak | Validation |
| --- | ---: | ---: | ---: | --- |
| 32,768 per slot × 8, F16 rolling cache | 262,144 | 8,827 MiB | 1,032 MiB | Eight simultaneous 31,000-token prompts passed |
| 32,768 per slot × 8, Q8_0 rolling cache | 262,144 | 7,788 MiB | 2,071 MiB | Eight simultaneous 31,000-token prompts passed |
| 32,768 per slot × 4, Q8_0 full SWA cache | 131,072 | 8,730 MiB | 1,128 MiB | Four simultaneous short prompts passed |

Each long request evaluated 31,000 tokens and generated 64 tokens. All returned
HTTP 200 with nonempty output. The workload used repeated synthetic text and
disabled prompt caching. End-to-end times were 39–60 seconds with F16 and
42–65 seconds with Q8_0. These are capacity checks, not representative answer
quality or interactive latency benchmarks. Concurrent image/audio workloads
were not exercised.

The matching llama.cpp memory estimator reports 14,336 MiB of F16 KV state for
32K × 8 with `--swa-full`, before model weights and compute buffers. That setting
does not fit this GPU. Rolling SWA retains the model's sliding attention window
while the global-attention layers retain the full context; it does not limit
the complete conversation to the sliding-window length. It can cause more
prompt reprocessing when conversations switch or cached prefixes are restored.

## Suggested profile

For more memory headroom, remove `--swa-full` and use:

```sh
-c 262144 --parallel 8 --cache-type-k q8_0 --cache-type-v q8_0
```

The application inference pool would also need eight slots and its configured
context budget would need 32,768 tokens. Q8_0 cache quality was not compared
against F16 in this test. F16 also fits if its smaller memory margin is acceptable.

The original 4,096-token-per-slot, two-slot server and the application controls
were restored after testing; this report does not deploy the suggested profile.
Test logs used bounded rotating capture (1 MiB plus one backup per test).
