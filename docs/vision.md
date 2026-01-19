# GLaDOS Vision Module

GLaDOS can see and react to its environment using Apple's FastVLM running locally via ONNX Runtime.

## Role in Architecture

Vision is a core input to the [autonomy loop](./autonomy.md). When enabled:

1. **Camera captures frames** at configured intervals
2. **Scene change detection** identifies meaningful changes
3. **FastVLM generates descriptions** of the current scene
4. **VisionUpdateEvent triggers** the autonomy loop
5. **Main agent decides** whether to act on what it sees

```
+-------------+     +------------------+     +----------------+
|   Camera    +---->| Scene Change     +---->| FastVLM        |
|   Capture   |     | Detection        |     | Inference      |
+-------------+     +------------------+     +-------+--------+
                                                     |
                                                     v
                                             +-------+--------+
                                             | VisionUpdate   |
                                             | Event          |
                                             +-------+--------+
                                                     |
                                                     v
                                             +-------+--------+
                                             | Autonomy Loop  |
                                             | (Main Agent)   |
                                             +----------------+
```

Vision takes priority over timer ticks - when vision is enabled, scene changes drive the autonomy loop instead of periodic timers.

## Quick Start

The vision module is disabled by default. To enable it:

```bash
uv run glados start --config ./configs/glados_vision_config.yaml
```

## Setup

### 1. Download FastVLM Models

```bash
huggingface-cli download onnx-community/FastVLM-0.5B-ONNX \
  --local-dir models/Vision \
  --include "onnx/vision_encoder_fp16.onnx" \
  --include "onnx/embed_tokens_int8.onnx" \
  --include "onnx/decoder_model_merged_q4f16.onnx" \
  --include "config.json" \
  --include "preprocessor_config.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "README.md" \
  --include "LICENSE"
```

Or using the newer command:

```bash
hf download onnx-community/FastVLM-0.5B-ONNX \
  --local-dir models/Vision \
  --include "onnx/vision_encoder_fp16.onnx" \
  --include "onnx/embed_tokens_int8.onnx" \
  --include "onnx/decoder_model_merged_q4f16.onnx" \
  --include "config.json" \
  --include "preprocessor_config.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "README.md" \
  --include "LICENSE"
```

This downloads the ONNX models (~640MB) to the default location.

### 2. Configure Vision

```yaml
vision:
  enabled: true
  model_dir: "models/Vision"
  camera_index: 0
  capture_interval_seconds: 5
  resolution: 384
  scene_change_threshold: 0.05
  max_tokens: 200
```

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable vision module |
| `model_dir` | string | `"models/Vision"` | Path to FastVLM ONNX models |
| `camera_index` | int | `0` | Camera device index |
| `capture_interval_seconds` | float | `5.0` | Time between frame captures |
| `resolution` | int | `384` | Scene-change detection resolution |
| `scene_change_threshold` | float | `0.05` | Minimum change to trigger inference (0=always, 1=never) |
| `max_tokens` | int | `200` | Maximum tokens in background description |

## Performance

FastVLM provides **85x faster time-to-first-token** compared to Ollama-based VLMs:

- **Direct ONNX inference** - no HTTP overhead
- **Runs on CPU or CUDA** - GPU acceleration when available
- **Small footprint** - ~640MB model files for 0.5B
- **Frame differencing** - skips unchanged scenes

## Context Injection

The vision system maintains a single `[vision]` slot that's injected into the LLM context:

```
[vision] A person sitting at a wooden desk with a laptop. There is a coffee mug
to their left and a window showing daylight behind them.
```

This snapshot is updated whenever a new inference completes. The main agent sees the current scene in every request.

## Detailed Lookups

For specific visual questions (e.g., "What color is my shirt?"), the LLM can call the `vision_look` tool:

```
vision_look(prompt="Describe the person's clothing in detail")
```

This triggers:
1. Fresh camera capture
2. Custom VLM prompt for the specific question
3. Detailed response returned to the LLM

Requires an LLM backend that supports tool calling.

## VisionProcessor Thread

Vision runs in a separate thread alongside other processors:

- **Captures frames** at `capture_interval_seconds`
- **Compares frames** using the configured threshold
- **Runs VLM inference** when scene changes detected
- **Updates VisionState** with latest description
- **Emits VisionUpdateEvent** to trigger autonomy

The thread is fully async and doesn't block voice or text processing.

## Troubleshooting

**Camera not opening:**
- Check `camera_index` in config (try 0, 1, 2...)
- Verify camera permissions
- Test with: `ls /dev/video*` (Linux) or check System Preferences (macOS)

**Models not found:**
- Ensure models downloaded to `models/Vision/`
- Check for `vision_encoder_fp16.onnx`, `embed_tokens_int8.onnx`, `decoder_model_merged_q4f16.onnx`

**Slow inference:**
- Increase `capture_interval_seconds`
- Ensure CUDA available (`CUDAExecutionProvider`)
- Raise `scene_change_threshold` (higher = fewer inferences)

**Too many triggers:**
- Increase `scene_change_threshold` (0.1 or higher)
- The threshold is a normalized difference score - adjust based on your environment

## Advanced

### Custom Model Path

```yaml
vision:
  model_dir: "/path/to/custom/fastvlm"
```

### Disable Vision

Remove the entire `vision:` section from your config, or set:

```yaml
vision:
  enabled: false
```

## Implementation Details

| Aspect | Value |
|--------|-------|
| **Model** | Apple FastVLM-0.5B (ONNX) |
| **Precision** | fp16 + q4f16 mix |
| **Architecture** | Vision encoder + text decoder |
| **Input** | 1024x1024 RGB images (center-cropped) |
| **Output** | Natural language scene descriptions |
| **Backend** | ONNX Runtime (CPU/CUDA) |
| **Integration** | Same ONNX patterns as ASR/TTS |

## TUI Commands

| Command | Description |
|---------|-------------|
| `/vision` | Show current vision status and last snapshot |
| `/observe` | View all events including vision updates |

## See Also

- [README](../README.md) - Full architecture diagram
- [autonomy.md](./autonomy.md) - How vision triggers the autonomy loop
- [vision_config.py](../src/glados/vision/vision_config.py) - Configuration source
- [constants.py](../src/glados/vision/constants.py) - Vision system prompts
