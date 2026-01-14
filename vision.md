# GLaDOS Vision Module

GLaDOS can capture the world with a camera and react to what it sees using Apple's FastVLM running locally via ONNX Runtime.

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

This downloads the selected ONNX models (~640MB) to the default location.

### 2. Configure Vision

See [vision_config.py](./src/glados/vision/vision_config.py) for configuration options:

- `model_dir`: Path to FastVLM ONNX models (uses `models/Vision` by default)
- `camera_index`: Camera device index (usually 0 for default webcam)
- `capture_interval_seconds`: Time between frame captures (default: 5s)
- `resolution`: Scene-change detection resolution (default: 384px)
- `scene_change_threshold`: Minimum change to trigger inference (0=always, 1=never, default: 0.05)
- `max_tokens`: Maximum tokens in background description (tool calls can override)

## Performance

FastVLM provides **85× faster time-to-first-token** compared to Ollama-based VLMs:
- Direct ONNX inference (no HTTP overhead)
- Runs on CPU or CUDA
- Small footprint (~640MB model files for 0.5B)
- Frame differencing skips unchanged scenes

## Architecture

1. **Camera Capture**: OpenCV captures frames at configured intervals
2. **Scene Change Detection**: Compares frames to skip redundant processing
3. **FastVLM Inference**: Local ONNX models generate a short scene snapshot
4. **Context Injection**: Latest snapshot is injected as a single `[vision]` system message when the LLM runs

## Detailed Lookups

If the user asks for a detailed visual check (e.g., outfit questions), the LLM calls the `vision_look` tool.
The tool triggers a fresh capture and uses a custom prompt for the VLM to answer that specific question.
Requires an LLM backend that supports tool calling.

## Usage Notes

- Vision runs in a separate thread like other processors (ASR, TTS)
- A single `[vision]` snapshot is maintained and updated as new inferences complete
- GLaDOS can react to or ignore vision observations based on context
- The system prompt explains how to use vision snapshots and when to call `vision_look` (see [constants.py](./src/glados/vision/constants.py))

## Troubleshooting

**Camera not opening:**
- Check `camera_index` in config (try 0, 1, 2...)
- Verify camera permissions
- Test with: `ls /dev/video*` (Linux)

**Models not found:**
- Ensure models downloaded to `models/Vision/`
- Check for `vision_encoder_fp16.onnx`, `embed_tokens_int8.onnx`, `decoder_model_merged_q4f16.onnx`

**Slow inference:**
- Increase `capture_interval_seconds`
- Ensure CUDA available (`CUDAExecutionProvider`)
- Check `scene_change_threshold` (higher = fewer inferences)

## Advanced

### Custom Model Path

```yaml
vision:
  model_dir: "/path/to/custom/fastvlm"
  # ... other settings
```

### Disable Vision

Remove the entire `vision:` section from your config, or use a config without vision.

## Implementation Details

- **Model**: Apple FastVLM-0.5B (ONNX, fp16 + q4f16 mix)
- **Architecture**: Vision encoder + text decoder (autoregressive)
- **Input**: 1024×1024 RGB images (center-cropped)
- **Output**: Natural language scene descriptions
- **Backend**: ONNX Runtime (CPU/CUDA)
- **Integration**: Follows GLaDOS ONNX patterns (same as ASR/TTS)
