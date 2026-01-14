# Vision Implementation Status

## Current State
- Continuous vision snapshot pipeline implemented with FastVLM (ONNX) and injected as a single `[vision]` system message at LLM request time.
- Detailed vision queries are handled via the `vision_look` tool (tool call triggers a fresh capture + custom prompt).
- Tool calling infrastructure (ToolExecutor + tool registry) merged into this branch.
- FastVLM tokenizer/preprocessing fixed to use Qwen2 chat template and model configs (ByteLevel BPE, 1024px center-crop).
- Smoke test added and passing locally once deps are available.

## Git
- Latest commit on branch `vision`: `bab2537` (Add FastVLM snapshot vision pipeline).
- Uncommitted fixes (post-test):
  - `src/glados/__init__.py` (lazy imports to avoid sounddevice load at import time)
  - `src/glados/utils/resources.py` (resource_path no longer imports `glados`)
  - `src/glados/vision/fastvlm.py` (decoder requires position_ids + past_key_values)

## Models and Assets
- Committed: `models/Vision/` (configs + tokenizer files only), `data/Golden_Gate_Bridge.jpeg`.
- ONNX files are intentionally not tracked; install script should fetch them later.
- Selected ONNX variants: `vision_encoder_fp16.onnx`, `embed_tokens_int8.onnx`, `decoder_model_merged_q4f16.onnx`.

## Dependencies
- `opencv-python` and `regex` are in `pyproject.toml`.
- `onnxruntime` is provided by the `cpu` extra (or `onnxruntime-gpu` via `cuda` extra).

## Tests
- `python -m pytest tests/test_fastvlm_smoke.py` passes after fixing FastVLM decoder inputs.
- In sandboxed Docker, `nvidia-smi` fails (NVML unavailable), so GPU detection is not reliable here.

## TODOs
- Commit the uncommitted fixes listed above.
- Add FastVLM ONNX downloads to `glados download` flow (or `scripts/install.py`) so the repo remains lightweight.
- Validate GPU execution outside sandbox and switch to `onnxruntime-gpu` if available.

## Next Steps (outside sandbox)
1. Confirm GPU visibility: `nvidia-smi`.
2. Update environment: `uv pip install -e .[cuda,dev]` (or `.[cpu,dev]` if no GPU).
3. Run smoke test: `python -m pytest tests/test_fastvlm_smoke.py`.
4. Run vision config: `uv run glados start --config ./configs/glados_vision_config.yaml`.
