# Feature Request: Add ROCm Support for AMD GPUs

## Summary

Add support for AMD GPUs via ROCm (Radeon Open Compute) to enable hardware acceleration for users with AMD graphics cards.

## Current State

GLaDOS currently supports:
- CPU inference via `onnxruntime`
- NVIDIA GPU acceleration via `onnxruntime-gpu` (CUDA)

AMD GPU users cannot leverage hardware acceleration, resulting in significantly slower inference for:
- Speech recognition (Parakeet TDT)
- Voice activity detection (Silero VAD)
- Vision processing (FastVLM)
- Text-to-speech synthesis (Kokoro)

## Proposed Changes

### 1. Add ROCm Dependency Option

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
cuda = ["onnxruntime-gpu>=1.16.0"]
rocm = ["onnxruntime-rocm>=1.16.0"]
cpu = ["onnxruntime>=1.16.0"]
```

### 2. Update Installation Documentation

Add ROCm installation instructions:

**AMD GPU (ROCm):**
```bash
pip install glados[rocm]
```

Prerequisites:
- ROCm 5.6+ installed
- Compatible AMD GPU (RDNA2/CDNA2 or newer)
- Linux platform (ROCm support on Windows is limited)

### 3. Update GPU Setup Section

Expand the GPU setup documentation to include all three platforms:

| Platform | Package | Command |
|----------|---------|---------|
| NVIDIA | onnxruntime-gpu | `pip install glados[cuda]` |
| AMD | onnxruntime-rocm | `pip install glados[rocm]` |
| CPU | onnxruntime | `pip install glados[cpu]` |

## Benefits

1. **Broader Hardware Support**: Opens GLaDOS to AMD GPU users
2. **Performance**: Comparable acceleration to CUDA for supported workloads
3. **Cost-Effective**: AMD GPUs often provide better price-to-performance ratios

## Implementation Notes

- ONNX Runtime has supported ROCm since version 1.11
- The same Python API works across CUDA, ROCm, and CPU backends
- No code changes required beyond dependency management
- May need platform-specific installation notes (ROCm primarily supports Linux)

## Testing

Suggested test matrix:
- AMD Radeon RX 6000/7000 series (RDNA2/3)
- AMD Instinct MI series (CDNA)
- ROCm versions 5.6, 6.0+

## Related

- [ONNX Runtime ROCm Documentation](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- [ROCm Compatibility Guide](https://rocm.docs.amd.com/)
