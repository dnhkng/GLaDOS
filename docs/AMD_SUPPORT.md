# AMD ROCm / MIGraphX Support Guide

## Overview

GLaDOS now supports AMD GPU acceleration via the MIGraphX Execution Provider for ONNX models. This provides significant performance improvements on RDNA 3.5 and RDNA 4 GPUs.

## Requirements

- **ROCm Version**: 6.4+ (required for RDNA 3.5/4 support)
- **ONNX Runtime**: `onnxruntime-migraphx>=1.21.0`
- **GPU**: AMD RDNA 3.5 or RDNA 4 (e.g., RX 7900 XTX, RX 8900 XTX)

## Installation

### 1. Install ROCm 6.4+

Follow AMD's official ROCm installation guide for your distribution:
https://rocm.docs.amd.com/

### 2. Install GLaDOS with MIGraphX

```bash
cd GLaDOS
pip install -e ".[migraphx]"
```

## Configuration

Use the provided AMD configuration file:

```bash
python -m glados --config configs/glados_config_amd.yaml
```

### Configuration Options

The `glados_config_amd.yaml` file includes these ONNX Runtime settings:

```yaml
onnx_runtime:
  migraphx:
    device_id: 0              # GPU device ID
    fp16_enable: true         # Enable FP16 quantization (recommended for RDNA 3.5/4)
    bf16_enable: false        # Enable BF16 quantization
    int8_enable: false        # Enable INT8 quantization
    exhaustive_tune: false    # Enable exhaustive kernel tuning (slow first run)
    cache_path: null          # Model compilation cache path
    mem_limit: null           # Memory arena limit
```

### Custom Configuration

To create a custom configuration:

1. Copy `configs/glados_config_amd.yaml` to a new file
2. Modify the `onnx_runtime.migraphx` section as needed
3. Launch with: `python -m glados --config your_config.yaml`

## Provider Priority

GLaDOS automatically detects and prioritizes execution providers:

1. **MIGraphXExecutionProvider** (AMD ROCm - highest priority when available)
2. **CUDAExecutionProvider** (NVIDIA GPU)
3. **CPUExecutionProvider** (fallback)

You can override this in your config:

```yaml
onnx_runtime:
  provider_priority:
    - MIGraphXExecutionProvider
    - CPUExecutionProvider
```

## Performance Tips

### FP16 Enablement

For RDNA 3.5/4 GPUs, enable FP16 for best performance:

```yaml
onnx_runtime:
  migraphx:
    fp16_enable: true
```

### Compilation Cache

Enable model compilation caching to speed up subsequent runs:

```yaml
onnx_runtime:
  migraphx:
    cache_path: "/path/to/cache"
    exhaustive_tune: false  # Set true for first run, then false
```

### Thread Configuration

Silero VAD automatically uses single-threaded inference to prevent CPU overhead. Other models use optimal multi-threading by default.

## Supported Models

All ONNX models in GLaDOS support MIGraphX acceleration:

- **TTS**: GladosTTS, KokoroTTS, Phonemizer
- **ASR**: TDT-ASR, CTC-ASR
- **VAD**: Silero VAD
- **Vision**: FastVLM

## Verification

Check that MIGraphX is active:

```python
import onnxruntime as ort

print("Available providers:", ort.get_available_providers())
# Should show: ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
```

## Troubleshooting

### MIGraphXExecutionProvider not available

1. Verify ROCm 6.4+ is installed: `rocm-smi`
2. Reinstall with migraphx extra: `pip install -e ".[migraphx]" --force-reinstall`
3. Check GPU compatibility: RDNA 3.5/4 required

### FP16 errors

Some models may not support FP16. Try disabling it:

```yaml
onnx_runtime:
  migraphx:
    fp16_enable: false
```

### Memory issues

Reduce memory usage with:

```yaml
onnx_runtime:
  migraphx:
    mem_limit: 4294967296  # 4GB limit
```

## Migration from CUDA

If migrating from NVIDIA to AMD:

1. Uninstall `onnxruntime-gpu`: `pip uninstall onnxruntime-gpu`
2. Install `onnxruntime-migraphx`: `pip install -e ".[migraphx]"`
3. Use `glados_config_amd.yaml` instead of CUDA config
4. No code changes required - provider selection is automatic

## Documentation

- [MIGraphX Execution Provider Docs](https://onnxruntime.ai/docs/execution-providers/MIGraphxExecutionProvider.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Instinct GPU Docs](https://www.amd.com/en/products/accelerators/instinct.html)
