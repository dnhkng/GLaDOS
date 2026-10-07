# AMD GPU setup

GLaDOS can select AMD's MIGraphX execution provider for its local ONNX audio
models: Parakeet/CTC transcription, speech synthesis, phonemization and VAD.
This does not install or configure the separate llama.cpp model server.

The legacy ROCm execution provider was removed in ONNX Runtime 1.23. Use
[MIGraphX](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
for new installations. Provider selection still accepts a legacy ROCm provider
when an older runtime is already installed.

## Install

1. Install ROCm and the appropriate GPU driver using
   [AMD's instructions](https://rocm.docs.amd.com/), and check that your GPU and
   operating system are supported. The GLaDOS installer does not change system
   drivers or install ROCm.
2. On Linux x86_64, run:

   ```bash
   python scripts/install.py --backend amd
   ```

   The installer reads `ROCM_PATH` or `ROCM_HOME`, defaulting to `/opt/rocm`, and
   selects an official AMD wheel for the detected release. If automatic
   detection cannot find the version, specify the release you have installed:

   ```bash
   python scripts/install.py --backend amd --rocm-version 7.2.1
   ```

3. Launch using the installed environment:

   ```bash
   uv run --no-sync glados
   # Or launch the console:
   uv run --no-sync glados webapp
   ```

The installer creates a Python 3.12 environment. Its AMD wheel mapping follows
the [official ONNX Runtime compatibility table](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html):

| ROCm | ONNX Runtime MIGraphX |
| --- | --- |
| 7.1 / 7.1.0 | 1.23.1 |
| 7.2 / 7.2.0 | 1.23.2 |
| 7.2.1 | 1.23.2 |

These wheels require a compatible Linux system (manylinux 2.28 or newer).
Other ROCm releases, Windows and ARM are not handled by this installer. Use
`--backend cpu` if a matching AMD wheel is unavailable. Without `--backend`,
the installer selects CUDA when its utility is available, then an installed
ROCm environment, then CPU. Explicit selection overrides detection.

## Verification and updates

Installation checks that MIGraphX is available, downloads the models and opens
the VAD model with the AMD runtime. It fails if that session falls back entirely
to CPU. At startup, each audio model logs its session providers; unsupported
operators may still run on CPU even when MIGraphX is registered.

The AMD wheel is installed separately from the project's CPU/CUDA extras.
Always use `uv run --no-sync` for this environment. Do not combine the AMD
runtime with `uv sync --extra cpu` or `uv sync --extra cuda`: these packages
share the `onnxruntime` module and can overwrite one another. To update an AMD
installation, rerun the installer with the same backend and ROCm release.

The installer preserves an existing virtual environment during updates.
When switching backends, it removes the previous ONNX Runtime
distribution before installing the selected one. It stops on package or
verification failures instead of proceeding to model download after a failed
package installation.

If MIGraphX is unavailable or the VAD session cannot initialize, check the
driver, installed ROCm libraries, GPU support and selected release. Discovery
of a provider is not proof that every model is accelerated. Full Parakeet and
TTS compatibility and latency still need validation on AMD hardware; the
automated tests cover installer behavior and provider selection without a GPU.
