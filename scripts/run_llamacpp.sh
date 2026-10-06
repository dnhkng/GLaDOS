#!/usr/bin/env bash
# Start a Gemma 4-capable CUDA llama.cpp server for the default GLaDOS profile.
set -euo pipefail

GLADOS_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GLADOS_MODEL_DIR="${GLADOS_MODEL_DIR:-$GLADOS_REPO_DIR/models/gemma4-benchmark}"
LLAMA_SERVER_BIN="${LLAMA_SERVER:-llama-server}"

if ! command -v -- "$LLAMA_SERVER_BIN" >/dev/null 2>&1; then
    echo "Set LLAMA_SERVER to a Gemma 4-capable CUDA llama-server executable." >&2
    exit 1
fi
for model in gemma-4-E4B-it-Q4_0.gguf mmproj-gemma-4-E4B-it-Q8_0.gguf; do
    if [[ ! -r "$GLADOS_MODEL_DIR/$model" ]]; then
        echo "Missing model: $GLADOS_MODEL_DIR/$model (see docs/gemma4.md)." >&2
        exit 1
    fi
done

exec "$LLAMA_SERVER_BIN" \
    -m "$GLADOS_MODEL_DIR/gemma-4-E4B-it-Q4_0.gguf" \
    --mmproj "$GLADOS_MODEL_DIR/mmproj-gemma-4-E4B-it-Q8_0.gguf" \
    --alias gemma-4-E4B --host 127.0.0.1 --port 18080 \
    -c 65536 -ngl all --parallel 4 --cont-batching --reasoning off \
    --swa-full --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --cache-prompt --cache-ram 8192 \
    --no-cache-idle-slots -b 2048 -ub 512 \
    --log-verbosity 2 \
    "$@"
