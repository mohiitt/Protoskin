#!/usr/bin/env bash
# Download the three models required for the first end-to-end build.
# Run this on the ZGX Nano before the demo. Weights stay in models/ and are not committed.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v hf >/dev/null 2>&1; then
  echo "Install the Hugging Face CLI first: python -m pip install -U huggingface_hub"
  exit 1
fi

mkdir -p models

hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir models/sdxl-base-1.0

hf download diffusers/controlnet-canny-sdxl-1.0 \
  --local-dir models/controlnet-canny-sdxl

hf download Qwen/Qwen3-8B \
  --local-dir models/qwen3-8b

echo "Primary models are in models/. Optional depth and fallback models are listed in the implementation plan."
