#!/usr/bin/env bash
# Build the isolated `protoskin3d-os` conda env for the 3D reconstruction
# service (visual_engine/reconstruct3d.py via scripts/run_3d_service.sh).
#
# Why a separate env: TripoSG's code targets diffusers 0.32 /
# transformers 4.x, while the main `zgx` env runs newer versions for SDXL
# and the Qwen3 explainer. Keeping them apart avoids any version clash.
#
# Everything here is open source and installs from prebuilt wheels on
# aarch64 (HP ZGX Nano) -- no CUDA extensions to compile. Deliberately NOT
# installed: `diso` (CC BY-NC 4.0; TripoSG's flash extractor, which
# ProtoSkin never uses -- see third_party/README.md) and Bria RMBG-1.4
# (non-open licence; ProtoSkin uses rembg's isnet/u2net instead).

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-protoskin3d-os}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
conda activate "$CONDA_ENV_NAME"
echo "Using interpreter: $(command -v python)"

# torch: match the cu13x build used by the main `zgx` env (GB10 / CUDA 13).
# Check yours with: <zgx python> -m pip show torch
pip install torch==2.14.0 torchvision

# Versions pinned to what was verified end-to-end with TripoSG on the
# ZGX Nano. rembg 2.0.57 is the last release supporting Python 3.10.
pip install \
  "diffusers==0.32.2" "transformers==4.49.0" peft accelerate \
  einops jaxtyping typeguard omegaconf \
  scikit-image scipy trimesh fast-simplification "rembg==2.0.57" \
  fastapi uvicorn python-multipart httpx

# Background-removal models. rembg downloads on first use, which would
# break the fully-offline demo, so fetch them now (~350MB into ~/.u2net).
python -c "import rembg; rembg.new_session('isnet-general-use'); rembg.new_session('u2net')"

echo
echo "== Verifying =="
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$REPO_ROOT/third_party/TripoSG" python -c "
import importlib.util, torch
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
assert importlib.util.find_spec('diso') is None, 'diso (non-open) must not be installed'
print('TripoSG imports OK without diso. CUDA available:', torch.cuda.is_available())
"

echo
echo "== Done. Environment '$CONDA_ENV_NAME' is ready. =="
echo "Set PROTOSKIN_3D_PYTHON to: $(command -v python)"
