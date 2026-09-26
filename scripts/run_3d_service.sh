#!/usr/bin/env bash
# Launch the sidecar 3D reconstruction service (TripoSG) under the
# protoskin3d-os environment (see scripts/setup_3d_env.sh). The main
# gateway process (gateway_and_ui/backend/main.py) talks to this over HTTP;
# it never imports visual_engine/reconstruct3d.py or service_3d.py directly,
# since they need TripoSG's separate dependency versions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

PYTHON="${PROTOSKIN_3D_PYTHON:-$HOME/miniforge3/envs/protoskin3d-os/bin/python}"
PORT="${PROTOSKIN_3D_SERVICE_PORT:-8100}"

if [ ! -x "$PYTHON" ]; then
  echo "error: $PYTHON not found or not executable."
  echo "Run scripts/setup_3d_env.sh first, or set PROTOSKIN_3D_PYTHON."
  exit 1
fi

# Everything the service needs is local (weights, rembg masks); refuse any
# silent Hugging Face download at runtime so the demo stays offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "Starting TripoSG 3D reconstruction service on port $PORT using $PYTHON"
exec "$PYTHON" -m uvicorn gateway_and_ui.backend.service_3d:app --host 127.0.0.1 --port "$PORT"
