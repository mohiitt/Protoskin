#!/usr/bin/env bash
# Launch the sidecar 3D reconstruction service under the protoskin3d
# environment (see scripts/setup_sf3d_env.sh). The main gateway process
# (gateway_and_ui/backend/main.py) talks to this over HTTP; it never
# imports visual_engine/reconstruct3d.py or service_3d.py directly, since
# they need SF3D's incompatible pinned dependencies.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

PYTHON="${PROTOSKIN_SF3D_PYTHON:-$HOME/miniforge3/envs/protoskin3d/bin/python}"
PORT="${PROTOSKIN_SF3D_SERVICE_PORT:-8100}"

if [ ! -x "$PYTHON" ]; then
  echo "error: $PYTHON not found or not executable."
  echo "Run scripts/setup_sf3d_env.sh first, or set PROTOSKIN_SF3D_PYTHON."
  exit 1
fi

echo "Starting SF3D reconstruction service on port $PORT using $PYTHON"
exec "$PYTHON" -m uvicorn gateway_and_ui.backend.service_3d:app --host 127.0.0.1 --port "$PORT"
