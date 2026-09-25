"""Sidecar 3D reconstruction service.

Runs as its own process under the `protoskin3d` conda environment (see
scripts/setup_sf3d_env.sh and scripts/run_sf3d_service.sh) -- separate
from the main gateway process because Stable Fast 3D's pinned
dependencies (transformers==4.42.3, trimesh==4.4.1) conflict with what
the local LLM explainer needs in the main `zgx` environment.

The main gateway (gateway_and_ui/backend/main.py) talks to this over
plain HTTP on PROTOSKIN_SF3D_SERVICE_URL (default
http://127.0.0.1:8100), never by importing this module directly.

Start with:
    PROTOSKIN_SF3D_PYTHON -m uvicorn gateway_and_ui.backend.service_3d:app --port 8100

or via the wrapper script, which also picks the right environment.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("gateway_and_ui.backend.service_3d")

ROOT = Path(__file__).resolve().parents[2]
DEMO_OUTPUTS = ROOT / "demo_outputs"

app = FastAPI(title="ProtoSkin 3D Reconstruction Service", version="0.1.0")

DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=DEMO_OUTPUTS), name="outputs")

_SKIP_WARMUP = os.environ.get("PROTOSKIN_SKIP_WARMUP") == "1"


@app.on_event("startup")
def _warm_up() -> None:
    if _SKIP_WARMUP:
        logger.info("PROTOSKIN_SKIP_WARMUP=1 set; skipping SF3D warm-up")
        return

    def _run():
        from visual_engine.reconstruct3d import warm_up

        try:
            logger.info("Warming up Stable Fast 3D...")
            warm_up()
            logger.info("Stable Fast 3D warm.")
        except Exception:
            logger.exception("SF3D warm-up failed; will load on first request instead.")

    threading.Thread(target=_run, name="sf3d-warmup", daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


class ReconstructRequest(BaseModel):
    image_path: str
    remesh_option: str = "quad"
    foreground_ratio: float = 0.85
    texture_resolution: int = 1024
    use_model: bool = True


@app.post("/reconstruct")
def reconstruct(request: ReconstructRequest):
    from visual_engine.reconstruct3d import Reconstruction3DError, reconstruct_3d

    try:
        return reconstruct_3d(
            request.image_path,
            remesh_option=request.remesh_option,
            foreground_ratio=request.foreground_ratio,
            texture_resolution=request.texture_resolution,
            use_model=request.use_model,
        )
    except Reconstruction3DError as exc:
        # A structured error the gateway can fold into
        # Reconstruction3DResult(status="error"), not a 500 -- this is an
        # expected, handled failure mode (missing model, bad image, OOM),
        # not a bug in the service itself.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
