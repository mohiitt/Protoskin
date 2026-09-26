"""Sidecar 3D reconstruction service (TripoSG, MIT).

Runs as its own process under the `protoskin3d-os` conda environment (see
scripts/setup_3d_env.sh and scripts/run_3d_service.sh) -- separate from
the main gateway process because TripoSG needs diffusers/transformers
versions that differ from what the rest of the app uses in the main
`zgx` environment.

The main gateway (gateway_and_ui/backend/main.py) talks to this over
plain HTTP on PROTOSKIN_3D_SERVICE_URL (default http://127.0.0.1:8100),
never by importing this module directly.

Start with:
    $PROTOSKIN_3D_PYTHON -m uvicorn gateway_and_ui.backend.service_3d:app --port 8100

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
        logger.info("PROTOSKIN_SKIP_WARMUP=1 set; skipping TripoSG warm-up")
        return

    def _run():
        from visual_engine.reconstruct3d import warm_up

        try:
            logger.info("Warming up TripoSG...")
            warm_up()
            logger.info("TripoSG warm.")
        except Exception:
            logger.exception("TripoSG warm-up failed; will load on first request instead.")

    threading.Thread(target=_run, name="triposg-warmup", daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


class ReconstructRequest(BaseModel):
    image_path: str
    # Largest real-world dimension to scale the mesh to, from the product
    # profile's nominal_size_mm; omitted = unitless (1 m largest dimension).
    target_max_mm: float | None = None
    # Supplied by the gateway from the selected material/finish (see
    # visual_engine.prompts.pbr_params); omitted = neutral defaults.
    roughness: float | None = None
    metallic: float | None = None
    use_model: bool = True


@app.post("/reconstruct")
def reconstruct(request: ReconstructRequest):
    from visual_engine.reconstruct3d import Reconstruction3DError, reconstruct_3d

    try:
        return reconstruct_3d(
            request.image_path,
            target_max_mm=request.target_max_mm,
            roughness=request.roughness,
            metallic=request.metallic,
            use_model=request.use_model,
        )
    except Reconstruction3DError as exc:
        # A structured error the gateway can fold into
        # Reconstruction3DResult(status="error"), not a 500 -- this is an
        # expected, handled failure mode (missing model, bad image, OOM),
        # not a bug in the service itself. Logged so it's diagnosable here.
        logger.warning("Reconstruction failed for %s: %s", request.image_path, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
