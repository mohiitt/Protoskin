"""ProtoSkin gateway. Sequential: materials, then image, then explanation."""

import logging
import os
import threading
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gateway_and_ui.backend.mock_material import compare_materials, explain_result, template_explanation
from gateway_and_ui.backend.mock_visual import generate_concept
from shared.config import (
    COLORS,
    CONTROL_MODES,
    DEMO_OUTPUTS,
    DISCLAIMER,
    FINISHES,
    FRONTEND_DIR,
    TEXTURES,
    load_materials,
    load_profiles,
)
from shared.schemas import ConceptResponse, Reconstruction3DResult, VisualResult

logger = logging.getLogger("gateway_and_ui.backend.main")

app = FastAPI(title="ProtoSkin", version="0.1.0")

DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=DEMO_OUTPUTS), name="outputs")
# Serves self-hosted static assets (e.g. model-viewer.min.js) referenced by
# index.html -- keeps the "zero external calls" claim true for the 3D
# viewer too, not just the model inference.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Set PROTOSKIN_SKIP_WARMUP=1 to skip loading real models at startup, e.g.
# for fast local iteration on the gateway/UI without a GPU or model weights.
_SKIP_WARMUP = os.environ.get("PROTOSKIN_SKIP_WARMUP") == "1"

# The 3D reconstruction sidecar (gateway_and_ui/backend/service_3d.py) runs
# as its own process under a separate Python environment -- see
# scripts/run_sf3d_service.sh and third_party/README.md for why. This
# gateway only ever talks to it over plain HTTP, never imports it.
_SF3D_SERVICE_URL = os.environ.get("PROTOSKIN_SF3D_SERVICE_URL", "http://127.0.0.1:8100")
_SF3D_TIMEOUT_S = float(os.environ.get("PROTOSKIN_SF3D_TIMEOUT_S", "60"))


@app.on_event("startup")
def _warm_up_models() -> None:
    """Pre-load SDXL/ControlNet and the local LLM so the first live demo
    request doesn't pay model-load latency in front of an audience (plan
    section 7, "warm the model before the demo"). Runs in a background
    thread so the server starts accepting requests immediately; a request
    that arrives before warm-up finishes just loads on demand instead.
    """
    if _SKIP_WARMUP:
        logger.info("PROTOSKIN_SKIP_WARMUP=1 set; skipping model warm-up")
        return

    def _run():
        from material_intelligence.llm_explainer import warm_up as warm_up_llm
        from visual_engine.pipeline import warm_up as warm_up_visual

        try:
            logger.info("Warming up visual engine (SDXL + ControlNet-Canny)...")
            warm_up_visual()
            logger.info("Visual engine warm.")
        except Exception:
            logger.exception("Visual engine warm-up failed; will load on first request instead.")
        try:
            logger.info("Warming up local LLM (Qwen3-8B)...")
            warm_up_llm()
            logger.info("Local LLM warm.")
        except Exception:
            logger.exception("LLM warm-up failed; explanations will use the template fallback.")

    threading.Thread(target=_run, name="protoskin-warmup", daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "inference": "local", "cloud_ai_calls": 0}


@app.get("/api/options")
def options():
    materials = load_materials()
    profiles = load_profiles()
    return {
        "materials": [{"id": key, "label": value["label"]} for key, value in materials.items()],
        "profiles": [
            {
                "id": key,
                "label": value["label"],
                "product_type": value["product_type"],
                "baseline_material": value["baseline_material"],
            }
            for key, value in profiles.items()
        ],
        "finishes": FINISHES,
        "textures": TEXTURES,
        "colors": COLORS,
        "control_modes": [
            {"id": "canny", "label": "Sketch / Wireframe"},
            {"id": "depth", "label": "CAD / Depth-style Render"},
        ],
        "disclaimer": DISCLAIMER,
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/concept", response_model=ConceptResponse)
async def concept(
    image: UploadFile = File(...),
    product_profile_id: str = Form(...),
    baseline_material_id: str = Form(...),
    candidate_material_id: str = Form(...),
    finish: str = Form("matte"),
    texture: str = Form("fine"),
    color: str = Form("graphite"),
    visual_prompt: str = Form(""),
    control_mode: str = Form("canny"),
    seed: int = Form(42),
):
    if control_mode not in CONTROL_MODES:
        raise HTTPException(status_code=400, detail="control_mode must be 'canny' or 'depth'")
    if finish not in FINISHES or texture not in TEXTURES or color not in COLORS:
        raise HTTPException(status_code=400, detail="Unknown finish, texture, or color")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Image file is empty")

    suffix = Path(image.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        suffix = ".png"
    upload_dir = DEMO_OUTPUTS / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    saved.write_bytes(contents)

    try:
        materials = compare_materials(product_profile_id, baseline_material_id, candidate_material_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        visual = generate_concept(
            str(saved),
            material_id=candidate_material_id,
            finish=finish,
            texture=texture,
            color=color,
            visual_prompt=visual_prompt,
            control_mode=control_mode,
            seed=seed,
            product_type=materials.product_type,
        )
    except Exception as exc:
        visual = VisualResult(status="error", error=str(exc), control_mode=control_mode, seed=seed)

    try:
        explanation = explain_result(materials)
    except Exception as exc:
        explanation = template_explanation(materials)
        explanation.error = str(exc)

    # Best-effort 3D preview, reconstructed from the 2D concept image via
    # the SF3D sidecar. Additive and non-blocking: any failure here (the
    # service being down, a timeout, a bad image) degrades to
    # Reconstruction3DResult(status="error") without affecting the 2D
    # image or material report already computed above -- mirrors how a
    # failed `visual` generation above still returns a usable response.
    if visual.status == "success" and visual.image_path:
        try:
            # visual.image_path is repo-root-relative (e.g.
            # "demo_outputs/concept-xxx.png"); DEMO_OUTPUTS is that same
            # directory as an absolute path, so re-derive the absolute path
            # from the filename rather than assuming the process CWD.
            image_path = str(DEMO_OUTPUTS / Path(visual.image_path).name)
            sf3d_response = httpx.post(
                f"{_SF3D_SERVICE_URL}/reconstruct",
                json={"image_path": image_path, "remesh_option": "quad"},
                timeout=_SF3D_TIMEOUT_S,
            )
            sf3d_response.raise_for_status()
            reconstruction = Reconstruction3DResult(**sf3d_response.json())
        except Exception as exc:
            reconstruction = Reconstruction3DResult(status="error", error=str(exc))
    else:
        reconstruction = Reconstruction3DResult(
            status="error", error="Skipped: no 2D concept image to reconstruct from."
        )

    return ConceptResponse(
        visual=visual,
        materials=materials,
        explanation=explanation,
        reconstruction=reconstruction,
        disclaimer=DISCLAIMER,
    )
