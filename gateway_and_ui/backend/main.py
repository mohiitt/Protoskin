"""ProtoSkin gateway: materials, then image, then 3D; LLM wording via /api/explain."""

import asyncio
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
    get_material,
    load_materials,
    load_profiles,
)
from shared.schemas import (
    ConceptResponse,
    ExplanationResult,
    MaterialComparison,
    Reconstruction3DResult,
    VisualResult,
)
from visual_engine.prompts import pbr_params

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
# scripts/run_3d_service.sh and third_party/README.md for why. This
# gateway only ever talks to it over plain HTTP, never imports it.
_3D_SERVICE_URL = os.environ.get("PROTOSKIN_3D_SERVICE_URL", "http://127.0.0.1:8100")
_3D_TIMEOUT_S = float(os.environ.get("PROTOSKIN_3D_TIMEOUT_S", "180"))

# One request at a time per model (see the comment in concept()).
_IMAGE_LOCK = asyncio.Lock()
_LLM_LOCK = asyncio.Lock()


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
        # Depth mode ("CAD / Depth-style Render") is part of the shared
        # contract but not built yet (visual_engine raises for it), so it's
        # not offered. Canny handles CAD renders too.
        "control_modes": [
            {"id": "canny", "label": "Sketch / Wireframe / CAD render"},
        ],
        "disclaimer": DISCLAIMER,
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/concept", response_model=ConceptResponse)
async def concept(
    image: UploadFile = File(...),
    # Either a preset profile id, or a typed product (product_name) with
    # optional nominal largest dimension and shell volume.
    product_profile_id: str = Form(""),
    product_name: str = Form(""),
    product_max_mm: float | None = Form(None),
    product_volume_cm3: float | None = Form(None),
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
    if control_mode == "depth":
        raise HTTPException(
            status_code=400,
            detail="Depth mode isn't available yet; use 'canny' (Sketch / Wireframe / CAD render).",
        )
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

    product_type, target_max_mm, materials = _resolve_product(
        product_profile_id,
        product_name,
        product_max_mm,
        product_volume_cm3,
        baseline_material_id,
        candidate_material_id,
    )

    # Model calls run in worker threads so the event loop (health checks,
    # other users) stays responsive. Each model gets a lock: the diffusers /
    # transformers pipelines aren't safe to drive from two threads at once.
    try:
        async with _IMAGE_LOCK:
            visual = await asyncio.to_thread(
                generate_concept,
                str(saved),
                material_id=candidate_material_id,
                finish=finish,
                texture=texture,
                color=color,
                visual_prompt=visual_prompt,
                control_mode=control_mode,
                seed=seed,
                product_type=product_type,
            )
    except Exception as exc:
        visual = VisualResult(status="error", error=str(exc), control_mode=control_mode, seed=seed)

    # 3D next, with the GPU to itself. Measured on the ZGX Nano: image ~3.6s
    # + 3D ~12.5s when each runs alone, but running the Qwen explanation
    # alongside 3D slowed it to ~26s -- all three models share the same
    # ~273 GB/s unified memory bandwidth. So this request returns the
    # deterministic template explanation (same numbers, since the LLM only
    # rephrases them), and the page fetches the LLM wording afterwards from
    # /api/explain.
    reconstruction = await _reconstruct(visual, target_max_mm, candidate_material_id, finish)
    explanation = (
        template_explanation(materials)
        if materials is not None
        else await _explain(None)  # "enter a shell volume" note
    )

    return ConceptResponse(
        visual=visual,
        materials=materials,
        explanation=explanation,
        reconstruction=reconstruction,
        disclaimer=DISCLAIMER,
    )


def _resolve_product(
    product_profile_id,
    product_name,
    product_max_mm,
    product_volume_cm3,
    baseline_material_id,
    candidate_material_id,
):
    """Return (product_type for the image prompt, nominal largest dimension
    in mm or None, MaterialComparison or None).

    A preset profile supplies all three. A typed product uses its own name
    in the prompt and only the size/volume the user entered: without a
    volume there are no mass/cost numbers (never guessed).
    """
    profiles = load_profiles()
    try:
        if product_profile_id in profiles:
            profile = profiles[product_profile_id]
            nominal = profile.get("nominal_size_mm")
            materials = compare_materials(product_profile_id, baseline_material_id, candidate_material_id)
            return profile["product_type"], (max(nominal) if nominal else None), materials

        # Collapse whitespace/newlines: this text goes into the image prompt.
        name = " ".join(product_name.split())[:80]
        if not name:
            raise HTTPException(status_code=400, detail="Choose a product profile or type a product name")
        for label, value in (("product_max_mm", product_max_mm), ("product_volume_cm3", product_volume_cm3)):
            if value is not None and value <= 0:
                raise HTTPException(status_code=400, detail=f"{label} must be positive")
        get_material(baseline_material_id)  # validate ids even without metrics
        get_material(candidate_material_id)
        materials = None
        if product_volume_cm3 is not None:
            materials = compare_materials(
                "custom",
                baseline_material_id,
                candidate_material_id,
                custom_product_type=name,
                custom_volume_cm3=product_volume_cm3,
            )
        return name, product_max_mm, materials
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/explain", response_model=ExplanationResult)
async def explain(materials: MaterialComparison):
    """Local-LLM wording of an already-computed MaterialComparison.

    Called by the page after /api/concept returns, so the ~LLM time never
    delays the image or 3D model. The numbers come only from the
    deterministic calculator (explain_result verifies the LLM repeats them
    exactly, else falls back to the template).
    """
    return await _explain(materials)


async def _explain(materials):
    if materials is None:
        return ExplanationResult(
            summary=(
                "Enter an approximate shell volume (cm³) for this product to estimate "
                "mass and raw-material cost."
            ),
            source="template",
        )
    try:
        async with _LLM_LOCK:
            return await asyncio.to_thread(explain_result, materials)
    except Exception as exc:
        explanation = template_explanation(materials)
        explanation.error = str(exc)
        return explanation


async def _reconstruct(visual, target_max_mm, candidate_material_id, finish):
    """Best-effort 3D preview, reconstructed from the 2D concept image via
    the TripoSG sidecar. Additive and non-blocking: any failure here (the
    service being down, a timeout, a bad image) degrades to
    Reconstruction3DResult(status="error") without affecting the 2D image
    or material report -- mirrors how a failed `visual` generation still
    returns a usable response."""
    if visual.status == "success" and visual.image_path:
        try:
            # visual.image_path is repo-root-relative (e.g.
            # "demo_outputs/concept-xxx.png"); DEMO_OUTPUTS is that same
            # directory as an absolute path, so re-derive the absolute path
            # from the filename rather than assuming the process CWD.
            image_path = str(DEMO_OUTPUTS / Path(visual.image_path).name)
            roughness, metallic = pbr_params(candidate_material_id, finish)
            # Async so the ~15s reconstruction doesn't block the event loop
            # (and with it /api/health and every other request).
            async with httpx.AsyncClient(timeout=_3D_TIMEOUT_S) as client:
                response = await client.post(
                    f"{_3D_SERVICE_URL}/reconstruct",
                    json={
                        "image_path": image_path,
                        "target_max_mm": target_max_mm,
                        "roughness": roughness,
                        "metallic": metallic,
                    },
                )
            if response.status_code == 422:
                # The service's own reason (missing weights, no object found...).
                raise RuntimeError(response.json().get("detail", response.text))
            response.raise_for_status()
            reconstruction = Reconstruction3DResult(**response.json())
        except Exception as exc:
            reconstruction = Reconstruction3DResult(status="error", error=str(exc))
    else:
        reconstruction = Reconstruction3DResult(
            status="error", error="Skipped: no 2D concept image to reconstruct from."
        )
    return reconstruction
