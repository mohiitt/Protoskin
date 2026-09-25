"""3D reconstruction from a generated concept image, using Stable Fast 3D.

Runs only inside the `protoskin3d` conda environment (see
scripts/setup_sf3d_env.sh) -- SF3D's pinned dependencies (transformers,
trimesh) conflict with the main app's environment, which the local LLM
explainer needs at different versions. This module is imported by
gateway_and_ui/backend/service_3d.py, a small standalone FastAPI process
run under that separate environment, never by the main gateway process
directly.

Additive and best-effort by design: 3D reconstruction is a bonus preview
on top of the 2D concept image, which remains the primary, reliable
output (see the implementation plan's Contract A). A failure here must
never be treated as a pipeline failure by its caller.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger("visual_engine.reconstruct3d")

# Resolution order matches the other model paths in this project
# (visual_engine/pipeline.py, material_intelligence/llm_explainer.py):
# explicit env var -> repo-relative default.
ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SF3D_PATH = ROOT / "models" / "stable-fast-3d"
_DEFAULT_SF3D_SRC = ROOT / "third_party" / "stable-fast-3d"

DEMO_OUTPUTS = ROOT / "demo_outputs"


def _sf3d_model_path() -> Path:
    configured = os.environ.get("PROTOSKIN_SF3D")
    return Path(configured) if configured else _DEFAULT_SF3D_PATH


def _sf3d_src_path() -> Path:
    configured = os.environ.get("PROTOSKIN_SF3D_SRC")
    return Path(configured) if configured else _DEFAULT_SF3D_SRC


class Reconstruction3DError(Exception):
    """Raised for any failure in the 3D reconstruction path.

    The caller (service_3d.py's endpoint, or the gateway's HTTP client to
    it) is responsible for catching this and reporting
    Reconstruction3DResult(status="error") rather than failing the whole
    /api/concept request -- see shared/schemas.py's docstring on that
    model for the full rationale.
    """


class _SF3DPipeline:
    """Lazily-loaded singleton wrapper around the Stable Fast 3D model.

    Mirrors the pattern in visual_engine/pipeline.py's
    _SDXLCannyPipeline and material_intelligence/llm_explainer.py's
    _QwenExplainer: pay the model-load cost once per process, reuse the
    instance across requests.
    """

    _instance: "_SF3DPipeline | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        src = _sf3d_src_path()
        if not src.exists():
            raise Reconstruction3DError(
                f"Vendored Stable Fast 3D source not found at {src}. "
                "See third_party/README.md."
            )
        model_path = _sf3d_model_path()
        if not model_path.exists():
            raise Reconstruction3DError(
                f"Local Stable Fast 3D weights not found at {model_path}. "
                "Download with scripts/download_models.sh (requires "
                "requesting access on the model's Hugging Face page first; "
                "see third_party/README.md)."
            )

        import sys

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        import rembg
        import torch
        from sf3d.system import SF3D
        from sf3d.utils import get_device, remove_background, resize_foreground

        self._torch = torch
        self._remove_background = remove_background
        self._resize_foreground = resize_foreground

        self.device = get_device()
        self.model = SF3D.from_pretrained(
            str(model_path),
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.model.to(self.device)
        self.model.eval()
        self.rembg_session = rembg.new_session()

    @classmethod
    def get(cls) -> "_SF3DPipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(
        self,
        image_path: str,
        foreground_ratio: float,
        remesh_option: str,
        texture_resolution: int,
    ):
        from PIL import Image

        image = self._remove_background(
            Image.open(image_path).convert("RGBA"), self.rembg_session
        )
        image = self._resize_foreground(image, foreground_ratio)

        # Matches run.py's own inference call exactly (the project's
        # reference CLI usage), including the autocast/nullcontext choice.
        if self.device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        with self._torch.no_grad():
            with (
                self._torch.autocast(device_type=self.device, dtype=self._torch.bfloat16)
                if "cuda" in self.device
                else nullcontext()
            ):
                # A length-1 list input makes run_image return a single
                # trimesh.Trimesh directly (not a list) -- see
                # sf3d/system.py's run_image: `if batch_size == 1: return
                # meshes[0], global_dict`.
                mesh, _glob_dict = self.model.run_image(
                    [image],
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                )
        return mesh


def warm_up() -> None:
    """Force-load the SF3D pipeline ahead of the live demo.

    Mirrors visual_engine.pipeline.warm_up() and
    material_intelligence.llm_explainer.warm_up(). Best-effort: if the
    model can't load, reconstruction requests still fail gracefully at
    call time and the gateway falls back to 2D-only output.
    """
    _SF3DPipeline.get()


def reconstruct_3d(
    image_path: str,
    remesh_option: str = "quad",
    foreground_ratio: float = 0.85,
    texture_resolution: int = 1024,
    use_model: bool = True,
) -> dict:
    """Reconstruct an orbit-able 3D mesh from a generated concept image.

    Returns a dict matching shared.schemas.Reconstruction3DResult's
    fields (this module intentionally has no dependency on the main
    app's `shared` package, since it runs in a separate Python
    environment with a different, incompatible transformers/trimesh
    pin -- see the module docstring). The sidecar service
    (service_3d.py) constructs the actual Pydantic model from this dict
    on the gateway side, where `shared` is importable.

    Raises Reconstruction3DError on any failure; callers must catch this
    (see the docstring on that exception).

    ``use_model=False`` skips loading SF3D and returns a placeholder
    result without touching the input image, mirroring
    ``visual_engine.pipeline.generate_concept(use_model=False)`` and
    ``material_intelligence.llm_explainer.explain_result(use_llm=False)``.
    Keeps the contract (schema, validation, error handling) testable in
    the fast smoke-test loop without GPU inference or the SF3D weights.
    """
    started = time.perf_counter()
    source = Path(image_path)
    if not source.is_file():
        raise Reconstruction3DError(f"Input image not found: {image_path}")
    if remesh_option not in {"none", "triangle", "quad"}:
        raise Reconstruction3DError(
            f"remesh_option must be 'none', 'triangle', or 'quad', got {remesh_option!r}"
        )

    if not use_model:
        elapsed = round(time.perf_counter() - started, 3)
        return {
            "glb_path": None,
            "glb_url": None,
            "reconstruction_time_s": elapsed,
            "status": "success",
        }

    try:
        pipeline = _SF3DPipeline.get()
        mesh = pipeline.run(
            str(source),
            foreground_ratio=foreground_ratio,
            remesh_option=remesh_option,
            texture_resolution=texture_resolution,
        )
    except Reconstruction3DError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error to the caller
        raise Reconstruction3DError(f"3D reconstruction failed: {exc}") from exc

    DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
    filename = f"mesh-{uuid.uuid4().hex[:8]}.glb"
    destination = DEMO_OUTPUTS / filename
    mesh.export(str(destination), include_normals=True)

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "glb_path": f"demo_outputs/{filename}",
        "glb_url": f"/outputs/{filename}",
        "reconstruction_time_s": elapsed,
        "status": "success",
    }
