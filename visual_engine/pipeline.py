"""Visual engine entry point. Person A owns this file.

Loads SDXL + the Canny ControlNet from local disk (no hub calls at
demo time) and generates a material concept image guided by the input
sketch/wireframe's structure. See the implementation plan section 7
for the design rationale (Canny-first, fixed-seed determinism, control
strength as the key knob).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from shared.config import CONTROL_MODES, DEMO_OUTPUTS, ROOT, get_material
from shared.schemas import VisualResult
from visual_engine.preprocess import load_and_resize, make_canny
from visual_engine.prompts import build_prompt

logger = logging.getLogger("visual_engine.pipeline")

# Resolution order: explicit env var -> repo-relative default (matches
# scripts/download_models.sh and .env.example). Mirrors the pattern used
# by material_intelligence/llm_explainer.py for the LLM path.
_DEFAULT_SDXL_PATH = ROOT / "models" / "sdxl-base-1.0"
_DEFAULT_CONTROLNET_PATH = ROOT / "models" / "controlnet-canny-sdxl"

# Depth mode is optional/post-MVP (implementation plan, section 4.2/4.3).
# Fail clearly instead of silently mis-guiding generation with a Canny
# edge map when depth was requested.
_SUPPORTED_CONTROL_MODES = {"canny"}

# 20 steps: visually indistinguishable from 30 on the ZGX Nano test inputs
# (compared side by side), ~3.6s vs ~5.3s -- keeps image + 3D within ~20s.
_DEFAULT_NUM_INFERENCE_STEPS = 20
_DEFAULT_GUIDANCE_SCALE = 6.5
# How strongly the ControlNet conditioning is followed. This is the main
# knob for "preserve structure vs. allow material restyling" called out
# in the plan. Tuned empirically against the shared sample inputs.
_DEFAULT_CONTROLNET_CONDITIONING_SCALE = 0.65
_MAX_SIDE = 1024


def _sdxl_model_path() -> Path:
    configured = os.environ.get("PROTOSKIN_SDXL")
    return Path(configured) if configured else _DEFAULT_SDXL_PATH


def _controlnet_model_path() -> Path:
    configured = os.environ.get("PROTOSKIN_CONTROLNET")
    return Path(configured) if configured else _DEFAULT_CONTROLNET_PATH


class _SDXLCannyPipeline:
    """Lazily-loaded singleton wrapper around SDXL + Canny ControlNet.

    Loading the ~7B-parameter pair of models takes real time, so we pay
    that cost once per process (the "warm the model before the demo"
    guidance in the plan) and reuse the instance across requests.
    """

    _instance: "_SDXLCannyPipeline | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        sdxl_path = _sdxl_model_path()
        controlnet_path = _controlnet_model_path()
        if not sdxl_path.exists():
            raise FileNotFoundError(f"Local SDXL base model not found at {sdxl_path}")
        if not controlnet_path.exists():
            raise FileNotFoundError(f"Local ControlNet model not found at {controlnet_path}")

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(str(controlnet_path), torch_dtype=dtype)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            str(sdxl_path),
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        self.pipe.to(self.device)
        if self.device == "cuda" and hasattr(self.pipe, "enable_vae_slicing"):
            # Keeps memory headroom for the LLM component running alongside
            # it; the ZGX Nano's unified memory makes this optional rather
            # than required, but it costs little throughput here. Guarded
            # since some diffusers versions moved/removed this method.
            self.pipe.enable_vae_slicing()

    @classmethod
    def get(cls) -> "_SDXLCannyPipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(
        self,
        control_image,
        init_size: tuple[int, int],
        prompt: str,
        negative_prompt: str,
        seed: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
    ):
        generator = None
        if seed is not None:
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
        width, height = init_size
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )
        return result.images[0]


def warm_up() -> None:
    """Load the pipeline and run one tiny generation ahead of the live demo.

    Call this once at process startup. Loading alone isn't enough: the
    first real generation also pays one-time costs (CUDA kernel selection,
    the VAE's float32 upcast path) -- measured at over a minute for the
    first request on the ZGX Nano. A 2-step 512px run here absorbs them so
    the first request in front of an audience is as fast as the rest.
    """
    from PIL import Image

    pipeline = _SDXLCannyPipeline.get()
    blank_edges = Image.new("RGB", (512, 512))
    pipeline.run(
        control_image=blank_edges,
        init_size=blank_edges.size,
        prompt="product",
        negative_prompt="",
        seed=0,
        num_inference_steps=2,
        guidance_scale=_DEFAULT_GUIDANCE_SCALE,
        controlnet_conditioning_scale=_DEFAULT_CONTROLNET_CONDITIONING_SCALE,
    )


def generate_concept(
    image_path: str,
    material_id: str,
    finish: str = "matte",
    texture: str = "fine",
    color: str | None = None,
    visual_prompt: str | None = None,
    control_mode: str = "canny",
    seed: int | None = 42,
    product_type: str = "product chassis",
    num_inference_steps: int = _DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = _DEFAULT_GUIDANCE_SCALE,
    controlnet_conditioning_scale: float = _DEFAULT_CONTROLNET_CONDITIONING_SCALE,
    use_model: bool = True,
) -> VisualResult:
    """Generate a material concept image with SDXL guided by ControlNet-Canny.

    Preserves the recognizable silhouette/structure of ``image_path`` via
    a Canny edge map while changing material appearance, finish, texture,
    and lighting per the shared material config and appearance controls.
    Falls back to raising rather than silently degrading; the gateway is
    responsible for catching this and reporting VisualResult(status="error").

    ``use_model=False`` skips loading SDXL/ControlNet and copies the
    source image through instead, mirroring
    ``material_intelligence.llm_explainer.explain_result(use_llm=False)``.
    This keeps the contract (schema, prompt building, error handling)
    testable in the fast smoke-test loop without GPU inference.
    """
    started = time.perf_counter()
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if control_mode not in CONTROL_MODES:
        raise ValueError("control_mode must be 'canny' or 'depth'")
    if use_model and control_mode not in _SUPPORTED_CONTROL_MODES:
        raise NotImplementedError(
            f"control_mode '{control_mode}' is not implemented yet; only 'canny' is supported"
        )
    get_material(material_id)  # raises ValueError for an unknown material id

    prompt, negative_prompt = build_prompt(
        material_id,
        finish=finish,
        texture=texture,
        color=color,
        visual_prompt=visual_prompt,
        product_type=product_type,
    )

    if not use_model:
        DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
        filename = f"concept-{uuid.uuid4().hex[:8]}{source.suffix.lower() or '.png'}"
        destination = DEMO_OUTPUTS / filename
        shutil.copyfile(source, destination)
        elapsed = round(time.perf_counter() - started, 3)
        return VisualResult(
            image_path=f"demo_outputs/{filename}",
            image_url=f"/outputs/{filename}",
            generation_time_s=elapsed,
            seed=seed,
            control_mode=control_mode,
            placeholder=True,
            status="success",
        )

    control_image = make_canny(str(source), max_side=_MAX_SIDE)
    init_size = control_image.size  # already resized + rounded to a multiple of 8
    # Keep the resized source around for reference/debugging parity with
    # the control image, even though only control_image drives generation.
    load_and_resize(str(source), max_side=_MAX_SIDE)

    pipeline = _SDXLCannyPipeline.get()
    image = pipeline.run(
        control_image=control_image,
        init_size=init_size,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
    filename = f"concept-{uuid.uuid4().hex[:8]}.png"
    destination = DEMO_OUTPUTS / filename
    image.save(destination)

    elapsed = round(time.perf_counter() - started, 3)
    return VisualResult(
        image_path=f"demo_outputs/{filename}",
        image_url=f"/outputs/{filename}",
        generation_time_s=elapsed,
        seed=seed,
        control_mode=control_mode,
        placeholder=False,
        status="success",
    )
