"""Visual engine entry point. Person A replaces the body, not the signature."""

import shutil
import time
import uuid
from pathlib import Path

from shared.config import CONTROL_MODES, DEMO_OUTPUTS, get_material
from shared.schemas import VisualResult
from visual_engine.preprocess import make_canny
from visual_engine.prompts import build_prompt


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
) -> VisualResult:
    """Generate a material concept image.

    Placeholder copies the input and builds the prompt so the contract
    can be tested with no model weights. Replace this body with SDXL +
    ControlNet. Keep the arguments and the VisualResult fields.
    """
    started = time.perf_counter()
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if control_mode not in CONTROL_MODES:
        raise ValueError("control_mode must be 'canny' or 'depth'")
    get_material(material_id)
    build_prompt(
        material_id,
        finish=finish,
        texture=texture,
        color=color,
        visual_prompt=visual_prompt,
        product_type=product_type,
    )
    if control_mode == "canny":
        make_canny(str(source))

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
