"""Build the SDXL prompt from the shared material config. Person A owns this."""

from shared.config import get_material

# Physically-based roughness per selected finish, for the 3D preview's
# material (0 = mirror, 1 = fully diffuse).
FINISH_ROUGHNESS = {
    "gloss": 0.15,
    "semi-gloss": 0.35,
    "satin": 0.5,
    "matte": 0.75,
}

DEFAULT_NEGATIVE = (
    "extra ports, extra hinges, extra screens, extra keyboards, "
    "warped chassis, text, logo, watermark, additional objects, "
    "changed product geometry"
)


def build_prompt(
    material_id: str,
    finish: str,
    texture: str,
    color: str | None = None,
    visual_prompt: str | None = None,
    product_type: str = "product chassis",
) -> tuple[str, str]:
    material = get_material(material_id)
    visual = material["visual"]
    parts = [
        product_type,
        material["label"],
        visual.get("appearance", ""),
        visual.get("prompt", ""),
        f"{finish} finish",
        f"{texture} texture",
        f"{visual.get('reflectivity', 'controlled')} reflectivity",
    ]
    if color:
        parts.append(f"{color} color")
    if visual_prompt and visual_prompt.strip():
        parts.append(visual_prompt.strip())
    parts.append(
        "clean industrial design, photorealistic studio product photography, "
        "soft controlled lighting, neutral background, single product"
    )
    prompt = ", ".join(part for part in parts if part)
    negative = visual.get("negative_prompt") or DEFAULT_NEGATIVE
    return prompt, negative


def pbr_params(material_id: str, finish: str) -> tuple[float, float]:
    """Return (roughness, metallic) for the 3D preview's PBR material.

    The 3D model reconstructs shape, not material, and guessing a
    material from a photo often misreads it (e.g. a metal shell as matte
    plastic). The material and finish are known here, so derive them
    instead: roughness from the selected finish, metallic from the shared
    material config.
    """
    visual = get_material(material_id)["visual"]
    roughness = FINISH_ROUGHNESS.get(finish, FINISH_ROUGHNESS.get(visual.get("finish"), 0.5))
    metallic = float(visual.get("metallic", 0.0))
    return roughness, metallic
