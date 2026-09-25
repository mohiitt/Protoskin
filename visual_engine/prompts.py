"""Build the SDXL prompt from the shared material config. Person A owns this."""

from shared.config import get_material

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
