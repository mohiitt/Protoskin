"""Paths and small lookups shared by every component."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIALS_PATH = ROOT / "shared" / "materials_config.json"
PROFILES_PATH = ROOT / "shared" / "product_profiles.json"
SAMPLE_INPUTS = ROOT / "shared" / "sample_inputs"
SAMPLE_WIREFRAME = SAMPLE_INPUTS / "laptop_wireframe.png"
DEMO_OUTPUTS = ROOT / "demo_outputs"
FRONTEND_DIR = ROOT / "gateway_and_ui" / "frontend"

FINISHES = ["matte", "semi-gloss", "gloss"]
TEXTURES = ["fine", "medium", "coarse"]
COLORS = ["graphite", "silver", "warm white"]
CONTROL_MODES = ["canny", "depth"]

DISCLAIMER = (
    "Concept-stage screening estimate. Not a manufacturing quote, "
    "BOM estimate, structural simulation, or thermal validation. "
    "3D preview is a best-effort reconstruction from the 2D concept image, "
    "not a verified 3D scan or CAD model. 3D preview powered by Stability AI "
    "(Stable Fast 3D), licensed under the Stability AI Community License."
)


def load_materials() -> dict:
    return json.loads(MATERIALS_PATH.read_text())


def load_profiles() -> dict:
    return json.loads(PROFILES_PATH.read_text())


def get_material(material_id: str) -> dict:
    materials = load_materials()
    if material_id not in materials:
        known = ", ".join(materials)
        raise ValueError(f"Unknown material '{material_id}'. Known: {known}")
    return materials[material_id]


def get_profile(profile_id: str) -> dict:
    profiles = load_profiles()
    if profile_id not in profiles:
        known = ", ".join(profiles)
        raise ValueError(f"Unknown product profile '{profile_id}'. Known: {known}")
    return profiles[profile_id]
