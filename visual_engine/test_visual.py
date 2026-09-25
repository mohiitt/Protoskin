"""Standalone checks for the visual-engine contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import SAMPLE_WIREFRAME
from visual_engine.pipeline import generate_concept
from visual_engine.prompts import build_prompt


def test_prompt_uses_material_config():
    prompt, negative = build_prompt(
        "ocean_bound_polymer",
        finish="matte",
        texture="fine",
        color="graphite",
        visual_prompt="soft studio lighting",
        product_type="13-inch laptop top shell",
    )
    assert "Ocean-Bound Recycled Polymer" in prompt
    assert "matte finish" in prompt
    assert "soft studio lighting" in prompt
    assert "extra screens" in negative


def test_generate_concept_placeholder():
    assert SAMPLE_WIREFRAME.is_file(), "Missing shared/sample_inputs/laptop_wireframe.png"
    result = generate_concept(
        str(SAMPLE_WIREFRAME),
        material_id="ocean_bound_polymer",
        finish="matte",
        texture="fine",
        color="graphite",
    )
    assert result.status == "success"
    assert result.placeholder is True
    assert result.control_mode == "canny"
    assert result.seed == 42
    assert Path(result.image_path).is_file()


def test_missing_image_raises():
    try:
        generate_concept("/tmp/protoskin-missing.png", material_id="standard_abs")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    test_prompt_uses_material_config()
    test_generate_concept_placeholder()
    test_missing_image_raises()
    print("visual engine ok")
