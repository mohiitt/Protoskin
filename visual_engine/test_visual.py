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
    """Fast contract check: no model load, mirrors explain_result(use_llm=False)."""
    assert SAMPLE_WIREFRAME.is_file(), "Missing shared/sample_inputs/laptop_wireframe.png"
    result = generate_concept(
        str(SAMPLE_WIREFRAME),
        material_id="ocean_bound_polymer",
        finish="matte",
        texture="fine",
        color="graphite",
        use_model=False,
    )
    assert result.status == "success"
    assert result.placeholder is True
    assert result.control_mode == "canny"
    assert result.seed == 42
    assert Path(result.image_path).is_file()


def test_missing_image_raises():
    try:
        generate_concept("/tmp/protoskin-missing.png", material_id="standard_abs", use_model=False)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_canny_preprocessing_on_sample_input():
    """Canny preprocessing works without loading SDXL/ControlNet."""
    from visual_engine.preprocess import make_canny

    edges = make_canny(str(SAMPLE_WIREFRAME))
    assert edges.mode == "RGB"
    assert edges.size[0] % 8 == 0 and edges.size[1] % 8 == 0


def test_generate_concept_real_model():
    """Real SDXL + ControlNet-Canny generation. Slow; requires local model
    weights and a GPU. Not part of scripts/smoke_test.py's fast loop --
    run explicitly to validate the live pipeline before the demo.
    """
    result = generate_concept(
        str(SAMPLE_WIREFRAME),
        material_id="ocean_bound_polymer",
        finish="matte",
        texture="fine",
        color="graphite",
        seed=42,
        num_inference_steps=20,
    )
    assert result.status == "success"
    assert result.placeholder is False
    assert result.control_mode == "canny"
    assert result.seed == 42
    assert Path(result.image_path).is_file()
    print(f"real generation time: {result.generation_time_s}s")


if __name__ == "__main__":
    test_prompt_uses_material_config()
    test_generate_concept_placeholder()
    test_missing_image_raises()
    test_canny_preprocessing_on_sample_input()
    print("visual engine ok (fast contract checks)")
    print("run test_generate_concept_real_model() separately to validate real inference")
