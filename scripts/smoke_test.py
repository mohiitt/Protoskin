"""One command that checks schemas, calculators, the visual placeholder, and the API."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from gateway_and_ui.backend.main import app
from material_intelligence.test_materials import (
    test_explanation_repeats_calculator_numbers,
    test_mass_and_cost_are_fixed,
    test_same_inputs_match,
    test_unknown_material_raises,
)
from shared.config import SAMPLE_WIREFRAME
from visual_engine.test_visual import (
    test_generate_concept_placeholder,
    test_missing_image_raises,
    test_prompt_uses_material_config,
)


def test_api_flow():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["cloud_ai_calls"] == 0

    options = client.get("/api/options")
    assert options.status_code == 200
    assert any(item["id"] == "ocean_bound_polymer" for item in options.json()["materials"])

    with SAMPLE_WIREFRAME.open("rb") as handle:
        response = client.post(
            "/api/concept",
            files={"image": ("laptop_wireframe.png", handle, "image/png")},
            data={
                "product_profile_id": "laptop_13",
                "baseline_material_id": "recycled_aluminum",
                "candidate_material_id": "ocean_bound_polymer",
                "finish": "matte",
                "texture": "fine",
                "color": "graphite",
                "visual_prompt": "soft studio lighting",
                "control_mode": "canny",
                "seed": "42",
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["materials"]["mass_delta_percent"] == -57.4
    assert body["visual"]["status"] == "success"
    assert body["visual"]["placeholder"] is True
    assert "+57.4%" not in body["explanation"]["summary"]
    assert "-57.4%" in body["explanation"]["summary"]
    assert body["disclaimer"]
    image = client.get(body["visual"]["image_url"])
    assert image.status_code == 200
    page = client.get("/")
    assert page.status_code == 200
    assert "Generate Concept" in page.text


def main():
    test_mass_and_cost_are_fixed()
    test_same_inputs_match()
    test_unknown_material_raises()
    test_explanation_repeats_calculator_numbers()
    test_prompt_uses_material_config()
    test_generate_concept_placeholder()
    test_missing_image_raises()
    test_api_flow()
    print("smoke test ok")


if __name__ == "__main__":
    main()
