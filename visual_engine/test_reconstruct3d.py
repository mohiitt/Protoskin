"""Standalone checks for the 3D reconstruction contract.

Run these under the `protoskin3d` environment (see
scripts/setup_sf3d_env.sh), not the main `zgx` environment -- this module
imports visual_engine.reconstruct3d, which needs SF3D's pinned
dependencies. Not part of scripts/smoke_test.py's fast loop for the same
reason (that script runs under the main environment); run this file
directly instead:

    $PROTOSKIN_SF3D_PYTHON visual_engine/test_reconstruct3d.py

No dependency on shared.config, deliberately: reconstruct3d.py itself
avoids importing `shared` since it runs in a separate Python environment
with incompatible pinned versions (see that module's docstring); this
test mirrors that same self-contained approach.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_engine.reconstruct3d import Reconstruction3DError, reconstruct_3d

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WIREFRAME = ROOT / "shared" / "sample_inputs" / "laptop_wireframe.png"


def test_reconstruct_3d_placeholder():
    """Fast contract check: no model load, mirrors generate_concept(use_model=False)."""
    assert SAMPLE_WIREFRAME.is_file(), "Missing shared/sample_inputs/laptop_wireframe.png"
    result = reconstruct_3d(str(SAMPLE_WIREFRAME), use_model=False)
    assert result["status"] == "success"
    assert result["glb_path"] is None
    assert result["glb_url"] is None


def test_missing_image_raises():
    try:
        reconstruct_3d("/tmp/protoskin-missing.png", use_model=False)
    except Reconstruction3DError:
        return
    raise AssertionError("expected Reconstruction3DError")


def test_invalid_remesh_option_raises():
    try:
        reconstruct_3d(str(SAMPLE_WIREFRAME), remesh_option="bogus", use_model=False)
    except Reconstruction3DError:
        return
    raise AssertionError("expected Reconstruction3DError")


def test_reconstruct_3d_real_model():
    """Real Stable Fast 3D reconstruction. Slow; requires local model
    weights, the vendored SF3D source, and a GPU. Run explicitly to
    validate the live pipeline before the demo -- use a real generated
    concept image (not the raw sketch) for a meaningful result, since
    SF3D expects a photorealistic single-object image, not a line drawing.
    """
    import glob

    candidates = sorted(glob.glob(str(ROOT / "demo_outputs" / "concept-*.png")))
    if not candidates:
        raise AssertionError(
            "No generated concept image found in demo_outputs/. "
            "Run the visual engine first (see visual_engine/test_visual.py's "
            "test_generate_concept_real_model), then re-run this test."
        )
    result = reconstruct_3d(candidates[-1], remesh_option="quad")
    assert result["status"] == "success"
    assert Path(result["glb_path"]).is_file() or (ROOT / result["glb_path"]).is_file()
    print(f"real reconstruction time: {result['reconstruction_time_s']}s")


if __name__ == "__main__":
    test_reconstruct_3d_placeholder()
    test_missing_image_raises()
    test_invalid_remesh_option_raises()
    print("3D reconstruction contract ok (fast contract checks)")
    print("run test_reconstruct_3d_real_model() separately to validate real inference")
