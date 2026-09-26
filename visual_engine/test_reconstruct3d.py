"""Standalone checks for the 3D reconstruction contract.

Run these under the `protoskin3d-os` environment (see
scripts/setup_3d_env.sh), not the main `zgx` environment -- this module
imports visual_engine.reconstruct3d, whose helpers need trimesh, rembg and
scikit-image from that environment. Not part of scripts/smoke_test.py's
fast loop for the same reason; run this file directly instead:

    $PROTOSKIN_3D_PYTHON visual_engine/test_reconstruct3d.py

The fast checks need no model weights or GPU. No dependency on
shared.config, deliberately: reconstruct3d.py itself avoids importing
`shared` since it runs in a separate Python environment.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_engine.reconstruct3d import (
    Reconstruction3DError,
    _colorize,
    _fit_camera,
    _lay_flat_if_panel,
    _prepare_image,
    _project,
    _rembg_model_name,
    _scale_to_real_size,
    _set_glb_material,
    _solidify_alpha,
    reconstruct_3d,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WIREFRAME = ROOT / "shared" / "sample_inputs" / "laptop_wireframe.png"


def _plane(z: float, facing: int, n: int = 21):
    """An n x n vertex grid spanning x, y in [-1, 1] at depth z.

    facing=+1 winds triangles so the normal points +Z, -1 for -Z.
    """
    import numpy as np
    import trimesh

    g = np.linspace(-1, 1, n)
    xs, ys = np.meshgrid(g, g)
    verts = np.stack([xs.ravel(), ys.ravel(), np.full(n * n, z)], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
            faces += [[a, b, c], [a, c, d]] if facing > 0 else [[a, c, b], [a, d, c]]
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def test_reconstruct_3d_placeholder():
    """Fast contract check: no model load, mirrors generate_concept(use_model=False)."""
    assert SAMPLE_WIREFRAME.is_file(), "Missing shared/sample_inputs/laptop_wireframe.png"
    result = reconstruct_3d(str(SAMPLE_WIREFRAME), use_model=False)
    assert result["status"] == "success"
    assert result["glb_path"] is None and result["glb_url"] is None
    assert result["dimensions_mm"] is None and result["view_orbit_deg"] is None


def test_missing_image_raises():
    try:
        reconstruct_3d("/tmp/protoskin-missing.png", use_model=False)
    except Reconstruction3DError:
        return
    raise AssertionError("expected Reconstruction3DError")


def test_invalid_inputs_raise():
    for kwargs in ({"roughness": 1.5}, {"metallic": -0.1}, {"target_max_mm": 0}):
        try:
            reconstruct_3d(str(SAMPLE_WIREFRAME), use_model=False, **kwargs)
        except Reconstruction3DError:
            continue
        raise AssertionError(f"expected Reconstruction3DError for {kwargs}")


def test_solidify_alpha_fills_interior_holes():
    """A low-contrast region inside the product (e.g. a grey screen) must
    not stay see-through, or the 3D model reconstructs a hole there."""
    import numpy as np
    from PIL import Image

    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[8:56, 8:56, 3] = 255  # product silhouette
    rgba[20:44, 20:44, 3] = 60  # semi-transparent interior (the "screen")
    rgba[2, 2, 3] = 255  # stray speck outside the product
    out = np.array(_solidify_alpha(Image.fromarray(rgba, "RGBA")))[:, :, 3]
    assert (out[20:44, 20:44] == 255).all(), "interior hole was not filled"
    assert out[2, 2] == 0, "stray speck was not removed"
    assert (out[8:56, 8:56] == 255).all()


def test_solidify_alpha_restores_original_colour():
    """rembg zeroes the colour of removed pixels; filled holes must get the
    original image's colour back, not black."""
    import numpy as np
    from PIL import Image

    original = np.full((64, 64, 4), [120, 130, 140, 255], dtype=np.uint8)
    masked = original.copy()
    masked[:, :, 3] = 0
    masked[8:56, 8:56, 3] = 255
    masked[20:44, 20:44] = [0, 0, 0, 0]  # rembg-style removed interior: black + transparent
    out = np.array(_solidify_alpha(Image.fromarray(masked, "RGBA"), Image.fromarray(original, "RGBA")))
    assert (out[30, 30] == [120, 130, 140, 255]).all(), out[30, 30]


def test_lay_flat_rotates_straight_on_panel():
    """A thin panel standing and facing the camera (a straight-on photo of
    e.g. a keyboard) is laid flat with the photographed face up."""
    import numpy as np
    import trimesh

    panel = trimesh.creation.box([1.0, 0.5, 0.05])  # thin along Z, facing +Z camera
    front_tag = np.argmax(panel.vertices[:, 2])  # a vertex on the photographed face
    yaw, pitch = _lay_flat_if_panel(panel, {"yaw": 0.0, "pitch": 0.0})
    ext = np.ptp(panel.vertices, axis=0)
    assert np.isclose(ext[1], 0.05), f"thin axis should now be vertical, extents {ext}"
    assert panel.vertices[front_tag, 1] > 0, "photographed face should be on top"
    assert pitch == 60.0 and yaw == 0.0


def test_lay_flat_leaves_other_objects_alone():
    import numpy as np
    import trimesh

    cube = trimesh.creation.box([1.0, 1.0, 1.0])
    before = cube.vertices.copy()
    assert _lay_flat_if_panel(cube, {"yaw": 10.0, "pitch": 5.0}) == (10.0, 5.0)
    panel = trimesh.creation.box([1.0, 0.5, 0.05])
    assert _lay_flat_if_panel(panel, {"yaw": 0.0, "pitch": 45.0}) == (0.0, 45.0), "angled view: keep"
    edge_on = trimesh.creation.box([0.05, 0.5, 1.0])  # thin axis sideways to the camera
    assert _lay_flat_if_panel(edge_on, {"yaw": 0.0, "pitch": 0.0}) == (0.0, 0.0), "edge-on: keep"
    assert np.allclose(cube.vertices, before)


def test_rembg_falls_back_when_model_not_cached():
    import os
    import tempfile

    old = os.environ.get("U2NET_HOME")
    with tempfile.TemporaryDirectory() as empty:
        os.environ["U2NET_HOME"] = empty
        try:
            assert _rembg_model_name() == "u2net"
        finally:
            if old is None:
                os.environ.pop("U2NET_HOME")
            else:
                os.environ["U2NET_HOME"] = old


def test_prepare_image_square_on_white():
    """Matches TripoSG's own preprocessing: square crop, 10% margin, white bg."""
    import numpy as np
    from PIL import Image

    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[40:60, 50:150] = [200, 30, 30, 255]  # a 100 x 20 red object
    cond, square = _prepare_image(Image.fromarray(rgba, "RGBA"))
    assert cond.mode == "RGB" and cond.size[0] == cond.size[1] == 120
    assert square.shape == (120, 120, 4)
    c = np.array(cond)
    assert (c[0, 0] == 255).all(), "background should be white"
    assert tuple(c[60, 60]) == (200, 30, 30), "object should be centred"


def test_camera_fit_recovers_known_view():
    """Render the silhouette of an asymmetric object from a known view,
    then check the fit recovers that view."""
    import numpy as np
    import trimesh

    parts = [
        trimesh.creation.box([1.0, 0.4, 0.6]),
        trimesh.creation.box([0.2, 0.9, 0.2], transform=trimesh.transformations.translation_matrix([0.4, 0.45, 0.2])),
        trimesh.creation.box([0.3, 0.2, 0.5], transform=trimesh.transformations.translation_matrix([-0.3, 0.3, -0.1])),
    ]
    mesh = trimesh.util.concatenate(parts)
    yaw, pitch, side = 40.0, 25.0, 200
    pts, _ = trimesh.sample.sample_surface(mesh, 60000, seed=3)
    u, v, _ = _project(pts, yaw, pitch, (70.0, side / 2, side / 2), np.zeros(3))
    square = np.zeros((side, side, 4), dtype=np.uint8)
    ok = (u >= 0) & (u < side) & (v >= 0) & (v < side)
    square[v[ok].astype(int), u[ok].astype(int)] = [128, 128, 128, 255]
    from scipy import ndimage

    square[:, :, 3] = ndimage.binary_closing(square[:, :, 3] > 0, iterations=2) * 255

    cam = _fit_camera(mesh, square)
    yaw_err = abs((cam["yaw"] - yaw + 180) % 360 - 180)
    assert yaw_err <= 6 and abs(cam["pitch"] - pitch) <= 6, cam
    assert cam["iou"] > 0.9, cam


def test_colorize_respects_visibility():
    """Front plane (seen by the camera) takes the photo's colours; the back
    plane (hidden behind it) gets one uniform material colour instead."""
    import numpy as np
    import trimesh

    front, back = _plane(0.1, +1), _plane(-0.1, -1)
    mesh = trimesh.util.concatenate([front, back])
    n_front = len(front.vertices)

    side = 100
    square = np.zeros((side, side, 4), dtype=np.uint8)
    square[:, : side // 2] = [255, 0, 0, 255]  # left half red
    square[:, side // 2 :] = [0, 0, 255, 255]  # right half blue
    camera = {"yaw": 0.0, "pitch": 0.0, "box": (40.0, side / 2, side / 2), "center": np.zeros(3)}

    covered = _colorize(mesh, square, camera)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(int)
    x = np.asarray(mesh.vertices)[:, 0]
    left, right = (x < -0.3), (x > 0.3)
    fl, fr = colors[:n_front][left[:n_front]], colors[:n_front][right[:n_front]]
    assert (fl[:, 0] > 200).all() and (fl[:, 2] < 30).all(), "front-left should be red"
    assert (fr[:, 2] > 200).all() and (fr[:, 0] < 30).all(), "front-right should be blue"
    back_colors = colors[n_front:]
    assert (back_colors == back_colors[0]).all(), "hidden back plane should be one uniform colour"
    assert 0.3 < covered < 0.7


def test_glb_material_patch_roundtrip():
    """Vertex-coloured GLB gets one PBR material assigned to every primitive."""
    import numpy as np
    import trimesh

    mesh = trimesh.creation.box([1, 1, 1])
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.full((8, 4), 200, dtype=np.uint8))
    glb = _set_glb_material(mesh.export(file_type="glb"), metallic=1.0, roughness=0.35)
    json_len = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20 : 20 + json_len])
    pbr = doc["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0 and pbr["roughnessFactor"] == 0.35
    for m in doc["meshes"]:
        for p in m["primitives"]:
            assert p["material"] == 0 and "COLOR_0" in p["attributes"]
    assert struct.unpack_from("<I", glb, 8)[0] == len(glb), "GLB total length header"
    import io

    reloaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    assert len(reloaded.faces) == 12


def test_scale_to_real_size():
    import numpy as np
    import trimesh

    mesh = trimesh.creation.box([2.0, 1.0, 0.5])  # x, y (height), z
    dims = _scale_to_real_size(mesh, 300)
    assert dims == [300.0, 75.0, 150.0], dims  # [width, depth, height] mm
    assert abs(mesh.bounds[0][1]) < 1e-9, "should rest on the ground (y=0)"
    assert np.isclose(np.ptp(mesh.vertices, axis=0).max(), 0.3), "metres"
    assert _scale_to_real_size(trimesh.creation.box([2, 1, 1]), None) is None


def test_reconstruct_3d_real_model():
    """Real TripoSG reconstruction. Slow (~45s); requires the local weights
    and a GPU. Run explicitly to validate the live pipeline before the
    demo -- use a real generated concept image (not the raw sketch), since
    TripoSG expects a single photoreal object."""
    import glob

    candidates = sorted(glob.glob(str(ROOT / "demo_outputs" / "concept-*.png")))
    if not candidates:
        raise AssertionError(
            "No generated concept image found in demo_outputs/. "
            "Run the visual engine first (see visual_engine/test_visual.py's "
            "test_generate_concept_real_model), then re-run this test."
        )
    result = reconstruct_3d(candidates[-1], target_max_mm=304, roughness=0.5, metallic=1.0)
    assert result["status"] == "success"
    glb = (ROOT / result["glb_path"]).read_bytes()
    doc = json.loads(glb[20 : 20 + struct.unpack_from("<I", glb, 12)[0]])
    assert "COLOR_0" in doc["meshes"][0]["primitives"][0]["attributes"]
    assert doc["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 1.0
    assert max(result["dimensions_mm"]) == 304.0
    assert len(result["view_orbit_deg"]) == 2
    print(
        f"real reconstruction time: {result['reconstruction_time_s']}s, "
        f"dimensions {result['dimensions_mm']} mm"
    )


if __name__ == "__main__":
    test_reconstruct_3d_placeholder()
    test_missing_image_raises()
    test_invalid_inputs_raise()
    test_solidify_alpha_fills_interior_holes()
    test_solidify_alpha_restores_original_colour()
    test_lay_flat_rotates_straight_on_panel()
    test_lay_flat_leaves_other_objects_alone()
    test_rembg_falls_back_when_model_not_cached()
    test_prepare_image_square_on_white()
    test_camera_fit_recovers_known_view()
    test_colorize_respects_visibility()
    test_glb_material_patch_roundtrip()
    test_scale_to_real_size()
    print("3D reconstruction contract ok (fast contract checks)")
    print("run test_reconstruct_3d_real_model() separately to validate real inference")
