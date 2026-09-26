"""3D reconstruction from a generated concept image, using TripoSG (MIT).

Every component on this path is open source: TripoSG code and weights
(MIT), its bundled DINOv2 image encoder (Apache-2.0), scikit-image
marching cubes (BSD) for surface extraction, rembg's isnet/u2net masks
(Apache-2.0/MIT), and fast-simplification (MIT). TripoSG's two non-open
defaults are deliberately avoided: the `diso` flash extractor (CC BY-NC
4.0, see the patch note in third_party/TripoSG/triposg/inference_utils.py)
and the Bria RMBG-1.4 background remover.

Runs only inside the `protoskin3d-os` conda environment (see
scripts/setup_3d_env.sh) -- TripoSG needs diffusers/transformers versions
that differ from the main app's environment. This module is imported by
gateway_and_ui/backend/service_3d.py, a small standalone FastAPI process
run under that environment, never by the main gateway process directly.

Additive and best-effort by design: 3D reconstruction is a bonus preview
on top of the 2D concept image, which remains the primary, reliable
output (see the implementation plan's Contract A). A failure here must
never be treated as a pipeline failure by its caller.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("visual_engine.reconstruct3d")

# Resolution order matches the other model paths in this project
# (visual_engine/pipeline.py, material_intelligence/llm_explainer.py):
# explicit env var -> repo-relative default.
ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRIPOSG_PATH = ROOT / "models" / "TripoSG"
_DEFAULT_TRIPOSG_SRC = ROOT / "third_party" / "TripoSG"

DEMO_OUTPUTS = ROOT / "demo_outputs"

# Generation settings, tuned on the ZGX Nano (TripoSG's own defaults are 50
# steps and a dense 256^3 grid refined to 512^3, ~60s per mesh). Measured:
# 20 steps + a dense 128^3 grid refined to 256^3 near the surface gives a
# near-identical mesh in ~15s -- the dense 256^3 grid alone meant 16.7M
# decoder queries, mostly in empty space. Below ~16 steps shapes break
# (duplicated/floating parts), so don't go lower than 20 without checking.
_STEPS = int(os.environ.get("PROTOSKIN_3D_STEPS", "20"))
_DENSE_DEPTH = int(os.environ.get("PROTOSKIN_3D_DENSE_DEPTH", "7"))
_OCTREE_DEPTH = int(os.environ.get("PROTOSKIN_3D_OCTREE_DEPTH", "8"))
_TARGET_FACES = int(os.environ.get("PROTOSKIN_3D_FACES", "250000"))
_SEED = int(os.environ.get("PROTOSKIN_3D_SEED", "42"))


def _triposg_model_path() -> Path:
    configured = os.environ.get("PROTOSKIN_TRIPOSG")
    return Path(configured) if configured else _DEFAULT_TRIPOSG_PATH


def _triposg_src_path() -> Path:
    configured = os.environ.get("PROTOSKIN_TRIPOSG_SRC")
    return Path(configured) if configured else _DEFAULT_TRIPOSG_SRC


class Reconstruction3DError(Exception):
    """Raised for any failure in the 3D reconstruction path.

    The caller (service_3d.py's endpoint, or the gateway's HTTP client to
    it) is responsible for catching this and reporting
    Reconstruction3DResult(status="error") rather than failing the whole
    /api/concept request -- see shared/schemas.py's docstring on that
    model for the full rationale.
    """


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------


def _solidify_alpha(image, original=None):
    """Make the product silhouette a single solid, hole-free mask.

    Background removal (both rembg's u2net and isnet) marks low-contrast
    *interior* regions as partly transparent -- e.g. a light-grey laptop
    screen or keyboard keys against a grey studio backdrop. A 3D model
    reads that transparency as empty space and builds holes there. A
    product is one solid object, so: threshold, keep the largest connected
    component (drops stray specks), and fill enclosed holes. Pixels on the
    outer silhouette keep their original soft alpha for clean edges.

    rembg also zeroes the colour of every pixel it removes, so filled holes
    would come back black. Pass the ``original`` image to take colours from
    it and use ``image`` only for the mask.
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    rgba = np.array(image.convert("RGBA"))
    if original is not None:
        rgba[:, :, :3] = np.array(original.convert("RGB"))
    alpha = rgba[:, :, 3]
    solid = alpha > 127
    labels, count = ndimage.label(solid)
    if count == 0:
        return image
    if count > 1:
        sizes = ndimage.sum(solid, labels, index=range(1, count + 1))
        solid = labels == (int(np.argmax(sizes)) + 1)
    solid = ndimage.binary_fill_holes(solid)
    # Inside the solid region: fully opaque. Just outside it (a 2px band):
    # keep the original soft edge. Everything else: transparent.
    edge_band = ndimage.binary_dilation(solid, iterations=2) & ~solid
    new_alpha = np.where(solid, 255, np.where(edge_band, alpha, 0)).astype(np.uint8)
    rgba[:, :, 3] = new_alpha
    return Image.fromarray(rgba, mode="RGBA")


def _rembg_model_name() -> str:
    """Pick the background-removal model, preferring isnet-general-use.

    isnet gives cleaner edges than rembg's default u2net on studio shots.
    rembg downloads models on first use, which would break the
    fully-offline demo, so only use isnet if it's already cached
    (scripts/setup_3d_env.sh pre-fetches it).
    """
    preferred = os.environ.get("PROTOSKIN_3D_REMBG_MODEL", "isnet-general-use")
    home = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    if (home / f"{preferred}.onnx").is_file():
        return preferred
    logger.warning(
        "rembg model %s not cached in %s; falling back to u2net. Run "
        "scripts/setup_3d_env.sh to pre-fetch it.",
        preferred,
        home,
    )
    return "u2net"


def _prepare_image(rgba_image, pad_ratio: float = 0.1):
    """Crop to the object and pad to a square, as TripoSG expects.

    Mirrors TripoSG's own scripts/image_process.py (bounding-box crop,
    ``pad_ratio`` margin on each side, square, composited on white) but
    takes an already-masked RGBA image, so it needs no Bria RMBG model.

    Returns (condition RGB image for TripoSG, the same square crop as an
    RGBA numpy array for colour projection).
    """
    import numpy as np
    from PIL import Image

    rgba = np.array(rgba_image.convert("RGBA"))
    ys, xs = np.nonzero(rgba[:, :, 3] > 127)
    if len(ys) == 0:
        raise Reconstruction3DError("No object found in the image after background removal")
    crop = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape[:2]
    side = int(round(max(h, w) * (1 + 2 * pad_ratio)))
    square = np.zeros((side, side, 4), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    square[oy : oy + h, ox : ox + w] = crop
    alpha = square[:, :, 3:4].astype(np.float32) / 255.0
    rgb = square[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)
    return Image.fromarray(rgb.round().astype(np.uint8), mode="RGB"), square


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class _TripoSGPipeline:
    """Lazily-loaded singleton wrapper around TripoSG.

    Mirrors the pattern in visual_engine/pipeline.py's
    _SDXLCannyPipeline and material_intelligence/llm_explainer.py's
    _QwenExplainer: pay the model-load cost once per process, reuse the
    instance across requests.
    """

    _instance: "_TripoSGPipeline | None" = None
    _lock = threading.Lock()
    # One generation at a time: the diffusers pipeline keeps per-run state
    # (e.g. the scheduler's timesteps) on shared objects, so concurrent runs
    # corrupt each other -- and running several at once only makes every one
    # of them slower on a single GPU. Extra requests queue here.
    _run_lock = threading.Lock()

    def __init__(self) -> None:
        src = _triposg_src_path()
        if not (src / "triposg").is_dir():
            raise Reconstruction3DError(
                f"Vendored TripoSG source not found at {src}. See third_party/README.md."
            )
        model_path = _triposg_model_path()
        if not (model_path / "model_index.json").is_file():
            raise Reconstruction3DError(
                f"Local TripoSG weights not found at {model_path}. "
                "Download with scripts/download_models.sh."
            )
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        import rembg
        import torch
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = TripoSGPipeline.from_pretrained(str(model_path)).to(self.device, dtype)
        self.rembg_session = rembg.new_session(_rembg_model_name())

    @classmethod
    def get(cls) -> "_TripoSGPipeline":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, image_path: str, seed: int):
        """Return (raw trimesh, square RGBA crop the model was conditioned on)."""
        import numpy as np
        import rembg
        import trimesh
        from PIL import Image

        original = Image.open(image_path).convert("RGBA")
        masked = rembg.remove(original, session=self.rembg_session)
        cond, square_rgba = _prepare_image(_solidify_alpha(masked, original))

        with self._run_lock, self._torch.no_grad():
            output = self.pipe(
                image=cond,
                generator=self._torch.Generator(device=self.device).manual_seed(seed),
                num_inference_steps=_STEPS,
                guidance_scale=7.0,
                # Scikit-image marching cubes (BSD). The flash decoder needs
                # the non-open `diso` package; see the module docstring.
                use_flash_decoder=False,
                dense_octree_depth=_DENSE_DEPTH,
                hierarchical_octree_depth=_OCTREE_DEPTH,
            ).samples[0]
        vertices, faces = output
        if vertices is None or len(vertices) == 0:
            raise Reconstruction3DError("TripoSG produced an empty surface")
        mesh = trimesh.Trimesh(
            np.asarray(vertices, dtype=np.float32), np.ascontiguousarray(faces), process=True
        )
        return mesh, square_rgba


# --------------------------------------------------------------------------
# Mesh post-processing
# --------------------------------------------------------------------------


def _clean_mesh(mesh, target_faces: int):
    """Drop floating fragments and decimate to a browser-friendly size."""
    import trimesh

    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        total = sum(p.area for p in parts)
        keep = [p for p in parts if p.area >= 0.01 * total]
        mesh = trimesh.util.concatenate(keep)
    if target_faces and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh


def _view_axes(yaw_deg: float, pitch_deg: float):
    """Orthonormal camera frame for an orbit camera around +Y (up).

    Direction to the camera d = (sin(yaw)cos(pitch), sin(pitch),
    cos(yaw)cos(pitch)) -- the same spherical convention model-viewer uses
    for camera-orbit (theta = yaw from +Z towards +X, phi = 90 - pitch), so
    a fitted view maps directly onto the viewer's starting angle.
    """
    import numpy as np

    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    d = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    right = np.cross([0.0, 1.0, 0.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    return right, up, d


def _project(points, yaw_deg, pitch_deg, box, center):
    """Orthographic projection into square-crop pixel coordinates.

    ``box`` = (scale, u0, v0) maps view-plane units to pixels, anchored so
    view-plane point ``center`` lands on pixel (u0, v0). Returns (u, v,
    depth) where larger depth = closer to the camera.
    """
    right, up, d = _view_axes(yaw_deg, pitch_deg)
    rel = points - center
    s, u0, v0 = box
    return u0 + s * (rel @ right), v0 - s * (rel @ up), rel @ d


def _silhouette_fit(points, mask, yaw_deg, pitch_deg):
    """Align one view's projected silhouette to the mask; return (IoU, box, center)."""
    import numpy as np
    from scipy import ndimage

    right, up, _ = _view_axes(yaw_deg, pitch_deg)
    x, y = points @ right, points @ up
    ys, xs = np.nonzero(mask)
    mw, mh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    pw, ph = max(x.max() - x.min(), 1e-6), max(y.max() - y.min(), 1e-6)
    scale = float(np.sqrt((mw / pw) * (mh / ph)))
    # Anchor projected bbox centre on mask bbox centre.
    cx_px, cy_px = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    center3d = right * (x.max() + x.min()) / 2 + up * (y.max() + y.min()) / 2
    u, v, _ = _project(points, yaw_deg, pitch_deg, (scale, cx_px, cy_px), center3d)
    h, w = mask.shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sil = np.zeros_like(mask)
    sil[v[ok].astype(int), u[ok].astype(int)] = True
    sil = ndimage.binary_closing(sil, iterations=2)
    inter = np.logical_and(sil, mask).sum()
    union = np.logical_or(sil, mask).sum()
    return inter / max(union, 1), (scale, cx_px, cy_px), center3d


def _edge_mismatch(mesh, photo_edges, yaw, pitch, box, center, shape):
    """How badly the mesh's creases disagree with the photo's edges.

    Silhouettes alone can't tell a view from its mirror opposite (a
    laptop seen from front-left vs back-right has nearly the same outline).
    Interior geometry can: keys, bezels and seams produce normal
    discontinuities that line up with the photo's edges only from the
    correct side. Returns the mean distance (px) from rendered crease
    pixels to the nearest photo edge -- lower is better.
    """
    import numpy as np
    from scipy import ndimage

    h, w = shape
    u, v, depth = _project(np.asarray(mesh.vertices), yaw, pitch, box, center)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = v[ok].astype(int) * w + u[ok].astype(int)
    zbuf = np.full(h * w, -np.inf)
    np.maximum.at(zbuf, idx, depth[ok])
    front = depth[ok] >= zbuf[idx] - 1e-3
    normals = np.zeros((h * w, 3))
    normals[idx[front]] = np.asarray(mesh.vertex_normals)[ok][front]
    normals = normals.reshape(h, w, 3)
    gx = np.abs(np.diff(normals, axis=1, append=normals[:, -1:])).sum(-1)
    gy = np.abs(np.diff(normals, axis=0, append=normals[-1:])).sum(-1)
    covered = zbuf.reshape(h, w) > -np.inf
    creases = ((gx + gy) > 0.6) & ndimage.binary_erosion(covered, iterations=2)
    if creases.sum() < 20:
        return float("inf")
    dist_to_edge = ndimage.distance_transform_edt(~photo_edges)
    return float(dist_to_edge[creases].mean())


def _fit_camera(mesh, square_rgba, work_px: int = 160):
    """Find the orthographic view from which the photo was taken.

    TripoSG generates in its own object-centric frame, so the input view
    is unknown. Coarse-to-fine search over yaw/pitch for the projected
    silhouette that best overlaps the photo's object mask (IoU), with
    ambiguous front/back pairs broken by crease-vs-photo edge agreement.
    Returns a dict in square-crop pixel units.
    """
    import numpy as np
    import trimesh
    from PIL import Image
    from scipy import ndimage

    side = square_rgba.shape[0]
    k = work_px / side
    small = np.array(Image.fromarray(square_rgba).resize((work_px, work_px), Image.BILINEAR))
    mask = small[:, :, 3] > 127
    points, _ = trimesh.sample.sample_surface(mesh, 40000, seed=0)

    scored = []
    for yaw in range(0, 360, 10):
        for pitch in range(-30, 70, 10):
            iou, box, center = _silhouette_fit(points, mask, yaw, pitch)
            scored.append((iou, yaw, pitch))
    scored.sort(reverse=True)

    # Refine the top distinct candidates locally.
    refined = []
    seeds = []
    for iou, yaw, pitch in scored:
        if all(abs((yaw - sy + 180) % 360 - 180) > 25 or abs(pitch - sp) > 25 for _, sy, sp in seeds):
            seeds.append((iou, yaw, pitch))
        if len(seeds) == 4:
            break
    for _, yaw0, pitch0 in seeds:
        best = (-1.0, yaw0, pitch0)
        for step, radius in ((2, 8), (1, 2)):
            cy, cp = best[1], best[2]
            for dy in range(-radius, radius + 1, step):
                for dp in range(-radius, radius + 1, step):
                    pitch = float(np.clip(cp + dp, -60, 85))
                    iou = _silhouette_fit(points, mask, (cy + dy) % 360, pitch)[0]
                    if iou > best[0]:
                        best = (iou, (cy + dy) % 360, pitch)
        refined.append(best)
    refined.sort(reverse=True)

    # Among candidates whose silhouettes fit about equally well, prefer the
    # one whose interior creases match the photo's edges.
    top_iou = refined[0][0]
    close = [c for c in refined if c[0] >= top_iou - 0.03]
    if len(close) > 1:
        import cv2

        gray = np.array(Image.fromarray(small[:, :, :3]).convert("L"))
        photo_edges = (cv2.Canny(gray, 50, 150) > 0) & ndimage.binary_erosion(mask, iterations=2)
        ranked = []
        for iou, yaw, pitch in close:
            _, box, center = _silhouette_fit(points, mask, yaw, pitch)
            ranked.append(
                (_edge_mismatch(mesh, photo_edges, yaw, pitch, box, center, mask.shape), -iou, yaw, pitch)
            )
        ranked.sort()
        _, neg_iou, yaw, pitch = ranked[0]
        iou = -neg_iou
    else:
        iou, yaw, pitch = refined[0]

    _, (scale, u0, v0), center = _silhouette_fit(points, mask, yaw, pitch)
    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "iou": float(iou),
        # Rescale the fitted box from the working resolution to the full crop.
        "box": (scale / k, u0 / k, v0 / k),
        "center": center,
    }


def _srgb_to_linear(c):
    import numpy as np

    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _colorize(mesh, square_rgba, camera) -> float:
    """Colour vertices from the photo where the camera saw them.

    Visible, camera-facing vertices take the photo's pixel colour; everything
    the camera couldn't see gets the photo's median object colour -- the
    material's real colour -- rather than an invented one. Colours are
    stored linear, as glTF's COLOR_0 requires. Returns the fraction of
    vertices that received photo colour.
    """
    import numpy as np
    import trimesh
    from scipy import ndimage

    side = square_rgba.shape[0]
    verts = np.asarray(mesh.vertices)
    u, v, depth = _project(verts, camera["yaw"], camera["pitch"], camera["box"], camera["center"])

    # Occlusion: z-buffer of dense surface samples at half crop resolution.
    zres = max(side // 2, 64)
    zk = zres / side
    samples, _ = trimesh.sample.sample_surface(mesh, 400000, seed=1)
    su, sv, sd = _project(samples, camera["yaw"], camera["pitch"], camera["box"], camera["center"])
    ok = (su >= 0) & (su < side) & (sv >= 0) & (sv < side)
    zbuf = np.full(zres * zres, -np.inf)
    np.maximum.at(zbuf, (sv[ok] * zk).astype(int) * zres + (su[ok] * zk).astype(int), sd[ok])
    zbuf = ndimage.maximum_filter(zbuf.reshape(zres, zres), size=3).ravel()
    extent = float(np.ptp(verts, axis=0).max())
    inside = (u >= 0) & (u < side - 1) & (v >= 0) & (v < side - 1)
    zi = np.clip((v * zk).astype(int), 0, zres - 1) * zres + np.clip((u * zk).astype(int), 0, zres - 1)
    visible = inside & (depth >= zbuf[zi] - 0.01 * extent)

    _, _, d = _view_axes(camera["yaw"], camera["pitch"])
    facing = np.clip(((np.asarray(mesh.vertex_normals) @ d) - 0.1) / 0.4, 0, 1)

    photo = square_rgba.astype(np.float32) / 255.0
    uc, vc = np.clip(u, 0, side - 1.001), np.clip(v, 0, side - 1.001)
    x0, y0 = uc.astype(int), vc.astype(int)
    fx, fy = (uc - x0)[:, None], (vc - y0)[:, None]
    sampled = (
        photo[y0, x0] * (1 - fx) * (1 - fy)
        + photo[y0, x0 + 1] * fx * (1 - fy)
        + photo[y0 + 1, x0] * (1 - fx) * fy
        + photo[y0 + 1, x0 + 1] * fx * fy
    )
    weight = (facing * visible * sampled[:, 3])[:, None]

    object_px = photo[photo[:, :, 3] > 0.5][:, :3]
    base = np.median(object_px, axis=0) if len(object_px) else np.array([0.7, 0.7, 0.7])
    rgb = sampled[:, :3] * weight + base[None, :] * (1 - weight)

    linear = _srgb_to_linear(np.clip(rgb, 0, 1))
    rgba = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=(rgba * 255).round().astype(np.uint8))
    return float((weight[:, 0] > 0.5).mean())


def _lay_flat_if_panel(mesh, camera, max_thin_ratio: float = 0.15, max_pitch: float = 20.0):
    """Lay a thin panel flat when it was photographed straight-on.

    A single straight-on photo can't say whether a flat object (a
    keyboard, a lid) is standing on its edge or lying flat and shot from
    above; TripoSG then builds it standing up, facing the camera. Physically
    a thin panel rests flat, so when the view is near-level, the object's
    thinnest dimension is under ``max_thin_ratio`` of its largest, and the
    camera was looking at its broad face, rotate it so the photographed face
    points up. Returns the camera (yaw, pitch) to open the viewer from:
    unchanged if nothing was rotated, otherwise looking down from above.
    Must run after colouring (which uses the original camera frame).
    """
    import numpy as np
    import trimesh

    if abs(camera["pitch"]) > max_pitch:
        return camera["yaw"], camera["pitch"]
    # Thinnest direction via the oriented bounding box (robust to yaw).
    to_obb, extents = trimesh.bounds.oriented_bounds(mesh)
    axes = np.linalg.inv(to_obb)[:3, :3]  # columns: OBB axes in mesh frame
    thin = int(np.argmin(extents))
    if extents[thin] > max_thin_ratio * extents.max():
        return camera["yaw"], camera["pitch"]
    thin_axis = axes[:, thin] / np.linalg.norm(axes[:, thin])
    _, _, view = _view_axes(camera["yaw"], 0.0)  # horizontal direction to the camera
    if abs(thin_axis @ view) < 0.8:  # camera wasn't looking at the broad face
        return camera["yaw"], camera["pitch"]
    # Rotate +90deg about (view x up): maps `view` -> +Y, so the photographed
    # face ends up on top.
    axis = np.cross(view, [0.0, 1.0, 0.0])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, axis))
    logger.info("Thin panel photographed straight-on: laid flat, photographed face up")
    return camera["yaw"], 60.0


def _set_glb_material(glb: bytes, metallic: float, roughness: float) -> bytes:
    """Give every primitive in a GLB one PBR material with these factors.

    trimesh exports vertex-coloured meshes without a material, and glTF's
    default material is metallic=1, roughness=1 (dark rough metal). Edit
    the GLB's JSON chunk directly -- a small, dependency-free change.
    """
    magic, version, _ = struct.unpack_from("<III", glb, 0)
    if magic != 0x46546C67:  # b"glTF"
        raise Reconstruction3DError("Not a GLB file")
    json_len, json_type = struct.unpack_from("<II", glb, 12)
    doc = json.loads(glb[20 : 20 + json_len].decode("utf-8"))
    rest = glb[20 + json_len :]

    doc["materials"] = [
        {
            "name": "protoskin_material",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": float(metallic),
                "roughnessFactor": float(roughness),
            },
            "doubleSided": True,
        }
    ]
    for gltf_mesh in doc.get("meshes", []):
        for primitive in gltf_mesh.get("primitives", []):
            primitive["material"] = 0

    new_json = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)  # chunks are 4-byte aligned
    body = struct.pack("<II", len(new_json), json_type) + new_json + rest
    return struct.pack("<III", magic, version, 12 + len(body)) + body


def _scale_to_real_size(mesh, target_max_mm: float | None) -> list[float] | None:
    """Rest the mesh on the ground (y=0), centre it, and scale it in metres.

    With ``target_max_mm`` the largest dimension is set to that nominal
    real-world size (from the product profile); the others follow the
    model's own proportions. Returns [width, depth, height] in mm, where
    width/depth are the longer/shorter sides of the footprint (TripoSG's
    object frame doesn't fix which horizontal axis is "width") and height
    is along +Y (glTF up), or None when no real size was given.
    """
    import numpy as np

    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    extent = float(np.ptp(mesh.vertices, axis=0).max())
    target_m = (target_max_mm / 1000.0) if target_max_mm else 1.0
    mesh.apply_scale(target_m / extent)
    if not target_max_mm:
        return None
    x, h, z = np.ptp(mesh.vertices, axis=0) * 1000.0
    return [round(float(max(x, z)), 1), round(float(min(x, z)), 1), round(float(h), 1)]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def warm_up() -> None:
    """Force-load TripoSG ahead of the live demo.

    Mirrors visual_engine.pipeline.warm_up() and
    material_intelligence.llm_explainer.warm_up(). Best-effort: if the
    model can't load, reconstruction requests still fail gracefully at
    call time and the gateway falls back to 2D-only output.
    """
    _TripoSGPipeline.get()


def reconstruct_3d(
    image_path: str,
    target_max_mm: float | None = None,
    roughness: float | None = None,
    metallic: float | None = None,
    seed: int = _SEED,
    use_model: bool = True,
) -> dict:
    """Reconstruct an orbit-able 3D mesh from a generated concept image.

    Returns a dict matching shared.schemas.Reconstruction3DResult's
    fields (this module intentionally has no dependency on the main
    app's `shared` package, since it runs in a separate Python
    environment -- see the module docstring). The sidecar service
    (service_3d.py) returns it as JSON; the gateway builds the Pydantic
    model on its side.

    ``target_max_mm`` scales the model so its largest dimension matches
    the selected product's nominal size (the gateway reads it from
    shared/product_profiles.json); ``roughness``/``metallic`` set the
    material (from visual_engine.prompts.pbr_params). Both are optional.

    Raises Reconstruction3DError on any failure; callers must catch this
    (see the docstring on that exception).

    ``use_model=False`` skips loading TripoSG and returns a placeholder
    result without touching the input image, mirroring
    ``visual_engine.pipeline.generate_concept(use_model=False)`` and
    ``material_intelligence.llm_explainer.explain_result(use_llm=False)``.
    Keeps the contract (schema, validation, error handling) testable in
    the fast smoke-test loop without GPU inference or the model weights.
    """
    started = time.perf_counter()
    source = Path(image_path)
    if not source.is_file():
        raise Reconstruction3DError(f"Input image not found: {image_path}")
    for name, value in (("roughness", roughness), ("metallic", metallic)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise Reconstruction3DError(f"{name} must be within [0, 1], got {value!r}")
    if target_max_mm is not None and target_max_mm <= 0:
        raise Reconstruction3DError(f"target_max_mm must be positive, got {target_max_mm!r}")

    if not use_model:
        elapsed = round(time.perf_counter() - started, 3)
        return {
            "glb_path": None,
            "glb_url": None,
            "reconstruction_time_s": elapsed,
            "status": "success",
            "dimensions_mm": None,
            "view_orbit_deg": None,
        }

    try:
        pipeline = _TripoSGPipeline.get()
        mesh, square_rgba = pipeline.run(str(source), seed=seed)
        mesh = _clean_mesh(mesh, _TARGET_FACES)
        camera = _fit_camera(mesh, square_rgba)
        logger.info(
            "Fitted view yaw=%.0f pitch=%.0f (silhouette IoU %.2f)",
            camera["yaw"],
            camera["pitch"],
            camera["iou"],
        )
        covered = _colorize(mesh, square_rgba, camera)
        logger.info("Photo colour covered %.1f%% of vertices", covered * 100)
        view_yaw, view_pitch = _lay_flat_if_panel(mesh, camera)
        dimensions_mm = _scale_to_real_size(mesh, target_max_mm)
    except Reconstruction3DError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error to the caller
        raise Reconstruction3DError(f"3D reconstruction failed: {exc}") from exc

    glb = mesh.export(file_type="glb")
    glb = _set_glb_material(
        glb,
        metallic=0.0 if metallic is None else metallic,
        roughness=0.6 if roughness is None else roughness,
    )
    DEMO_OUTPUTS.mkdir(parents=True, exist_ok=True)
    filename = f"mesh-{uuid.uuid4().hex[:8]}.glb"
    (DEMO_OUTPUTS / filename).write_bytes(glb)

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "glb_path": f"demo_outputs/{filename}",
        "glb_url": f"/outputs/{filename}",
        "reconstruction_time_s": elapsed,
        "status": "success",
        "dimensions_mm": dimensions_mm,
        # model-viewer camera-orbit: theta = yaw, phi = 90 - pitch.
        "view_orbit_deg": [round(view_yaw, 1), round(90.0 - view_pitch, 1)],
    }
