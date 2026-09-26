# third_party/

Vendored source for dependencies that aren't installable as plain pip
packages. Tracked in git (unlike `models/`) because these are copies with
small, documented ProtoSkin patches.

## TripoSG/

The `triposg/` Python package from
[`VAST-AI-Research/TripoSG`](https://github.com/VAST-AI-Research/TripoSG)
(upstream commit in `UPSTREAM_COMMIT`), used by
`visual_engine/reconstruct3d.py` for the 3D preview via the sidecar
process started by `scripts/run_3d_service.sh`.

**License: MIT** (code and weights) -- see `LICENSE` and `NOTICE` in this
directory. Only the `triposg/` package is vendored; upstream's `scripts/`
(which pull in the non-open Bria RMBG-1.4 background remover) and example
assets are not.

### Why this is vendored instead of pip-installed

TripoSG isn't published as a pip package; its README runs it from a
checkout. `reconstruct3d.py` adds this directory to `sys.path` and imports
`triposg.pipelines.pipeline_triposg.TripoSGPipeline` directly.

### The one ProtoSkin patch: `diso` made optional

`triposg/inference_utils.py` imported `diso` (`DiffDMC`) at module top
level. `diso` is licensed **CC BY-NC 4.0** -- non-commercial, not open
source -- and is only used by TripoSG's optional *flash* surface extractor.
The patch moves that import inside `flash_extract_geometry()`, so the
package imports without `diso` installed. ProtoSkin always calls the
pipeline with `use_flash_decoder=False`, which uses
`hierarchical_extract_geometry()` -- scikit-image marching cubes (BSD) on a
512^3 grid. `scripts/setup_3d_env.sh` asserts `diso` is not installed.

### Open-source audit of the 3D path

| Component | Licence |
|---|---|
| TripoSG code + weights | MIT |
| DINOv2 image encoder (bundled in the TripoSG weights) | Apache-2.0 |
| scikit-image marching cubes | BSD-3-Clause |
| rembg + isnet-general-use / u2net masks | MIT / Apache-2.0 |
| fast-simplification (mesh decimation) | MIT |
| diffusers / transformers / peft | Apache-2.0 |
| model-viewer (browser 3D viewer, self-hosted bundle) | Apache-2.0 (google/model-viewer); the bundle also carries BSD-3-Clause (Lit) and MIT (three.js) headers |

Deliberately **not** used: `diso` (CC BY-NC 4.0) and Bria RMBG-1.4
(Bria licence), TripoSG's two non-open defaults.

## Not committing model weights here

TripoSG's weights (~7.5GB), like the SDXL / ControlNet / Qwen weights, are
*not* in this directory or in git. They're downloaded separately to
`PROTOSKIN_TRIPOSG` (see `.env.example`) by `scripts/download_models.sh`.
