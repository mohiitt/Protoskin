# ProtoSkin

**Local AI Material Concept Studio for Industrial Design**

ProtoSkin is a fully local generative AI application that helps industrial designers visualize how product concepts could look with alternative materials and finishes. It uses **SDXL + ControlNet** to preserve the structure of an uploaded product sketch or CAD-derived image while applying photorealistic material concepts, and a local LLM generates a grounded material-analysis summary from predefined engineering data.

Designed for the **HP ZGX Nano GB10**, ProtoSkin keeps sensitive product concepts on-device and avoids external AI API and rendering costs.

## What It Does

1. Upload a product wireframe, sketch, or CAD-derived image.
2. Select a material such as recycled aluminum or ocean-bound plastic.
3. Optionally specify finish, texture, color, or lighting.
4. Generate a structure-guided photorealistic concept locally.
5. Compare the baseline and candidate material.
6. View deterministic material metrics plus a local-LLM explanation.

> ProtoSkin is a concept-visualization and early material-screening tool, not an engineering simulation or CAD validation system.

## Run the boilerplate

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
uvicorn gateway_and_ui.backend.main:app --reload
```

Open http://127.0.0.1:8000. Image generation is a placeholder copy of the upload. Material numbers come from the shared config. No model download is required for this flow.

On the ZGX Nano, when you are ready to connect real models:

```bash
pip install -r requirements-ml.txt
bash scripts/download_models.sh
```

## Optional: 3D preview

`visual_engine/reconstruct3d.py` adds a best-effort, orbit-able 3D preview
reconstructed from the generated 2D concept image, using
[TripoSG](https://github.com/VAST-AI-Research/TripoSG) (MIT). Every
component on the 3D path is open source -- see `third_party/README.md` for
the licence audit and the one patch that keeps TripoSG's non-open `diso`
extractor out. This is additive on top of the 2D image (which remains the
primary, reliable output).

What it does: background removal (rembg) → TripoSG shape generation
(512³ surface extraction) → fits the camera the photo was taken from →
colours the mesh from the photo (hidden sides get the material's real
colour) → scales it to the product profile's nominal size in mm.

Quick start:

```bash
bash scripts/setup_3d_env.sh            # one-time: builds the protoskin3d-os env
bash scripts/download_models.sh         # say yes when asked about TripoSG
bash scripts/run_3d_service.sh          # run alongside the main gateway
```

The main gateway (`gateway_and_ui/backend/main.py`) calls this sidecar
service over HTTP (`PROTOSKIN_3D_SERVICE_URL`, default
`http://127.0.0.1:8100`) and folds the result into `/api/concept`'s
response as `reconstruction`. On an idle ZGX Nano the concept image takes
~4s and the 3D model ~13s, so a result is back in under 20s; the local-LLM
explanation is fetched afterwards (`/api/explain`) because running it
alongside the 3D step slows both -- every model shares the same unified
memory bandwidth. Requests queue rather than run concurrently for the same
reason. If the 3D service isn't running or reconstruction fails, the 2D
image and material report still return normally -- the 3D viewer just
doesn't appear.

Limits: shape comes from a single image, so a flat, straight-on view gives
the model little depth to work with -- angled (3/4) views reconstruct
best. The mm size is nominal (from the product profile), not measured.

### Who edits what

| Folder | Owner | Frozen entry point |
|---|---|---|
| `visual_engine/` | Person A | `generate_concept(...)` |
| `material_intelligence/` | Person B | `compare_materials(...)`, `explain_result(...)` |
| `gateway_and_ui/` | Person C | FastAPI and `frontend/` |
| `shared/` | whole team | change only together |

Do not change those signatures or the fields in `shared/schemas.py` without agreeing first. The gateway imports `mock_visual.generate_concept`, which currently calls the placeholder pipeline. At integration, point that import at the real SDXL pipeline.

Suggested branches: `feature/visual-engine`, `feature/material-intelligence`, `feature/ui-gateway`.

## Architecture

```text
                     ProtoSkin Studio
                           |
                           v
                    FastAPI Gateway
                     /           \
                    /             \
                   v               v
        Visual Generation      Material Intelligence
        SDXL + ControlNet      Material Database
                |                    |
                |             Deterministic Metrics
                |                    |
                |               Local LLM
                \                   /
                 \                 /
                  v               v
               Render + Material Report
```

All AI inference is designed to run locally on the **HP ZGX Nano**.

## Open-Source Models

Models are downloaded from Hugging Face before use and then served locally.

### Required

- `stabilityai/stable-diffusion-xl-base-1.0`
  - Base image-generation model.

- `diffusers/controlnet-canny-sdxl-1.0`
  - Structural conditioning for sketches and wireframes.

- `Qwen/Qwen3-8B`
  - Local language model for converting grounded material metrics into a concise human-readable report.

### Optional

- `diffusers/controlnet-depth-sdxl-1.0-small`
  - Alternative conditioning for depth/CAD-style inputs.

- `Intel/dpt-hybrid-midas`
  - Depth estimation when a depth map is not already available.

- `Qwen/Qwen2.5-7B-Instruct`
  - Fallback local LLM.

## Example Workflow

```text
Input:
Generic laptop chassis wireframe

Material:
Ocean-Bound Recycled Polymer

Prompt:
"Matte graphite finish, fine recycled speckle,
studio product lighting"

              |
              v

      SDXL + ControlNet
              |
              v

Photorealistic Concept Render

              +

Deterministic Material Calculator
              |
              v
Density / mass / raw-material cost /
thermal-property comparison
              |
              v
Local Qwen Explanation
```

## Material Analysis

ProtoSkin does **not** ask the LLM to invent engineering numbers.

Material properties are stored in a shared material configuration and numerical comparisons are calculated deterministically. The local LLM only explains those results.

Example:

```text
Candidate Material: Recycled Polymer

Estimated relative mass:       -11.8%
Raw-material cost difference:  -$0.74 / shell
Thermal conductivity:          Lower than aluminum baseline
Recycled content:              75%

Assumptions:
- Same approximate shell volume
- Raw-material comparison only
- Manufacturing, tooling and assembly costs excluded
```

## Repository Structure

```text
protoskin/
|
|-- shared/
|   |-- materials_config.*
|   |-- schemas/
|   `-- sample_inputs/
|
|-- visual_engine/
|   `-- SDXL + ControlNet pipeline
|
|-- material_intelligence/
|   |-- deterministic calculations
|   `-- local LLM explanation
|
|-- gateway_and_ui/
|   |-- backend/
|   `-- frontend/
|
|-- scripts/
|   `-- download_models.sh
|
|-- tests/
|
`-- README.md
```

## Team Split

### Visual Engine
Owns:
- SDXL
- ControlNet
- Canny/depth preprocessing
- prompt construction
- generation performance and consistency

### Material Intelligence
Owns:
- material dataset/configuration
- deterministic calculations
- structured analysis output
- local Qwen explanation

### Gateway + UI
Owns:
- FastAPI
- frontend
- engine orchestration
- A/B comparison interface
- loading/error states
- final integration

Each module follows shared input/output contracts so the three workstreams can be developed independently.

## Model Download

Install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Example downloads:

```bash
hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir models/sdxl-base

hf download diffusers/controlnet-canny-sdxl-1.0 \
  --local-dir models/controlnet-canny

hf download Qwen/Qwen3-8B \
  --local-dir models/qwen3-8b
```

Optional:

```bash
hf download diffusers/controlnet-depth-sdxl-1.0-small \
  --local-dir models/controlnet-depth

hf download Intel/dpt-hybrid-midas \
  --local-dir models/dpt-hybrid-midas
```

Do not commit downloaded model weights to Git.

Add the model directory to `.gitignore`:

```text
models/
```

## Suggested Stack

- **AI / Vision:** PyTorch, Hugging Face Diffusers, ControlNet
- **LLM:** Qwen
- **Backend:** FastAPI
- **Frontend:** React, Gradio, or Streamlit
- **Image Processing:** Pillow, OpenCV
- **Hardware:** HP ZGX Nano GB10
- **Model Source:** Hugging Face

## Why Local?

Industrial product concepts can contain confidential pre-release design information.

ProtoSkin keeps:

- product sketches,
- CAD-derived imagery,
- prompts,
- generated concepts, and
- material-analysis context

inside the local environment rather than sending them to external generative-AI APIs.

Local execution also removes per-request external AI inference and rendering API charges, making repeated design exploration practical.

## Hackathon Demo

The live demo will:

1. Upload a laptop or printer wireframe.
2. Select a candidate sustainable material.
3. Generate a structure-guided material visualization locally.
4. Show the original and generated concept side-by-side.
5. Compare the baseline and candidate material metrics.
6. Generate a concise local-LLM explanation.
7. Demonstrate that no cloud AI API is used.

## Project Status

Hackathon prototype — active development.

## Disclaimer

ProtoSkin provides AI-generated concept visualizations and early-stage material screening estimates. Generated imagery is not CAD-accurate engineering output, and material estimates should not be treated as manufacturing, structural, thermal, or BOM validation.
