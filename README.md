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
|-- material_engine/
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
