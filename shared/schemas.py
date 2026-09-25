"""Frozen request and response shapes.

Do not rename fields without team agreement. Gateway, visual engine,
and material engine all import these models.
"""

from typing import Literal

from pydantic import BaseModel, Field


class VisualResult(BaseModel):
    image_path: str | None = None
    image_url: str | None = None
    generation_time_s: float = 0
    seed: int | None = None
    control_mode: Literal["canny", "depth"] = "canny"
    placeholder: bool = False
    status: Literal["success", "error"] = "success"
    error: str | None = None


class MaterialComparison(BaseModel):
    baseline_material_id: str
    candidate_material_id: str
    baseline_label: str
    candidate_label: str
    product_profile_id: str
    product_type: str
    baseline_mass_g: float
    candidate_mass_g: float
    mass_delta_percent: float
    baseline_raw_material_cost_usd: float
    candidate_raw_material_cost_usd: float
    raw_material_cost_delta_usd: float
    baseline_thermal_conductivity_w_mk: float
    candidate_thermal_conductivity_w_mk: float
    thermal_comparison: str
    baseline_recycled_content_percent: float
    recycled_content_percent: float
    assumptions: list[str] = Field(default_factory=list)
    status: Literal["success", "error"] = "success"
    error: str | None = None


class ExplanationResult(BaseModel):
    summary: str
    tradeoffs: list[str] = Field(default_factory=list)
    source: Literal["template", "llm"] = "template"
    status: Literal["success", "error"] = "success"
    error: str | None = None


class Reconstruction3DResult(BaseModel):
    """Best-effort orbit-able 3D preview, reconstructed from the generated
    concept image via Stable Fast 3D. Additive and non-blocking: a failure
    here never prevents the 2D image + material report from returning
    successfully (mirrors VisualResult's own error-status contract).
    """

    glb_path: str | None = None
    glb_url: str | None = None
    reconstruction_time_s: float = 0
    status: Literal["success", "error"] = "success"
    error: str | None = None


class ConceptResponse(BaseModel):
    visual: VisualResult
    materials: MaterialComparison
    explanation: ExplanationResult
    reconstruction: Reconstruction3DResult
    disclaimer: str
