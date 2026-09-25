"""Explain calculator output. Person B swaps in local Qwen later.

The template is the fallback. Keep it: if the LLM fails or invents
numbers, return template_explanation instead of crashing.
"""

from shared.config import get_material
from shared.schemas import ExplanationResult, MaterialComparison


def template_explanation(comparison: MaterialComparison) -> ExplanationResult:
    summary = (
        f"Under the same shell-volume assumption, {comparison.candidate_label} "
        f"is estimated to change shell mass by {comparison.mass_delta_percent:+.1f}% "
        f"and raw-material cost by {comparison.raw_material_cost_delta_usd:+.2f} USD "
        f"per shell versus {comparison.baseline_label}. "
        f"{comparison.thermal_comparison}"
    )
    tradeoffs = [
        comparison.thermal_comparison,
        (
            f"Recycled content indicator: {comparison.recycled_content_percent:.0f}% "
            f"for the candidate, {comparison.baseline_recycled_content_percent:.0f}% "
            f"for the baseline."
        ),
        "These figures are screening estimates, not a BOM, quote, or engineering validation.",
    ]
    notes = get_material(comparison.candidate_material_id)["analytical"].get("notes")
    if notes:
        tradeoffs.append(notes)
    return ExplanationResult(summary=summary, tradeoffs=tradeoffs, source="template", status="success")


def explain_result(comparison: MaterialComparison) -> ExplanationResult:
    """Return a short explanation of a MaterialComparison.

    Placeholder uses the template only. The real local LLM must repeat
    these numbers and must not invent new ones. On any model or parse
    failure, return template_explanation(comparison).
    """
    return template_explanation(comparison)
