"""Re-export of the shared material contract. Do not define a second schema."""

from shared.schemas import ExplanationResult, MaterialComparison

__all__ = ["ExplanationResult", "MaterialComparison"]
