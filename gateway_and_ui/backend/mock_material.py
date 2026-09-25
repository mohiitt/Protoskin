"""Gateway entry for material screening.

The calculator and template explainer have no model dependency, so the
UI can use them before the local LLM is connected.
"""

from material_intelligence.calculator import compare_materials
from material_intelligence.llm_explainer import explain_result, template_explanation

__all__ = ["compare_materials", "explain_result", "template_explanation"]
