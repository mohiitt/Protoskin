"""Standalone checks for deterministic material screening."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from material_intelligence.calculator import compare_materials
from material_intelligence.llm_explainer import explain_result, template_explanation


def test_mass_and_cost_are_fixed():
    result = compare_materials("laptop_13", "recycled_aluminum", "ocean_bound_polymer")
    assert result.baseline_mass_g == 221.4
    assert result.candidate_mass_g == 94.3
    assert result.mass_delta_percent == -57.4
    assert result.baseline_raw_material_cost_usd == 0.69
    assert result.candidate_raw_material_cost_usd == 0.23
    assert result.raw_material_cost_delta_usd == -0.46
    assert result.recycled_content_percent == 70
    assert "lower" in result.thermal_comparison
    assert result.assumptions
    assert result.status == "success"


def test_same_inputs_match():
    first = compare_materials("laptop_15", "recycled_aluminum", "recycled_magnesium")
    second = compare_materials("laptop_15", "recycled_aluminum", "recycled_magnesium")
    assert first == second


def test_unknown_material_raises():
    try:
        compare_materials("laptop_13", "not_a_material", "standard_abs")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_explanation_repeats_calculator_numbers():
    comparison = compare_materials("printer_compact", "standard_abs", "recycled_fiber_composite")
    explanation = explain_result(comparison, use_llm=False)
    assert explanation.source == "template"
    assert f"{comparison.mass_delta_percent:+.1f}%" in explanation.summary
    assert f"{comparison.raw_material_cost_delta_usd:+.2f}" in explanation.summary
    assert explanation.tradeoffs


def test_template_explanation_matches_explain_result_fallback():
    comparison = compare_materials("laptop_13", "recycled_aluminum", "recycled_magnesium")
    assert template_explanation(comparison) == explain_result(comparison, use_llm=False)


def test_explain_result_falls_back_without_crashing_when_llm_disabled():
    comparison = compare_materials("laptop_15", "recycled_aluminum", "ocean_bound_polymer")
    explanation = explain_result(comparison, use_llm=False)
    assert explanation.status == "success"
    assert explanation.source == "template"


if __name__ == "__main__":
    test_mass_and_cost_are_fixed()
    test_same_inputs_match()
    test_unknown_material_raises()
    test_explanation_repeats_calculator_numbers()
    test_template_explanation_matches_explain_result_fallback()
    test_explain_result_falls_back_without_crashing_when_llm_disabled()
    print("material intelligence ok")
