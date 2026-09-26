"""Deterministic material screening. Person B owns the formulas."""

from shared.config import get_material, get_profile
from shared.schemas import MaterialComparison


def _mass_g(density_kg_m3: float, volume_cm3: float) -> float:
    # kg/m^3 * cm^3 -> grams: density * volume / 1000
    return round(density_kg_m3 * volume_cm3 / 1000, 1)


def _thermal_comparison(baseline_label: str, candidate_label: str, base_k: float, cand_k: float) -> str:
    if cand_k < base_k * 0.9:
        relation = "lower"
    elif cand_k > base_k * 1.1:
        relation = "higher"
    else:
        relation = "similar"
    return (
        f"{candidate_label} thermal conductivity is {relation} than "
        f"{baseline_label} ({cand_k:g} vs {base_k:g} W/m·K)."
    )


def compare_materials(
    product_profile_id: str,
    baseline_material_id: str,
    candidate_material_id: str,
    *,
    custom_product_type: str | None = None,
    custom_volume_cm3: float | None = None,
) -> MaterialComparison:
    """Compare two materials for one product profile.

    mass = density * assumed shell volume
    raw material cost = mass * cost per kg

    Screening only. Same inputs always return the same numbers.

    For a product that isn't one of the preset profiles, pass
    ``custom_product_type`` and the user-supplied ``custom_volume_cm3``
    instead of a profile (``product_profile_id`` is then recorded as-is,
    e.g. "custom"). The volume is never guessed: callers without one must
    not call this.
    """
    if custom_product_type is not None:
        if custom_volume_cm3 is None or custom_volume_cm3 <= 0:
            raise ValueError("A custom product needs a positive shell volume (cm³)")
        profile = {"product_type": custom_product_type, "estimated_shell_volume_cm3": custom_volume_cm3}
    else:
        profile = get_profile(product_profile_id)
    baseline = get_material(baseline_material_id)
    candidate = get_material(candidate_material_id)
    base_props = baseline["analytical"]
    cand_props = candidate["analytical"]
    volume = float(profile["estimated_shell_volume_cm3"])

    baseline_mass_g = _mass_g(base_props["density_kg_m3"], volume)
    candidate_mass_g = _mass_g(cand_props["density_kg_m3"], volume)
    if baseline_mass_g == 0:
        mass_delta_percent = 0.0
    else:
        mass_delta_percent = round((candidate_mass_g - baseline_mass_g) / baseline_mass_g * 100, 1)

    baseline_cost = round(baseline_mass_g / 1000 * base_props["raw_material_cost_usd_per_kg"], 2)
    candidate_cost = round(candidate_mass_g / 1000 * cand_props["raw_material_cost_usd_per_kg"], 2)
    cost_delta = round(candidate_cost - baseline_cost, 2)

    assumptions = [
        f"Identical shell volume of {volume:g} cm³ ({profile['product_type']}"
        + (", user-entered)." if custom_product_type is not None else ")."),
        "Material-only estimate from the shared screening dataset.",
        "Excludes tooling, coatings, labor, yield, assembly, logistics, and supplier negotiation.",
        "Values are hackathon screening inputs, not HP supplier data.",
    ]
    return MaterialComparison(
        baseline_material_id=baseline_material_id,
        candidate_material_id=candidate_material_id,
        baseline_label=baseline["label"],
        candidate_label=candidate["label"],
        product_profile_id=product_profile_id,
        product_type=profile["product_type"],
        baseline_mass_g=baseline_mass_g,
        candidate_mass_g=candidate_mass_g,
        mass_delta_percent=mass_delta_percent,
        baseline_raw_material_cost_usd=baseline_cost,
        candidate_raw_material_cost_usd=candidate_cost,
        raw_material_cost_delta_usd=cost_delta,
        baseline_thermal_conductivity_w_mk=base_props["thermal_conductivity_w_mk"],
        candidate_thermal_conductivity_w_mk=cand_props["thermal_conductivity_w_mk"],
        thermal_comparison=_thermal_comparison(
            baseline["label"],
            candidate["label"],
            base_props["thermal_conductivity_w_mk"],
            cand_props["thermal_conductivity_w_mk"],
        ),
        baseline_recycled_content_percent=base_props["recycled_content_percent"],
        recycled_content_percent=cand_props["recycled_content_percent"],
        assumptions=assumptions,
        status="success",
    )
