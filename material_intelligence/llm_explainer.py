"""Explain calculator output using a local Qwen3-8B model.

The template in this file is the guaranteed fallback: if the LLM is
unavailable, fails to load, times out, returns malformed output, or
produces text that does not reproduce the calculator's own numbers,
explain_result() falls back to template_explanation() instead of
raising or inventing numbers. The app must never crash because the
LLM misbehaved.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from shared.config import ROOT, get_material
from shared.schemas import ExplanationResult, MaterialComparison

logger = logging.getLogger("material_intelligence.llm_explainer")

# Resolution order: explicit env var -> repo-relative default (matches
# scripts/download_models.sh and .env.example).
_DEFAULT_LLM_PATH = ROOT / "models" / "qwen3-8b"


def _llm_model_path() -> Path:
    configured = os.environ.get("PROTOSKIN_LLM")
    return Path(configured) if configured else _DEFAULT_LLM_PATH


_MAX_NEW_TOKENS = 320
_GENERATION_TIMEOUT_S = 60.0

_SYSTEM_PROMPT = (
    "You are a materials-explanation assistant for ProtoSkin, a concept-screening tool. "
    "You will be given deterministic, already-calculated material metrics as grounded facts. "
    "Write a short plain-language explanation of those facts for an industrial designer.\n\n"
    "Rules you must follow exactly:\n"
    "- Use only the numbers given to you. Copy them exactly, with the same sign and precision.\n"
    "- Do not calculate, estimate, round differently, or invent any new numeric value.\n"
    "- Do not invent claims about manufacturing readiness, BOM savings, structural validation, "
    "or thermal simulation.\n"
    "- Do not claim the result is a supplier quote or production data.\n"
    "- Respond with a single JSON object only, no markdown fences, matching this shape:\n"
    '{"summary": "2-4 sentences", "tradeoffs": ["short bullet", "short bullet"]}'
)


def _build_user_prompt(comparison: MaterialComparison) -> str:
    candidate_notes = get_material(comparison.candidate_material_id)["analytical"].get("notes")
    baseline_notes = get_material(comparison.baseline_material_id)["analytical"].get("notes")
    facts = {
        "product_type": comparison.product_type,
        "baseline_material": comparison.baseline_label,
        "candidate_material": comparison.candidate_label,
        "baseline_mass_g": comparison.baseline_mass_g,
        "candidate_mass_g": comparison.candidate_mass_g,
        "mass_delta_percent": comparison.mass_delta_percent,
        "baseline_raw_material_cost_usd": comparison.baseline_raw_material_cost_usd,
        "candidate_raw_material_cost_usd": comparison.candidate_raw_material_cost_usd,
        "raw_material_cost_delta_usd": comparison.raw_material_cost_delta_usd,
        "thermal_comparison": comparison.thermal_comparison,
        "baseline_recycled_content_percent": comparison.baseline_recycled_content_percent,
        "candidate_recycled_content_percent": comparison.recycled_content_percent,
        "assumptions": comparison.assumptions,
        "baseline_notes": baseline_notes,
        "candidate_notes": candidate_notes,
    }
    return (
        "Grounded facts (JSON, already calculated deterministically):\n"
        f"{json.dumps(facts, indent=2)}\n\n"
        "Write the explanation now. Report mass_delta_percent as a signed percentage "
        "(e.g. '-57.4%') and raw_material_cost_delta_usd as a signed dollar amount "
        "(e.g. '-$0.46') exactly as given."
    )


class _QwenExplainer:
    """Lazily-loaded singleton wrapper around the local Qwen3-8B model."""

    _instance: "_QwenExplainer | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = _llm_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Local LLM not found at {model_path}")

        self._torch = torch
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    @classmethod
    def get(cls) -> "_QwenExplainer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def generate(self, comparison: MaterialComparison) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(comparison)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def warm_up() -> None:
    """Force-load the local LLM ahead of the live demo.

    Mirrors visual_engine.pipeline.warm_up(): call once at process startup
    so the first real request isn't paying model-load latency in front of
    an audience. Best-effort — if the model can't load, requests still
    fall back to the template explanation at call time.
    """
    _QwenExplainer.get()


def _run_with_timeout(func, timeout_s: float):
    result: dict[str, Any] = {}

    def target():
        try:
            result["value"] = func()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller via result
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"LLM generation exceeded {timeout_s}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Grab the first {...} block in case the model added stray text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


def _format_signed_percent(value: float) -> str:
    return f"{value:+.1f}%"


def _format_signed_dollars(value: float) -> str:
    return f"{value:+.2f}"


def _numbers_are_grounded(summary: str, comparison: MaterialComparison) -> bool:
    """Verify the LLM copied the calculator's numbers rather than inventing new ones."""
    mass_str = _format_signed_percent(comparison.mass_delta_percent)
    cost_str = _format_signed_dollars(comparison.raw_material_cost_delta_usd)
    # Accept the dollar delta with or without a leading $ sign.
    cost_variants = (cost_str, f"${cost_str.lstrip('+') if cost_str.startswith('+') else cost_str}")
    has_mass = mass_str in summary
    has_cost = any(variant in summary for variant in cost_variants) or (
        f"{abs(comparison.raw_material_cost_delta_usd):.2f}" in summary
    )
    return has_mass and has_cost


def _parse_llm_response(raw_text: str, comparison: MaterialComparison) -> ExplanationResult:
    data = _extract_json(raw_text)
    summary = str(data["summary"]).strip()
    tradeoffs = [str(item).strip() for item in data.get("tradeoffs", []) if str(item).strip()]
    if not summary:
        raise ValueError("LLM summary was empty")
    if not _numbers_are_grounded(summary, comparison):
        raise ValueError("LLM summary did not reproduce the calculator's numbers")
    return ExplanationResult(summary=summary, tradeoffs=tradeoffs, source="llm", status="success")


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


def explain_result(comparison: MaterialComparison, use_llm: bool = True) -> ExplanationResult:
    """Return a short explanation of a MaterialComparison.

    Tries the local Qwen3-8B model first (if use_llm and the model is
    available). The model must repeat the calculator's numbers exactly;
    if it doesn't, or generation fails/times out/loads incorrectly for
    any reason, this falls back to template_explanation(comparison) so
    the app never crashes and never shows an invented number.
    """
    if not use_llm:
        return template_explanation(comparison)

    try:
        explainer = _QwenExplainer.get()
        raw_text = _run_with_timeout(lambda: explainer.generate(comparison), _GENERATION_TIMEOUT_S)
        return _parse_llm_response(raw_text, comparison)
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the template
        logger.warning("Local LLM explanation failed, using template fallback: %s", exc)
        return template_explanation(comparison)
