"""Tool definitions exposed to the Claude agent.

The agent calls these as tools (not LLM math). Each tool wraps a deterministic
analyzer so reasoning happens in the LLM and engineering math stays trusted.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .analyzers.decline_curve import fit_decline, project_eur, analyze_type_curve
from .analyzers.economics import evaluate_intervention, simulate_intervention
from .analyzers.esp_diagnostics import evaluate_esp
from .data_loader import WellFile


# Tool schemas for Claude (Anthropic tool-use API)
TOOL_SCHEMAS = [
    {
        "name": "fit_decline_curve",
        "description": (
            "Fit an Arps decline model to the well's production history. "
            "Returns initial rate, decline rate, hyperbolic b, R², the full-fit "
            "residual, and a type_curve block: a type curve fit on early/established "
            "decline and extrapolated forward, giving today's rate deviation and the "
            "cumulative deferred production (bbl and $) vs that type curve "
            "(negative deviation = underperforming)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["exponential", "harmonic", "hyperbolic"],
                    "default": "hyperbolic",
                },
            },
        },
    },
    {
        "name": "evaluate_esp_health",
        "description": (
            "Evaluate the ESP's operating health: is it within the Preferred "
            "Operating Range (POR)? Are there flags on intake pressure, motor "
            "temperature, or amperage? Returns likely failure modes."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "evaluate_intervention",
        "description": (
            "Run economics on a proposed intervention (acid stim, ESP swap, "
            "ESP-to-beam conversion, workover). Returns NPV @ 10%, payout in "
            "months, incremental EUR, and discounted profitability index "
            "(PV of inflows / investment; >1.0 = value-accretive)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Intervention name"},
                "treatment_cost_usd": {"type": "number"},
                "incremental_rate_bopd": {
                    "type": "number",
                    "description": "Expected initial uplift in oil rate (bbl/d)",
                },
                "uplift_decline_per_yr": {"type": "number", "default": 0.6},
            },
            "required": ["name", "treatment_cost_usd", "incremental_rate_bopd"],
        },
    },
    {
        "name": "simulate_intervention_economics",
        "description": (
            "Monte-Carlo (~10,000 trials) of an intervention's economics over "
            "uncertain inputs: incremental rate (lognormal ±30%), uplift decline "
            "(±0.15 abs), and realized price (sd ~$12). Returns P10/P50/P90 NPV "
            "(P10=optimistic, P90=conservative), probability of payout (NPV>0 AND "
            "payout < 24 months), and a tornado sensitivity (NPV swing per variable). "
            "Use this when the deterministic NPV is borderline or the user asks about "
            "risk / downside / confidence in an intervention's economics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Intervention name"},
                "treatment_cost_usd": {"type": "number"},
                "incremental_rate_bopd": {
                    "type": "number",
                    "description": "Expected initial uplift in oil rate (bbl/d)",
                },
                "uplift_decline_per_yr": {"type": "number", "default": 0.6},
            },
            "required": ["name", "treatment_cost_usd", "incremental_rate_bopd"],
        },
    },
    {
        "name": "project_recovery",
        "description": (
            "Project remaining recoverable oil to an economic limit using the "
            "fitted decline. Call fit_decline_curve first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "economic_limit_bopd": {"type": "number", "default": 5.0},
            },
        },
    },
]


# Valid interventions for the AFE-Copilot handoff. MUST stay in sync with
# afe-copilot/src/cost_db.py COST_TEMPLATES keys — this is the pe→afe contract.
AFE_INTERVENTIONS = (
    "acid_stimulation",
    "scale_treatment",
    "esp_swap",
    "esp_to_beam_conversion",
    "rod_pump_workover",
    "gas_lift_optimization",
    "paraffin_treatment",
    "p_and_a",
)

# Map free-text / report phrasing -> canonical AFE intervention key. Order matters:
# more specific signatures first (e.g. esp-to-beam before plain esp).
_AFE_PHRASE_MAP = [
    ("esp-to-beam", "esp_to_beam_conversion"),
    ("esp to beam", "esp_to_beam_conversion"),
    ("beam pump conversion", "esp_to_beam_conversion"),
    ("convert to beam", "esp_to_beam_conversion"),
    ("scale", "scale_treatment"),
    ("acid", "acid_stimulation"),
    ("matrix acid", "acid_stimulation"),
    ("esp swap", "esp_swap"),
    ("pump swap", "esp_swap"),
    ("replace the esp", "esp_swap"),
    ("right-siz", "esp_swap"),
    ("gas lift", "gas_lift_optimization"),
    ("injection rate", "gas_lift_optimization"),
    ("paraffin", "paraffin_treatment"),
    ("hot oil", "paraffin_treatment"),
    ("wax", "paraffin_treatment"),
    ("parted rod", "rod_pump_workover"),
    ("rod string", "rod_pump_workover"),
    ("workover", "rod_pump_workover"),
    ("rod replacement", "rod_pump_workover"),
    ("plug and abandon", "p_and_a"),
    ("p&a", "p_and_a"),
    ("abandon", "p_and_a"),
]


def _coerce_intervention(value: str) -> str:
    """Map a possibly-loose intervention string to a valid AFE key, or raise."""
    v = (value or "").strip().lower().replace(" ", "_")
    if v in AFE_INTERVENTIONS:
        return v
    text = (value or "").lower()
    for phrase, key in _AFE_PHRASE_MAP:
        if phrase in text:
            return key
    raise ValueError(
        f"Cannot map intervention {value!r} to a valid AFE intervention. "
        f"Must be one of: {', '.join(AFE_INTERVENTIONS)}"
    )


def export_afe_diagnosis(well: WellFile, review: dict[str, Any]) -> dict:
    """Emit a validated dict matching AFE-Copilot's AFEDiagnosis schema EXACTLY.

    This is the pe→afe chaining contract. `review` carries the engineer's (or
    agent's) selected intervention and economics; `well` supplies identity fields.
    Output keys/types are exactly what afe-copilot AFEDiagnosis.from_json() consumes:
      {well_id, api_number, field, operator, intervention, primary_diagnosis,
       incremental_rate_bopd, expected_uplift_decline_per_yr, requested_by}

    `intervention` is validated/coerced to one of the eight AFE keys.
    Raises ValueError if the intervention cannot be resolved.
    """
    intervention = _coerce_intervention(str(review.get("intervention", "")))

    try:
        rate = float(review.get("incremental_rate_bopd"))
    except (TypeError, ValueError):
        raise ValueError("incremental_rate_bopd is required and must be numeric.")
    if rate < 0:
        raise ValueError("incremental_rate_bopd must be >= 0.")

    decline = review.get("expected_uplift_decline_per_yr", 0.6)
    try:
        decline = float(decline)
    except (TypeError, ValueError):
        decline = 0.6

    operator = (
        review.get("operator")
        or getattr(well, "operator", None)
        or "Operator (synthetic)"
    )
    diagnosis = str(review.get("primary_diagnosis") or "").strip()
    if not diagnosis:
        raise ValueError("primary_diagnosis is required (free-form text).")

    return {
        "well_id": well.well_id,
        "api_number": well.api_number,
        "field": well.field,
        "operator": str(operator),
        "intervention": intervention,
        "primary_diagnosis": diagnosis,
        "incremental_rate_bopd": rate,
        "expected_uplift_decline_per_yr": decline,
        "requested_by": str(review.get("requested_by") or "Senior Production Engineer"),
    }


class ToolExecutor:
    """Executes tool calls against a single well's data. Stateful across the agent loop."""

    def __init__(self, well: WellFile):
        self.well = well
        self._last_fit = None

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            result = getattr(self, f"_tool_{name}")(**args)
            return json.dumps(result, default=float, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": name})

    def _tool_fit_decline_curve(self, model: str = "hyperbolic") -> dict:
        history = self.well.production_history
        days = np.array([row["day"] for row in history])
        rates = np.array([row["oil_bopd"] for row in history])
        fit = fit_decline(days, rates, model=model)
        self._last_fit = fit
        out = {
            "model": fit.model,
            "qi_bopd": fit.qi,
            "di_per_day": fit.di,
            "b": fit.b,
            "r_squared": fit.r_squared,
            "last_actual_bopd": fit.last_actual,
            "last_predicted_bopd": fit.last_predicted,
            # Quality-of-fit residual on the full history (NOT a type-curve call).
            "fit_residual_pct": fit.fit_residual_pct,
        }
        # True type-curve benchmark: fit early/established decline, extrapolate, and
        # measure rate deviation today + cumulative deferred production.
        try:
            tc = analyze_type_curve(days, rates, model=model)
            out["type_curve"] = {
                "deviation_pct": round(tc.deviation_pct, 1),
                "type_curve_bopd_today": round(tc.type_curve_at_last, 1),
                "deferred_bbl": round(tc.deferred_bbl, 0),
                "deferred_pct": round(tc.deferred_pct, 1),
                "deferred_value_usd": round(tc.deferred_value_usd, 0),
                "established_on_n_points": tc.established_days,
                "interpretation": (
                    "Underperforming type curve" if tc.deviation_pct < -10
                    else "Outperforming type curve" if tc.deviation_pct > 10
                    else "On type curve"
                ),
            }
        except ValueError as e:
            out["type_curve"] = {"error": str(e)}
        return out

    def _tool_evaluate_esp_health(self) -> dict:
        if self.well.artificial_lift.get("type") != "ESP":
            return {"applicable": False, "reason": "Well is not on ESP"}
        diag = evaluate_esp(self.well.esp_readings, self.well.artificial_lift["pump_spec"])
        return {
            "applicable": True,
            "in_por": diag.in_por,
            "current_bfpd": diag.current_bfpd,
            "por_window_bfpd": [diag.por_min_bfpd, diag.por_max_bfpd],
            "intake_pressure_psi": diag.intake_pressure_psi,
            "motor_temp_f": diag.motor_temp_f,
            "motor_amps": diag.motor_amps,
            "flags": diag.flags,
            "likely_issues": diag.likely_issues,
        }

    def _tool_evaluate_intervention(self, **kwargs) -> dict:
        econ = evaluate_intervention(**kwargs)
        return {
            "name": econ.name,
            "treatment_cost_usd": econ.treatment_cost_usd,
            "first_year_incremental_bbl": econ.incremental_first_year_bbl,
            "incremental_eur_bbl": econ.incremental_eur_bbl,
            "npv_10pct_usd": econ.npv_10pct_usd,
            # None (not Infinity) when the intervention never pays out — keeps the
            # tool output valid JSON for any downstream strict parser.
            "payout_months": econ.payout_months if math.isfinite(econ.payout_months) else None,
            "profitability_index": round(econ.profitability_index, 2),
            "recommendation": (
                "STRONG" if econ.npv_10pct_usd > 100_000 and econ.payout_months < 12
                else "MARGINAL" if econ.npv_10pct_usd > 0
                else "REJECT"
            ),
        }

    def _tool_simulate_intervention_economics(self, **kwargs) -> dict:
        sim = simulate_intervention(**kwargs)
        # Round for compact, readable tool output (the agent reasons over these).
        return {
            "name": sim["name"],
            "n_trials": sim["n_trials"],
            "treatment_cost_usd": round(sim["treatment_cost_usd"], 0),
            "npv_p90_conservative_usd": round(sim["npv_p90_usd"], 0),
            "npv_p50_median_usd": round(sim["npv_p50_usd"], 0),
            "npv_p10_optimistic_usd": round(sim["npv_p10_usd"], 0),
            "npv_mean_usd": round(sim["npv_mean_usd"], 0),
            "probability_of_payout": round(sim["probability_of_payout"], 3),
            "payout_cutoff_months": sim["payout_cutoff_months"],
            "tornado_swing_usd": {
                k: round(v["swing"], 0) for k, v in sim["tornado"].items()
            },
            "risk_verdict": (
                "ROBUST" if sim["npv_p90_usd"] > 0 and sim["probability_of_payout"] > 0.8
                else "MARGINAL" if sim["npv_p50_usd"] > 0
                else "HIGH RISK"
            ),
        }

    def _tool_project_recovery(self, economic_limit_bopd: float = 5.0) -> dict:
        if self._last_fit is None:
            return {"error": "Call fit_decline_curve first."}
        eur = project_eur(self._last_fit, economic_limit_bopd=economic_limit_bopd)
        return {"remaining_eur_bbl": eur, "economic_limit_bopd": economic_limit_bopd}
