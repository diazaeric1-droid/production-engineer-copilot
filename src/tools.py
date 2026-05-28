"""Tool definitions exposed to the Claude agent.

The agent calls these as tools (not LLM math). Each tool wraps a deterministic
analyzer so reasoning happens in the LLM and engineering math stays trusted.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .analyzers.decline_curve import fit_decline, project_eur
from .analyzers.economics import evaluate_intervention
from .analyzers.esp_diagnostics import evaluate_esp
from .data_loader import WellFile


# Tool schemas for Claude (Anthropic tool-use API)
TOOL_SCHEMAS = [
    {
        "name": "fit_decline_curve",
        "description": (
            "Fit an Arps decline model to the well's production history. "
            "Returns initial rate, decline rate, hyperbolic b, R², and how "
            "far the most recent rate deviates from the fit (negative = "
            "underperforming type curve)."
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
            "months, incremental EUR, and rate of return."
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
        return {
            "model": fit.model,
            "qi_bopd": fit.qi,
            "di_per_day": fit.di,
            "b": fit.b,
            "r_squared": fit.r_squared,
            "last_actual_bopd": fit.last_actual,
            "last_predicted_bopd": fit.last_predicted,
            "deviation_pct": fit.deviation_pct,
            "interpretation": (
                "Underperforming type curve" if fit.deviation_pct < -10
                else "Outperforming type curve" if fit.deviation_pct > 10
                else "On type curve"
            ),
        }

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
            "payout_months": econ.payout_months,
            "rate_of_return_pct": econ.rate_of_return_pct,
            "recommendation": (
                "STRONG" if econ.npv_10pct_usd > 100_000 and econ.payout_months < 12
                else "MARGINAL" if econ.npv_10pct_usd > 0
                else "REJECT"
            ),
        }

    def _tool_project_recovery(self, economic_limit_bopd: float = 5.0) -> dict:
        if self._last_fit is None:
            return {"error": "Call fit_decline_curve first."}
        eur = project_eur(self._last_fit, economic_limit_bopd=economic_limit_bopd)
        return {"remaining_eur_bbl": eur, "economic_limit_bopd": economic_limit_bopd}
