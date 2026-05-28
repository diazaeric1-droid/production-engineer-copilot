"""Smoke tests for the deterministic analyzers."""
import numpy as np

from src.analyzers.decline_curve import fit_decline, project_eur
from src.analyzers.economics import evaluate_intervention
from src.analyzers.esp_diagnostics import evaluate_esp


def test_fit_decline_recovers_known_curve():
    days = np.arange(30, 1000, 30)
    qi_true, di_true, b_true = 1000, 0.003, 0.9
    rates = qi_true / np.power(1 + b_true * di_true * days, 1 / b_true)
    fit = fit_decline(days, rates, model="hyperbolic")
    assert fit.r_squared > 0.99
    assert abs(fit.qi - qi_true) / qi_true < 0.05


def test_esp_flags_below_por():
    readings = [{"bfpd": 1200, "intake_pressure_psi": 30, "motor_temp_f": 300, "motor_amps": 70}]
    spec = {"por_min_bfpd": 1800, "por_max_bfpd": 3400, "motor_temp_max_f": 350, "motor_amps_nameplate": 65}
    diag = evaluate_esp(readings, spec)
    assert not diag.in_por
    assert any("BELOW POR" in f for f in diag.flags)


def test_intervention_economics_positive_npv():
    econ = evaluate_intervention(
        name="Acid Stim",
        treatment_cost_usd=150_000,
        incremental_rate_bopd=120,
    )
    assert econ.npv_10pct_usd > 0
    assert econ.payout_months < 12
