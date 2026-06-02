"""Smoke tests for the deterministic analyzers."""
import numpy as np

from src.analyzers.decline_curve import fit_decline, project_eur
from src.analyzers.economics import evaluate_intervention, simulate_intervention
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


def test_monte_carlo_percentile_ordering_and_centering():
    sim = simulate_intervention(
        name="Acid Stim",
        treatment_cost_usd=150_000,
        incremental_rate_bopd=120,
        n_trials=10_000,
        seed=7,
    )
    # P90 (conservative) < P50 (median) < P10 (optimistic).
    assert sim["npv_p90_usd"] < sim["npv_p50_usd"] < sim["npv_p10_usd"]
    # Stochastic median should sit near the deterministic NPV (same math, mean-preserving draws).
    det = evaluate_intervention(
        name="Acid Stim", treatment_cost_usd=150_000, incremental_rate_bopd=120,
    ).npv_10pct_usd
    assert abs(sim["npv_p50_usd"] - det) / abs(det) < 0.25
    assert 0.0 <= sim["probability_of_payout"] <= 1.0
    assert set(sim["tornado"]) == {
        "incremental_rate_bopd", "uplift_decline_per_yr", "realized_price_per_bbl"
    }
    # Rate is the dominant driver -> largest tornado swing for this strong case.
    swings = {k: v["swing"] for k, v in sim["tornado"].items()}
    assert swings["incremental_rate_bopd"] == max(swings.values())


def test_export_afe_diagnosis_schema():
    from src.tools import export_afe_diagnosis, AFE_INTERVENTIONS

    class _W:
        well_id = "ED-001H"
        api_number = "42-109-12345"
        field = "Delaware Basin"
        operator = "Synthetic Operator"

    out = export_afe_diagnosis(_W(), {
        "intervention": "scale inhibitor squeeze + acid stimulation",
        "primary_diagnosis": "Scale + low intake + below POR",
        "incremental_rate_bopd": 220,
        "expected_uplift_decline_per_yr": 0.7,
    })
    assert set(out) == {
        "well_id", "api_number", "field", "operator", "intervention",
        "primary_diagnosis", "incremental_rate_bopd",
        "expected_uplift_decline_per_yr", "requested_by",
    }
    assert out["intervention"] in AFE_INTERVENTIONS
    assert out["intervention"] == "scale_treatment"
    assert out["incremental_rate_bopd"] == 220.0
