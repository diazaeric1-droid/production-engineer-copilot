"""Locks the Phase-A agent fixes derived from the strict dev-set re-grade (35/41).

The five strict-grader confusion pairs (acid-stim<->scale, gas-separator<->esp-swap,
gas-lift-normal->optimization, insufficient-data->monitor, below-POR->acid-stim) traced
to three root causes: the prompt conflated acid-stim with scale ("surface BOTH terms"),
listed no healthy-gas-lift criteria under the monitor rule, and left the insufficient-data
call to LLM judgment on a bare error string. These tests pin the fixes: the deterministic
data-sufficiency verdict in fit_decline_curve, and the prompt's new decision rules.
All deterministic — no API key.
"""
from src.agent import SYSTEM_PROMPT
from src.data_loader import WellFile
from src.tools import ToolExecutor


def _well(n_points: int, esp_readings=None, dyno_cards=None) -> WellFile:
    history = [
        {"day": 30 * (i + 1), "oil_bopd": 1000 - 20 * i, "water_bwpd": 500, "gas_mcfd": 1700}
        for i in range(n_points)
    ]
    return WellFile(
        well_id="TEST-001H", api_number="42-109-99999", field="Test (synthetic)",
        operator="Test Operator", spud_date="2025-11-01", first_prod_date="2026-01-15",
        completion={"formation": "Wolfcamp A"},
        artificial_lift={"type": "ESP", "pump_spec": {"por_min_bfpd": 1800,
                                                      "por_max_bfpd": 3400,
                                                      "motor_amps_nameplate": 65}},
        production_history=history,
        esp_readings=esp_readings or [],
        dyno_cards=dyno_cards or [],
    )


# ---------- deterministic data-sufficiency verdict --------------------------

def test_sparse_history_no_lift_data_returns_must_verdict():
    out = ToolExecutor(_well(4))._tool_fit_decline_curve()
    suff = out["data_sufficiency"]
    assert suff["sufficient"] is False
    assert suff["valid_production_points"] == 4
    assert suff["esp_readings_available"] is False
    assert suff["dyno_cards_available"] is False
    # The verdict must dictate the exact primary-recommendation wording and forbid
    # the 'monitor' default — this is what the agent failed at under LLM judgment.
    assert "Insufficient data to make a recommendation" in suff["verdict"]
    assert "monitor" in suff["verdict"]


def test_sparse_history_with_lift_data_gets_softer_verdict():
    readings = [{"date": "2026-05-25", "bfpd": 2200, "intake_pressure_psi": 120,
                 "motor_temp_f": 290, "motor_amps": 62}]
    out = ToolExecutor(_well(4, esp_readings=readings))._tool_fit_decline_curve()
    suff = out["data_sufficiency"]
    assert suff["sufficient"] is False
    assert suff["esp_readings_available"] is True
    # With lift diagnostics available the well is NOT an insufficient-data case —
    # the verdict must not force that recommendation.
    assert "MUST" not in suff["verdict"]


def test_sufficient_history_fits_normally():
    out = ToolExecutor(_well(12))._tool_fit_decline_curve()
    assert "data_sufficiency" not in out
    assert "qi_bopd" in out and "r_squared" in out


# ---------- prompt decision rules (regression guards) ------------------------

def test_prompt_no_longer_conflates_acid_stim_with_scale():
    # The old scale rule said: surface BOTH "scale treatment" and "acid stimulation" —
    # written to game the lenient grader, and the direct cause of the
    # acid-stim<->scale confusions under strict grading.
    assert "surface BOTH terms" not in SYSTEM_PROMPT


def test_prompt_has_dedicated_acid_stim_rule():
    assert "Acid stimulation (matrix acid)" in SYSTEM_PROMPT
    # The discriminator: starved intake + over-amping = inflow problem, not pump deposition.
    assert "starved AND dragging" in SYSTEM_PROMPT


def test_prompt_separates_gas_from_scale_by_amp_direction():
    # Free gas cuts pump load; scale drags it. High amps must rule OUT gas as primary.
    assert "High amps CONTRADICT gas" in SYSTEM_PROMPT


def test_prompt_has_healthy_gas_lift_criteria():
    # gas_lift_normal -> "gas_lift_optimization" was action bias by omission: the clean
    # rule listed no gas-lift health criteria, so optimization looked like the default.
    assert "stable tubing-head pressure" in SYSTEM_PROMPT
    assert "never a default for the lift type" in SYSTEM_PROMPT


def test_prompt_keys_insufficient_data_off_tool_verdict():
    assert "data_sufficiency.sufficient: false" in SYSTEM_PROMPT
    assert "do not overrule the tool" in SYSTEM_PROMPT
