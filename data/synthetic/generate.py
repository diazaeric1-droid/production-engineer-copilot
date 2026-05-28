"""Generate a diverse synthetic well dataset for evals.

Each scenario is hand-tuned so the expected primary intervention is obvious to
an expert reviewer, giving the agent a meaningful eval signal.

Usage:
    python data/synthetic/generate.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
random.seed(42)
np.random.seed(42)


# ---------- helpers ----------------------------------------------------------

def hyperbolic_history(qi: float, di: float, b: float, days: list[int], noise: float = 0.05) -> list[dict]:
    """Generate production history with a hyperbolic decline + multiplicative noise."""
    out = []
    for d in days:
        oil = qi / ((1 + b * di * d) ** (1 / b)) * (1 + np.random.normal(0, noise))
        water = oil * (0.5 + d / 1500)
        gas = oil * (1.8 - d / 2000)
        out.append({"day": d, "oil_bopd": round(max(oil, 0), 1),
                    "water_bwpd": round(max(water, 0), 1),
                    "gas_mcfd": round(max(gas, 0), 1)})
    return out


def underperform(history: list[dict], from_day: int, factor: float) -> list[dict]:
    """Knock recent rates below the curve to simulate degradation."""
    for row in history:
        if row["day"] >= from_day:
            row["oil_bopd"] = round(row["oil_bopd"] * factor, 1)
    return history


STANDARD_DAYS = [30, 60, 90, 120, 180, 240, 300, 365, 450, 540, 630, 720, 810, 900, 990]


# ---------- scenario builders ------------------------------------------------

def esp_below_por(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H",
        "api_number": f"42-109-{10000 + idx}",
        "field": "Delaware Basin (synthetic)",
        "operator": "Synthetic Operator LLC",
        "spud_date": "2022-04-10",
        "first_prod_date": "2022-07-01",
        "completion": {"lateral_length_ft": 9800, "stages": 48, "proppant_lb_per_ft": 2300,
                       "fluid_bbl_per_ft": 52, "formation": "Wolfcamp B"},
        "artificial_lift": {
            "type": "ESP", "installed_date": "2024-01-15",
            "pump_spec": {"model": "REDA 538-D2700N", "stages": 180,
                          "por_min_bfpd": 1800, "por_max_bfpd": 3400,
                          "motor_temp_max_f": 350, "motor_amps_nameplate": 65},
        },
        "production_history": underperform(hyperbolic_history(1100, 0.0035, 0.85, STANDARD_DAYS), 720, 0.75),
        "esp_readings": [
            {"date": "2026-05-21", "bfpd": 1650, "intake_pressure_psi": 42, "motor_temp_f": 290, "motor_amps": 70},
            {"date": "2026-05-25", "bfpd": 1580, "intake_pressure_psi": 36, "motor_temp_f": 295, "motor_amps": 72},
        ],
        "scada_recent": [], "dyno_cards": [],
        "notes": ["Pump rate has trended below POR over last 60 days."],
    }


def esp_gas_interference(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Midland Basin (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2021-08-22", "first_prod_date": "2021-11-15",
        "completion": {"lateral_length_ft": 10200, "stages": 50, "proppant_lb_per_ft": 2500,
                       "fluid_bbl_per_ft": 58, "formation": "Wolfcamp A"},
        "artificial_lift": {
            "type": "ESP", "installed_date": "2023-06-10",
            "pump_spec": {"model": "Centrilift FC-2200", "stages": 200,
                          "por_min_bfpd": 1500, "por_max_bfpd": 3000,
                          "motor_temp_max_f": 350, "motor_amps_nameplate": 60},
        },
        "production_history": hyperbolic_history(1300, 0.004, 0.9, STANDARD_DAYS),
        "esp_readings": [
            {"date": "2026-05-21", "bfpd": 2100, "intake_pressure_psi": 18, "motor_temp_f": 320, "motor_amps": 55},
            {"date": "2026-05-25", "bfpd": 1980, "intake_pressure_psi": 14, "motor_temp_f": 325, "motor_amps": 52},
        ],
        "scada_recent": [], "dyno_cards": [],
        "notes": ["High GLR field. Casing pressure trending up.",
                  "Gas separator not currently installed."],
    }


def esp_scale(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Delaware Basin (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2021-05-01", "first_prod_date": "2021-08-01",
        "completion": {"lateral_length_ft": 10800, "stages": 54, "proppant_lb_per_ft": 2400,
                       "fluid_bbl_per_ft": 55, "formation": "Bone Spring 3rd"},
        "artificial_lift": {
            "type": "ESP", "installed_date": "2023-09-15",
            "pump_spec": {"model": "REDA 538-D2700N", "stages": 180,
                          "por_min_bfpd": 1800, "por_max_bfpd": 3400,
                          "motor_temp_max_f": 350, "motor_amps_nameplate": 65},
        },
        "production_history": underperform(hyperbolic_history(1200, 0.0038, 0.88, STANDARD_DAYS), 810, 0.70),
        "esp_readings": [
            {"date": "2026-05-21", "bfpd": 2200, "intake_pressure_psi": 95, "motor_temp_f": 330, "motor_amps": 82},
            {"date": "2026-05-25", "bfpd": 2150, "intake_pressure_psi": 92, "motor_temp_f": 335, "motor_amps": 85},
        ],
        "scada_recent": [], "dyno_cards": [],
        "notes": ["High calcium carbonate scale in field. Last scale inhibitor batch 4 months ago.",
                  "Amps trending up over last 90 days."],
    }


def esp_normal(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Delaware Basin (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2023-02-10", "first_prod_date": "2023-05-01",
        "completion": {"lateral_length_ft": 10500, "stages": 52, "proppant_lb_per_ft": 2400,
                       "fluid_bbl_per_ft": 55, "formation": "Wolfcamp A"},
        "artificial_lift": {
            "type": "ESP", "installed_date": "2024-08-01",
            "pump_spec": {"model": "REDA 538-D2700N", "stages": 180,
                          "por_min_bfpd": 1800, "por_max_bfpd": 3400,
                          "motor_temp_max_f": 350, "motor_amps_nameplate": 65},
        },
        "production_history": hyperbolic_history(1150, 0.0032, 0.92, STANDARD_DAYS),
        "esp_readings": [
            {"date": "2026-05-21", "bfpd": 2400, "intake_pressure_psi": 130, "motor_temp_f": 285, "motor_amps": 62},
            {"date": "2026-05-25", "bfpd": 2380, "intake_pressure_psi": 128, "motor_temp_f": 286, "motor_amps": 63},
        ],
        "scada_recent": [], "dyno_cards": [],
        "notes": ["Performing on type curve."],
    }


def esp_to_beam_candidate(idx: int) -> dict:
    well = esp_below_por(idx)
    well["production_history"] = underperform(hyperbolic_history(950, 0.004, 0.85, STANDARD_DAYS), 540, 0.5)
    well["esp_readings"] = [
        {"date": "2026-05-21", "bfpd": 950, "intake_pressure_psi": 75, "motor_temp_f": 305, "motor_amps": 45},
        {"date": "2026-05-25", "bfpd": 920, "intake_pressure_psi": 73, "motor_temp_f": 308, "motor_amps": 44},
    ]
    well["notes"] = ["Rate well below ESP POR floor. End of ESP economic life. Adjacent wells converted to beam pumps successfully."]
    return well


def beam_pump_pumpoff(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Permian conventional (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2015-03-01", "first_prod_date": "2015-06-01",
        "completion": {"lateral_length_ft": 0, "stages": 0, "proppant_lb_per_ft": 0,
                       "fluid_bbl_per_ft": 0, "formation": "San Andres"},
        "artificial_lift": {
            "type": "Beam Pump", "installed_date": "2019-04-10",
            "pump_spec": {"model": "C-228D-200-74", "stroke_in": 74, "spm": 6.5,
                          "plunger_in": 1.75},
        },
        "production_history": hyperbolic_history(120, 0.001, 0.5, STANDARD_DAYS, noise=0.08),
        "esp_readings": [], "scada_recent": [],
        "dyno_cards": [{"date": "2026-05-25",
                        "pattern": "Fluid pound — incomplete pump fillage, sharp downstroke drop.",
                        "fillage_pct": 55}],
        "notes": ["Cycles trending up. Operator suspects pump-off. Last dyno: severe fluid pound."],
    }


def beam_pump_normal(idx: int) -> dict:
    w = beam_pump_pumpoff(idx)
    w["dyno_cards"] = [{"date": "2026-05-25", "pattern": "Full pump fillage, healthy stroke profile.", "fillage_pct": 95}]
    w["notes"] = ["Well performing normally. Annual integrity check due."]
    return w


def beam_pump_parted_rods(idx: int) -> dict:
    w = beam_pump_pumpoff(idx)
    w["production_history"] = underperform(w["production_history"], 900, 0.05)
    w["dyno_cards"] = [{"date": "2026-05-25", "pattern": "Flat card — no fluid load. Likely parted rod string.", "fillage_pct": 5}]
    w["notes"] = ["Sudden production drop to near-zero. Dyno indicates parted rods. Rig needed."]
    return w


def gas_lift_liquid_loading(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "GoM Offshore (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2018-09-01", "first_prod_date": "2019-02-15",
        "completion": {"lateral_length_ft": 0, "stages": 0, "proppant_lb_per_ft": 0,
                       "fluid_bbl_per_ft": 0, "formation": "Pliocene"},
        "artificial_lift": {
            "type": "Gas Lift", "installed_date": "2019-02-15",
            "pump_spec": {"injection_rate_mscfd_design": 1500, "valve_count": 6},
        },
        "production_history": underperform(hyperbolic_history(800, 0.0028, 0.8, STANDARD_DAYS), 720, 0.55),
        "esp_readings": [], "scada_recent": [], "dyno_cards": [],
        "notes": ["Slugging behavior at surface. THP cycling. Suspected liquid loading.",
                  "Injection rate may be below critical Turner velocity."],
    }


def gas_lift_normal(idx: int) -> dict:
    w = gas_lift_liquid_loading(idx)
    w["production_history"] = hyperbolic_history(750, 0.003, 0.85, STANDARD_DAYS)
    w["notes"] = ["Stable production, on type curve. Routine surveillance."]
    return w


def plunger_lift_sticking(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Eagle Ford (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2017-11-01", "first_prod_date": "2018-02-01",
        "completion": {"lateral_length_ft": 7500, "stages": 30, "proppant_lb_per_ft": 1800,
                       "fluid_bbl_per_ft": 40, "formation": "Eagle Ford"},
        "artificial_lift": {
            "type": "Plunger Lift", "installed_date": "2020-06-01",
            "pump_spec": {"plunger_type": "Bypass", "cycles_per_day_target": 24},
        },
        "production_history": underperform(hyperbolic_history(300, 0.002, 0.7, STANDARD_DAYS), 810, 0.6),
        "esp_readings": [], "scada_recent": [], "dyno_cards": [],
        "notes": ["Cycle count down 40%. Plunger arrival times erratic.",
                  "Suspected plunger sticking — paraffin buildup likely."],
    }


def low_recovery_p_and_a_candidate(idx: int) -> dict:
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Permian marginal (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2005-01-15", "first_prod_date": "2005-04-01",
        "completion": {"lateral_length_ft": 0, "stages": 0, "proppant_lb_per_ft": 0,
                       "fluid_bbl_per_ft": 0, "formation": "Spraberry"},
        "artificial_lift": {
            "type": "Beam Pump", "installed_date": "2010-01-01",
            "pump_spec": {"model": "C-114D-143-64", "stroke_in": 64, "spm": 5,
                          "plunger_in": 1.5},
        },
        "production_history": [{"day": d, "oil_bopd": 3.5 + np.random.normal(0, 0.4),
                                "water_bwpd": 90, "gas_mcfd": 8} for d in STANDARD_DAYS],
        "esp_readings": [], "scada_recent": [],
        "dyno_cards": [{"date": "2026-05-25", "pattern": "Low fluid load, marginal economics.", "fillage_pct": 70}],
        "notes": ["20+ year old well. <5 bopd for 6 months. Workover OPEX exceeds revenue.",
                  "P&A candidate."],
    }


def acid_stim_candidate(idx: int) -> dict:
    """Like well_001 — classic acid stim signature."""
    return {
        "well_id": f"ED-{idx:03d}H", "api_number": f"42-109-{10000 + idx}",
        "field": "Delaware Basin (synthetic)", "operator": "Synthetic Operator LLC",
        "spud_date": "2021-12-01", "first_prod_date": "2022-03-15",
        "completion": {"lateral_length_ft": 10500, "stages": 52, "proppant_lb_per_ft": 2400,
                       "fluid_bbl_per_ft": 55, "formation": "Wolfcamp A"},
        "artificial_lift": {
            "type": "ESP", "installed_date": "2023-08-01",
            "pump_spec": {"model": "REDA 538-D2700N", "stages": 180,
                          "por_min_bfpd": 1800, "por_max_bfpd": 3400,
                          "motor_temp_max_f": 350, "motor_amps_nameplate": 65},
        },
        "production_history": underperform(hyperbolic_history(1200, 0.004, 0.9, STANDARD_DAYS), 810, 0.72),
        "esp_readings": [
            {"date": "2026-05-21", "bfpd": 1700, "intake_pressure_psi": 38, "motor_temp_f": 300, "motor_amps": 78},
            {"date": "2026-05-25", "bfpd": 1640, "intake_pressure_psi": 32, "motor_temp_f": 305, "motor_amps": 81},
        ],
        "scada_recent": [], "dyno_cards": [],
        "notes": ["Scale signature: pressure decline + amp increase. 3 months since last scale treatment.",
                  "Adjacent wells responded well to diverted acid (+150 bopd avg)."],
    }


# ---------- generate ---------------------------------------------------------

# Each tuple: (builder, count, expected_recommendation, keywords)
SCENARIOS = [
    (esp_below_por,                 3, "esp swap",                ["below POR", "downthrust"]),
    (esp_gas_interference,          3, "gas separator",           ["low intake pressure", "gas interference"]),
    (esp_scale,                     2, "scale treatment",         ["scale", "high amps"]),
    (esp_normal,                    2, "monitor",                 ["on type curve", "in POR"]),
    (esp_to_beam_candidate,         2, "esp-to-beam conversion",  ["below POR", "end of ESP"]),
    (beam_pump_pumpoff,             1, "pump-off controller",     ["fluid pound", "pump-off"]),
    (beam_pump_normal,              1, "monitor",                 ["normal", "fillage"]),
    (beam_pump_parted_rods,         1, "workover",                ["parted rods", "rig"]),
    (gas_lift_liquid_loading,       1, "gas lift optimization",   ["liquid loading", "slugging"]),
    (gas_lift_normal,               1, "monitor",                 ["stable", "on type curve"]),
    (plunger_lift_sticking,         1, "paraffin treatment",      ["plunger", "paraffin"]),
    (low_recovery_p_and_a_candidate, 1, "p&a",                    ["uneconomic", "p&a candidate"]),
    (acid_stim_candidate,           0, "", []),  # idx 1 already exists
]


def main():
    cases_yaml_lines = ["# Auto-generated by data/synthetic/generate.py", "cases:"]
    cases_yaml_lines.append(
        "  - id: case_001\n"
        "    well_file: data/synthetic/well_001.json\n"
        "    expected_primary_recommendation: acid_stimulation\n"
        "    expected_diagnosis_keywords: [\"below POR\", \"low intake pressure\", \"scale\"]\n"
        "    notes: hand-crafted seed case"
    )

    next_idx = 2
    for builder, count, rec, keywords in SCENARIOS:
        for _ in range(count):
            well = builder(next_idx)
            path = OUT / f"well_{next_idx:03d}.json"
            with path.open("w") as f:
                json.dump(well, f, indent=2)
            cases_yaml_lines.append(
                f"  - id: case_{next_idx:03d}\n"
                f"    well_file: data/synthetic/well_{next_idx:03d}.json\n"
                f"    expected_primary_recommendation: {rec.replace(' ', '_')}\n"
                f"    expected_diagnosis_keywords: {json.dumps(keywords)}\n"
                f"    notes: {builder.__name__}"
            )
            next_idx += 1

    cases_path = OUT.parent.parent / "evals" / "cases.yaml"
    cases_path.write_text("\n".join(cases_yaml_lines) + "\n")
    print(f"Wrote {next_idx - 1} wells. Updated {cases_path}.")


if __name__ == "__main__":
    main()
