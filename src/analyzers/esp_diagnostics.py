"""ESP health diagnostics — POR check, common failure-mode pattern recognition."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ESPDiagnostic:
    in_por: bool
    por_min_bfpd: float
    por_max_bfpd: float
    current_bfpd: float
    intake_pressure_psi: float
    motor_temp_f: float
    motor_amps: float
    flags: list[str]
    likely_issues: list[str]


def evaluate_esp(esp_readings: list[dict], pump_spec: dict) -> ESPDiagnostic:
    """Evaluate ESP health from recent readings against the pump's POR.

    pump_spec keys: model, stages, por_min_bfpd, por_max_bfpd,
                   motor_temp_max_f, motor_amps_nameplate
    """
    if not esp_readings:
        raise ValueError("No ESP readings provided.")

    latest = esp_readings[-1]
    bfpd = latest.get("bfpd", 0)
    intake_p = latest.get("intake_pressure_psi", 0)
    motor_t = latest.get("motor_temp_f", 0)
    amps = latest.get("motor_amps", 0)

    por_min = pump_spec["por_min_bfpd"]
    por_max = pump_spec["por_max_bfpd"]
    in_por = por_min <= bfpd <= por_max

    flags = []
    issues = []

    if bfpd < por_min:
        flags.append(f"BELOW POR ({bfpd:.0f} < {por_min:.0f} bfpd)")
        issues.append("Pump downthrust risk — consider VSD reduction or smaller pump")
    if bfpd > por_max:
        flags.append(f"ABOVE POR ({bfpd:.0f} > {por_max:.0f} bfpd)")
        issues.append("Pump upthrust risk — consider VSD boost or larger pump")

    if intake_p < 50:
        flags.append(f"LOW INTAKE PRESSURE ({intake_p:.0f} psi)")
        issues.append("Gas interference / pump-off — check gas separator, casing pressure")

    if motor_t > pump_spec.get("motor_temp_max_f", 350) * 0.95:
        flags.append(f"HIGH MOTOR TEMP ({motor_t:.0f} F)")
        issues.append("Cooling concern — possible scale, low rate, or motor degradation")

    amps_nominal = pump_spec.get("motor_amps_nameplate", 50)
    if amps > amps_nominal * 1.15:
        flags.append(f"HIGH AMPS ({amps:.0f} A vs {amps_nominal} nameplate)")
        issues.append("Mechanical load increase — possible scale, sand, or worn stages")
    if amps < amps_nominal * 0.6:
        flags.append(f"LOW AMPS ({amps:.0f} A vs {amps_nominal} nameplate)")
        issues.append("Possible gas locking or broken shaft")

    return ESPDiagnostic(
        in_por=in_por,
        por_min_bfpd=por_min,
        por_max_bfpd=por_max,
        current_bfpd=bfpd,
        intake_pressure_psi=intake_p,
        motor_temp_f=motor_t,
        motor_amps=amps,
        flags=flags,
        likely_issues=issues,
    )
