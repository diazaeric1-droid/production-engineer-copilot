"""Arps decline curve analysis (exponential, harmonic, hyperbolic)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit


DeclineModel = Literal["exponential", "harmonic", "hyperbolic"]


@dataclass
class DeclineFit:
    model: DeclineModel
    qi: float            # initial rate (bbl/d or mcf/d)
    di: float            # initial decline rate (1/day)
    b: float             # hyperbolic exponent (0 = exp, 1 = harmonic)
    r_squared: float
    last_actual: float
    last_predicted: float
    deviation_pct: float  # negative = underperforming type curve


def _exponential(t, qi, di):
    return qi * np.exp(-di * t)


def _harmonic(t, qi, di):
    return qi / (1 + di * t)


def _hyperbolic(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


def fit_decline(
    days: np.ndarray,
    rates: np.ndarray,
    model: DeclineModel = "hyperbolic",
) -> DeclineFit:
    """Fit an Arps decline model to rate-time data."""
    days = np.asarray(days, dtype=float)
    rates = np.asarray(rates, dtype=float)
    mask = (rates > 0) & np.isfinite(rates)
    days, rates = days[mask], rates[mask]

    if len(days) < 5:
        raise ValueError("Need at least 5 valid production points to fit decline.")

    qi_guess = rates[0]

    if model == "exponential":
        popt, _ = curve_fit(_exponential, days, rates, p0=[qi_guess, 0.001], maxfev=5000)
        qi, di, b = popt[0], popt[1], 0.0
        predicted = _exponential(days, qi, di)
    elif model == "harmonic":
        popt, _ = curve_fit(_harmonic, days, rates, p0=[qi_guess, 0.001], maxfev=5000)
        qi, di, b = popt[0], popt[1], 1.0
        predicted = _harmonic(days, qi, di)
    else:
        popt, _ = curve_fit(
            _hyperbolic, days, rates,
            p0=[qi_guess, 0.001, 0.5],
            bounds=([0, 0, 0], [np.inf, 1, 2]),
            maxfev=5000,
        )
        qi, di, b = popt
        predicted = _hyperbolic(days, qi, di, b)

    ss_res = np.sum((rates - predicted) ** 2)
    ss_tot = np.sum((rates - rates.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    deviation = (rates[-1] - predicted[-1]) / predicted[-1] * 100 if predicted[-1] > 0 else 0

    return DeclineFit(
        model=model,
        qi=float(qi),
        di=float(di),
        b=float(b),
        r_squared=float(r_squared),
        last_actual=float(rates[-1]),
        last_predicted=float(predicted[-1]),
        deviation_pct=float(deviation),
    )


def project_eur(fit: DeclineFit, economic_limit_bopd: float = 5.0, horizon_days: int = 365 * 30) -> float:
    """Estimated ultimate recovery to economic limit (bbl)."""
    t = np.arange(1, horizon_days)
    if fit.model == "exponential":
        q = _exponential(t, fit.qi, fit.di)
    elif fit.model == "harmonic":
        q = _harmonic(t, fit.qi, fit.di)
    else:
        q = _hyperbolic(t, fit.qi, fit.di, fit.b)
    above_limit = q[q >= economic_limit_bopd]
    return float(above_limit.sum())
