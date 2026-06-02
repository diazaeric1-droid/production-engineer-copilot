"""Arps decline curve analysis (exponential, harmonic, hyperbolic)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

# np.trapz was renamed to np.trapezoid in numpy 2.0 (trapz deprecated); support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz"))


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
    fit_residual_pct: float  # (last actual - full-fit value at last day) / full-fit value.
                             # Quality-of-fit residual, NOT a type-curve deviation —
                             # use analyze_type_curve() for that.


@dataclass
class TypeCurveResult:
    """Benchmark of actual production against a type curve fit on early/established
    decline and extrapolated forward. This is the honest 'is the well underperforming
    its type curve' answer: the curve is NOT dragged down by the degraded tail."""
    model: DeclineModel
    established_days: int          # number of early points the type curve was fit on
    qi: float
    di: float
    b: float
    last_actual: float
    type_curve_at_last: float      # extrapolated type-curve rate at the latest day
    deviation_pct: float           # negative = actual is BELOW type curve
    cum_actual_bbl: float
    cum_type_curve_bbl: float
    deferred_bbl: float            # type-curve cum - actual cum (positive = production left behind)
    deferred_pct: float
    deferred_value_usd: float      # deferred_bbl * net margin (price - opex)


def _exponential(t, qi, di):
    return qi * np.exp(-di * t)


def _harmonic(t, qi, di):
    return qi / (1 + di * t)


def _hyperbolic(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


def _curve(model: DeclineModel, qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
    """Evaluate the fitted Arps model at times t (pure numpy — no scipy)."""
    if model == "exponential":
        return _exponential(t, qi, di)
    if model == "harmonic":
        return _harmonic(t, qi, di)
    return _hyperbolic(t, qi, di, max(b, 1e-6))


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

    residual = (rates[-1] - predicted[-1]) / predicted[-1] * 100 if predicted[-1] > 0 else 0

    return DeclineFit(
        model=model,
        qi=float(qi),
        di=float(di),
        b=float(b),
        r_squared=float(r_squared),
        last_actual=float(rates[-1]),
        last_predicted=float(predicted[-1]),
        fit_residual_pct=float(residual),
    )


def analyze_type_curve(
    days: np.ndarray,
    rates: np.ndarray,
    model: DeclineModel = "hyperbolic",
    established_frac: float = 0.4,
    price_per_bbl: float = 65.0,
    opex_per_bbl: float = 12.0,
) -> TypeCurveResult:
    """Fit a type curve on the EARLY/established portion of the well's history and
    extrapolate it forward, then measure how far actual production has fallen below
    that curve (rate deviation today + cumulative deferred production).

    Fitting only early data is the key difference from fit_decline(): a curve fit
    over the whole history is dragged down by the degraded tail, so a rolling-over
    well looks 'on type curve'. Here the type curve reflects what the well *should*
    be doing.
    """
    days = np.asarray(days, dtype=float)
    rates = np.asarray(rates, dtype=float)
    mask = (rates > 0) & np.isfinite(rates)
    days, rates = days[mask], rates[mask]
    if len(days) < 6:
        raise ValueError("Need at least 6 valid production points for type-curve analysis.")

    n_est = max(5, int(round(len(days) * established_frac)))
    n_est = min(n_est, len(days) - 1)  # always hold out at least one point to evaluate
    fit = fit_decline(days[:n_est], rates[:n_est], model=model)

    tc = _curve(model, fit.qi, fit.di, fit.b, days)
    tc_last = float(tc[-1])
    deviation = (rates[-1] - tc_last) / tc_last * 100 if tc_last > 0 else 0.0

    cum_actual = float(_trapezoid(rates, days))
    cum_tc = float(_trapezoid(tc, days))
    deferred = cum_tc - cum_actual
    deferred_pct = (deferred / cum_tc * 100) if cum_tc > 0 else 0.0
    deferred_value = max(0.0, deferred) * (price_per_bbl - opex_per_bbl)

    return TypeCurveResult(
        model=model,
        established_days=int(n_est),
        qi=float(fit.qi),
        di=float(fit.di),
        b=float(fit.b),
        last_actual=float(rates[-1]),
        type_curve_at_last=tc_last,
        deviation_pct=float(deviation),
        cum_actual_bbl=cum_actual,
        cum_type_curve_bbl=cum_tc,
        deferred_bbl=float(deferred),
        deferred_pct=float(deferred_pct),
        deferred_value_usd=float(deferred_value),
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
