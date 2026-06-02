"""Quick economics: NPV, payout, $/BOE for intervention candidates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterventionEconomics:
    name: str
    treatment_cost_usd: float
    incremental_eur_bbl: float
    incremental_first_year_bbl: float
    npv_10pct_usd: float
    payout_months: float
    profitability_index: float  # discounted PV of inflows / investment (>1.0 = value-accretive)


def evaluate_intervention(
    name: str,
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    uplift_decline_per_yr: float = 0.6,
    horizon_years: int = 5,
    realized_price_per_bbl: float = 65.0,
    discount_rate: float = 0.10,
    opex_per_bbl: float = 12.0,
) -> InterventionEconomics:
    """Simple NPV of an intervention assuming exponential decline of the uplift."""
    days_per_month = 365.25 / 12  # avoid the 360-day-year undercount
    months = np.arange(1, horizon_years * 12 + 1)
    monthly_rate = incremental_rate_bopd * np.exp(-uplift_decline_per_yr * (months / 12))
    monthly_vol = monthly_rate * days_per_month  # bbl/month
    monthly_revenue = monthly_vol * (realized_price_per_bbl - opex_per_bbl)
    discount_factors = (1 + discount_rate / 12) ** months
    npv = float(np.sum(monthly_revenue / discount_factors) - treatment_cost_usd)

    # Payout: cumulative[i] is the cumulative net revenue at the END of month i+1,
    # so the recovery month is the 1-based (payout_idx + 1).
    cumulative = np.cumsum(monthly_revenue)
    payout_idx = int(np.searchsorted(cumulative, treatment_cost_usd))
    payout_months = float(payout_idx + 1) if payout_idx < len(months) else float("inf")

    first_year_bbl = float(monthly_vol[:12].sum())
    eur = float(monthly_vol.sum())
    # Discounted profitability index = PV of inflows / investment (NOT an IRR / rate of return).
    pi = (npv + treatment_cost_usd) / treatment_cost_usd if treatment_cost_usd > 0 else 0.0

    return InterventionEconomics(
        name=name,
        treatment_cost_usd=treatment_cost_usd,
        incremental_eur_bbl=eur,
        incremental_first_year_bbl=first_year_bbl,
        npv_10pct_usd=npv,
        payout_months=payout_months,
        profitability_index=float(pi),
    )


def _npv_payout_vectorized(
    treatment_cost_usd: float,
    incremental_rate_bopd: np.ndarray,
    uplift_decline_per_yr: np.ndarray,
    realized_price_per_bbl: np.ndarray,
    horizon_years: int = 5,
    discount_rate: float = 0.10,
    opex_per_bbl: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized NPV + payout-month across many Monte-Carlo draws.

    Each input array has shape (n_trials,). Returns (npv[n], payout_months[n]) and
    uses exactly the same monthly-decline / discount math as evaluate_intervention()
    so the deterministic and stochastic paths can never diverge.
    """
    n = incremental_rate_bopd.shape[0]
    days_per_month = 365.25 / 12
    months = np.arange(1, horizon_years * 12 + 1)                  # (M,)
    # (n, M) monthly rate: exp(-decline * year_fraction), decline varies per trial.
    rate = incremental_rate_bopd[:, None] * np.exp(
        -uplift_decline_per_yr[:, None] * (months[None, :] / 12)
    )
    monthly_vol = rate * days_per_month                            # bbl/month, (n, M)
    margin = (realized_price_per_bbl - opex_per_bbl)[:, None]      # (n, 1)
    monthly_revenue = monthly_vol * margin                         # (n, M)
    discount_factors = (1 + discount_rate / 12) ** months          # (M,)
    npv = np.sum(monthly_revenue / discount_factors[None, :], axis=1) - treatment_cost_usd

    # Payout: first month where cumulative net (undiscounted) revenue >= cost.
    cumulative = np.cumsum(monthly_revenue, axis=1)                # (n, M)
    reached = cumulative >= treatment_cost_usd
    any_reached = reached.any(axis=1)
    first_idx = np.argmax(reached, axis=1)                         # 0 where never reached
    payout = np.where(any_reached, first_idx + 1, np.inf).astype(float)
    return npv, payout


def simulate_intervention(
    name: str,
    treatment_cost_usd: float,
    incremental_rate_bopd: float,
    uplift_decline_per_yr: float = 0.6,
    horizon_years: int = 5,
    realized_price_per_bbl: float = 65.0,
    discount_rate: float = 0.10,
    opex_per_bbl: float = 12.0,
    n_trials: int = 10_000,
    rate_cv: float = 0.30,
    decline_abs_sd: float = 0.15,
    price_sd: float = 12.0,
    payout_cutoff_months: float = 24.0,
    seed: int | None = 42,
) -> dict:
    """Monte-Carlo of intervention economics over uncertain inputs.

    Uncertainty model (per the engineering judgement these are estimated under):
      - incremental_rate_bopd : lognormal, ~±30% (rate_cv) about the point estimate
                                (can't go negative, right-skewed like real uplift).
      - uplift_decline_per_yr : normal, sd = decline_abs_sd (0.15 absolute), clipped >= 0.
      - realized_price_per_bbl: normal, sd = price_sd ($12), clipped to a sane floor.

    Returns P10/P50/P90 NPV, probability_of_payout (NPV>0 AND payout < cutoff),
    and a one-at-a-time tornado dict (low/high NPV swing per variable, others held
    at their point estimate). P-naming follows reserves convention: P10 = optimistic
    (high) NPV, P90 = conservative (low) NPV.
    """
    rng = np.random.default_rng(seed)

    # Lognormal for rate: choose mu/sigma so the MEAN equals the point estimate.
    sigma = np.sqrt(np.log(1 + rate_cv ** 2))
    mu = np.log(max(incremental_rate_bopd, 1e-9)) - 0.5 * sigma ** 2
    rate_draws = rng.lognormal(mean=mu, sigma=sigma, size=n_trials)

    decline_draws = np.clip(
        rng.normal(uplift_decline_per_yr, decline_abs_sd, size=n_trials), 0.0, None
    )
    price_draws = np.clip(
        rng.normal(realized_price_per_bbl, price_sd, size=n_trials),
        opex_per_bbl + 1.0, None,  # never below opex+$1 (a non-economic price is nonsense here)
    )

    npv, payout = _npv_payout_vectorized(
        treatment_cost_usd, rate_draws, decline_draws, price_draws,
        horizon_years=horizon_years, discount_rate=discount_rate, opex_per_bbl=opex_per_bbl,
    )

    p90, p50, p10 = (float(x) for x in np.percentile(npv, [10, 50, 90]))  # P90=low, P10=high
    prob_payout = float(np.mean((npv > 0) & (payout < payout_cutoff_months)))

    # ---- tornado: one-at-a-time low/high, others at point estimate -------------
    def _scalar_npv(rate, decline, price):
        npv1, _ = _npv_payout_vectorized(
            treatment_cost_usd, np.array([rate]), np.array([decline]), np.array([price]),
            horizon_years=horizon_years, discount_rate=discount_rate, opex_per_bbl=opex_per_bbl,
        )
        return float(npv1[0])

    base = (incremental_rate_bopd, uplift_decline_per_yr, realized_price_per_bbl)
    # Use the P10/P90 draw quantiles of each variable as its low/high endpoints.
    rate_lo, rate_hi = (float(x) for x in np.percentile(rate_draws, [10, 90]))
    dec_lo, dec_hi = (float(x) for x in np.percentile(decline_draws, [10, 90]))
    prc_lo, prc_hi = (float(x) for x in np.percentile(price_draws, [10, 90]))

    tornado = {
        "incremental_rate_bopd": {
            "low_input": rate_lo, "high_input": rate_hi,
            # lower rate -> lower NPV, higher rate -> higher NPV
            "low_npv": _scalar_npv(rate_lo, base[1], base[2]),
            "high_npv": _scalar_npv(rate_hi, base[1], base[2]),
        },
        "uplift_decline_per_yr": {
            "low_input": dec_lo, "high_input": dec_hi,
            # FASTER decline (high) -> lower NPV; report swing endpoints honestly
            "low_npv": _scalar_npv(base[0], dec_lo, base[2]),
            "high_npv": _scalar_npv(base[0], dec_hi, base[2]),
        },
        "realized_price_per_bbl": {
            "low_input": prc_lo, "high_input": prc_hi,
            "low_npv": _scalar_npv(base[0], base[1], prc_lo),
            "high_npv": _scalar_npv(base[0], base[1], prc_hi),
        },
    }
    for v in tornado.values():
        v["swing"] = abs(v["high_npv"] - v["low_npv"])

    return {
        "name": name,
        "n_trials": int(n_trials),
        "treatment_cost_usd": float(treatment_cost_usd),
        "npv_p90_usd": p90,   # conservative
        "npv_p50_usd": p50,   # median
        "npv_p10_usd": p10,   # optimistic
        "npv_mean_usd": float(np.mean(npv)),
        "probability_of_payout": prob_payout,  # NPV>0 AND payout < cutoff
        "payout_cutoff_months": float(payout_cutoff_months),
        "tornado": tornado,
        "assumptions": {
            "rate_cv": rate_cv,
            "decline_abs_sd": decline_abs_sd,
            "price_sd": price_sd,
        },
    }
