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
    rate_of_return_pct: float


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
    months = np.arange(1, horizon_years * 12 + 1)
    monthly_rate = incremental_rate_bopd * np.exp(-uplift_decline_per_yr * (months / 12))
    monthly_vol = monthly_rate * 30  # bbl/month
    monthly_revenue = monthly_vol * (realized_price_per_bbl - opex_per_bbl)
    discount_factors = (1 + discount_rate / 12) ** months
    npv = float(np.sum(monthly_revenue / discount_factors) - treatment_cost_usd)

    # Payout
    cumulative = np.cumsum(monthly_revenue)
    payout_idx = np.searchsorted(cumulative, treatment_cost_usd)
    payout_months = float(payout_idx) if payout_idx < len(months) else float("inf")

    first_year_bbl = float(monthly_vol[:12].sum())
    eur = float(monthly_vol.sum())
    ror = (npv + treatment_cost_usd) / treatment_cost_usd * 100 if treatment_cost_usd > 0 else 0

    return InterventionEconomics(
        name=name,
        treatment_cost_usd=treatment_cost_usd,
        incremental_eur_bbl=eur,
        incremental_first_year_bbl=first_year_bbl,
        npv_10pct_usd=npv,
        payout_months=payout_months,
        rate_of_return_pct=float(ror),
    )
