"""
investment.py — Phase-2b investment rules.

Pure functions. Two simple rules that activate the I_F and I_E flows
and produce endogenous capital-stock dynamics without yet introducing
the profitability-weighted reinvestment that drives decarbonisation
(that lands in Phase 2c).

Rules implemented:

  * firm_investment    — I_F = i_F × nominal F-output
  * energy_replacement — I_E = δ_E × total K_E (pure replacement)
  * depreciation       — δ × K (both sectors)

All flows are non-negative nominal (in K-units, since K is stored 1:1
with its nominal value on the balance sheet).
"""

from __future__ import annotations


def firm_investment(p_F: float,
                    Y_F: float,
                    i_rate_F: float) -> float:
    """
    Firm nominal investment rule.

    I_F = i_rate_F × p_F × Y_F

    A fraction `i_rate_F` of current nominal output is plowed back as
    new real capital. Simple, positive, and keeps K_F growing in line
    with demand. A more structural rule (profit-margin-weighted,
    capacity-utilisation-responsive) can slot in here in Phase 2c.
    """
    return max(0.0, float(i_rate_F * p_F * Y_F))


def energy_replacement_investment(K_E_total: float,
                                  depreciation_rate_E: float) -> float:
    """
    Aggregate energy-sector replacement investment.

    I_E_total = δ_E × K_E_total

    Phase-2b assumption: capacity is kept constant by exactly replacing
    depreciated units. No expansion, no technology churn. The tech mix
    therefore stays fixed through Phase 2b, which is the point —
    Phase 2c introduces the profitability-weighted reinvestment rule
    that actually changes the mix.
    """
    return max(0.0, float(depreciation_rate_E * K_E_total))


def depreciation(K_total: float, rate: float) -> float:
    """Depreciation flow = rate × K. Non-negative."""
    return max(0.0, float(rate * K_total))
