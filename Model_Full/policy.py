"""
policy.py — Phase-2c policy instruments.

Three pure functions that together close the carbon-pricing loop:

  * carbon_tax_flow        — tax revenue and per-plant tax cost
  * reinvestment_weights   — profitability-weighted share of I_E per plant
  * lump_sum_rebate        — aggregate per-household rebate from revenue

All functions are stateless; they take arrays and scalars, return arrays
and scalars, and never mutate inputs. Units follow the Caiani convention
(see docs/units_convention.md): 1 unit of E = €1 of energy at 2019
prices, so emission factors are in tCO₂/unit_E and prices in €/unit_E.

Conventions
-----------
τ (carbon tax) is in **€/tCO₂ directly** — the EU ETS allowance price.
Under the convention `mc_effective = mc + τ · ef`, with ef in tCO₂/unit_E
and mc in €/unit_E, the product τ·ef has the same units as mc, so the
addition is well-defined. Reference EU ETS averages: ~25 (2019), ~85
(2022), ~83 (2023) €/tCO₂.

π_k (plant profit) is defined as producer surplus on the current-period
dispatch: π_k = y_k · (p_E - mc_effective_k), where mc_effective
already includes the carbon tax. This is the cleanest signal for a
reinvestment rule and it aligns the decarbonisation narrative into a
single mechanism: carbon tax ⇒ dirty plants' π_k collapses ⇒ they lose
share of new capacity ⇒ their K_E depreciates away ⇒ emissions fall.
"""

from __future__ import annotations

import numpy as np


def clear_ets_permit_market(demand: float,
                            mc: np.ndarray,
                            capacity: np.ndarray,
                            emission_factor: np.ndarray,
                            cap: float,
                            tau_max: float = 500.0,
                            tolerance: float = 0.01,
                            max_iter: int = 50) -> float:
    """
    Emissions-Trading-Scheme permit-price clearing (paper §5 Eq. 20,
    gap-closure Stage 5.1).

    Given an emissions cap (tCO₂ per period), bisect on τ ∈ [0, τ_max]
    to find the permit price that makes the period's dispatch-implied
    emissions equal the cap. The dispatch function is a pure call so
    this is a safe in-period root-finding routine.

    Returns τ (€/tCO₂).

    Edge cases:
      - At τ = 0 emissions already ≤ cap  ⇒ no tax needed; returns 0.
      - At τ = τ_max emissions still > cap ⇒ saturated; returns τ_max.
    """
    from energy_market import dispatch as _dispatch

    if cap <= 0.0 or demand <= 0.0:
        return 0.0

    def _emissions(tau: float) -> float:
        mc_eff = mc + tau * emission_factor
        _y, _p, em, _r = _dispatch(
            demand=demand, mc=mc_eff, capacity=capacity,
            emission_factor=emission_factor,
        )
        return float(em)

    em0 = _emissions(0.0)
    if em0 <= cap:
        return 0.0
    em_max = _emissions(float(tau_max))
    if em_max > cap:
        return float(tau_max)

    lo, hi = 0.0, float(tau_max)
    tau_mid = 0.0
    for _ in range(max_iter):
        tau_mid = 0.5 * (lo + hi)
        em_mid = _emissions(tau_mid)
        # Converged if emissions within tolerance of cap (relative).
        if abs(em_mid - cap) <= tolerance * cap:
            return tau_mid
        # emissions are monotonically decreasing in τ under merit order,
        # so if emissions > cap we need a higher τ.
        if em_mid > cap:
            lo = tau_mid
        else:
            hi = tau_mid
    return tau_mid


def carbon_tax_flow(y_per_plant: np.ndarray,
                    emission_factor: np.ndarray,
                    tau: float) -> tuple[float, np.ndarray]:
    """
    Carbon-tax flow on the current period's dispatch.

    Each plant pays τ · ef_k · y_k to the government. Total tax revenue
    is the sum across plants.

    Parameters
    ----------
    y_per_plant : (N_E,) array
        Real energy output dispatched from each plant this period.
    emission_factor : (N_E,) array
        tCO2 per unit of output for each plant.
    tau : float
        Carbon tax rate in model-price units per unit of emission
        factor. Must be non-negative.

    Returns
    -------
    total_revenue : float
        Aggregate tax flow from E to G this period (non-negative).
    per_plant_tax : (N_E,) array
        Per-plant tax cost (for diagnostics).
    """
    if tau <= 0.0 or y_per_plant.size == 0:
        return 0.0, np.zeros_like(y_per_plant, dtype=float)
    per_plant = np.maximum(0.0, tau * emission_factor * y_per_plant)
    return float(per_plant.sum()), per_plant


def reinvestment_weights(profit_per_plant: np.ndarray,
                         churn_alpha: float,
                         K_per_plant: np.ndarray | None = None
                         ) -> np.ndarray:
    """
    Profitability-weighted share of aggregate replacement investment.

        w_k = max(0, π_k)^α / Σ max(0, π_j)^α

    Plants with non-positive profits receive zero weight.

    **Status-quo fallback.** If ALL plants have non-positive (or zero)
    profit this period — degenerate case, e.g., at t=0 before any
    dispatch has happened, or later when the merit-order clearing price
    bottoms out and no plant earns producer surplus — we fall back to
    weights proportional to *current capital*:

        w_k = K_k / Σ_j K_j   (if `K_per_plant` is provided and > 0)

    With total replacement investment I_E = δ · K_E_total, this makes
    per-plant net growth exactly zero in the fallback case, preserving
    any heterogeneity that earlier profitability-weighted periods have
    produced. If `K_per_plant` is not provided (backward compatibility),
    we fall back to uniform weights.

    Parameters
    ----------
    profit_per_plant : (N_E,) array
        Last period's π_k for each plant. Can contain negatives.
    churn_alpha : float
        Sharpness of the reinvestment rule. α = 1 gives proportional
        reinvestment; α → ∞ concentrates everything on the best plant.
    K_per_plant : (N_E,) array, optional
        Current per-plant capital stock, used only for the status-quo
        fallback. When omitted, the fallback is uniform (legacy
        behaviour).

    Returns
    -------
    weights : (N_E,) array
        Non-negative, sums to 1.
    """
    N = profit_per_plant.shape[0]
    if N == 0:
        return np.zeros(0, dtype=float)

    def _fallback() -> np.ndarray:
        if K_per_plant is not None and K_per_plant.size == N:
            total_K = float(np.asarray(K_per_plant, dtype=float).sum())
            if total_K > 0.0:
                return np.asarray(K_per_plant, dtype=float) / total_K
        return np.full(N, 1.0 / N, dtype=float)

    pos = np.maximum(0.0, profit_per_plant)
    if pos.sum() <= 0.0:
        return _fallback()
    w = np.power(pos, churn_alpha)
    total = float(w.sum())
    if total <= 0.0:
        return _fallback()
    return w / total


def lump_sum_rebate(total_revenue: float, N_H: int) -> float:
    """
    Same-quarter lump-sum rebate of carbon-tax revenue to households.

    Phase-2c uses full recycling: the rebate flow equals the carbon-tax
    flow. Per-household share is (total_revenue / N_H) and is applied
    uniformly by the laws-of-motion helper. Partial rebate (with the
    residual used for a green subsidy) is a one-line extension.

    Returns
    -------
    rebate : float
        Aggregate rebate flow from G to H this period (non-negative).
    """
    if N_H <= 0 or total_revenue <= 0.0:
        return 0.0
    return float(total_revenue)
