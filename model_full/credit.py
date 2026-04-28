"""
credit.py — Phase-2b commercial-bank credit market.

Pure functions. The credit market sits between the real-sector demand
for financing (firms and energy producers wanting to cover investment
and working capital) and the balance-sheet capacity of the banking
sector (bounded by a Basel-style capital ratio).

Closures implemented:

  * loan_demand_F            — firm credit demand this period
  * loan_demand_E            — energy-producer credit demand this period
  * bank_lending_capacity    — aggregate new-lending headroom of banks
  * allocate_new_loans       — proportional rationing if demand > capacity

Convention: all flows are nominal, non-negative. Amortisation is
handled upstream as a fixed fraction of outstanding principal (see
dynamics.make_phase2b_flows).
"""

from __future__ import annotations

import numpy as np


def bank_leverage(L_F: np.ndarray,
                  L_E: np.ndarray,
                  NW_B: np.ndarray) -> np.ndarray:
    """
    Per-bank book leverage λ_b = (L_F_b + L_E_b) / NW_B_b.

    Guarded against zero/negative NW (returns a large but finite value
    so the risk-premium formula saturates at its ceiling rather than
    blowing up numerically).
    """
    loans = np.asarray(L_F, dtype=float) + np.asarray(L_E, dtype=float)
    nw = np.asarray(NW_B, dtype=float)
    safe_nw = np.where(nw > 1e-6, nw, 1e-6)
    lev = loans / safe_nw
    # Clamp at an upper bound consistent with the ceiling used in the
    # pricing rule. Higher leverage than λ_max simply saturates the
    # risk-premium term.
    return np.clip(lev, 0.0, 1e6)


def loan_rate_risk_adjusted(lev: np.ndarray,
                            r_CB: float,
                            mu_b: float,
                            delta_bar: float,
                            gamma: float,
                            lambda_max: float) -> np.ndarray:
    """
    Bank-specific loan-pricing rule (paper §3.6 Eq. 14, gap-closure
    Stage 3.1).

        r_L_b = r_CB + μ_b + δ̄ · (λ_b / λ_max)^γ

    Decomposition:
      - r_CB: central-bank rate (system-wide funding cost)
      - μ_b:  base markup over CB rate (operating cost + competitive rent)
      - δ̄:    maximum risk-premium at the leverage ceiling
      - γ:    convexity of the risk premium (γ = 2: quadratic)
      - λ_max: regulatory leverage ceiling (rates saturate here)

    With δ̄ = 0 the rule collapses to a flat r_CB + μ_b, which can
    reproduce the pre-Stage-3.1 scalar r_L when calibrated.
    """
    lev_arr = np.asarray(lev, dtype=float)
    lam_max = max(float(lambda_max), 1e-6)
    ratio = np.minimum(lev_arr / lam_max, 1.0)   # cap at 1 ⇒ saturate
    risk_premium = float(delta_bar) * np.power(ratio, float(gamma))
    return float(r_CB) + float(mu_b) + risk_premium


def loan_demand_F(I_F: float,
                  amort_F: float,
                  W_F: float,
                  working_capital_coverage: float) -> float:
    """
    Firm credit demand.

    Firms borrow to (a) finance net investment not covered by retained
    deposits and (b) cover a fraction `working_capital_coverage` of
    the wage bill ahead of sales revenue, plus (c) roll over the
    amortising share of the existing loan book.

    In this Phase-2b MVP we do not price-discriminate between the three
    components — we just sum them into a single gross loan request.
    The bank may then ration proportionally across sectors if its
    capital headroom is insufficient.
    """
    demand = I_F + amort_F + working_capital_coverage * W_F
    return max(0.0, float(demand))


def loan_demand_E(I_E: float,
                  amort_E: float,
                  W_E: float,
                  working_capital_coverage: float) -> float:
    """Energy-producer credit demand — symmetric to loan_demand_F."""
    demand = I_E + amort_E + working_capital_coverage * W_E
    return max(0.0, float(demand))


def bank_lending_capacity(bank_nw: float,
                          existing_assets: float,
                          target_capital_ratio: float) -> float:
    """
    Maximum new lending a bank sector can extend this period without
    breaching its capital ratio target.

    Capital ratio:  k = NW / total_assets.
    Target:         NW ≥ target_capital_ratio × total_assets.

    If new_loans are issued against fresh deposits, NW is unchanged
    by the act of lending itself. What changes is total assets (up by
    new_loans) and the associated increase in deposit liabilities.
    Solving  NW / (existing_assets + new_loans) ≥ k  for new_loans
    gives  new_loans ≤ NW / k - existing_assets.

    Interpretation: banks with high NW have more headroom; banks with
    existing_assets close to NW/k are already at their constraint and
    can't lend more this quarter.

    Returns a non-negative number. Zero (or negative, clamped) means
    the capital ratio is already binding — no new loans this quarter.
    """
    if target_capital_ratio <= 0.0:
        return float("inf")
    cap = bank_nw / target_capital_ratio - existing_assets
    return max(0.0, float(cap))


def risk_weighted_assets(L_F: float,
                         L_E: float,
                         GB: float,
                         Res: float,
                         rho_green: float = 1.0) -> float:
    """
    Risk-weighted bank assets used in the capital-ratio denominator
    (paper §3.6, gap-closure Stage 3.3).

    Green-preferential risk weight: E-sector loans are scaled by
    `rho_green ≤ 1`, reducing their contribution to the capital-ratio
    denominator. This is the model analogue of the EU taxonomy's
    "green supporting factor" or the BoE/BIS proposals for climate-
    aware prudential treatment.

    With `rho_green = 1` the function returns unmodified total assets
    (pre-Stage-3.3 behaviour). With `rho_green < 1` banks can hold
    more E-sector loans for the same NW before hitting the capital
    ratio — a regulatory nudge toward green financing.
    """
    return (float(L_F)
            + float(rho_green) * float(L_E)
            + float(GB)
            + float(Res))


def concentration_headroom(existing_L_sector: float,
                           aggregate_NW_B: float,
                           beta: float) -> float:
    """
    Sector-level concentration-limit headroom (paper §3.6 Eq. 13b,
    gap-closure Stage 3.2).

    Basel-style large-exposure rule: bank exposure to any single
    counterparty capped at β · NW_b. Aggregated across the banking
    sector, total new lending to sector s is bounded by
        (β · ΣNW_b) − existing_L_s
    clipped at zero. Returns the remaining headroom.

    When `beta ≤ 0` the limit is deactivated (returns +∞) so the
    pre-Stage-3.2 unrestricted behaviour is preserved for backward-
    compat regression.
    """
    if beta <= 0.0:
        return float("inf")
    cap = beta * float(aggregate_NW_B) - float(existing_L_sector)
    return max(0.0, cap)


def allocate_new_loans(demand_F: float,
                       demand_E: float,
                       capacity: float,
                       headroom_F: float = float("inf"),
                       headroom_E: float = float("inf")
                       ) -> tuple[float, float, float]:
    """
    Proportional rationing of aggregate loan demand against bank
    capacity, with optional per-sector concentration-limit headrooms.

    Sequencing:
      1. Each sector's effective demand is the lesser of the nominal
         demand and the concentration headroom (Stage 3.2 gating).
         If `headroom_F = +∞` (β = 0 ⇒ no concentration limit) the
         nominal demand passes through unchanged.
      2. The combined effective demand is then rationed proportionally
         against aggregate bank capacity if capacity-binding.

    Returns (new_loans_F, new_loans_E, rationing_ratio). The ratio is
    the capacity-rationing fraction — it does NOT include any concentration
    cuts that happened in step 1. For a combined "how much of the
    original demand was met" metric, compare
    (new_F + new_E) / (demand_F + demand_E) externally.
    """
    # Step 1: concentration-cap each sector.
    eff_demand_F = min(float(demand_F), float(headroom_F))
    eff_demand_E = min(float(demand_E), float(headroom_E))
    eff_demand_F = max(0.0, eff_demand_F)
    eff_demand_E = max(0.0, eff_demand_E)

    # Step 2: capacity rationing on the concentration-adjusted demands.
    total_demand = eff_demand_F + eff_demand_E
    if total_demand <= 0.0:
        return 0.0, 0.0, 1.0
    if total_demand <= capacity:
        return float(eff_demand_F), float(eff_demand_E), 1.0
    ratio = capacity / total_demand if total_demand > 0 else 0.0
    return (float(eff_demand_F * ratio),
            float(eff_demand_E * ratio),
            float(ratio))
