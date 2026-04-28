"""
accounting.py — TFM, BSM, laws of motion and the SFC residual check.

The residual check implements the redundant-equation principle of
Godley & Lavoie (2007): because any five of the six stock laws of
motion in v2.1 §3.10 can be derived from the sixth together with the
TFM and BSM constraints, we verify consistency by

  1. computing every stock at t+1 by direct integration of the flows
     that appear in the TFM column of its holder sector;
  2. verifying that the horizontal and vertical sums of the BSM hold
     exactly (up to floating-point noise);
  3. verifying the NW identity: sum of sectoral NW equals total real
     capital.

A period whose residual exceeds `RESIDUAL_TOL * GDP_proxy` raises
`SFCClosureError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from agents import Economy


RESIDUAL_TOL = 1e-10


class SFCClosureError(RuntimeError):
    """Raised when the SFC residual check fails."""


# --------------------------------------------------------------------------
# TFM — transaction flow matrix
# --------------------------------------------------------------------------
@dataclass
class TFMFlows:
    """
    Container for all nominal flows that occur during a single period.
    Every flow is a single non-negative number; the TFM signs are fixed
    by the cell in which the flow is recorded.

    The fields mirror Table 1 of v2.1. All quantities are already
    multiplied by price where applicable (i.e. they are nominal).
    """
    # Real-sector flows
    W:       float = 0.0   # total wages paid by F+E to H
    C_H_F:   float = 0.0   # nominal consumption of F-goods by H
    EB_H:    float = 0.0   # energy bill paid by H to E
    EB_F:    float = 0.0   # energy bill paid by F to E
    G_spend: float = 0.0   # gov purchase of F-goods

    # Tax / transfer flows
    T_H:    float = 0.0    # direct tax on H
    Tr_H:   float = 0.0    # transfer from G to H

    # Interest flows
    r_D_H:  float = 0.0    # interest on H deposits, from B to H
    r_D_F:  float = 0.0    # interest on F deposits, from B to F
    r_D_E:  float = 0.0    # interest on E deposits, from B to E
    r_L_F:  float = 0.0    # interest on F loans, from F to B
    r_L_E:  float = 0.0    # interest on E loans, from E to B
    r_GB_B: float = 0.0    # coupon on GB held by B, from G to B
    r_GB_CB: float = 0.0   # coupon on GB held by CB, from G to CB
    r_Res:  float = 0.0    # remuneration of reserves, from CB to B

    # Capital-account flows (phase 0 sets these all to zero)
    new_loans_F:  float = 0.0
    new_loans_E:  float = 0.0
    amort_F:      float = 0.0
    amort_E:      float = 0.0
    writeoff_F:   float = 0.0
    writeoff_E:   float = 0.0
    GB_issued:    float = 0.0     # new gov-bond issuance
    GB_amort:     float = 0.0
    I_F:          float = 0.0     # investment by F in real capital
    I_E:          float = 0.0     # investment by E in real capital
    depreciation_F: float = 0.0
    depreciation_E: float = 0.0
    dividends_F:  float = 0.0     # F→H
    dividends_E:  float = 0.0     # E→H (Phase-1 closure: energy-sector profits)
    dividends_B:  float = 0.0     # B→H

    # ---- Phase-2c policy flows (zero in phases 0-2b) ----
    carbon_tax:       float = 0.0   # E → G
    household_rebate: float = 0.0   # G → H
    # Stage-5.3 green-fiscal channel: direct government transfer to
    # E sector, financed by new green-bond issuance (Eq. 32). Zero by
    # default; set to a positive value in policy-experiment scenarios.
    green_subsidy_E:  float = 0.0   # G → E
    # ---- Stage-5.4: price-support instruments ----
    # Feed-in tariff: fixed €/unit_E paid to supported techs on top of
    # whatever merit-order clears at. G→E, always ≥ 0. Financed via
    # additional GB issuance in the same G-column accounting.
    fit_payment:      float = 0.0   # G → E
    # Contract for Difference (CfD): symmetric hedge against market
    # price. Positive value = G pays E (p_E below strike); negative =
    # E pays G (p_E above strike). Sign convention follows the G→E
    # direction, so `cfd_payment < 0` means net revenue to G.
    cfd_payment:      float = 0.0   # G → E (signed)

    # ---- Phase-2c per-plant capital allocation (None => uniform) ----
    # When these are set, apply_laws_of_motion updates eco.E.K per plant
    # using I_E_per_plant[k] - depreciation_E_per_plant[k]. The scalar
    # flows.I_E and flows.depreciation_E must still equal their array
    # sums so the TFM column sums remain correct.
    I_E_per_plant: "np.ndarray | None" = None
    depreciation_E_per_plant: "np.ndarray | None" = None

    # ------- Column sums of the TFM (= sector savings) -------
    def column_sums(self) -> Dict[str, float]:
        """
        Net cash flow into each sector during the period, i.e. each
        sector's saving / NAFA (net acquisition of financial assets).

        Individual sector savings are NOT zero in general (a sector
        saves when income exceeds expenditure). The correct consistency
        identity is the quadruple-entry one: the sum across all six
        sectors is zero, because every flow is bilateral.
        """
        # F/E split of the aggregate wage bill. Phase 0 carries this
        # split as private attributes `_W_F` and `_W_E` attached to the
        # flow bundle; defaults are the 80/20 split used by Caiani.
        W_F = getattr(self, "_W_F", 0.8 * self.W)
        W_E = getattr(self, "_W_E", 0.2 * self.W)

        H = (
            + self.W
            + self.Tr_H
            + self.r_D_H
            + self.dividends_F
            + self.dividends_E
            + self.dividends_B
            + self.household_rebate
            - self.T_H
            - self.C_H_F
            - self.EB_H
        )
        # Investment is self-financed (firms producing their own capital):
        # I_F and I_E are pure capital-account flows and do not drain
        # current-account cash. They are therefore absent from the column
        # sums here, and from delta_DEP_F / delta_DEP_E below.
        F = (
            + self.C_H_F
            + self.G_spend
            + self.r_D_F
            + self.new_loans_F
            - W_F
            - self.r_L_F
            - self.EB_F
            - self.amort_F
            - self.dividends_F
        )
        E = (
            + self.EB_H
            + self.EB_F
            + self.r_D_E
            + self.new_loans_E
            + self.green_subsidy_E           # Stage 5.3 G → E transfer
            + self.fit_payment               # Stage 5.4 FiT: G → E
            + self.cfd_payment               # Stage 5.4 CfD (signed)
            - W_E
            - self.r_L_E
            - self.amort_E
            - self.carbon_tax
            - self.dividends_E
        )
        B = (
            + self.r_L_F
            + self.r_L_E
            + self.r_GB_B
            + self.r_Res
            + self.amort_F
            + self.amort_E
            + self.GB_amort          # G repays B: cash inflow to B
            - self.r_D_H
            - self.r_D_F
            - self.r_D_E
            - self.new_loans_F
            - self.new_loans_E
            - self.writeoff_F
            - self.writeoff_E
            - self.dividends_B
            - self.GB_issued         # B buys new bond: cash outflow
        )
        G = (
            + self.T_H
            + self.GB_issued
            + self.carbon_tax
            - self.Tr_H
            - self.r_GB_B
            - self.r_GB_CB
            - self.G_spend
            - self.GB_amort
            - self.household_rebate
            - self.green_subsidy_E           # Stage 5.3 G → E transfer
            - self.fit_payment               # Stage 5.4 FiT outflow
            - self.cfd_payment               # Stage 5.4 CfD (signed)
        )
        CB = (
            + self.r_GB_CB
            - self.r_Res
        )
        return dict(H=H, F=F, E=E, B=B, G=G, CB=CB)


# --------------------------------------------------------------------------
# BSM (horizontal) cancellation check
# --------------------------------------------------------------------------
def bsm_residuals(eco: Economy) -> Dict[str, float]:
    """
    Every financial stock must have equal absolute value on its asset
    side and its liability side. The residuals returned here should all
    be zero up to floating-point noise.
    """
    r = {}
    r["DEP"] = eco.DEP_H + eco.DEP_F + eco.DEP_E - eco.DEP_liab
    r["L"]   = eco.L_asset - (eco.L_F_total + eco.L_E_total)
    r["GB"]  = (eco.GB_B + eco.GB_CB) - eco.G.GB_out
    r["Res"] = float(eco.B.Res.sum()) - eco.CB.Res_liab
    r["DEP_G"] = eco.G.DEP_CB - eco.CB.DEP_G
    return r


# --------------------------------------------------------------------------
# NW identity check
# --------------------------------------------------------------------------
def nw_identity_residual(eco: Economy) -> float:
    """
    Sum of sectoral NW must equal total real capital.
    """
    nw_sum = (
        eco.H.NW_total
        + eco.F.NW_total
        + eco.E.NW_total
        + eco.B.NW_total
        + eco.G.NW
        + eco.CB.NW
    )
    return nw_sum - eco.K_total


# --------------------------------------------------------------------------
# Global residual check
# --------------------------------------------------------------------------
@dataclass
class ResidualReport:
    t: int
    bsm: Dict[str, float]
    nw:  float
    col: Dict[str, float]
    max_abs: float


def residual_check(eco: Economy,
                   flows: TFMFlows | None = None,
                   tol: float = RESIDUAL_TOL,
                   raise_on_fail: bool = True) -> ResidualReport:
    """
    Run all three checks and return a ResidualReport. If any residual
    exceeds tol * eco.GDP_proxy, raise SFCClosureError.

    Three independent identities are verified:
      1. BSM horizontal cancellation:   bsm_residuals(eco) ≈ 0
      2. NW identity:                    nw_identity_residual(eco) ≈ 0
      3. Quadruple-entry TFM closure:    Σ column_sums(flows) ≈ 0
         (individual sector savings can be non-zero; their sum is
         zero because every flow is bilateral).
    """
    bsm = bsm_residuals(eco)
    nw  = nw_identity_residual(eco)
    col = flows.column_sums() if flows is not None else {}
    col_total = sum(col.values()) if col else 0.0

    scale = eco.GDP_proxy
    check_vals = list(bsm.values()) + [nw, col_total]
    max_abs = max(abs(v) for v in check_vals) if check_vals else 0.0

    rep = ResidualReport(t=eco.t, bsm=bsm, nw=nw, col=col, max_abs=max_abs)

    if raise_on_fail and max_abs > tol * scale:
        raise SFCClosureError(
            f"SFC closure failed at t={eco.t}: "
            f"max|residual|={max_abs:.3e}, tol={tol * scale:.3e}\n"
            f"  BSM residuals: {bsm}\n"
            f"  NW residual:   {nw:.3e}\n"
            f"  TFM column sums (savings, Σ must be 0): {col}\n"
            f"  Σ column sums: {col_total:.3e}"
        )
    return rep


# --------------------------------------------------------------------------
# Laws of motion (v2.1 §3.10, Eq. 27-32)
# --------------------------------------------------------------------------
def apply_laws_of_motion(eco: Economy, flows: TFMFlows) -> None:
    """
    Update every stock in `eco` in place using the flows defined over
    the current period. The implementation splits aggregate flows over
    agents proportionally to their current holdings, which is the
    natural closure in phase 0 where there is no heterogeneity in
    behaviour.
    """
    # --- Eq. (27): household deposits ---
    # Phase-2c: household_rebate enters as a transfer from G into the
    # H column (see TFMFlows.column_sums), so it augments YD_H.
    YD_H = (flows.W + flows.Tr_H + flows.r_D_H
            + flows.dividends_F + flows.dividends_E + flows.dividends_B
            + flows.household_rebate - flows.T_H)
    net_H = YD_H - flows.C_H_F - flows.EB_H
    _distribute(eco.H.DEP, net_H)

    # --- Eq. (29): firm loans (both sides symmetric) ---
    delta_L_F = flows.new_loans_F - flows.amort_F - flows.writeoff_F
    _distribute(eco.F.L,   delta_L_F)   # borrower liability
    _distribute(eco.B.L_F, delta_L_F)   # bank asset

    # --- Eq. (29 analogue for E) ---
    delta_L_E = flows.new_loans_E - flows.amort_E - flows.writeoff_E
    _distribute(eco.E.L,   delta_L_E)
    _distribute(eco.B.L_E, delta_L_E)

    # --- Firm deposits (retained cash on the F column of the TFM) ---
    # Investment I_F is self-financed and does NOT subtract from cash —
    # see column_sums above for the matching convention.
    delta_DEP_F = (
        flows.C_H_F + flows.G_spend + flows.r_D_F + flows.new_loans_F
        - _F_wages(flows) - flows.r_L_F - flows.EB_F
        - flows.amort_F - flows.dividends_F
    )
    _distribute(eco.F.DEP, delta_DEP_F)

    # --- Energy deposits ---
    # Phase-2c: carbon_tax is paid out of E-sector cash.
    # Stage-5.3: green_subsidy_E is received by E-sector (from G).
    # Stage-5.4: FiT (always +) and CfD (signed) from G to E.
    delta_DEP_E = (
        flows.EB_H + flows.EB_F + flows.r_D_E + flows.new_loans_E
        + flows.green_subsidy_E
        + flows.fit_payment
        + flows.cfd_payment
        - _E_wages(flows) - flows.r_L_E
        - flows.amort_E
        - flows.carbon_tax
        - flows.dividends_E
    )
    _distribute(eco.E.DEP, delta_DEP_E)

    # --- Eq. (31): capital stocks (zero in phase 0) ---
    _distribute(eco.F.K, flows.I_F - flows.depreciation_F)
    # Phase-2c allows per-plant I_E and depreciation_E. When both are
    # supplied, they override the uniform distribution (and must sum
    # to the scalar flows above so the TFM column remains closed).
    if (flows.I_E_per_plant is not None
            and flows.depreciation_E_per_plant is not None):
        eco.E.K += (flows.I_E_per_plant
                    - flows.depreciation_E_per_plant)
    else:
        _distribute(eco.E.K, flows.I_E - flows.depreciation_E)

    # --- Eq. (32): government bonds ---
    eco.G.GB_out += flows.GB_issued - flows.GB_amort

    # GB distribution between B and CB: in phase 0 all new bonds go to B
    _distribute(eco.B.GB, flows.GB_issued - flows.GB_amort)

    # --- Bank liability side: DEP_liab tracks the sum of all customer deposits ---
    delta_DEP_liab = net_H + delta_DEP_F + delta_DEP_E
    _distribute(eco.B.DEP_liab, delta_DEP_liab)

    # --- Bank reserves & government account at CB: phase 0 leaves untouched ---
    # (these flows are all zero in phase 0)

    # --- Advance time ---
    eco.t += 1


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------
def _distribute(arr: np.ndarray, delta: float) -> None:
    """
    Spread a scalar aggregate change uniformly across an array of
    agents. Chosen because in phase 0 there is no behavioural
    heterogeneity; in later phases this function is replaced by
    agent-specific flow assignments.
    """
    n = arr.shape[0]
    if n == 0:
        return
    arr += delta / n


def _F_wages(flows: TFMFlows) -> float:
    """
    Share of total wages W that flows out of the F column.
    Phase 0 uses a fixed split 0.8·W to F and 0.2·W to E, stored in
    `flows.W` as the aggregate; the split itself is carried by the
    dynamics builder (see `dynamics.make_phase0_flows`).
    """
    return getattr(flows, "_W_F", 0.8 * flows.W)


def _E_wages(flows: TFMFlows) -> float:
    return getattr(flows, "_W_E", 0.2 * flows.W)
