"""
run_phase2b.py — driver for the phase-2b closure test and credit/
investment diagnostics.

Phase 2b activates the investment, depreciation, amortisation, and
new-lending flows. The closure test is unchanged — max|residual|
must stay below `TOL * GDP_proxy`. On top of the residual plot we
write three KPI panels: outstanding loan stocks, capital stocks,
and the credit-rationing intensity (0 = fully binding, 1 = slack).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accounting import SFCClosureError
from dynamics import step
from init_stocks import make_initial_economy


N_QUARTERS = 500
TOL        = 1e-10
SEED       = 0
OUT_RES    = Path(__file__).parent / "residual_phase2b.png"
OUT_BS     = Path(__file__).parent / "balance_sheet_phase2b.png"
OUT_CREDIT = Path(__file__).parent / "credit_phase2b.png"


def main() -> int:
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy

    T = N_QUARTERS
    max_abs = np.zeros(T)

    # Stock traces (t=0 baseline, then one row per quarter)
    L_F   = np.zeros(T + 1); L_F[0]   = eco.L_F_total
    L_E   = np.zeros(T + 1); L_E[0]   = eco.L_E_total
    K_F   = np.zeros(T + 1); K_F[0]   = eco.K_F_total
    K_E   = np.zeros(T + 1); K_E[0]   = eco.K_E_total
    NW_B  = np.zeros(T + 1); NW_B[0]  = eco.B.NW_total
    DEP_H = np.zeros(T + 1); DEP_H[0] = eco.DEP_H

    # Flow traces (one row per quarter after the step)
    new_F   = np.zeros(T + 1)
    new_E   = np.zeros(T + 1)
    dem_F   = np.zeros(T + 1)
    dem_E   = np.zeros(T + 1)
    ration  = np.ones(T + 1)

    try:
        for t in range(T):
            rep = step(eco, phase=3, tol=TOL)
            max_abs[t] = rep.max_abs

            L_F[t + 1]   = eco.L_F_total
            L_E[t + 1]   = eco.L_E_total
            K_F[t + 1]   = eco.K_F_total
            K_E[t + 1]   = eco.K_E_total
            NW_B[t + 1]  = eco.B.NW_total
            DEP_H[t + 1] = eco.DEP_H

            dem_F[t + 1]  = getattr(eco, "_pending_loan_demand_F", 0.0)
            dem_E[t + 1]  = getattr(eco, "_pending_loan_demand_E", 0.0)
            ration[t + 1] = getattr(eco, "_pending_ration_ratio", 1.0)
            # new loans are simply demand × ration
            new_F[t + 1] = dem_F[t + 1] * ration[t + 1]
            new_E[t + 1] = dem_E[t + 1] * ration[t + 1]
    except SFCClosureError as exc:
        print(f"FAIL: SFC closure broke at quarter {eco.t}")
        print(exc)
        return 1

    overall_max = float(max_abs.max())
    tol_abs     = TOL * gdp_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    rationed_quarters = int((ration[1:] < 1.0).sum())
    print("=" * 64)
    print(f"Phase-2b closure test over {N_QUARTERS} quarters")
    print("=" * 64)
    print(f"  GDP_proxy (K_total)       : {gdp_proxy:.4e}")
    print(f"  Tolerance (TOL*GDP)       : {tol_abs:.4e}")
    print(f"  Max |residual|            : {overall_max:.4e}")
    print(f"  L_F initial → final       : {L_F[0]:.2e} → {L_F[-1]:.2e}")
    print(f"  L_E initial → final       : {L_E[0]:.2e} → {L_E[-1]:.2e}")
    print(f"  K_F initial → final       : {K_F[0]:.2e} → {K_F[-1]:.2e}")
    print(f"  K_E initial → final       : {K_E[0]:.2e} → {K_E[-1]:.2e}")
    print(f"  NW_B initial → final      : {NW_B[0]:.2e} → {NW_B[-1]:.2e}")
    print(f"  Rationed quarters         : {rationed_quarters} / {N_QUARTERS}")
    print(f"  Status                    : {status}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Plot 1: residuals
    # ------------------------------------------------------------------
    t_axis = np.arange(1, T + 1)
    floor = 1e-20
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
    ax.semilogy(t_axis, np.maximum(np.abs(max_abs), floor),
                color="black", lw=1.0, label="max|residual|")
    ax.axhline(tol_abs, color="red", ls="--", lw=1.0,
               label=f"tol = {tol_abs:.1e}")
    ax.set_xlabel("quarter")
    ax.set_ylabel("|residual| (log)")
    ax.set_title(f"Phase-2b SFC residual vs time — {status}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RES, dpi=150)
    print(f"Wrote residual plot  → {OUT_RES}")

    # ------------------------------------------------------------------
    # Plot 2: balance-sheet stocks
    # ------------------------------------------------------------------
    kpi_t = np.arange(T + 1)
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    axes2[0, 0].plot(kpi_t, L_F, color="tab:blue",   label="L_F")
    axes2[0, 0].plot(kpi_t, L_E, color="tab:orange", label="L_E")
    axes2[0, 0].set_title("Outstanding loan stocks")
    axes2[0, 0].grid(True, alpha=0.3); axes2[0, 0].legend()

    axes2[0, 1].plot(kpi_t, K_F, color="tab:green",  label="K_F")
    axes2[0, 1].plot(kpi_t, K_E, color="tab:purple", label="K_E")
    axes2[0, 1].set_title("Real capital stocks")
    axes2[0, 1].grid(True, alpha=0.3); axes2[0, 1].legend()

    axes2[1, 0].plot(kpi_t, NW_B, color="black", label="NW_B")
    axes2[1, 0].set_title("Bank sector net worth")
    axes2[1, 0].grid(True, alpha=0.3); axes2[1, 0].legend()
    axes2[1, 0].set_xlabel("quarter")

    axes2[1, 1].plot(kpi_t, DEP_H, color="tab:red", label="DEP_H")
    axes2[1, 1].set_title("Household deposits")
    axes2[1, 1].grid(True, alpha=0.3); axes2[1, 1].legend()
    axes2[1, 1].set_xlabel("quarter")

    fig2.suptitle("Phase-2b balance-sheet diagnostics")
    fig2.tight_layout()
    fig2.savefig(OUT_BS, dpi=150)
    print(f"Wrote balance plot   → {OUT_BS}")

    # ------------------------------------------------------------------
    # Plot 3: credit — demand, issued, rationing
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes3[0].plot(kpi_t, dem_F, color="tab:blue",   ls="--", label="demand F")
    axes3[0].plot(kpi_t, new_F, color="tab:blue",   lw=1.2,  label="issued F")
    axes3[0].plot(kpi_t, dem_E, color="tab:orange", ls="--", label="demand E")
    axes3[0].plot(kpi_t, new_E, color="tab:orange", lw=1.2,  label="issued E")
    axes3[0].set_ylabel("new lending (nominal)")
    axes3[0].set_title("Credit demand vs issuance")
    axes3[0].grid(True, alpha=0.3); axes3[0].legend(ncol=2, fontsize=9)

    axes3[1].plot(kpi_t, ration, color="black", lw=1.0)
    axes3[1].axhline(1.0, color="grey", ls=":", lw=0.7)
    axes3[1].set_ylim(-0.05, 1.1)
    axes3[1].set_xlabel("quarter")
    axes3[1].set_ylabel("rationing ratio")
    axes3[1].set_title("Credit rationing (1 = fully met, 0 = shut out)")
    axes3[1].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(OUT_CREDIT, dpi=150)
    print(f"Wrote credit plot    → {OUT_CREDIT}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
