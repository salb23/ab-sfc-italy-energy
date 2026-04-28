"""
run_phase1.py — driver for the phase-1 SFC closure test and diagnostics.

Adds to the phase-0 closure test three behavioural channels:

  - household consumption C = α1 YD + α2 NW
  - firm pricing p_F = (1+μ) UC
  - wage Phillips curve

The point of this driver remains closure: max|residual| must stay
below `TOL * |GDP|`. A second panel plots key macro variables
(nominal GDP, CPI, wage, unemployment) for a sanity check, but the
pass/fail decision is purely the residual test.
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
OUT_RES    = Path(__file__).parent / "residual_phase1.png"
OUT_KPI    = Path(__file__).parent / "kpi_phase1.png"


def main() -> int:
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy  # use fixed scale (K_total) for tolerance

    T = N_QUARTERS
    max_abs  = np.zeros(T)
    bsm_DEP  = np.zeros(T)
    nw_res   = np.zeros(T)
    col_sum  = np.zeros(T)

    # KPI traces
    w_ser    = np.zeros(T + 1);  w_ser[0]   = eco.w
    pF_ser   = np.zeros(T + 1);  pF_ser[0]  = eco.p_F
    cpi_ser  = np.zeros(T + 1);  cpi_ser[0] = eco.cpi
    u_ser    = np.zeros(T + 1);  u_ser[0]   = eco.u
    YF_ser   = np.zeros(T + 1);  YF_ser[0]  = 0.0
    YE_ser   = np.zeros(T + 1);  YE_ser[0]  = 0.0
    GDPn     = np.zeros(T + 1);  GDPn[0]    = 0.0
    DEP_H    = np.zeros(T + 1);  DEP_H[0]   = eco.DEP_H
    DEP_liab = np.zeros(T + 1);  DEP_liab[0] = eco.DEP_liab

    try:
        for t in range(T):
            rep = step(eco, phase=1, tol=TOL)
            max_abs[t]  = rep.max_abs
            bsm_DEP[t]  = rep.bsm["DEP"]
            nw_res[t]   = rep.nw
            col_sum[t]  = sum(rep.col.values()) if rep.col else 0.0
            w_ser[t + 1]    = eco.w
            pF_ser[t + 1]   = eco.p_F
            cpi_ser[t + 1]  = eco.cpi
            u_ser[t + 1]    = eco.u
            YF_ser[t + 1]   = eco.Y_F
            YE_ser[t + 1]   = eco.Y_E
            # nominal GDP proxy (phase 1): nominal household expenditure
            GDPn[t + 1]     = eco.Y_F * eco.p_F + eco.Y_E * eco.params["p_E"]
            DEP_H[t + 1]    = eco.DEP_H
            DEP_liab[t + 1] = eco.DEP_liab
    except SFCClosureError as exc:
        print(f"FAIL: SFC closure broke at quarter {eco.t}")
        print(exc)
        return 1

    overall_max = float(max_abs.max())
    tol_abs     = TOL * gdp_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    print("=" * 60)
    print(f"Phase-1 closure test over {N_QUARTERS} quarters")
    print("=" * 60)
    print(f"  GDP_proxy (K_total)   : {gdp_proxy:.4e}")
    print(f"  Tolerance (TOL*GDP)   : {tol_abs:.4e}")
    print(f"  Max |residual|        : {overall_max:.4e}")
    print(f"  w initial → final     : {w_ser[0]:.4f}  →  {w_ser[-1]:.4f}")
    print(f"  u initial → final     : {u_ser[0]:.4f}  →  {u_ser[-1]:.4f}")
    print(f"  nominal GDP final     : {GDPn[-1]:.4e}")
    print(f"  DEP_H initial → final : {DEP_H[0]:.4e}  →  {DEP_H[-1]:.4e}")
    print(f"  Status                : {status}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Plot 1: residuals
    # ------------------------------------------------------------------
    t_axis = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    floor = 1e-20
    ax = axes[0]
    ax.semilogy(t_axis, np.maximum(np.abs(max_abs), floor),
                color="black", lw=1.0, label="max|residual|")
    ax.axhline(tol_abs, color="red", ls="--", lw=1.0, label=f"tol = {tol_abs:.1e}")
    ax.set_ylabel("|residual| (log)")
    ax.set_title(f"Phase-1 SFC residual vs time — {status}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(t_axis, bsm_DEP, label="BSM DEP", lw=0.8)
    ax.plot(t_axis, nw_res,  label="NW ident", lw=0.8)
    ax.plot(t_axis, col_sum, label="Σ TFM col", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.4)
    ax.set_xlabel("quarter")
    ax.set_ylabel("residual (signed)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_RES, dpi=150)
    print(f"Wrote residual plot → {OUT_RES}")

    # ------------------------------------------------------------------
    # Plot 2: KPIs
    # ------------------------------------------------------------------
    kpi_t = np.arange(T + 1)
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    axes2[0, 0].plot(kpi_t, GDPn, color="black", lw=1.0)
    axes2[0, 0].set_title("Nominal GDP proxy"); axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].plot(kpi_t, cpi_ser, color="tab:blue", lw=1.0)
    axes2[0, 1].set_title("CPI index"); axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].plot(kpi_t, u_ser * 100, color="tab:red", lw=1.0)
    axes2[1, 0].axhline(eco.params["u_star"] * 100, color="red", ls="--", lw=0.7,
                        label="u* (NAIRU)")
    axes2[1, 0].set_title("Unemployment rate (%)"); axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend()
    axes2[1, 0].set_xlabel("quarter")

    axes2[1, 1].plot(kpi_t, w_ser, color="tab:green", lw=1.0, label="wage w")
    axes2[1, 1].plot(kpi_t, pF_ser, color="tab:orange", lw=1.0, label="p_F")
    axes2[1, 1].set_title("Wage & F-price"); axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend()
    axes2[1, 1].set_xlabel("quarter")

    fig2.suptitle("Phase-1 macro diagnostics (stylised, not calibrated)")
    fig2.tight_layout()
    fig2.savefig(OUT_KPI, dpi=150)
    print(f"Wrote KPI plot      → {OUT_KPI}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
