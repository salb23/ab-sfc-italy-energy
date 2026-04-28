"""
run_phase0.py — driver for the phase-0 SFC closure test.

Runs the phase-0 dynamics for `N_QUARTERS` periods and verifies that
the SFC residual never exceeds `TOL * |GDP|`. Outputs:

  - residual_phase0.png : plot of max|residual| vs time and of the
    three component residuals (BSM, NW, TFM column sums);
  - stdout summary      : max residual, tolerance used, pass/fail.

Exit code is 0 on success, 1 on any residual > tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from accounting import SFCClosureError
from dynamics import step
from init_stocks import make_initial_economy


# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
N_QUARTERS = 500
TOL        = 1e-10
SEED       = 0
OUTFILE    = Path(__file__).parent / "residual_phase0.png"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy

    # Pre-allocate residual arrays
    max_abs   = np.zeros(N_QUARTERS)
    bsm_DEP   = np.zeros(N_QUARTERS)
    bsm_L     = np.zeros(N_QUARTERS)
    bsm_GB    = np.zeros(N_QUARTERS)
    nw_res    = np.zeros(N_QUARTERS)
    col_sum   = np.zeros(N_QUARTERS)   # Σ across sectors, must be 0

    # Also track a couple of real quantities to verify stationarity
    K_total   = np.zeros(N_QUARTERS + 1)
    DEP_liab  = np.zeros(N_QUARTERS + 1)
    K_total[0]  = eco.K_total
    DEP_liab[0] = eco.DEP_liab

    try:
        for t in range(N_QUARTERS):
            rep = step(eco, phase=0, tol=TOL)
            max_abs[t]  = rep.max_abs
            bsm_DEP[t]  = rep.bsm["DEP"]
            bsm_L[t]    = rep.bsm["L"]
            bsm_GB[t]   = rep.bsm["GB"]
            nw_res[t]   = rep.nw
            col_sum[t]  = sum(rep.col.values()) if rep.col else 0.0
            K_total[t + 1]  = eco.K_total
            DEP_liab[t + 1] = eco.DEP_liab
    except SFCClosureError as exc:
        print(f"FAIL: SFC closure broke at quarter {eco.t}")
        print(exc)
        return 1

    # ----------------------------------------------------------------------
    # Summary stats
    # ----------------------------------------------------------------------
    overall_max = float(max_abs.max())
    tol_abs     = TOL * gdp_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    print("=" * 60)
    print(f"Phase-0 closure test over {N_QUARTERS} quarters")
    print("=" * 60)
    print(f"  GDP_proxy (K_total)   : {gdp_proxy:.4e}")
    print(f"  Tolerance (TOL*GDP)   : {tol_abs:.4e}")
    print(f"  Max |residual|        : {overall_max:.4e}")
    print(f"  K_total drift         : {K_total[-1] - K_total[0]:+.4e}")
    print(f"  DEP_liab drift        : {DEP_liab[-1] - DEP_liab[0]:+.4e}")
    print(f"  Status                : {status}")
    print("=" * 60)

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------
    t_axis = np.arange(1, N_QUARTERS + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax = axes[0]
    # Use np.maximum with a small floor so log plot doesn't break on zeros
    floor = 1e-20
    ax.semilogy(t_axis, np.maximum(np.abs(max_abs), floor),
                color="black", lw=1.2, label="max|residual|")
    ax.axhline(tol_abs, color="red", ls="--", lw=1.0, label=f"tol = {tol_abs:.1e}")
    ax.set_ylabel("|residual|  (log scale)")
    ax.set_title(f"Phase-0 SFC residual vs time  —  {status}")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, bsm_DEP, label="BSM DEP",  lw=0.8)
    ax.plot(t_axis, bsm_L,   label="BSM L",    lw=0.8)
    ax.plot(t_axis, bsm_GB,  label="BSM GB",   lw=0.8)
    ax.plot(t_axis, nw_res,  label="NW ident", lw=0.8)
    ax.plot(t_axis, col_sum, label="Σ TFM col", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel("quarter")
    ax.set_ylabel("residual (signed)")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=150)
    print(f"Wrote residual plot → {OUTFILE}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
