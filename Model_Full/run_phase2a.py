"""
run_phase2a.py — driver for the phase-2a closure test and merit-order
diagnostics.

Adds to the phase-1 driver a third channel: an endogenous wholesale
energy market. Each quarter the period's real energy demand (from the
phase-1 behavioural block) is cleared via a merit-order dispatch, the
marginal plant's MC becomes the next period's `p_E`, and emissions
accumulate.

The pass/fail test is still closure: max|residual| must stay below
`TOL * GDP_proxy`. The rest is diagnostic plots showing that the
energy side is doing something.

Three plots are written:
  - residual_phase2a.png  — residuals over the horizon (closure check)
  - merit_phase2a.png     — clearing price and per-technology dispatch shares
  - emissions_phase2a.png — cumulative emissions and flow per quarter
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
OUT_RES    = Path(__file__).parent / "residual_phase2a.png"
OUT_MERIT  = Path(__file__).parent / "merit_phase2a.png"
OUT_EMIT   = Path(__file__).parent / "emissions_phase2a.png"


def main() -> int:
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy

    T   = N_QUARTERS
    N_E = eco.E.N
    tech = eco.E.tech.copy()

    # Closure traces
    max_abs  = np.zeros(T)

    # KPI traces — one row per quarter after the step
    p_E_ser      = np.zeros(T + 1); p_E_ser[0]     = eco.params["p_E"]
    p_F_ser      = np.zeros(T + 1); p_F_ser[0]     = eco.p_F
    cpi_ser      = np.zeros(T + 1); cpi_ser[0]     = eco.cpi
    u_ser        = np.zeros(T + 1); u_ser[0]       = eco.u
    rationed_ser = np.zeros(T + 1)
    emis_stock_ser = np.zeros(T + 1); emis_stock_ser[0] = eco.emissions_stock
    emis_flow_ser  = np.zeros(T + 1)
    y_mat        = np.zeros((T + 1, N_E))  # per-plant dispatched output

    try:
        for t in range(T):
            rep = step(eco, phase=2, tol=TOL)
            max_abs[t] = rep.max_abs

            p_E_ser[t + 1]      = eco.params["p_E"]
            p_F_ser[t + 1]      = eco.p_F
            cpi_ser[t + 1]      = eco.cpi
            u_ser[t + 1]        = eco.u
            rationed_ser[t + 1] = getattr(eco, "_pending_rationed", 0.0)
            emis_stock_ser[t + 1] = eco.emissions_stock
            emis_flow_ser[t + 1]  = getattr(eco, "_pending_emissions_add", 0.0)
            y_mat[t + 1, :]     = eco.Y_per_plant
    except SFCClosureError as exc:
        print(f"FAIL: SFC closure broke at quarter {eco.t}")
        print(exc)
        return 1

    overall_max = float(max_abs.max())
    tol_abs     = TOL * gdp_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    print("=" * 60)
    print(f"Phase-2a closure test over {N_QUARTERS} quarters")
    print("=" * 60)
    print(f"  GDP_proxy (K_total)    : {gdp_proxy:.4e}")
    print(f"  Tolerance (TOL*GDP)    : {tol_abs:.4e}")
    print(f"  Max |residual|         : {overall_max:.4e}")
    print(f"  p_E initial → final    : {p_E_ser[0]:.4f} → {p_E_ser[-1]:.4f}")
    print(f"  cumulative emissions   : {emis_stock_ser[-1]:.4e}")
    print(f"  rationed (max over t)  : {rationed_ser.max():.4e}")
    print(f"  u initial → final      : {u_ser[0]:.4f} → {u_ser[-1]:.4f}")
    print(f"  Status                 : {status}")
    print("=" * 60)

    # Per-tech aggregation for the dispatch share plot
    unique_tech = sorted(set(tech.tolist()))
    y_by_tech = np.zeros((T + 1, len(unique_tech)))
    for j, tlabel in enumerate(unique_tech):
        mask = (tech == tlabel)
        y_by_tech[:, j] = y_mat[:, mask].sum(axis=1)

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
    ax.set_title(f"Phase-2a SFC residual vs time — {status}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RES, dpi=150)
    print(f"Wrote residual plot  → {OUT_RES}")

    # ------------------------------------------------------------------
    # Plot 2: clearing price and merit-order dispatch mix
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    kpi_t = np.arange(T + 1)

    axes2[0].plot(kpi_t, p_E_ser, color="black", lw=1.0, label="p_E (clearing)")
    axes2[0].plot(kpi_t, p_F_ser, color="tab:orange", lw=0.8, ls="--", label="p_F")
    axes2[0].set_ylabel("price (model units)")
    axes2[0].set_title("Market-clearing prices")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    tech_colors = {
        "coal":    "#2b2b2b",
        "gas":     "#c0392b",
        "nuclear": "#8e44ad",
        "solar":   "#f1c40f",
        "wind":    "#3498db",
        "biomass": "#27ae60",
    }
    bottom = np.zeros(T + 1)
    for j, tlabel in enumerate(unique_tech):
        col = tech_colors.get(tlabel, f"C{j}")
        axes2[1].fill_between(kpi_t, bottom, bottom + y_by_tech[:, j],
                              color=col, alpha=0.85, label=tlabel)
        bottom = bottom + y_by_tech[:, j]
    axes2[1].set_xlabel("quarter")
    axes2[1].set_ylabel("dispatched output (units)")
    axes2[1].set_title("Generation mix by technology (stacked)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(ncol=len(unique_tech), fontsize=8, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(OUT_MERIT, dpi=150)
    print(f"Wrote merit plot     → {OUT_MERIT}")

    # ------------------------------------------------------------------
    # Plot 3: emissions
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes3[0].plot(kpi_t, emis_stock_ser, color="tab:red", lw=1.2)
    axes3[0].set_ylabel("cumulative tCO2")
    axes3[0].set_title("Cumulative emissions (phase-2a baseline)")
    axes3[0].grid(True, alpha=0.3)
    axes3[1].plot(kpi_t, emis_flow_ser, color="tab:red", lw=0.8)
    axes3[1].set_xlabel("quarter")
    axes3[1].set_ylabel("tCO2 per quarter")
    axes3[1].set_title("Emissions flow per quarter")
    axes3[1].grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUT_EMIT, dpi=150)
    print(f"Wrote emissions plot → {OUT_EMIT}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
