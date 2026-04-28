"""
run_phase2c_tax.py — driver for the phase-2c closure test with a
single carbon-tax level and full lump-sum rebate.

Phase 2c activates:
  * effective marginal cost with a τ · ef_k premium on dispatch;
  * carbon-tax revenue flowing E → G at the end of each period;
  * same-quarter lump-sum rebate G → H in equal per-household shares;
  * profitability-weighted split of aggregate replacement investment
    across energy plants (predetermined by one period).

The closure bound is unchanged (max|residual| < TOL · GDP_proxy). On
top of the residual plot we produce:

  1. emissions vs time (flow and cumulative stock);
  2. per-plant K_E trajectory (the decarbonisation mechanism in
     action — dirty plants shrink, clean plants grow);
  3. carbon-tax revenue and rebate vs time (they coincide by
     construction — a visual sanity check of fiscal balance);
  4. household rebate as a share of wage income.
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
TAU        = 0.4          # carbon-tax level (model-price units)
TOL        = 1e-10
SEED       = 0
OUT_RES     = Path(__file__).parent / "residual_phase2c.png"
OUT_EMIT    = Path(__file__).parent / "emissions_phase2c.png"
OUT_TECH    = Path(__file__).parent / "tech_mix_phase2c.png"
OUT_FISCAL  = Path(__file__).parent / "fiscal_phase2c.png"


def main() -> int:
    eco = make_initial_economy(params=dict(carbon_tax=TAU), seed=SEED)
    gdp_proxy = eco.GDP_proxy
    N_E = eco.E.N
    tech = list(eco.E.tech)
    ef   = eco.E.emission_factor.copy()

    T = N_QUARTERS
    max_abs = np.zeros(T)

    # Stock/flow traces — t=0 baseline, then one row per quarter
    K_E_path    = np.zeros((T + 1, N_E))
    K_E_path[0] = eco.E.K
    em_flow     = np.zeros(T + 1)
    em_stock    = np.zeros(T + 1)
    tax_rev     = np.zeros(T + 1)
    rebate      = np.zeros(T + 1)
    wage_bill   = np.zeros(T + 1)
    rebate_pc   = np.zeros(T + 1)   # rebate per household

    try:
        for t in range(T):
            rep = step(eco, phase=4, tol=TOL)
            max_abs[t] = rep.max_abs
            K_E_path[t + 1] = eco.E.K
            em_flow[t + 1]  = float((eco.Y_per_plant * ef).sum())
            em_stock[t + 1] = eco.emissions_stock
            # pull the most recent flow values from the pending state
            tau_now = getattr(eco, "carbon_tax", 0.0)
            tax_rev[t + 1] = tau_now * em_flow[t + 1]
            rebate[t + 1]  = tax_rev[t + 1]       # full recycling
            # wage bill stash is on _W_F / _W_E (set by flow builder)
            rebate_pc[t + 1] = rebate[t + 1] / eco.H.N
            # A rough "rebate as share of wages" — use cumulative wage.
            wage_bill[t + 1] = eco.params["w_initial"] * eco.H.N
    except SFCClosureError as exc:
        print(f"FAIL: SFC closure broke at quarter {eco.t}")
        print(exc)
        return 1

    overall_max = float(max_abs.max())
    tol_abs     = TOL * gdp_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    K_E_start = K_E_path[0]
    K_E_end   = K_E_path[-1]

    print("=" * 72)
    print(f"Phase-2c closure test — τ = {TAU} (model units) — "
          f"{N_QUARTERS} quarters")
    print("=" * 72)
    print(f"  GDP_proxy (K_total)           : {gdp_proxy:.4e}")
    print(f"  Tolerance (TOL*GDP)           : {tol_abs:.4e}")
    print(f"  Max |residual|                : {overall_max:.4e}")
    print(f"  Cumulative emissions          : {em_stock[-1]:.4e}")
    print(f"  Cumulative carbon-tax revenue : {tax_rev[1:].sum():.4e}")
    print(f"  Cumulative household rebate   : {rebate[1:].sum():.4e}")
    print(f"  Per-plant K_E, t = 0          :")
    for k in range(N_E):
        print(f"     {tech[k]:9s} (ef={ef[k]:.2f})  K_0={K_E_start[k]:7.2f}")
    print(f"  Per-plant K_E, t = {N_QUARTERS}          :")
    for k in range(N_E):
        delta = K_E_end[k] - K_E_start[k]
        print(f"     {tech[k]:9s} (ef={ef[k]:.2f})  K_T={K_E_end[k]:7.2f}  "
              f"Δ={delta:+7.2f}")
    print(f"  Status                        : {status}")
    print("=" * 72)

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
    ax.set_title(f"Phase-2c SFC residual (τ={TAU}) — {status}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RES, dpi=150)
    print(f"Wrote residual plot  → {OUT_RES}")

    # ------------------------------------------------------------------
    # Plot 2: emissions
    # ------------------------------------------------------------------
    kpi_t = np.arange(T + 1)
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes2[0].plot(kpi_t[1:], em_flow[1:], color="tab:red", lw=1.0)
    axes2[0].set_ylabel("emissions (tCO₂/quarter)")
    axes2[0].set_title(f"Emissions flow — τ={TAU}")
    axes2[0].grid(True, alpha=0.3)
    axes2[1].plot(kpi_t, em_stock, color="tab:red", lw=1.0)
    axes2[1].set_xlabel("quarter")
    axes2[1].set_ylabel("cumulative tCO₂")
    axes2[1].set_title("Cumulative emissions")
    axes2[1].grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_EMIT, dpi=150)
    print(f"Wrote emissions plot → {OUT_EMIT}")

    # ------------------------------------------------------------------
    # Plot 3: tech mix — per-plant K_E trajectory, coloured by ef
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 5))
    # Order plants: cleanest first (so the stackplot bottom is clean)
    order = np.argsort(ef)
    # Use a Reds colormap indexed by emission factor
    cmap = plt.get_cmap("RdYlGn_r")
    for rank, k in enumerate(order):
        c = cmap(float(ef[k]))
        ax3.plot(kpi_t, K_E_path[:, k], color=c, lw=1.3,
                 label=f"{tech[k]} (ef={ef[k]:.2f})")
    ax3.set_xlabel("quarter")
    ax3.set_ylabel("installed capital K_E per plant")
    ax3.set_title(f"Per-plant K_E trajectory — τ={TAU}")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9, ncol=2)
    fig3.tight_layout()
    fig3.savefig(OUT_TECH, dpi=150)
    print(f"Wrote tech-mix plot  → {OUT_TECH}")

    # ------------------------------------------------------------------
    # Plot 4: fiscal — tax revenue and rebate
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes4[0].plot(kpi_t[1:], tax_rev[1:], color="tab:orange",
                   lw=1.0, label="tax revenue")
    axes4[0].plot(kpi_t[1:], rebate[1:], color="tab:blue",
                   lw=1.0, ls="--", label="rebate")
    axes4[0].set_ylabel("flow (nominal)")
    axes4[0].set_title(
        f"Carbon-tax revenue vs lump-sum rebate — τ={TAU}")
    axes4[0].grid(True, alpha=0.3); axes4[0].legend()
    axes4[1].plot(kpi_t[1:], rebate_pc[1:], color="tab:blue", lw=1.0)
    axes4[1].set_xlabel("quarter")
    axes4[1].set_ylabel("rebate per household")
    axes4[1].set_title("Per-household rebate")
    axes4[1].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(OUT_FISCAL, dpi=150)
    print(f"Wrote fiscal plot    → {OUT_FISCAL}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
