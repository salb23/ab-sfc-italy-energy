"""
run_phase2c_sweep.py — τ-sweep driver for Phase 2c.

Runs the closed economy at a grid of carbon-tax levels and assembles a
three-panel comparative figure that summarises the policy response:

  1. cumulative emissions (tCO₂) over the horizon, one curve per τ;
  2. terminal tech-mix (per-plant K_E), ordered by emission factor;
  3. cumulative per-household rebate, one curve per τ.

Closure is checked at every step; a FAIL in any run aborts the sweep so
we never plot corrupted series.
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
TAU_GRID   = (0.0, 0.2, 0.4, 0.6, 0.8)
TOL        = 1e-10
SEED       = 0
OUT_FIG    = Path(__file__).parent / "sweep_phase2c.png"


def run_one(tau: float, n: int = N_QUARTERS, tol: float = TOL,
            seed: int = SEED) -> dict:
    """Simulate Phase-2c for `n` quarters at a single τ level."""
    eco = make_initial_economy(params=dict(carbon_tax=tau), seed=seed)
    N_E = eco.E.N
    tech = list(eco.E.tech)
    ef   = eco.E.emission_factor.copy()
    N_H  = eco.H.N

    max_abs   = np.zeros(n)
    K_E_path  = np.zeros((n + 1, N_E))
    K_E_path[0] = eco.E.K
    em_flow   = np.zeros(n + 1)
    em_stock  = np.zeros(n + 1)
    tax_rev   = np.zeros(n + 1)
    rebate_pc = np.zeros(n + 1)

    try:
        for t in range(n):
            rep = step(eco, phase=4, tol=tol)
            max_abs[t]     = rep.max_abs
            K_E_path[t+1]  = eco.E.K
            em_flow[t+1]   = float((eco.Y_per_plant * ef).sum())
            em_stock[t+1]  = eco.emissions_stock
            tax_rev[t+1]   = tau * em_flow[t+1]
            rebate_pc[t+1] = tax_rev[t+1] / N_H
    except SFCClosureError as exc:
        return dict(tau=tau, status="FAIL", quarter=eco.t, err=str(exc))

    overall_max = float(max_abs.max())
    tol_abs     = TOL * eco.GDP_proxy
    status      = "PASS" if overall_max <= tol_abs else "FAIL"

    return dict(
        tau=tau,
        status=status,
        max_abs=overall_max,
        tol_abs=tol_abs,
        tech=tech,
        ef=ef,
        K_E_start=K_E_path[0].copy(),
        K_E_end=K_E_path[-1].copy(),
        em_flow=em_flow,
        em_stock=em_stock,
        tax_rev=tax_rev,
        rebate_pc=rebate_pc,
        cum_rebate_pc=np.cumsum(rebate_pc),
        N_H=N_H,
    )


def main() -> int:
    print("=" * 72)
    print(f"Phase-2c τ-sweep — {N_QUARTERS} quarters per level")
    print(f"  τ grid: {TAU_GRID}")
    print("=" * 72)

    results = []
    for tau in TAU_GRID:
        r = run_one(tau)
        status = r["status"]
        if status == "FAIL":
            err = r.get("err", "residual > tol")
            print(f"  τ={tau:.2f}  FAIL  ({err})")
            return 1
        print(f"  τ={tau:.2f}  {status}  "
              f"max|res|={r['max_abs']:.3e}  "
              f"cum emissions={r['em_stock'][-1]:.3e}  "
              f"cum rebate/hh={r['cum_rebate_pc'][-1]:.3e}")
        results.append(r)

    # ------------------------------------------------------------------
    # Assemble comparative figure
    # ------------------------------------------------------------------
    # Shared colour ramp (viridis) indexed by τ position
    cmap = plt.get_cmap("viridis")
    tau_colors = [cmap(i / max(len(TAU_GRID) - 1, 1))
                  for i in range(len(TAU_GRID))]

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_em     = fig.add_subplot(gs[0, 0])
    ax_reb    = fig.add_subplot(gs[0, 1])
    ax_mix    = fig.add_subplot(gs[1, :])

    kpi_t = np.arange(N_QUARTERS + 1)

    # Panel 1: cumulative emissions
    for r, c in zip(results, tau_colors):
        ax_em.plot(kpi_t, r["em_stock"], color=c, lw=1.3,
                   label=f"τ = {r['tau']:.1f}")
    ax_em.set_xlabel("quarter")
    ax_em.set_ylabel("cumulative tCO₂")
    ax_em.set_title("Cumulative emissions across τ levels")
    ax_em.grid(True, alpha=0.3)
    ax_em.legend(loc="best", fontsize=9)

    # Panel 2: cumulative per-household rebate
    for r, c in zip(results, tau_colors):
        ax_reb.plot(kpi_t, r["cum_rebate_pc"], color=c, lw=1.3,
                    label=f"τ = {r['tau']:.1f}")
    ax_reb.set_xlabel("quarter")
    ax_reb.set_ylabel("cumulative rebate per household")
    ax_reb.set_title("Cumulative per-household rebate")
    ax_reb.grid(True, alpha=0.3)
    ax_reb.legend(loc="best", fontsize=9)

    # Panel 3: terminal tech mix — grouped bars, one group per plant,
    # one bar per τ level, ordered by emission factor (dirty → clean).
    ref = results[0]
    ef   = ref["ef"]
    tech = ref["tech"]
    order = np.argsort(-ef)  # dirty first (left), clean last (right)
    labels = [f"{tech[k]}\n(ef={ef[k]:.2f})" for k in order]
    n_tau = len(results)
    bar_w = 0.8 / n_tau
    x = np.arange(len(order))

    for i, (r, c) in enumerate(zip(results, tau_colors)):
        heights = r["K_E_end"][order]
        offset  = (i - (n_tau - 1) / 2) * bar_w
        ax_mix.bar(x + offset, heights, width=bar_w, color=c,
                   label=f"τ = {r['tau']:.1f}")
    # Dashed reference line: starting K_E (identical across τ)
    start = ref["K_E_start"][order]
    for xi, h in zip(x, start):
        ax_mix.hlines(h, xi - 0.45, xi + 0.45, colors="black",
                      linestyles="--", lw=0.9)
    ax_mix.set_xticks(x)
    ax_mix.set_xticklabels(labels, fontsize=9)
    ax_mix.set_ylabel("terminal K_E per plant")
    ax_mix.set_title(
        "Terminal tech mix (bars) vs. starting K_E (dashed) — "
        "dirty plants shrink, clean plants grow with τ")
    ax_mix.grid(True, axis="y", alpha=0.3)
    ax_mix.legend(loc="best", fontsize=9, ncol=len(TAU_GRID))

    # Zoom to show the ±10-unit response around the starting K_E band.
    all_heights = np.concatenate(
        [r["K_E_end"] for r in results] + [start])
    pad = max(3.0, 0.25 * (all_heights.max() - all_heights.min()))
    ymin = float(all_heights.min() - pad)
    ymax = float(all_heights.max() + pad)
    ax_mix.set_ylim(ymin, ymax)

    fig.suptitle(
        f"Phase-2c carbon-tax sweep — {N_QUARTERS} quarters, "
        f"τ ∈ {list(TAU_GRID)}",
        fontsize=12)
    fig.savefig(OUT_FIG, dpi=150)
    print("=" * 72)
    print(f"Wrote comparative figure → {OUT_FIG}")

    # ------------------------------------------------------------------
    # Compact summary table
    # ------------------------------------------------------------------
    print()
    print(f"{'τ':>6}  {'cum emissions':>14}  {'cum rebate/hh':>14}  "
          f"{'max|res|':>10}  status")
    for r in results:
        print(f"{r['tau']:>6.2f}  "
              f"{r['em_stock'][-1]:>14.4e}  "
              f"{r['cum_rebate_pc'][-1]:>14.4e}  "
              f"{r['max_abs']:>10.3e}  "
              f"{r['status']}")
    print()

    # ------------------------------------------------------------------
    # Informational diagnostics (NOT pass/fail criteria)
    # ------------------------------------------------------------------
    # Monotonicity is NOT guaranteed: the rebate re-stimulates demand,
    # and capacity-constrained clean plants can't always absorb it, so
    # emissions can be non-monotonic in τ. This is an economically
    # meaningful response (revenue-recycling rebound), not a bug.
    em_terminal = np.array([r["em_stock"][-1] for r in results])
    print("Informational diagnostics:")
    mono_em = bool(np.all(np.diff(em_terminal) <= 0))
    print(f"  cumulative emissions monotone in τ? {mono_em} "
          f"(not required)")
    # Weighted-average ef of terminal K_E
    ef_end = np.array([
        float((r["K_E_end"] * r["ef"]).sum() / r["K_E_end"].sum())
        for r in results
    ])
    mono_ef = bool(np.all(np.diff(ef_end) <= 1e-12))
    print(f"  mean-ef(weighted) monotone in τ?    {mono_ef} "
          f"(not required)")
    for r, ef_val in zip(results, ef_end):
        print(f"    ef(τ={r['tau']:.1f}) = {ef_val:.6f}")

    all_pass = all(r["status"] == "PASS" for r in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
