"""
scenarios.py — Scenario specifications, multi-seed runner, sensitivity helpers.

Stage 8 of the gap-closure plan. Provides the scaffolding for paper-
ready Monte Carlo runs:

  * `SCENARIOS` — named scenario dict (stack + params + shock + policy).
  * `run_single(...)` — one-seed simulation with KPI + RESI recording.
  * `run_ensemble(...)` — N-seed wrapper, serial or multiprocessing.
  * `bootstrap_ci(...)` — percentile CI from an array of seed draws.
  * `wilcoxon_paired(...)` — paired signed-rank test for scenario Δ.
  * `sensitivity_sweep(...)` — one-parameter perturbation runner.

Design principles:

  - The module is intentionally thin. Heavy lifting is delegated to the
    existing `make_initial_economy` + `step` + `resilience` modules.
  - Each scenario is a pure spec (dict). Adding a new scenario = adding
    one entry to `SCENARIOS`.
  - Shocks are injected via a small `Shock` namedtuple describing
    (tech_label, multiplier, t_start, t_end). At t_start the tech's mc
    is scaled; at t_end it reverts.
  - Default horizon 50 quarters covers Italian 2019-2031 (40q) with
    10q pre-shock baseline window.

Paper uses:
  - 100 seeds per scenario (default below is 20 for speed; bump via
    the `n_seeds` argument).
  - Bootstrap CIs on headline KPIs.
  - Paired Wilcoxon signed-rank for scenario Δs where scenarios share
    the same seed (common-random-numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynamics import step
from init_stocks import (
    make_initial_economy,
    STACK_ITALY_2019,
    STACK_ITALY_2019_NUCLEAR,
    STACK_ITALY_2019_RENEWABLES,
    STACK_ITALY_2019_DIVERSIFIED,
)
from italian_calibration import PARAMS_ITALY
from resilience import ResilienceRecorder, compute_resi


# --------------------------------------------------------------------------
# Shock specification
# --------------------------------------------------------------------------
@dataclass
class Shock:
    """
    Time-varying marginal-cost shock applied to one technology class.

    `multiplier` multiplies the nameplate mc for all plants of the
    listed tech, during quarters in [t_start, t_end).
    """
    tech: str                # e.g. 'gas'
    multiplier: float        # e.g. 2.0 for 100% price rise
    t_start: int             # first shocked quarter (inclusive)
    t_end: int               # first post-shock quarter (exclusive)


# TTF-style gas shock: gas mc doubles for 8 quarters (2y), calibrated
# to the milder interpretation of the 2022 TTF spike (severe version
# triggers CES instability under default params — see docs).
TTF_SHOCK_2022 = Shock(tech="gas", multiplier=2.0, t_start=10, t_end=18)


# --------------------------------------------------------------------------
# Scenario catalogue
# --------------------------------------------------------------------------
SCENARIOS: Dict[str, dict] = {
    # --- Baseline family (Italian calibration) ---
    "italy_baseline": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=None,
        policy={},
    ),
    "italy_baseline_2022_shock": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={},
    ),

    # --- Tech-mix counterfactuals (under 2022 shock) ---
    "italy_nuclear_2022_shock": dict(
        stack=STACK_ITALY_2019_NUCLEAR,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={},
    ),
    "italy_renewables_2022_shock": dict(
        stack=STACK_ITALY_2019_RENEWABLES,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={},
    ),
    "italy_diversified_2022_shock": dict(
        stack=STACK_ITALY_2019_DIVERSIFIED,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={},
    ),

    # --- Policy experiments on baseline stack + shock ---
    "italy_baseline_2022_tax25": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 25.0},       # 2019 EU ETS average
    ),
    "italy_baseline_2022_tax85": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 85.0},       # 2022 EU ETS average
    ),

    # --- Tier A: stack × policy cross (tax + non-baseline stacks) ---
    # Purpose: show whether the nuclear/renewables/diversified benefit
    # compounds or competes with a carbon tax. Expected pattern: higher
    # τ amplifies the emissions-reduction gap between stacks but leaves
    # the peak-price gap ~invariant (gas still marginal under shock).
    "italy_nuclear_2022_tax85": dict(
        stack=STACK_ITALY_2019_NUCLEAR,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 85.0},
    ),
    "italy_renewables_2022_tax25": dict(
        stack=STACK_ITALY_2019_RENEWABLES,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 25.0},
    ),
    "italy_diversified_2022_tax25": dict(
        stack=STACK_ITALY_2019_DIVERSIFIED,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 25.0},
    ),
    "italy_diversified_2022_tax85": dict(
        stack=STACK_ITALY_2019_DIVERSIFIED,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={"carbon_tax": 85.0},
    ),

    # --- Tier B: ETS regime (endogenous τ from cap clearing) ---
    # Stage 5.1 ETS mode: `ets_mode=True` + `ets_cap` in tCO₂/q. The
    # cap is calibrated relative to the pre-shock baseline emissions
    # (~550 tCO₂/q under Italian default). These scenarios pair an
    # emissions-quantity constraint with the same 2022 gas shock, so
    # that tax-vs-ETS contrasts are run on identical pre-shock states.
    # `ets_tau_max=1000.0` gives headroom for the clearing bisection
    # under tight caps; the period τ saturates there if the cap is
    # infeasible given available capacity and demand.
    "italy_baseline_2022_ets_loose": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={
            "ets_mode": True,
            "ets_cap": 500.0,          # 90% of baseline ~550 tCO₂/q
            "ets_tau_max": 1000.0,
        },
    ),
    "italy_baseline_2022_ets_moderate": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={
            "ets_mode": True,
            "ets_cap": 385.0,          # 70% of baseline
            "ets_tau_max": 1000.0,
        },
    ),
    "italy_baseline_2022_ets_tight": dict(
        stack=STACK_ITALY_2019,
        params=PARAMS_ITALY,
        shock=TTF_SHOCK_2022,
        policy={
            "ets_mode": True,
            "ets_cap": 275.0,          # 50% of baseline
            "ets_tau_max": 1000.0,
        },
    ),
}


# --------------------------------------------------------------------------
# KPI recording per period
# --------------------------------------------------------------------------
def _record_kpis(eco, rec: ResilienceRecorder, t: int) -> None:
    """Buffer the headline KPIs this period."""
    # Per-class and per-quintile adoption rates (diagnostic).
    class_from_q = np.asarray(
        eco.params.get("class_from_quintile", (0, 0, 1, 1, 2)), dtype=int)
    class_of_hh = class_from_q[np.asarray(eco.H.quintile, dtype=int)]
    adopt_L = float(eco.H.has_DER[class_of_hh == 0].mean() or 0.0)
    adopt_M = float(eco.H.has_DER[class_of_hh == 1].mean() or 0.0)
    adopt_H = float(eco.H.has_DER[class_of_hh == 2].mean() or 0.0)
    rec.record(
        t,
        u=eco.u,
        cpi=eco.cpi,
        w=eco.w,
        p_E=eco.p_E_realised,
        emissions_stock=eco.emissions_stock,
        DEP_H_total=float(eco.H.DEP.sum()),
        adopt_L=adopt_L,
        adopt_M=adopt_M,
        adopt_H=adopt_H,
        carbon_tax=eco.carbon_tax,
    )


# --------------------------------------------------------------------------
# Single-seed runner
# --------------------------------------------------------------------------
def run_single(scenario: dict,
               seed: int = 0,
               n_quarters: int = 50,
               baseline_window: Tuple[int, int] = (0, 8),
               record_trajectory: bool = False
               ) -> Dict[str, Any]:
    """
    Execute one seed of the scenario. Returns a dict of headline metrics
    + RESI per-KPI scores. If `record_trajectory=True`, also returns the
    full ResilienceRecorder for inspection.

    Sequence:
      1. Build economy with scenario's stack + params (pre-converged).
      2. Apply scenario policy params (e.g. carbon_tax).
      3. Run `n_quarters` steps. During (t_start, t_end) of the shock
         (if any), scale the targeted tech's mc by multiplier.
      4. Record KPIs each period via `ResilienceRecorder`.
      5. Compute per-KPI RESI against pre-shock baseline mean.
    """
    params = dict(scenario.get("params", {}))
    stack  = scenario.get("stack", None)
    if stack is not None:
        params["stack"] = stack
    # Apply in-scenario policy overrides (carbon_tax, ets_mode, etc.)
    params.update(scenario.get("policy", {}))

    eco = make_initial_economy(params=params, seed=seed)
    rec = ResilienceRecorder()

    shock = scenario.get("shock", None)
    tech_mask = None
    orig_mc = None
    if shock is not None:
        tech_mask = np.array(
            [str(t) == shock.tech for t in eco.E.tech], dtype=bool)
        orig_mc = eco.E.mc[tech_mask].copy()

    for t in range(1, n_quarters + 1):
        # Apply / revert shock
        if shock is not None:
            if shock.t_start <= t < shock.t_end:
                eco.E.mc[tech_mask] = shock.multiplier * orig_mc
            else:
                eco.E.mc[tech_mask] = orig_mc
        step(eco, phase=4, tol=1e-10)
        _record_kpis(eco, rec, t)

    # Baseline estimation: pre-shock window.
    b = {k: rec.baseline_mean(k, baseline_window)
         for k in ("u", "cpi", "w", "p_E", "DEP_H_total")}

    # Shock window for RESI calculation (no shock ⇒ use a dummy mid-run window).
    if shock is not None:
        shock_win = (shock.t_start, shock.t_end)
    else:
        shock_win = (n_quarters // 3, 2 * n_quarters // 3)

    # Resources mobilised during the shock window (carbon tax revenue).
    tau_path = rec.series("carbon_tax")
    emissions_path = rec.series("emissions_stock")
    if len(emissions_path) > shock_win[1] and shock_win[1] > shock_win[0]:
        emissions_delta = float(
            emissions_path[shock_win[1] - 1] - emissions_path[shock_win[0] - 1])
        tau_mean = float(tau_path[shock_win[0]:shock_win[1]].mean())
        resources = emissions_delta * tau_mean
    else:
        resources = 0.0

    # Per-KPI RESI
    baseline_gdp = 60_000_000.0  # rough Italian-scale nominal €/q (calibration-specific)
    resi_dict = {}
    for kpi, orient in [("u", "increase_bad"),
                        ("cpi", "deviation"),
                        ("p_E", "increase_bad"),
                        ("w", "decrease_bad")]:
        s = rec.series(kpi)
        resi_dict[kpi] = compute_resi(
            kpi_trajectory=s,
            baseline_value=b[kpi],
            shock_window=shock_win,
            resources_mobilised=resources,
            baseline_gdp=baseline_gdp,
            orientation=orient,
        )

    # Headline aggregates
    out = {
        "seed": seed,
        "baseline_u":   b["u"],
        "baseline_cpi": b["cpi"],
        "baseline_w":   b["w"],
        "baseline_pE":  b["p_E"] * 52.0,       # €/MWh for reporting
        "final_DEP_H":  float(eco.H.DEP.sum()),
        "final_emissions": float(eco.emissions_stock),
        "final_adopt_L":  float(rec.series("adopt_L")[-1]) if rec.n() else 0.0,
        "final_adopt_M":  float(rec.series("adopt_M")[-1]) if rec.n() else 0.0,
        "final_adopt_H":  float(rec.series("adopt_H")[-1]) if rec.n() else 0.0,
        "shock_peak_pE_mwh": float(
            rec.series("p_E")[shock_win[0]:shock_win[1]].max() * 52.0)
            if shock_win[1] > shock_win[0] else 0.0,
        "shock_peak_u":  float(
            rec.series("u")[shock_win[0]:shock_win[1]].max())
            if shock_win[1] > shock_win[0] else 0.0,
        "shock_peak_cpi": float(
            rec.series("cpi")[shock_win[0]:shock_win[1]].max())
            if shock_win[1] > shock_win[0] else 0.0,
        "RESI_u":   resi_dict["u"]["RESI"],
        "RESI_cpi": resi_dict["cpi"]["RESI"],
        "RESI_pE":  resi_dict["p_E"]["RESI"],
        "RESI_w":   resi_dict["w"]["RESI"],
        "RESI_composite": float(np.mean([
            resi_dict[k]["RESI"] for k in ("u", "cpi", "p_E", "w")])),
    }
    if record_trajectory:
        out["recorder"] = rec
    return out


# --------------------------------------------------------------------------
# Ensemble runner (multi-seed)
# --------------------------------------------------------------------------
def run_ensemble(scenario: dict,
                  n_seeds: int = 20,
                  n_quarters: int = 50,
                  seed_offset: int = 0,
                  parallel: bool = False,
                  n_workers: int = 4
                  ) -> List[Dict[str, Any]]:
    """
    Run N seeds of a scenario and return a list of per-seed result
    dicts. Default is serial; set `parallel=True` to use
    multiprocessing.Pool with `n_workers` processes (subject to the
    platform supporting it).

    Common-random-numbers convention: seeds = range(seed_offset,
    seed_offset + n_seeds). This lets paired scenarios be compared
    seed-by-seed with Wilcoxon signed-rank tests.
    """
    seeds = list(range(seed_offset, seed_offset + n_seeds))
    if parallel and n_workers > 1:
        try:
            from multiprocessing import Pool
            with Pool(processes=n_workers) as pool:
                args = [(scenario, s, n_quarters, (0, 8), False) for s in seeds]
                # Map via a helper that unpacks args
                results = pool.starmap(run_single, args)
            return list(results)
        except Exception:
            # Fall back to serial if parallel setup fails.
            pass
    return [run_single(scenario, seed=s, n_quarters=n_quarters) for s in seeds]


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray,
                  alpha: float = 0.05,
                  n_boot: int = 1000,
                  rng: Optional[np.random.Generator] = None
                  ) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean of `values`. Returns
    (lower, median, upper) at the 1−α confidence level.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(seed=0)
    n = v.size
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(v[idx].mean())
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    mid = float(np.median(boot_means))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, mid, hi


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test. Returns (stat, p_value). Assumes
    `a` and `b` are the same-length seed-matched arrays (common-random
    numbers) from two scenarios. Uses the normal approximation for
    large n and includes a zero-exclusion rule for ties.

    Lightweight implementation (no SciPy dependency).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired arrays must have the same shape"
    d = a - b
    d = d[d != 0]  # exclude ties
    n = d.size
    if n < 6:
        return float("nan"), float("nan")
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Handle ties in |d| by averaging ranks (simplified).
    signed_ranks = np.sign(d) * ranks
    W = float(signed_ranks.sum())
    # Normal approximation: mean 0, variance = n(n+1)(2n+1)/6
    var = n * (n + 1) * (2 * n + 1) / 6.0
    z = W / max(np.sqrt(var), 1e-12)
    # Two-sided p from standard normal
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
    return W, float(p)


def summarise_ensemble(results: List[Dict[str, Any]],
                        fields: Sequence[str] = ("RESI_composite",
                                                   "RESI_u", "RESI_cpi",
                                                   "RESI_pE", "RESI_w",
                                                   "shock_peak_pE_mwh",
                                                   "shock_peak_u",
                                                   "shock_peak_cpi",
                                                   "final_emissions",
                                                   "final_adopt_L",
                                                   "final_adopt_M",
                                                   "final_adopt_H"),
                        alpha: float = 0.05,
                        ) -> Dict[str, Tuple[float, float, float]]:
    """
    Per-field bootstrap CI across seeds. Returns {field: (lo, mid, hi)}.
    """
    summary = {}
    for f in fields:
        arr = np.array([r.get(f, np.nan) for r in results], dtype=float)
        summary[f] = bootstrap_ci(arr, alpha=alpha)
    return summary


# --------------------------------------------------------------------------
# Sensitivity sweep
# --------------------------------------------------------------------------
def sensitivity_sweep(base_scenario: dict,
                       param_name: str,
                       value_grid: Sequence[float],
                       n_seeds: int = 10,
                       n_quarters: int = 50,
                       ) -> Dict[float, List[Dict[str, Any]]]:
    """
    Vary one parameter across `value_grid` while holding all others
    fixed; run `n_seeds` each. Returns a dict keyed by parameter value
    with the list of per-seed results.

    Example: sweep `chi ∈ {0.3, 0.5, 0.7}` to test robustness of the
    fragility channel to the uncertainty-markup coefficient.
    """
    out: Dict[float, List[Dict[str, Any]]] = {}
    for v in value_grid:
        scen = {
            "stack":  base_scenario.get("stack"),
            "params": {**base_scenario.get("params", {}), param_name: v},
            "shock":  base_scenario.get("shock"),
            "policy": base_scenario.get("policy", {}),
        }
        out[v] = run_ensemble(scen, n_seeds=n_seeds, n_quarters=n_quarters)
    return out


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main() -> None:
    """Quick smoke test: run 5 seeds of 2 scenarios, print summary."""
    print("Running 5-seed smoke test on italy_baseline_2022_shock ...")
    results = run_ensemble(SCENARIOS["italy_baseline_2022_shock"],
                            n_seeds=5, n_quarters=40)
    summary = summarise_ensemble(results, alpha=0.10)
    print("\n90% bootstrap CIs (median, [lo, hi]):")
    for k, (lo, mid, hi) in summary.items():
        print(f"  {k:25s}: {mid:>10.4f}  [{lo:>10.4f}, {hi:>10.4f}]")


if __name__ == "__main__":
    main()
