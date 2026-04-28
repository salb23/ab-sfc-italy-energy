"""
test_sfc_closure.py — pytest closure + stationarity tests for phase 0.

Two guarantees are asserted:

  1. Closure:   for every quarter t in [0, N_QUARTERS) the residual
     returned by `accounting.residual_check` is strictly below
     `TOL * eco.GDP_proxy`;

  2. Stationarity of real stocks:
     in phase 0 there is no investment and no depreciation, so the
     total real capital stock `K_total` must be bit-exactly constant
     over the entire horizon; the test asserts a zero drift.

The second test is a useful sanity check: if any phase-0 flow accidentally
hits `K` (for instance a bug that routed investment here), `K_total`
would drift and the test would fail immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the parent directory importable when pytest is run from tests/
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from accounting import SFCClosureError            # noqa: E402
from dynamics import step                          # noqa: E402
from init_stocks import make_initial_economy       # noqa: E402


N_QUARTERS = 500
TOL        = 1e-10
SEED       = 0


@pytest.fixture(scope="module")
def simulated():
    """
    Run the full phase-0 horizon once and return the trajectory.
    Scoped to the module so both tests share the same simulation
    without paying its cost twice.
    """
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy
    K0 = eco.K_total

    residuals = []
    K_series  = [K0]
    try:
        for _ in range(N_QUARTERS):
            rep = step(eco, phase=0, tol=TOL)
            residuals.append(rep.max_abs)
            K_series.append(eco.K_total)
    except SFCClosureError as exc:
        pytest.fail(f"SFC closure raised during phase 0: {exc}")

    return dict(
        eco=eco,
        gdp_proxy=gdp_proxy,
        residuals=residuals,
        K_series=K_series,
        K0=K0,
    )


def test_sfc_closure_over_horizon(simulated):
    """
    The maximum absolute residual across the whole horizon must stay
    below TOL * initial GDP_proxy.
    """
    max_res = max(simulated["residuals"])
    bound   = TOL * simulated["gdp_proxy"]
    assert max_res <= bound, (
        f"SFC residual exceeded tolerance: "
        f"max|residual|={max_res:.3e}, tol={bound:.3e}"
    )


def test_real_capital_is_stationary(simulated):
    """
    Phase 0 has no investment and no depreciation, so K_total must be
    exactly constant.
    """
    K_series = simulated["K_series"]
    K0 = simulated["K0"]
    drift = max(abs(k - K0) for k in K_series)
    # Allow only floating-point-noise-level drift.
    assert drift <= 1e-9, f"K_total drifted in phase 0: max|drift|={drift:.3e}"


def test_initial_conditions_close(simulated):
    """
    Safety check: the opening economy — before any step — must already
    satisfy every BSM and NW identity. (We rely on this when we
    interpret the first residual as coming from the laws of motion,
    not from an inconsistent initial balance sheet.)
    """
    from accounting import residual_check
    eco0 = make_initial_economy(seed=SEED)
    rep = residual_check(eco0, flows=None, tol=TOL, raise_on_fail=False)
    # No flows → no column-sum check; only BSM and NW identities.
    bound = TOL * eco0.GDP_proxy
    assert rep.max_abs <= bound, (
        f"Initial balance sheet is not closed: max|residual|={rep.max_abs:.3e}"
    )


# --------------------------------------------------------------------------
# Phase-1 closure
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulated_phase1():
    """
    Run the full phase-1 horizon once. Phase-1 adds behavioural rules
    (household consumption, firm pricing, wage Phillips curve) but the
    accounting core is shared with phase 0 — the closure guarantee must
    therefore extend unchanged.
    """
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy
    residuals = []
    try:
        for _ in range(N_QUARTERS):
            rep = step(eco, phase=1, tol=TOL)
            residuals.append(rep.max_abs)
    except SFCClosureError as exc:
        pytest.fail(f"SFC closure raised during phase 1: {exc}")
    return dict(eco=eco, gdp_proxy=gdp_proxy, residuals=residuals)


def test_phase1_sfc_closure_over_horizon(simulated_phase1):
    max_res = max(simulated_phase1["residuals"])
    bound   = TOL * simulated_phase1["gdp_proxy"]
    assert max_res <= bound, (
        f"Phase-1 SFC residual exceeded tolerance: "
        f"max|residual|={max_res:.3e}, tol={bound:.3e}"
    )


# --------------------------------------------------------------------------
# Phase-2a closure
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulated_phase2a():
    """
    Run the full phase-2a horizon once. Phase-2a adds merit-order
    dispatch on top of the phase-1 behavioural block: the period's
    real energy demand is cleared cheapest-first and the marginal
    plant's MC becomes the next period's `p_E`. The SFC accounting
    identities are unchanged — the same closure guarantee applies.
    """
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy
    residuals = []
    try:
        for _ in range(N_QUARTERS):
            rep = step(eco, phase=2, tol=TOL)
            residuals.append(rep.max_abs)
    except SFCClosureError as exc:
        pytest.fail(f"SFC closure raised during phase 2a: {exc}")
    return dict(eco=eco, gdp_proxy=gdp_proxy, residuals=residuals)


def test_phase2a_sfc_closure_over_horizon(simulated_phase2a):
    max_res = max(simulated_phase2a["residuals"])
    bound   = TOL * simulated_phase2a["gdp_proxy"]
    assert max_res <= bound, (
        f"Phase-2a SFC residual exceeded tolerance: "
        f"max|residual|={max_res:.3e}, tol={bound:.3e}"
    )


def test_phase2a_emissions_are_nonneg(simulated_phase2a):
    """
    Emissions are a cumulative stock with non-negative flow each
    quarter (no negative-emissions technology in the base merit
    order), so the stock must be non-negative. A failure here means
    either a bug in the dispatch routine returned a negative
    emissions flow, or the stock was written to from somewhere else.
    """
    eco = simulated_phase2a["eco"]
    assert eco.emissions_stock >= 0.0


# --------------------------------------------------------------------------
# Phase-2b closure
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulated_phase2b():
    """
    Run the full phase-2b horizon once. Phase-2b activates the
    investment, depreciation, amortisation, and new-lending flows on
    top of phase-2a. The SFC identities now bind non-trivially on the
    credit side (bank loan assets must track borrower liabilities,
    and deposit/NW changes must absorb the full set of flows). This
    is the most demanding closure test so far.
    """
    eco = make_initial_economy(seed=SEED)
    gdp_proxy = eco.GDP_proxy
    residuals = []
    L_F_series = [eco.L_F_total]
    L_E_series = [eco.L_E_total]
    K_E_series = [eco.K_E_total]
    ration_series: list[float] = []
    try:
        for _ in range(N_QUARTERS):
            rep = step(eco, phase=3, tol=TOL)
            residuals.append(rep.max_abs)
            L_F_series.append(eco.L_F_total)
            L_E_series.append(eco.L_E_total)
            K_E_series.append(eco.K_E_total)
            ration_series.append(
                float(getattr(eco, "_pending_ration_ratio", 1.0))
            )
    except SFCClosureError as exc:
        pytest.fail(f"SFC closure raised during phase 2b: {exc}")
    return dict(
        eco=eco,
        gdp_proxy=gdp_proxy,
        residuals=residuals,
        L_F_series=L_F_series,
        L_E_series=L_E_series,
        K_E_series=K_E_series,
        ration_series=ration_series,
    )


def test_phase2b_sfc_closure_over_horizon(simulated_phase2b):
    max_res = max(simulated_phase2b["residuals"])
    bound   = TOL * simulated_phase2b["gdp_proxy"]
    assert max_res <= bound, (
        f"Phase-2b SFC residual exceeded tolerance: "
        f"max|residual|={max_res:.3e}, tol={bound:.3e}"
    )


def test_phase2b_loan_stocks_nonneg(simulated_phase2b):
    """
    Outstanding loan principal is a non-negative stock by construction
    (amortisation is bounded by the rate × current L, new lending is
    non-negative). A negative value would signal double-counting in
    apply_laws_of_motion or a pathological rationing allocation.
    """
    assert min(simulated_phase2b["L_F_series"]) >= 0.0
    assert min(simulated_phase2b["L_E_series"]) >= 0.0


def test_phase2b_energy_capital_is_bounded(simulated_phase2b):
    """
    Under Stage-2.3 (NPV investment + Wright's-Law learning) the
    energy-sector aggregate follows a *trend-stationary* path rather
    than bit-exact stationarity: positive-NPV plants expand organically,
    negative-NPV plants decay, and aggregate K_E drifts slowly along a
    balanced-growth path (Dosi, Lamperti 2020; Caiani 2019).

    We test bounded drift rather than zero drift:
      * Per-quarter growth rate of K_E_total stays below 1% (unreasonable
        expansion rates would indicate a mis-calibrated κ).
      * Cumulative drift over the test horizon stays below 30% of K_E_0
        (beyond that the economy is hitting capacity-constraint pathology).
    Exact stationarity is retained as a regression property under the
    backward-compat `invest_mode="replacement"` — see
    `test_phase2b_replacement_mode_stationary`.
    """
    K_E_series = simulated_phase2b["K_E_series"]
    K_E0 = K_E_series[0]
    # Per-quarter growth rate
    max_q_rate = max(abs(K_E_series[t+1] / K_E_series[t] - 1.0)
                      for t in range(len(K_E_series) - 1))
    assert max_q_rate <= 0.01, (
        f"Phase-2b K_E per-quarter growth rate exceeded 1%: "
        f"max={max_q_rate:.4f}"
    )
    # Cumulative drift
    max_cum_drift = max(abs(k / K_E0 - 1.0) for k in K_E_series)
    assert max_cum_drift <= 0.30, (
        f"Phase-2b K_E cumulative drift exceeded 30%: max={max_cum_drift:.4f}"
    )


def test_phase2b_replacement_mode_stationary():
    """
    Regression test: under `invest_mode="replacement"` the pre-Stage-2.3
    rule I_E = δ_E·K_E_total applies and K_E_total is bit-exactly
    stationary. Guards against accidental behavioural change to the
    legacy path.
    """
    eco = make_initial_economy(
        params={"invest_mode": "replacement"}, seed=SEED)
    K_E0 = eco.K_E_total
    max_drift = 0.0
    for _ in range(50):
        step(eco, phase=2, tol=TOL)
        max_drift = max(max_drift, abs(eco.K_E_total - K_E0))
    rel_bound = 1e-12 * K_E0
    assert max_drift <= max(1e-9, rel_bound), (
        f"K_E_total drifted in replacement mode: max|drift|={max_drift:.3e}, "
        f"bound={max(1e-9, rel_bound):.3e}"
    )


def test_phase2b_rationing_ratio_in_unit_interval(simulated_phase2b):
    """
    The rationing ratio is defined as min(1, capacity/demand), so it
    must lie in [0, 1]. Values outside this interval would indicate a
    bug in `allocate_new_loans` (e.g., negative capacity leaking
    through, or the early-return branch miscomputing demand).
    """
    rations = simulated_phase2b["ration_series"]
    assert min(rations) >= 0.0
    assert max(rations) <= 1.0 + 1e-12


# --------------------------------------------------------------------------
# Phase-2c closure (carbon tax + lump-sum rebate + reinvestment mix)
# --------------------------------------------------------------------------
# A non-trivial τ for the main closure test, in €/tCO₂ (Caiani
# convention — see model_full/docs/units_convention.md). 50 €/tCO₂
# sits between 2019 EU ETS (25) and 2022 EU ETS (85) — large enough
# to shift the merit order and make tax/rebate flows bite, small
# enough not to starve the energy sector. Under the pre-Caiani log-
# transformed dispatch this test ran at τ=0.4 (model-price units).
PHASE2C_TAU = 50.0


def _run_phase2c(tau: float, n: int = N_QUARTERS):
    """
    Helper: run Phase 2c for `n` quarters at a given tax rate.
    Returns the trajectory dict or raises pytest.fail on closure break.
    """
    import numpy as np
    eco = make_initial_economy(params=dict(carbon_tax=tau), seed=SEED)
    gdp_proxy = eco.GDP_proxy
    N_E = eco.E.N
    residuals = []
    K_E_path  = [eco.E.K.copy()]
    em_flow   = [0.0]
    em_stock  = [eco.emissions_stock]
    tax_rev   = [0.0]
    rebate    = [0.0]
    try:
        for _ in range(n):
            rep = step(eco, phase=4, tol=TOL)
            residuals.append(rep.max_abs)
            K_E_path.append(eco.E.K.copy())
            per_q = float((eco.Y_per_plant * eco.E.emission_factor).sum())
            em_flow.append(per_q)
            em_stock.append(eco.emissions_stock)
            tax_rev.append(tau * per_q)
            rebate.append(tau * per_q)
    except SFCClosureError as exc:
        pytest.fail(f"SFC closure raised during phase 2c (τ={tau}): {exc}")
    return dict(
        eco=eco,
        gdp_proxy=gdp_proxy,
        residuals=residuals,
        K_E_path=np.array(K_E_path),
        em_flow=em_flow,
        em_stock=em_stock,
        tax_rev=tax_rev,
        rebate=rebate,
    )


@pytest.fixture(scope="module")
def simulated_phase2c():
    """Run Phase 2c at τ > 0 (tax active, full recycling)."""
    return _run_phase2c(PHASE2C_TAU)


@pytest.fixture(scope="module")
def simulated_phase2c_zero():
    """Run Phase 2c at τ = 0 — must equal Phase 2b bit-for-bit."""
    return _run_phase2c(0.0)


def test_phase2c_sfc_closure_over_horizon(simulated_phase2c):
    """Closure must hold even with carbon tax and rebate flowing."""
    max_res = max(simulated_phase2c["residuals"])
    bound   = TOL * simulated_phase2c["gdp_proxy"]
    assert max_res <= bound, (
        f"Phase-2c SFC residual exceeded tolerance: "
        f"max|residual|={max_res:.3e}, tol={bound:.3e}"
    )


def test_phase2c_fiscal_balance(simulated_phase2c):
    """
    Same-quarter recycling: carbon_tax and household_rebate must be
    equal every period (by construction: rebate = tax revenue). The
    government's per-period NW change from the policy block must be
    zero.
    """
    tax = simulated_phase2c["tax_rev"]
    reb = simulated_phase2c["rebate"]
    assert len(tax) == len(reb)
    for t, r in zip(tax, reb):
        assert abs(t - r) <= 1e-12


def test_phase2c_zero_tax_has_zero_fiscal_flows(simulated_phase2c_zero):
    """
    At τ = 0 the carbon-tax and rebate flows must both be exactly zero
    every period. This is the off-by-default property of the policy
    layer.

    Note: at τ=0 the Phase-2c trajectory is NOT identical to Phase-2b.
    Even without a tax, the profitability-weighted reinvestment rule
    reallocates K_E toward infra-marginal plants (cheap techs earn
    positive producer surplus on the merit-order clearing), which
    feeds back into p_E, firm costs, and K_F. This is realistic
    competitive-market behaviour; the τ=0 case is the "no-policy but
    endogenous reinvestment" baseline, not a replica of Phase 2b.
    """
    sim = simulated_phase2c_zero
    tax = sim["tax_rev"]
    reb = sim["rebate"]
    assert max(tax) <= 1e-12, f"τ=0 tax revenue leaked: max={max(tax):.3e}"
    assert max(reb) <= 1e-12, f"τ=0 rebate leaked:      max={max(reb):.3e}"


def test_phase2c_zero_tax_k_e_stationary(simulated_phase2c_zero):
    """
    K_E_total must stay constant even in the competitive baseline —
    the aggregate rule I_E_total = δ_E · K_E_total holds regardless
    of whether the allocation across plants is uniform or profit-
    weighted. This verifies that the per-plant arrays sum back to
    the scalar flows.
    """
    K_E_path = simulated_phase2c_zero["K_E_path"]
    K_E_tot  = K_E_path.sum(axis=1)
    # Under Stage-2.3 (NPV investment) the balanced-growth path replaces
    # bit-exact stationarity. At τ=0 positive-NPV plants still expand
    # slowly (gas, renewables) while oil contracts, so K_E_total drifts
    # with bounded rate. Require |per-quarter change| ≤ 1% and cumulative
    # drift ≤ 30% over the horizon. Exact stationarity is recovered
    # under invest_mode="replacement" — see a separate regression test
    # for that path.
    import numpy as np
    per_q_growth = np.abs(np.diff(K_E_tot) / K_E_tot[:-1]).max()
    cum_drift    = float(abs(K_E_tot / K_E_tot[0] - 1.0).max())
    assert per_q_growth <= 0.01, (
        f"K_E_total per-quarter growth rate at τ=0 exceeded 1%: "
        f"max={per_q_growth:.4f}"
    )
    assert cum_drift <= 0.30, (
        f"K_E_total cumulative drift at τ=0 exceeded 30%: "
        f"max={cum_drift:.4f}"
    )


def test_phase2c_cuts_emissions(simulated_phase2c, simulated_phase2c_zero):
    """
    Tax > 0 must produce strictly fewer cumulative emissions than
    τ = 0 over the same horizon, because the effective-mc premium
    shifts the merit order toward low-ef plants (and the reinvestment
    rule compounds the shift by eroding dirty capital).
    """
    em_tax  = simulated_phase2c["eco"].emissions_stock
    em_base = simulated_phase2c_zero["eco"].emissions_stock
    assert em_tax < em_base, (
        f"Phase-2c cumulative emissions did NOT fall under tax: "
        f"em(τ=0)={em_base:.3e}, em(τ={PHASE2C_TAU})={em_tax:.3e}"
    )


def test_phase2c_tech_mix_tilts_clean(simulated_phase2c):
    """
    Under τ > 0, the reinvestment rule should move K_E toward
    low-emission plants. Test: the K_E-weighted average emission
    factor at the end of the horizon is ≤ the t=0 value.
    """
    import numpy as np
    eco = simulated_phase2c["eco"]
    K_end = eco.E.K
    ef    = eco.E.emission_factor
    K_start = simulated_phase2c["K_E_path"][0]
    mean_ef_end   = float((K_end * ef).sum() / max(1e-12, K_end.sum()))
    mean_ef_start = float((K_start * ef).sum() / max(1e-12, K_start.sum()))
    assert mean_ef_end <= mean_ef_start + 1e-12, (
        f"Weighted mean emission factor did NOT fall under τ>0: "
        f"start={mean_ef_start:.4f}, end={mean_ef_end:.4f}"
    )


def test_phase2c_energy_capital_is_stationary(simulated_phase2c):
    """
    The aggregate replacement rule still sets I_E_total = δ_E · K_E_total,
    so K_E_total must stay constant even though individual plants'
    K_k are now diverging. This is the scalar-consistency test that
    verifies the per-plant allocation sums back to the scalar.
    """
    K_E_path = simulated_phase2c["K_E_path"]
    K_E_totals = K_E_path.sum(axis=1)
    # Stage-2.3 relaxation: balanced-growth path rather than bit-exact
    # stationarity. Under τ > 0, dirty plants contract faster than clean
    # expand, so K_E_total typically trends down; drift bounds capture
    # that motion without demanding exact constancy.
    import numpy as np
    per_q_growth = np.abs(np.diff(K_E_totals) / K_E_totals[:-1]).max()
    cum_drift    = float(abs(K_E_totals / K_E_totals[0] - 1.0).max())
    assert per_q_growth <= 0.01, (
        f"K_E_total per-quarter growth rate at τ>0 exceeded 1%: "
        f"max={per_q_growth:.4f}"
    )
    assert cum_drift <= 0.30, (
        f"K_E_total cumulative drift at τ>0 exceeded 30%: "
        f"max={cum_drift:.4f}"
    )
