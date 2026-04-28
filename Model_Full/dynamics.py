"""
dynamics.py — period-step function for the full AB-SFC framework.

Phase 0 is pure accounting: no heterogeneous behaviour, no stochastic
draws, no policy shocks. Every flow is defined by a deterministic
rule whose coefficients come from `Economy.params`. The sole purpose
of phase 0 is to prove that the laws of motion in v2.1 §3.10 together
with the TFM/BSM identities close exactly.

Later phases replace `make_phase0_flows` with behaviourally-richer
flow builders but keep the same interface, so `apply_laws_of_motion`
and `residual_check` stay unchanged.
"""

from __future__ import annotations

from accounting import (
    ResidualReport,
    TFMFlows,
    apply_laws_of_motion,
    residual_check,
)
from agents import Economy
from behaviour import (
    adopt_der,
    ces_share_F,
    composite_labour_coeff,
    dividend_flow,
    efficiency_investment,
    firm_pricing,
    solve_consumption,
    wage_update,
    wage_update_by_class,
)
from energy_market import dispatch
from credit import (
    allocate_new_loans,
    bank_lending_capacity,
    bank_leverage,
    concentration_headroom,
    loan_demand_E,
    loan_demand_F,
    loan_rate_risk_adjusted,
    risk_weighted_assets,
)


def _loan_interest_flows(eco, p) -> tuple[float, float]:
    """
    Aggregate interest flows from F and E borrowers to the banking sector,
    computed bank-by-bank under the Stage-3.1 risk-adjusted pricing rule.

    Under `loan_rate_mode = "scalar"` we fall back to the pre-Stage-3.1
    flat-rate path (r_L · L_sector_total) for bit-exact regression.

    Returns (r_L_F_amt, r_L_E_amt): nominal interest flows paid by F and
    E respectively.
    """
    mode = p.get("loan_rate_mode", "scalar")
    if mode != "risk_based":
        return (
            float(p["r_L"]) * float(eco.L_F_total),
            float(p["r_L"]) * float(eco.L_E_total),
        )
    # Risk-based pricing: each bank charges its own rate, interest flows
    # aggregate across the (r_L_b × L_sector_b) products.
    import numpy as _np
    lev = bank_leverage(eco.B.L_F, eco.B.L_E, eco.B.NW)
    r_L_b = loan_rate_risk_adjusted(
        lev=lev,
        r_CB=float(p.get("r_CB", 0.0)),
        mu_b=float(p.get("mu_b", p.get("r_L", 0.0))),
        delta_bar=float(p.get("delta_bar", 0.0)),
        gamma=float(p.get("gamma_lev", 2.0)),
        lambda_max=float(p.get("lambda_max", 20.0)),
    )
    r_L_F_amt = float(_np.sum(r_L_b * _np.asarray(eco.B.L_F, dtype=float)))
    r_L_E_amt = float(_np.sum(r_L_b * _np.asarray(eco.B.L_E, dtype=float)))
    return r_L_F_amt, r_L_E_amt
from investment import (
    depreciation,
    energy_replacement_investment,
    firm_investment,
)
from policy import (
    carbon_tax_flow,
    lump_sum_rebate,
    reinvestment_weights,
)

import numpy as np


# --------------------------------------------------------------------------
# Phase-0 flow builder
# --------------------------------------------------------------------------
def make_phase0_flows(eco: Economy) -> TFMFlows:
    """
    Build the period's TFM flows under the phase-0 rules:

      - aggregate wage bill W = wage_per_q · N_H, split 0.8/0.2 between F and E;
      - direct tax T_H = tax_rate · W;
      - transfer Tr chosen so that the government runs a balanced budget
        (T_H - Tr - r_GB·GB_B = 0);
      - interest income/expense proportional to current stocks;
      - disposable income YD_H = W + Tr + r_D·DEP_H - T_H;
      - nominal consumption C_H = mpc · YD_H, split cons_share_F / cons_share_E;
      - firm energy bill EB_F = 0.30 · W_F (a fixed fraction of the non-energy
        wage bill used as a stand-in for energy input cost);
      - no new lending, no investment, no dividends, no new bond issuance.
    """
    p = eco.params
    wage = p["wage_per_q"]
    N_H = eco.H.N

    # Wages
    W = wage * N_H
    W_F = 0.80 * W
    W_E = 0.20 * W

    # Taxes
    T_H = p["tax_rate"] * W

    # Interest flows. Loan interest uses per-bank risk-adjusted rates
    # under the Stage-3.1 pricing rule (falls back to scalar r_L if
    # loan_rate_mode = "scalar").
    r_D_H  = p["r_D"]  * eco.DEP_H
    r_D_F  = p["r_D"]  * eco.DEP_F
    r_D_E  = p["r_D"]  * eco.DEP_E
    r_L_F, r_L_E = _loan_interest_flows(eco, p)
    r_GB_B = p["r_GB"] * eco.GB_B

    # Balanced-budget transfer:  T_H - Tr_H - r_GB_B = 0
    Tr_H = T_H - r_GB_B

    # Disposable income and consumption
    YD_H = W + Tr_H + r_D_H - T_H
    C_H  = p["mpc"] * YD_H
    C_H_F = p["cons_share_F"] * C_H
    EB_H  = p["cons_share_E"] * C_H

    # Firm energy bill — fixed fraction of non-energy wage bill
    EB_F = 0.30 * W_F

    flows = TFMFlows(
        W=W,
        C_H_F=C_H_F,
        EB_H=EB_H,
        EB_F=EB_F,
        G_spend=0.0,
        T_H=T_H,
        Tr_H=Tr_H,
        r_D_H=r_D_H,
        r_D_F=r_D_F,
        r_D_E=r_D_E,
        r_L_F=r_L_F,
        r_L_E=r_L_E,
        r_GB_B=r_GB_B,
        r_GB_CB=0.0,
        r_Res=0.0,
        new_loans_F=0.0,
        new_loans_E=0.0,
        amort_F=0.0,
        amort_E=0.0,
        writeoff_F=0.0,
        writeoff_E=0.0,
        GB_issued=0.0,
        GB_amort=0.0,
        I_F=0.0,
        I_E=0.0,
        depreciation_F=0.0,
        depreciation_E=0.0,
        dividends_F=0.0,
        dividends_B=0.0,
    )
    # Attach the F/E wage split for the accounting helper to pick up
    flows._W_F = W_F   # type: ignore[attr-defined]
    flows._W_E = W_E   # type: ignore[attr-defined]
    return flows


# --------------------------------------------------------------------------
# Phase-1 flow builder  (household + firm + labour behaviour)
# --------------------------------------------------------------------------
def make_phase1_flows(eco: Economy) -> TFMFlows:
    """
    Build the period's TFM flows under phase-1 behavioural rules.

    Decisions are timed so that everything reduces to one closed-form
    equation for nominal consumption:

      1. Firms set the representative F-price from current wage & energy price
         (mark-up rule).
      2. Households choose nominal consumption from the two-term rule
         C = α1·YD + α2·NW_H. Substituting W = w·C·λ (aggregate wage bill
         as a function of total demand) gives a linear equation in C
         with closed-form solution (see behaviour.solve_consumption).
      3. Firm and energy output follow from CES demand shares.
      4. Labour demand is the total workers required to produce that output.
         If it exceeds labour supply N_H, production is scaled back
         proportionally (simple one-sided rationing).
      5. Government runs a balanced budget (Tr_H = T_H - r_GB_B), as in
         phase 0. No investment, no new loans, no dividends.
      6. End of period: the wage Phillips curve sets w for t+1; p_F and
         cpi are stored for next period's decision problem.
    """
    p = eco.params

    # ----- Parameters -----
    w        = eco.w
    # Paper §3.7 Eq. 15: firm pricing and household consumption decisions
    # read the adaptive price expectation, not the raw last-realised
    # price. The expectation is updated at commit time (_commit_phase2a_state)
    # from the realised clearing price delivered by the dispatch block.
    p_E      = eco.p_E_expected
    xi_F     = p["productivity_F"]
    xi_E     = p["productivity_E"]
    ei_avg   = float(eco.F.energy_intensity.mean())
    taste_F  = p["taste_F"]
    sigma    = p["ces_sigma"]
    markup   = p["markup"]
    alpha1   = p["alpha1"]
    alpha2   = p["alpha2"]
    tax_rate = p["tax_rate"]
    theta    = p.get("theta_div", 0.0)
    N_H      = eco.H.N

    # ----- Lagged dividends (predetermined from previous period) -----
    # Caiani closure: each period, F, E and B pay out θ of their PRIOR-
    # period cash profit to households. 1-period lag keeps the
    # consumption fixed-point closed form in this period.
    #
    # Bank dividends are necessary, not cosmetic: banks collect
    # (r_L_F + r_L_E + r_GB_B) and pay (r_D_H + r_D_F + r_D_E) as
    # interest. The gap is a structural "net interest margin" that
    # otherwise accumulates as bank NW forever and drains household
    # deposits one-for-one. With Div_B = θ·π_B the B column closes too.
    Div_F = dividend_flow(last_profit=getattr(eco, "last_profit_F", 0.0),
                          theta_div=theta)
    Div_E = dividend_flow(last_profit=getattr(eco, "last_profit_E", 0.0),
                          theta_div=theta)
    Div_B = dividend_flow(last_profit=getattr(eco, "last_profit_B", 0.0),
                          theta_div=theta)
    Div_prev_total = Div_F + Div_E + Div_B

    # ----- 1. Firm pricing (uncertainty-loaded markup, Stage 2.1) -----
    # σ̂_PE is the rolling coefficient of variation of the dispatch
    # clearing price, maintained by _commit_phase2a_state. At steady
    # state σ̂_PE → 0 and the markup collapses to the base ψ̅.
    chi      = float(p.get("chi", 0.0))
    sigma_PE = float(getattr(eco, "p_E_volatility", 0.0))
    p_F = firm_pricing(w=w, p_E=p_E, ei_avg=ei_avg,
                       xi_F=xi_F, markup=markup,
                       chi=chi, sigma_PE=sigma_PE)

    # ----- 2. CES demand shares and composite labour coefficient -----
    s_F = ces_share_F(p_F=p_F, p_E=p_E, taste_F=taste_F, sigma=sigma)
    lam = composite_labour_coeff(s_F=s_F, p_F=p_F, p_E=p_E,
                                 ei_avg=ei_avg, xi_F=xi_F, xi_E=xi_E)

    # ----- 3. Interest flows (based on current stocks) -----
    r_D_H_amt  = p["r_D"]  * eco.DEP_H
    r_D_F_amt  = p["r_D"]  * eco.DEP_F
    r_D_E_amt  = p["r_D"]  * eco.DEP_E
    # Stage-3.1 risk-adjusted per-bank loan pricing (falls back to
    # scalar r_L under loan_rate_mode="scalar").
    r_L_F_amt, r_L_E_amt = _loan_interest_flows(eco, p)
    r_GB_B_amt = p["r_GB"] * eco.GB_B

    # ----- 4a. Stage 4.2 DER adoption (Rai & Robinson 2015) -----
    # Evaluate adoption decisions; stash the results as pending state
    # for commit after the SFC check succeeds. We do NOT mutate H stocks
    # here — the capex is instead routed through C_H_F as an extra
    # F-sector purchase, which flows through apply_laws_of_motion and
    # reduces DEP_H by the correct amount via normal accounting.
    der_suppressed = bool(eco.params.get("_der_suppressed", False))
    der_capex_per_adopter = (
        float(p.get("der_capacity_per_adopter", 0.0))
        * float(p.get("der_unit_cost", 0.0))
    )
    aggregate_capex_DER = 0.0
    new_adopter_mask = np.zeros(eco.H.N, dtype=bool)
    if (not der_suppressed) and (der_capex_per_adopter > 0.0):
        # Stage 4.3 peer-effect stimulus. Compute adoption share within
        # each income class, broadcast to per-household stimulus = ρ_peer
        # · adoption_share_of_own_class. Bollinger & Gillingham (2012).
        rho_peer = float(p.get("rho_peer", 0.0))
        peer_stim = None
        if rho_peer > 0.0:
            class_from_q_vec = np.asarray(
                p.get("class_from_quintile", (0, 0, 1, 1, 2)), dtype=int)
            class_of_hh = class_from_q_vec[
                np.asarray(eco.H.quintile, dtype=int)]
            # Per-class adoption share
            adopt_mask_now = eco.H.has_DER
            per_class_adoption = np.zeros(3, dtype=float)
            for c in range(3):
                in_class = (class_of_hh == c)
                if in_class.any():
                    per_class_adoption[c] = float(
                        adopt_mask_now[in_class].mean())
            peer_stim = rho_peer * per_class_adoption[class_of_hh]
        # Stage 5.2: weight the adoption threshold by policy credibility.
        # κ_G < 1 discounts households' expected future savings — they
        # adopt less under wobbly policy. At κ_G = 1 (full credibility)
        # adoption matches Stage-4.3 behaviour.
        kappa_G = float(getattr(eco, "credibility", 1.0))
        adoption_threshold_eff = kappa_G * np.asarray(
            eco.H.adoption_threshold, dtype=float)
        new_adopter_mask = adopt_der(
            has_DER=eco.H.has_DER,
            DEP=eco.H.DEP,
            credit_access=eco.H.credit_access,
            adoption_threshold=adoption_threshold_eff,
            p_E_expected=eco.p_E_expected,
            der_capacity_per_adopter=p["der_capacity_per_adopter"],
            der_unit_cost=p["der_unit_cost"],
            peer_stimulus=peer_stim,
        )
        if new_adopter_mask.any():
            aggregate_capex_DER = float(new_adopter_mask.sum()) * der_capex_per_adopter
    # Stash pending DER state updates for commit (after SFC check).
    eco._pending_DER_new_mask = new_adopter_mask
    eco._pending_DER_capex_per_adopter = der_capex_per_adopter
    eco._pending_DER_lifetime = int(p.get("der_lifetime_quarters", 80))

    # ----- 4b. Solve for nominal consumption -----
    # NW_H uses current DEP (pre-capex) — adoption decisions above
    # have ALREADY used the `DEP >= capex` gate, and the capex will
    # be subtracted via C_H_F accounting below.
    NW_H = float(eco.H.NW.sum())
    C_nom = solve_consumption(w=w, lam=lam,
                              alpha1=alpha1, alpha2=alpha2,
                              NW_H=NW_H,
                              r_D_H=r_D_H_amt,
                              r_GB_B=r_GB_B_amt,
                              Div_prev=Div_prev_total)

    # Split into F-goods and energy spending by households
    C_H_F = s_F * C_nom + aggregate_capex_DER  # DER capex is extra F-sector revenue
    EB_H_intended = (1.0 - s_F) * C_nom

    # ----- 4c. DER generation offsets household energy demand -----
    # Existing-DER households self-supply some energy each period. The
    # nominal value of that self-supply reduces the cash outflow from
    # H to E (EB_H falls, DEP_H retains the difference as saving).
    # Monasterolo & Raberto (2018): "green investment lowers household
    # energy bill" — this is the flow through which that shows up.
    DER_output_total = float(
        (eco.H.has_DER.astype(float) * eco.H.DER_capacity).sum()
    )
    DER_savings = DER_output_total * p_E
    # Cap DER savings at the household's intended energy budget: no
    # exports to the grid in v1. Surplus self-generation is just
    # unused capacity (the paper-2 extension can add prosumer-sale
    # behaviour).
    DER_savings = min(DER_savings, EB_H_intended)
    EB_H = max(0.0, EB_H_intended - DER_savings)

    # ----- 5. Production and firm energy input -----
    Y_F = C_H_F / p_F                  # real F output (includes DER capex bump)
    EB_F = ei_avg * Y_F * p_E          # firm energy bill (nominal)
    # Real energy produced: only the purchased portion (EB_H, not
    # EB_H_intended) flows through the E sector. Self-generation is
    # outside the merit order.
    Y_E = (EB_H + EB_F) / p_E

    # ----- 6. Labour demand; ration production if it exceeds supply -----
    N_F = Y_F / xi_F
    N_E = Y_E / xi_E
    N_total = N_F + N_E
    if N_total > N_H:
        scale = N_H / N_total
        Y_F   *= scale
        Y_E   *= scale
        C_H_F *= scale
        EB_H  *= scale
        EB_F  *= scale
        N_F   *= scale
        N_E   *= scale
        N_total = N_H

    # ----- 7. Wage bill and tax / transfer -----
    W   = w * N_total
    W_F = w * N_F
    W_E = w * N_E
    T_H  = tax_rate * W
    Tr_H = T_H - r_GB_B_amt            # balanced budget

    # ----- 8. Stash post-period state updates on the economy -----
    u_new   = 1.0 - N_total / N_H
    cpi_new = s_F * p_F + (1.0 - s_F) * p_E
    # Stage-4.1 income-class Phillips curves. The effective markup
    # drives the L-class profit-margin pass-through; at baseline this
    # equals params["markup"] and the margin-deviation term is zero.
    # Under Stage-2.1 uncertainty-loaded markup it widens with σ̂_PE,
    # bidding up L-class wages relative to M/H.
    effective_markup_now = (p_F / max(1e-12, (w / xi_F + ei_avg * p_E))) - 1.0
    w_by_class_new, qtrs_since_new, cpi_at_last_new = wage_update_by_class(
        w_by_class=eco.w_by_class,
        qtrs_since_update=eco.qtrs_since_update,
        cpi_at_last_update=eco.cpi_at_last_update,
        cpi_new=cpi_new,
        cpi_old=eco.cpi,
        u=u_new,
        margin=effective_markup_now,
        margin_reference=eco.margin_reference,
        phi_pi=p["phi_pi"],
        phi_u=p["phi_u"],
        u_star=p["u_star"],
        phi_pm=float(p.get("phi_pm", 0.0)),
        contract_length=p.get("contract_length", (1, 4, 4)),
        max_q_wage_growth=float(p.get("max_q_wage_growth", 0.10)),
    )
    # Aggregate scalar `w` used by solve_consumption and λ is the
    # employment-weighted average across classes. Under uniform
    # unemployment (current simplification) this equals N_weighted mean.
    class_from_q = np.asarray(p.get("class_from_quintile", (0, 0, 1, 1, 2)),
                              dtype=int)
    q_of_household = np.asarray(eco.H.quintile, dtype=int)
    class_of_household = class_from_q[q_of_household]
    class_counts = np.bincount(class_of_household, minlength=3).astype(float)
    class_weights = class_counts / max(1.0, float(class_counts.sum()))
    w_new = float(np.dot(w_by_class_new, class_weights))

    # Store for the NEXT period; applied after the SFC check succeeds.
    eco._pending_w                 = w_new
    eco._pending_w_by_class        = w_by_class_new
    eco._pending_qtrs_since_update = qtrs_since_new
    eco._pending_cpi_at_last       = cpi_at_last_new
    eco._pending_pF                = p_F
    eco._pending_cpi               = cpi_new
    eco._pending_u                 = u_new
    eco._pending_YF                = Y_F
    eco._pending_YE                = Y_E

    # ----- 9. Assemble TFMFlows -----
    flows = TFMFlows(
        W=W,
        C_H_F=C_H_F,
        EB_H=EB_H,
        EB_F=EB_F,
        G_spend=0.0,
        T_H=T_H,
        Tr_H=Tr_H,
        r_D_H=r_D_H_amt,
        r_D_F=r_D_F_amt,
        r_D_E=r_D_E_amt,
        r_L_F=r_L_F_amt,
        r_L_E=r_L_E_amt,
        r_GB_B=r_GB_B_amt,
        r_GB_CB=0.0,
        r_Res=0.0,
        new_loans_F=0.0,
        new_loans_E=0.0,
        amort_F=0.0,
        amort_E=0.0,
        writeoff_F=0.0,
        writeoff_E=0.0,
        GB_issued=0.0,
        GB_amort=0.0,
        I_F=0.0,
        I_E=0.0,
        depreciation_F=0.0,
        depreciation_E=0.0,
        dividends_F=Div_F,
        dividends_E=Div_E,
        dividends_B=Div_B,
    )
    # Carry the F/E wage split into the laws-of-motion helper.
    flows._W_F = W_F   # type: ignore[attr-defined]
    flows._W_E = W_E   # type: ignore[attr-defined]
    # Stash consumption totals on the economy for diagnostics.
    eco._last_C = C_nom
    eco._last_C_nom = C_nom
    return flows


def _commit_phase1_state(eco: Economy) -> None:
    """Apply the pending state updates stashed by make_phase1_flows."""
    eco.w   = getattr(eco, "_pending_w",   eco.w)
    eco.p_F = getattr(eco, "_pending_pF",  eco.p_F)
    eco.cpi = getattr(eco, "_pending_cpi", eco.cpi)
    eco.u   = getattr(eco, "_pending_u",   eco.u)
    eco.Y_F = getattr(eco, "_pending_YF",  eco.Y_F)
    eco.Y_E = getattr(eco, "_pending_YE",  eco.Y_E)
    # Stage 4.1 — commit per-class wage state.
    pending_wbc = getattr(eco, "_pending_w_by_class", None)
    if pending_wbc is not None:
        eco.w_by_class = np.asarray(pending_wbc, dtype=float).copy()
    pending_qtrs = getattr(eco, "_pending_qtrs_since_update", None)
    if pending_qtrs is not None:
        eco.qtrs_since_update = np.asarray(pending_qtrs, dtype=int).copy()
    pending_cpi_at = getattr(eco, "_pending_cpi_at_last", None)
    if pending_cpi_at is not None:
        eco.cpi_at_last_update = np.asarray(pending_cpi_at, dtype=float).copy()
    # Stage 4.2 — commit DER adoption/retirement state. Capex flow was
    # accounted for via C_H_F during make_phase1_flows; here we only
    # flip the `has_DER` / `DER_capacity` / `DER_age` bookkeeping.
    new_mask = getattr(eco, "_pending_DER_new_mask", None)
    if new_mask is not None and bool(new_mask.any()):
        cap = float(getattr(eco, "_pending_DER_capex_per_adopter", 0.0))
        # Installed capacity per adopter is capex / unit_cost. We stored
        # capex_per_adopter on the pending field; derive capacity from
        # the params dict below for clarity.
        capacity_per_adopter = (
            float(eco.params.get("der_capacity_per_adopter", 0.0))
            if cap > 0.0 else 0.0
        )
        eco.H.has_DER      = eco.H.has_DER | new_mask
        eco.H.DER_capacity = np.where(
            new_mask, capacity_per_adopter, eco.H.DER_capacity)
        eco.H.DER_age      = np.where(new_mask, 0, eco.H.DER_age)
    # Retire aged-out DER (age >= lifetime, v1 hard cutoff).
    lifetime = int(getattr(eco, "_pending_DER_lifetime", 80))
    if lifetime > 0 and eco.H.has_DER.any():
        retired = eco.H.has_DER & (eco.H.DER_age >= lifetime)
        if bool(retired.any()):
            eco.H.has_DER      = eco.H.has_DER & (~retired)
            eco.H.DER_capacity = np.where(retired, 0.0, eco.H.DER_capacity)
            eco.H.DER_age      = np.where(retired, 0, eco.H.DER_age)
    # Advance DER age for households still holding.
    eco.H.DER_age = eco.H.DER_age + eco.H.has_DER.astype(int)

    # ---- Firm energy-efficiency investment (gap-closure Stage 2.2) ----
    # Decision rule: x_f = min(x_max, η · max(0, p_E_expected − p_E_ref)).
    # Same decision across firms in this scalar version (all firms see
    # the same expectation). Law of motion:
    #   energy_intensity_{f,t+1} = energy_intensity_{f,t} · (1 − g_e · x_f)
    # Applied AFTER this period's production so next period uses the
    # improved intensity (matches Eq. 6–7 timing). No monetary flow is
    # charged in this version — the efficiency gains are treated as
    # within-firm process improvements absorbed by regular operations.
    # Stage 3 (capital investment) can add an explicit cash outlay.
    g_e    = float(eco.params.get("g_e", 0.0))
    eta_xe = float(eco.params.get("eta_xe", 0.0))
    x_max  = float(eco.params.get("x_max", 0.0))
    if g_e > 0.0 and x_max > 0.0:
        x_val = efficiency_investment(
            p_E_expected=eco.p_E_expected,
            p_E_reference=eco.p_E_reference,
            eta_xe=eta_xe,
            x_max=x_max,
        )
        eco.F.x[:] = x_val
        if x_val > 0.0:
            eco.F.energy_intensity *= (1.0 - g_e * x_val)


def _commit_last_profits(eco: Economy, flows) -> None:
    """
    Record this period's aggregate cash-flow profits for F, E and B.
    These are read at the START of the NEXT period's Phase-1 block to
    compute dividends paid from prior-period retained earnings.

    The formulas match the F/E/B column-sum conventions in
    accounting.py (I_F, I_E are self-financed capital-account flows
    and hence do NOT reduce cash profit; amort, new_loans,
    carbon_tax, dividends already paid in-period ARE included so
    next period's dividend is based on what the sector *actually
    retained* after paying everything else).

    Each pi_k is computed as the k-column sum with dividends_k = 0,
    i.e. the residual cash flow that — absent a dividend — would
    accumulate as balance-sheet growth forever.
    """
    W_F = getattr(flows, "_W_F", 0.8 * flows.W)
    W_E = getattr(flows, "_W_E", 0.2 * flows.W)
    # F column sum ex-dividends:
    pi_F = (flows.C_H_F + flows.G_spend + flows.r_D_F + flows.new_loans_F
            - W_F - flows.r_L_F - flows.EB_F
            - flows.amort_F)
    # E column sum ex-dividends:
    pi_E = (flows.EB_H + flows.EB_F + flows.r_D_E + flows.new_loans_E
            - W_E - flows.r_L_E - flows.amort_E
            - flows.carbon_tax)
    # B column sum ex-dividends: net interest margin + amort receipts
    # - new loans extended - writeoffs.  Note new_loans here are
    # a *cash outflow* in the TFM convention (the bank hands cash to
    # the borrower; the matching reflux to DEP_liab is a balance-sheet
    # item, not a B column flow), so they reduce π_B and — with
    # predetermined amort — are re-added next period.
    pi_B = (flows.r_L_F + flows.r_L_E + flows.r_GB_B + flows.r_Res
            + flows.amort_F + flows.amort_E
            - flows.r_D_H - flows.r_D_F - flows.r_D_E
            - flows.new_loans_F - flows.new_loans_E
            - flows.writeoff_F - flows.writeoff_E)
    eco.last_profit_F = float(pi_F)
    eco.last_profit_E = float(pi_E)
    eco.last_profit_B = float(pi_B)


# --------------------------------------------------------------------------
# Phase-2a flow builder  (phase-1 behaviour + merit-order dispatch)
# --------------------------------------------------------------------------
def make_phase2a_flows(eco: Economy) -> TFMFlows:
    """
    Phase-2a = phase-1 behaviour + endogenous energy market clearing.

    The period's decisioning is *identical* to phase 1: household
    consumption, firm pricing, labour demand, taxes, and transfers all
    use the current `eco.params["p_E"]` (the previous period's clearing
    price, seeded to the initial parameter at t=0). This keeps the SFC
    closure proof intact — every cash flow is consistent with the same
    p_E used by the behavioural block.

    *After* the phase-1 block has produced the period's real energy
    demand (the pending `Y_E`), the dispatch routine is called:

      - plants are stacked cheapest-first up to their capacity;
      - the marginal plant's MC becomes the *new* p_E, stashed for
        the next period's decisions;
      - per-plant output and per-plant emissions are recorded for
        diagnostics;
      - cumulative emissions are updated at commit time.

    No new TFM flows are added — dispatch changes *who* produces the
    energy, not how much is paid for it. Phase-2b will add producer-
    surplus redistribution and start moving money to cover genuine
    cost differences.
    """
    flows = make_phase1_flows(eco)

    # Real energy demand produced by the phase-1 block, already
    # reflecting labour-rationing scale-downs if any.
    demand = getattr(eco, "_pending_YE", 0.0)

    # Per-plant capacity this quarter. cap_factor is a per-plant
    # array (built in init_stocks from the tech → factor dict) so
    # capacity reflects real cross-technology differences in
    # utilisation — solar ≈ 0.15, wind ≈ 0.22, gas ≈ 0.40,
    # coal ≈ 0.65, nuclear ≈ 0.88.
    capacity = eco.E.cap_factor * eco.E.K

    # Stage 5.1: if ETS mode is active, solve the permit market for τ
    # that clears this period's emissions to the cap. The dispatch then
    # uses this endogenous τ. Under `ets_mode = False` the scalar
    # `carbon_tax` remains an exogenous policy lever (Carbon Tax regime).
    if eco.params.get("ets_mode", False):
        from policy import clear_ets_permit_market
        eco.carbon_tax = clear_ets_permit_market(
            demand=demand,
            mc=eco.E.mc,
            capacity=capacity,
            emission_factor=eco.E.emission_factor,
            cap=float(eco.params.get("ets_cap", 0.0)),
            tau_max=float(eco.params.get("ets_tau_max", 500.0)),
            tolerance=float(eco.params.get("ets_tolerance", 0.01)),
        )

    # Effective marginal cost includes the carbon tax (τ·ef). In
    # Phase 2a/2b τ defaults to 0 so mc_effective == eco.E.mc and the
    # dispatch is unchanged. Phase 2c sets τ > 0 and the merit order
    # tilts accordingly. Under Stage 5.1 ETS mode τ is set above by
    # clear_ets_permit_market rather than read from config.
    tau = getattr(eco, "carbon_tax", 0.0)
    mc_effective = eco.E.mc + tau * eco.E.emission_factor

    y, p_E_new, emissions, rationed = dispatch(
        demand=demand,
        mc=mc_effective,
        capacity=capacity,
        emission_factor=eco.E.emission_factor,
    )
    per_plant_em = y * eco.E.emission_factor

    # Stash phase-2a pending state — applied after closure check.
    eco._pending_p_E = p_E_new
    eco._pending_Y_per_plant = y
    eco._pending_emissions_per_plant = per_plant_em
    eco._pending_emissions_add = emissions
    eco._pending_rationed = rationed
    # Phase 2c consumers read mc_effective to compute per-plant profits.
    eco._pending_mc_effective = mc_effective
    return flows


def _update_credibility(eco: Economy) -> None:
    """
    Stage-5.2 policy-credibility update (paper §3.7).

    Each period compare realised τ (which under Stage 5.1 ETS mode may
    be endogenous) to the government's announced target. Normalise the
    deviation into a per-period signal in [0, 1]:

        signal = 1 − |realised − announced| / max(1, |announced|)

    Then update κ_G via exponential smoothing:

        κ_G ← (1 − β) · κ_G + β · signal

    `announced_carbon_tax = None` means "the government announces what it
    actually does", so signal = 1 always and κ_G converges to 1.

    Skipped when `_cred_suppressed` flag is set (during pre-convergence)
    so κ_G stays at its initial value through bootstrap — this lets
    scenario runners start with full credibility and introduce policy
    shocks mid-run rather than having the bootstrap absorb them.
    """
    p = eco.params
    if p.get("_cred_suppressed", False):
        return
    announced = p.get("announced_carbon_tax", None)
    if announced is None:
        announced = float(eco.carbon_tax)
    announced = float(announced)
    realised = float(eco.carbon_tax)
    denom = max(1.0, abs(announced))
    signal = max(0.0, 1.0 - abs(realised - announced) / denom)
    beta = float(p.get("credibility_update_rate", 0.10))
    floor = float(p.get("credibility_min", 0.05))
    eco.credibility = max(floor, (1.0 - beta) * eco.credibility + beta * signal)


def _commit_phase2a_state(eco: Economy) -> None:
    """Apply phase-1 commits, then update dispatch-specific state."""
    _commit_phase1_state(eco)
    _update_credibility(eco)
    # Realised clearing price from the dispatch just completed.
    p_E_realised = float(getattr(
        eco, "_pending_p_E", eco.params.get("p_E", 1.0)))
    eco.p_E_realised = p_E_realised
    # Adaptive price expectations (paper §3.7 Eq. 15, gap-closure Stage 1.1):
    #   p_E_expected ← α_p · p_E_realised + (1 − α_p) · p_E_expected
    alpha_p = float(eco.params.get("alpha_p", 1.0))
    eco.p_E_expected = (alpha_p * p_E_realised
                        + (1.0 - alpha_p) * eco.p_E_expected)
    # Volatility tracker σ̂_PE (gap-closure Stage 1.2). Roll the rolling
    # window one step forward, append the fresh realised price, and
    # recompute the coefficient of variation over the new window.
    hist = eco.p_E_history
    if hist is not None and hist.size > 0:
        hist[:-1] = hist[1:]
        hist[-1] = p_E_realised
        mean = float(hist.mean())
        eco.p_E_volatility = (float(hist.std()) / mean) if mean > 1e-12 else 0.0
    # `params["p_E"]` is preserved only as an *initial* parameter for
    # warm-start consumers. Legacy reads that expect a post-dispatch
    # price should switch to `eco.p_E_expected` (Phase 1 decisions) or
    # `eco.p_E_realised` (diagnostics). We still update params for
    # backward compatibility with external scripts; the value written
    # is the expectation so mid-run reads behave the same as before
    # when α_p = 1.
    eco.params["p_E"] = eco.p_E_expected
    eco.Y_per_plant = getattr(eco, "_pending_Y_per_plant", eco.Y_per_plant)
    eco.E.emissions_flow = getattr(
        eco, "_pending_emissions_per_plant", eco.E.emissions_flow)
    eco.emissions_stock += getattr(eco, "_pending_emissions_add", 0.0)
    # ---- Wright's-Law cumulative output update (Stage 2.3) ----
    # Accumulate each plant's real output for the learning curve. The
    # anchor Q_ref is the pre-converged cumulative output (captured at
    # make_initial_economy), so new history adds multiplicatively.
    if eco.E.cumulative_output is not None:
        eco.E.cumulative_output = (eco.E.cumulative_output
                                   + getattr(eco, "_pending_Y_per_plant",
                                             np.zeros(eco.E.N, dtype=float)))
        # Refresh I_unit per plant from Wright's Law:
        #   I_unit_k = (cum_Q_k / Q_ref_k)^(−b_k)
        # Q_ref_k was baked into the initial cumulative_output so that at
        # t=0 the ratio equals 1 and I_unit = 1. As cum_Q grows beyond
        # that anchor, learning techs (b > 0) see I_unit decay.
        Q_ref_mult = int(eco.params.get("Q_ref_multiplier", 4))
        b_dict = eco.params.get("b_by_tech", {}) or {}
        initial_anchor = Q_ref_mult * 4.0 * (eco.E.cap_factor * eco.E.K)
        # Guard: if any anchor is zero (e.g. a plant with K=0), clamp to
        # a tiny positive number to avoid divide-by-zero in Wright's Law.
        anchor = np.where(initial_anchor > 1e-12,
                          initial_anchor,
                          1e-12)
        ratio = eco.E.cumulative_output / anchor
        b_vec = np.array(
            [float(b_dict.get(str(t), 0.0)) for t in eco.E.tech],
            dtype=float,
        )
        # I_unit = ratio^(-b). b=0 ⇒ I_unit = 1 regardless.
        # Use np.where to avoid raising ratio to 0 power unnecessarily.
        eco.E.I_unit = np.where(
            b_vec > 0.0,
            np.power(np.maximum(ratio, 1e-12), -b_vec),
            np.ones_like(ratio),
        )


# --------------------------------------------------------------------------
# Phase-2b flow builder  (phase-2a + credit + investment)
# --------------------------------------------------------------------------
def make_phase2b_flows(eco: Economy) -> TFMFlows:
    """
    Phase-2b = phase-2a + investment, depreciation, amortisation, and
    new lending under a bank capital-ratio constraint.

    Sequencing:

      1. Phase-2a block fires (phase-1 behaviour + merit-order dispatch).
         All of its cash flows are in place on `flows`, and its pending
         state is stashed for commit.

      2. Real-sector investment rules fire:
           I_F = i_rate_F · p_F · Y_F          (firm reinvestment)
           I_E = δ_E · K_E_total               (replacement-only)
         Depreciation for each sector is computed.

      3. Amortisation is a fixed fraction of the current loan book.

      4. Loan demand is assembled: investment + amortisation rollover +
         working-capital coverage of the current wage bill.

      5. Bank aggregate lending capacity is computed from total bank NW
         and existing assets under the capital-ratio target. If demand
         exceeds capacity, both F and E are rationed proportionally.

      6. The I_F, I_E, amort_F, amort_E, depreciation_F, depreciation_E,
         new_loans_F, new_loans_E fields on `flows` are populated.
         Everything else is inherited from phase-2a unchanged.

    The dispatch clearing price still feeds next period's decisions via
    the phase-2a commit path; phase-2b adds no new state commits beyond
    diagnostic fields (rationing ratio).
    """
    flows = make_phase2a_flows(eco)
    p = eco.params

    # Current-period prices/outputs — use the values just computed by
    # the phase-1 block (stashed as pending state, applied at commit).
    p_F   = getattr(eco, "_pending_pF", eco.p_F)
    Y_F   = getattr(eco, "_pending_YF", eco.Y_F)

    # ----- 2. Investment -----
    I_F = firm_investment(p_F=p_F, Y_F=Y_F, i_rate_F=p["i_rate_F"])
    depr_rate_E = p["depreciation_rate_E"]
    invest_mode = p.get("invest_mode", "replacement")
    if invest_mode == "npv":
        # Stage 2.3: per-plant NPV-based investment with Wright's-Law
        # learning. Each plant computes its marginal NPV and chooses
        # ΔK_k = κ · max(0, margin_k) · K_k expansion on top of pure
        # replacement δ·K_k. Negative-margin plants receive no
        # investment — their capital decays at rate δ (organic exit).
        tau_eff  = getattr(eco, "carbon_tax", 0.0)
        mc_eff_k = eco.E.mc + tau_eff * eco.E.emission_factor
        p_E_exp  = eco.p_E_expected
        horizon  = int(p.get("horizon_quarters", 80))
        kappa    = float(p.get("kappa_invest", 0.01))
        r_p_map  = p.get("r_p_by_tech", {}) or {}
        r_p_vec  = np.array(
            [float(r_p_map.get(str(t), 0.025)) for t in eco.E.tech],
            dtype=float,
        )
        # Marginal NPV per plant (vectorised computation of npv_marginal).
        # Positive values mean the plant would expand.
        annuity = np.where(
            r_p_vec > 0.0,
            (1.0 - (1.0 + r_p_vec) ** (-float(horizon))) / r_p_vec,
            float(horizon),
        )
        revenue_per_eur_K = np.maximum(0.0, p_E_exp - mc_eff_k) * eco.E.cap_factor
        margin_vec = revenue_per_eur_K * annuity - eco.E.I_unit
        # Positive-margin plants: replacement + κ-scaled expansion.
        # Negative-margin plants: zero investment, pure decay.
        positive = margin_vec > 0.0
        I_E_per_plant = np.where(
            positive,
            depr_rate_E * eco.E.K + kappa * np.maximum(0.0, margin_vec) * eco.E.K,
            0.0,
        )
        I_E = float(I_E_per_plant.sum())
        # Stash per-plant vector for Phase 2c to use (bypassing the
        # profitability reinvestment allocation). Phase 2c checks this
        # attribute first and falls back to the legacy weights if absent.
        eco._pending_I_E_per_plant = I_E_per_plant
        eco._pending_npv_margin    = margin_vec
    else:
        # "replacement" mode (legacy): uniform δ·K_E_total aggregate,
        # allocation deferred to Phase-2c reinvestment weights.
        I_E = energy_replacement_investment(
            K_E_total=eco.K_E_total,
            depreciation_rate_E=depr_rate_E,
        )
        eco._pending_I_E_per_plant = None
        eco._pending_npv_margin    = None

    # ----- 2b. Depreciation -----
    depr_F = depreciation(eco.K_F_total, p["depreciation_rate_F"])
    depr_E = depreciation(eco.K_E_total, p["depreciation_rate_E"])

    # ----- 3. Amortisation of outstanding loan books -----
    amort_F = p["amort_rate_F"] * eco.L_F_total
    amort_E = p["amort_rate_E"] * eco.L_E_total

    # ----- 4. Loan demand -----
    W_F = getattr(flows, "_W_F", 0.0)
    W_E = getattr(flows, "_W_E", 0.0)
    wcc = p["working_capital_coverage"]
    dem_F = loan_demand_F(I_F=I_F, amort_F=amort_F, W_F=W_F,
                          working_capital_coverage=wcc)
    dem_E = loan_demand_E(I_E=I_E, amort_E=amort_E, W_E=W_E,
                          working_capital_coverage=wcc)

    # ----- 5. Bank lending capacity and rationing -----
    # Stage-3.3 green preferential risk weight: E-sector loans enter the
    # capital-ratio denominator at ρ_green ≤ 1. ρ_green = 1 (default)
    # recovers legacy behaviour.
    bank_nw = eco.B.NW_total
    rho_green = float(p.get("rho_green", 1.0))
    existing_assets = risk_weighted_assets(
        L_F=eco.L_F_total,
        L_E=eco.L_E_total,
        GB=eco.GB_B,
        Res=float(eco.B.Res.sum()),
        rho_green=rho_green,
    )
    capacity = bank_lending_capacity(
        bank_nw=bank_nw,
        existing_assets=existing_assets,
        target_capital_ratio=p["bank_capital_target"],
    )
    # Stage-3.2 concentration-limit headroom per sector. β = 0 (default)
    # deactivates the check; a positive β enforces sector-level caps at
    # β · ΣNW_b minus existing exposure.
    beta_conc = float(p.get("beta_concentration", 0.0))
    head_F = concentration_headroom(
        existing_L_sector=float(eco.L_F_total),
        aggregate_NW_B=float(bank_nw),
        beta=beta_conc,
    )
    head_E = concentration_headroom(
        existing_L_sector=float(eco.L_E_total),
        aggregate_NW_B=float(bank_nw),
        beta=beta_conc,
    )
    new_F, new_E, ration_ratio = allocate_new_loans(
        demand_F=dem_F, demand_E=dem_E, capacity=capacity,
        headroom_F=head_F, headroom_E=head_E,
    )

    # ----- 6. Populate the Phase-2b fields on `flows` -----
    flows.I_F = I_F
    flows.I_E = I_E
    flows.depreciation_F = depr_F
    flows.depreciation_E = depr_E
    flows.amort_F     = amort_F
    flows.amort_E     = amort_E
    flows.new_loans_F = new_F
    flows.new_loans_E = new_E

    # Diagnostic stash — not committed, just read by the driver/tests.
    eco._pending_ration_ratio = ration_ratio
    eco._pending_bank_capacity = capacity
    eco._pending_loan_demand_F = dem_F
    eco._pending_loan_demand_E = dem_E
    return flows


def _commit_phase2b_state(eco: Economy) -> None:
    """No new scalar state beyond phase-2a's; delegate."""
    _commit_phase2a_state(eco)


# --------------------------------------------------------------------------
# Phase-2c flow builder  (phase-2b + carbon tax + profitability-weighted I_E)
# --------------------------------------------------------------------------
def make_phase2c_flows(eco: Economy) -> TFMFlows:
    """
    Phase-2c = phase-2b + carbon pricing + reinvestment mix dynamics.

    Sequencing:

      1. Phase-2b block fires (phase-1 behaviour + merit-order dispatch
         already using the tax-adjusted mc + investment + credit). All
         TFM flows except the policy ones are in place on `flows`.

      2. Compute carbon-tax revenue from this period's per-plant
         dispatch and the current τ. Set `flows.carbon_tax` and
         `flows.household_rebate = carbon_tax` (full rebate, same
         quarter — government runs a balanced budget every period).

      3. Compute this period's per-plant profit as producer surplus on
         the current dispatch:  π_k = y_k · (p_E_new - mc_eff_k).
         Stash for commit, so next period's reinvestment rule can
         read it.

      4. Reinvestment weights: use the profits from the PREVIOUS
         period (already committed on `eco.profit_per_plant`). This
         makes the rule 1-period predetermined, matching standard
         Phase-n accounting closure (decisions on t use information
         from ≤ t-1). Weights sum to 1; multiplied by flows.I_E to
         get per-plant investment. Per-plant depreciation follows
         rate × K_k. Both arrays written to `flows`.

    The only new state commits are profit_per_plant (used next period
    by the reinvestment rule). Everything else is inherited from
    phase-2a via the phase-2b / 2a commit chain.
    """
    flows = make_phase2b_flows(eco)
    p = eco.params

    tau = getattr(eco, "carbon_tax", 0.0)

    # ----- 2. Carbon tax + rebate -----
    y         = getattr(eco, "_pending_Y_per_plant",
                        np.zeros(eco.E.N, dtype=float))
    mc_eff    = getattr(eco, "_pending_mc_effective",
                        eco.E.mc.copy())
    p_E_new   = float(getattr(eco, "_pending_p_E",
                              eco.params.get("p_E", 1.5)))
    tax_total, _ = carbon_tax_flow(
        y_per_plant=y,
        emission_factor=eco.E.emission_factor,
        tau=tau,
    )
    rebate = lump_sum_rebate(total_revenue=tax_total, N_H=eco.H.N)
    flows.carbon_tax       = tax_total
    flows.household_rebate = rebate
    # Stage 5.3 — green fiscal channel. Direct transfer from G to E,
    # financed by new GB issuance. Paper §3.10 Eq. 32: ΔGB_out = DEF_G.
    # With carbon tax fully rebated and a separate green-subsidy outlay,
    # GB_issued = green_subsidy_E to balance the G column.
    gs_E = float(p.get("green_subsidy_E", 0.0))
    if gs_E > 0.0:
        flows.green_subsidy_E = gs_E
        flows.GB_issued = flows.GB_issued + gs_E

    # Stage 5.4 — feed-in tariff and contract-for-difference support.
    # Both instruments transfer cash between G and E per unit of output
    # in supported technologies. Net fiscal position funded by GB.
    fit_price = float(p.get("fit_price", 0.0))
    cfd_strike = float(p.get("cfd_strike", 0.0))
    if (fit_price > 0.0) or (cfd_strike > 0.0):
        # Build per-plant eligibility masks from tech labels.
        fit_supp = set(p.get("fit_supported_techs", ())) if fit_price > 0.0 else set()
        cfd_supp = set(p.get("cfd_supported_techs", ())) if cfd_strike > 0.0 else set()
        tech_arr = [str(t) for t in eco.E.tech]
        fit_mask = np.array([t in fit_supp for t in tech_arr], dtype=float)
        cfd_mask = np.array([t in cfd_supp for t in tech_arr], dtype=float)
        # FiT: producers receive a per-unit premium on top of market.
        fit_flow = fit_price * float((fit_mask * y).sum())
        # CfD: symmetric settlement. Positive ⇒ G pays E (hedged floor).
        cfd_flow = (cfd_strike - p_E_new) * float((cfd_mask * y).sum())
        flows.fit_payment = fit_flow
        flows.cfd_payment = cfd_flow
        # Fund the net deficit via GB issuance (may be negative ⇒ G
        # retires some bonds out of CfD revenue when p_E > strike).
        net_deficit = fit_flow + cfd_flow
        flows.GB_issued = flows.GB_issued + max(0.0, net_deficit)
        flows.GB_amort  = flows.GB_amort  + max(0.0, -net_deficit)

    # ----- 3. Per-plant profit on current dispatch -----
    # Producer surplus: y_k · (p_E - mc_effective_k). Marginal plant
    # earns 0 (p_E == mc_eff by definition); infra-marginal plants
    # earn positive surplus.
    profit_now = y * (p_E_new - mc_eff)
    eco._pending_profit_per_plant = profit_now

    # ----- 4. Per-plant investment allocation -----
    # Under Stage-2.3 NPV mode, Phase 2b has already computed a per-
    # plant vector on `_pending_I_E_per_plant`. We use it as-is. Under
    # legacy "replacement" mode we fall back to the profitability-
    # weighted reinvestment rule.
    pending_I_E = getattr(eco, "_pending_I_E_per_plant", None)
    if pending_I_E is not None:
        I_E_per_plant = np.asarray(pending_I_E, dtype=float).copy()
    else:
        weights = reinvestment_weights(
            profit_per_plant=eco.profit_per_plant,
            churn_alpha=p["churn_alpha"],
            K_per_plant=eco.E.K,
        )
        I_E_total = float(flows.I_E)
        I_E_per_plant = weights * I_E_total
    depr_rate_E = p["depreciation_rate_E"]
    depr_E_per_plant = depr_rate_E * eco.E.K.copy()

    # Keep the scalar flows.depreciation_E consistent with the per-
    # plant array. The Phase-2b rule δ_E·K_E_total equals the sum of
    # per-plant δ_E·K_k; if heterogeneity has developed, the scalar
    # is simply the array sum. This keeps TFM column sums exact.
    flows.depreciation_E = float(depr_E_per_plant.sum())
    flows.I_E_per_plant          = I_E_per_plant
    flows.depreciation_E_per_plant = depr_E_per_plant
    return flows


def _commit_phase2c_state(eco: Economy) -> None:
    """Commit phase-2b state plus the new per-plant profit vector."""
    _commit_phase2b_state(eco)
    pending = getattr(eco, "_pending_profit_per_plant", None)
    if pending is not None:
        # Take a copy so subsequent mutation of the pending array
        # (unlikely, but defensively) can't reach the commit.
        eco.profit_per_plant = np.asarray(pending, dtype=float).copy()


# --------------------------------------------------------------------------
# Period step (phase-gated)
# --------------------------------------------------------------------------
def step(eco: Economy, phase: int = 0,
         tol: float = 1e-10) -> ResidualReport:
    """
    Advance the economy by one quarter and run the SFC residual check.
    """
    if phase == 0:
        flows = make_phase0_flows(eco)
    elif phase == 1:
        flows = make_phase1_flows(eco)
    elif phase == 2:
        flows = make_phase2a_flows(eco)
    elif phase == 3:
        flows = make_phase2b_flows(eco)
    elif phase == 4:
        flows = make_phase2c_flows(eco)
    else:
        raise NotImplementedError(
            f"Phase {phase} dynamics not implemented yet."
        )

    # Update all stocks using the laws of motion (v2.1 §3.10)
    apply_laws_of_motion(eco, flows)

    # Phase-specific state commits (prices, wages, diagnostics)
    if phase == 1:
        _commit_phase1_state(eco)
    elif phase == 2:
        _commit_phase2a_state(eco)
    elif phase == 3:
        _commit_phase2b_state(eco)
    elif phase == 4:
        _commit_phase2c_state(eco)

    # Phases 1+ use the Caiani lagged-dividend closure. Compute this
    # period's aggregate F / E cash profits AFTER the flows are fully
    # populated (so Phase-2b/2c amort, new_loans, carbon_tax, etc.
    # enter the retained-earnings calculation). Consumed by the next
    # step's solve_consumption via make_phase1_flows.
    if phase >= 1:
        _commit_last_profits(eco, flows)

    # Run the residual check; raises SFCClosureError if it fails
    return residual_check(eco, flows=flows, tol=tol, raise_on_fail=True)
