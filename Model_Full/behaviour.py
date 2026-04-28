"""
behaviour.py — phase-1 behavioural functions.

Pure functions only: no side effects, no mutation. The caller (dynamics.py)
feeds these functions the relevant state scalars and receives the flows
or the updated parameters. This isolates economic behaviour from
accounting, so `accounting.py` stays phase-agnostic.

Phase-1 closures implemented here:

  * firm_pricing               — fixed mark-up over unit cost
  * ces_share_F                — CES demand allocation between F and E
  * composite_labour_coeff     — λ = workers per unit of nominal consumption
  * solve_consumption          — closed-form C_nom from fixed-point identity
  * wage_update                — Phillips curve for next-period wage
"""

from __future__ import annotations


def firm_pricing(w: float,
                 p_E: float,
                 ei_avg: float,
                 xi_F: float,
                 markup: float,
                 chi: float = 0.0,
                 sigma_PE: float = 0.0) -> float:
    """
    Representative F-goods price with uncertainty-loaded mark-up
    (paper §3.4, Eq. 8; gap-closure Stage 2.1).

    Unit cost has two components:
      - wage per unit of output   = w / xi_F     (labour productivity xi_F)
      - energy per unit of output = ei_avg * p_E (average energy intensity)

    Effective mark-up widens with electricity-price volatility:
        ψ_f,t = ψ̅_f · (1 + χ · σ̂_PE,t)
        p_F   = (1 + ψ_f,t) · UC

    With χ = 0 (Stage-0 default) the rule reduces to the pre-Stage-2.1
    fixed-mark-up formula p_F = (1 + μ)·UC. With χ > 0 firms widen their
    profit cushion when electricity prices become volatile, reflecting
    both genuine risk compensation and a precautionary pass-through of
    uncertainty into posted prices. This is the central fragility
    amplifier of the paper's Story 1.
    """
    UC = w / xi_F + ei_avg * p_E
    effective_markup = markup * (1.0 + chi * sigma_PE)
    return (1.0 + effective_markup) * UC


def ces_share_F(p_F: float,
                p_E: float,
                taste_F: float,
                sigma: float) -> float:
    """
    Nominal-expenditure share of F-goods under a CES aggregator with
    elasticity of substitution `sigma`. Standard form:

        s_F = α · p_F^{1-σ} / (α · p_F^{1-σ} + (1-α) · p_E^{1-σ})

    where α = taste_F. Returns a float in (0, 1).
    """
    num = taste_F      * p_F ** (1.0 - sigma)
    den = num + (1.0 - taste_F) * p_E ** (1.0 - sigma)
    return num / den


def composite_labour_coeff(s_F: float,
                           p_F: float,
                           p_E: float,
                           ei_avg: float,
                           xi_F: float,
                           xi_E: float) -> float:
    """
    λ ≡ total labour demand per unit of nominal household consumption.

    Derivation:
      Real F-demand                  =  s_F · C_nom / p_F
      Real E-demand by households    =  (1-s_F) · C_nom / p_E
      Real E-demand by firms (input) =  ei_avg · (real F output)
                                     =  ei_avg · s_F · C_nom / p_F
      Labour to produce F            =  (real F) / xi_F
      Labour to produce E            =  (real E total) / xi_E

      N_total = C_nom · [ s_F / (p_F · xi_F)
                         + (1 - s_F) / (p_E · xi_E)
                         + ei_avg · s_F / (p_F · xi_E) ]
              = C_nom · λ
    """
    return (s_F / (p_F * xi_F)
            + (1.0 - s_F) / (p_E * xi_E)
            + ei_avg * s_F / (p_F * xi_E))


def solve_consumption(w: float,
                      lam: float,
                      alpha1: float,
                      alpha2: float,
                      NW_H: float,
                      r_D_H: float,
                      r_GB_B: float,
                      Div_prev: float = 0.0) -> float:
    """
    Closed-form household nominal consumption, with optional
    Caiani-style dividend recycling via a predetermined dividend flow.

    Consumption rule:     C = α1 · YD + α2 · NW_H
    Disposable income:    YD = W + Div + Tr + r_D_H - T
                             = W + Div + (T - r_GB_B) + r_D_H - T
                             = W + Div - r_GB_B + r_D_H    (balanced budget)
    Wage bill:            W = w · N_total = w · C · λ

    `Div_prev` is the aggregate dividend flow paid THIS period out of
    firm + energy cash surplus accumulated LAST period. Because it is
    predetermined (1-period lag), it enters as a constant in the
    fixed-point equation and solve_consumption remains closed form:

        C · (1 - α1 · w · λ) = α1 · (Div_prev + r_D_H - r_GB_B) + α2·NW_H

    With Div_prev = 0 this reduces to the original formula (no
    dividends — econ collapses slowly because income < expenditure).
    With Div_prev > 0 each period's income is topped up by the prior
    period's retained firm profit, so YD → C at the steady state and
    a true flow equilibrium exists.

    Returns C (clamped to zero in degenerate regions of state space).
    """
    denom = 1.0 - alpha1 * w * lam
    if denom <= 0.0:
        raise ValueError(
            f"solve_consumption: instability — (1 - α1·w·λ) = {denom:.3e} ≤ 0. "
            "Wage or labour coefficient too high relative to propensity α1."
        )
    num = alpha1 * (Div_prev + r_D_H - r_GB_B) + alpha2 * NW_H
    C = num / denom
    return max(0.0, C)


def dividend_flow(last_profit: float, theta_div: float) -> float:
    """
    Dividend paid this period out of last period's cash profit.

        Div_k = theta_div · max(0, last_profit_k)

    Clamp at zero: sectors running a cash loss don't pay dividends.
    """
    return max(0.0, theta_div * last_profit)


def wright_law_unit_cost(cumulative_output,
                         Q_ref,
                         b: float) -> float:
    """
    Unit cost of new capacity under Wright's Law (paper §3.5 Eq. 12,
    gap-closure Stage 2.3).

        I_unit = (cumulative_output / Q_ref)^(−b)

    with the convention that `I_unit = 1` when cumulative_output = Q_ref.
    For learning techs (b > 0) the unit cost decays monotonically with
    experience. For non-learning techs (b = 0) it stays at 1 forever.
    Scalar or array-compatible; accepts np.ndarray for per-plant
    computation.
    """
    import numpy as _np
    # Guard against division/log of zero when a plant has no history yet.
    ratio = _np.maximum(
        _np.asarray(cumulative_output, dtype=float) / float(Q_ref),
        1e-12,
    )
    if b == 0.0:
        return _np.ones_like(ratio)
    return _np.power(ratio, -float(b))


def npv_marginal(p_E_expected: float,
                 mc_effective: float,
                 cap_factor: float,
                 I_unit: float,
                 r_p: float,
                 horizon: int) -> float:
    """
    Marginal NPV of adding one unit (€) of capacity to a plant
    (paper §3.5 Eq. 11, gap-closure Stage 2.3).

    Annual-cash-flow annuity factor:
        A(r, T) = Σ_{t=1..T} (1+r)^{−t} = (1 − (1+r)^{−T}) / r

    Marginal NPV per € of K added:
        margin = [ (p_E_expected − mc_eff) · cap_factor ] · A(r_p, T) − I_unit

    Positive margin ⇒ plant would expand; negative ⇒ let capital decay.
    All arguments are per-plant scalars (the function is vector-friendly
    via numpy broadcasting at the caller's discretion).
    """
    if r_p <= 0.0:
        annuity = float(horizon)
    else:
        annuity = (1.0 - (1.0 + r_p) ** (-float(horizon))) / r_p
    revenue_per_euro_K = max(0.0, float(p_E_expected) - float(mc_effective)) \
                         * float(cap_factor)
    return revenue_per_euro_K * annuity - float(I_unit)


def adopt_der(has_DER,
              DEP,
              credit_access,
              adoption_threshold,
              p_E_expected: float,
              der_capacity_per_adopter: float,
              der_unit_cost: float,
              peer_stimulus=None) -> "np.ndarray":
    """
    Per-household DER adoption decision rule (paper §3.3 Eq. 3–4,
    Stages 4.2 and 4.3).

    Micro foundation: Rai & Robinson (2015), *Ecological Economics* —
    ABM of residential PV adoption with heterogeneous payback-threshold
    rule and peer effects. The payback period is the household's main
    decision criterion:

        payback_i = capex / (4q · DER_capacity · p_E_expected)

    A household adopts iff
      (a) it has not yet adopted,
      (b) it has credit access (always True for M/H class, Bernoulli(η_κ)
          for L — captures Italian EU-SILC 2019 data on liquidity-
          constrained low-income households),
      (c) payback < θ_i + S_i (effective threshold = base + peer stimulus),
      (d) it has enough deposits to cover the capex this period
          (self-financing under option-1 SFC treatment — see
          HouseholdArray docstring).

    Peer stimulus S_i (Stage 4.3, Bollinger & Gillingham 2012) raises
    the effective acceptance threshold in proportion to how many peers
    in the same income class have already adopted — generating the
    characteristic S-curve of technology diffusion. Pass None to
    deactivate the peer channel (pre-Stage-4.3 behaviour).

    Returns a boolean mask of new adopters.
    """
    import numpy as _np
    capex = float(der_capacity_per_adopter) * float(der_unit_cost)
    if p_E_expected <= 0.0 or der_capacity_per_adopter <= 0.0:
        return _np.zeros_like(has_DER, dtype=bool)
    # Payback in years: capex / annual_savings, where annual_savings =
    # 4 quarters · DER_capacity · p_E_expected.
    annual_savings = 4.0 * float(der_capacity_per_adopter) * float(p_E_expected)
    payback_years = capex / max(annual_savings, 1e-12)
    # Effective threshold = base + peer stimulus (both in years).
    thresh = _np.asarray(adoption_threshold, dtype=float)
    if peer_stimulus is not None:
        thresh = thresh + _np.asarray(peer_stimulus, dtype=float)
    mask = ((~_np.asarray(has_DER, dtype=bool))
            & _np.asarray(credit_access, dtype=bool)
            & (payback_years < thresh)
            & (_np.asarray(DEP, dtype=float) >= capex))
    return mask


def efficiency_investment(p_E_expected: float,
                           p_E_reference: float,
                           eta_xe: float,
                           x_max: float) -> float:
    """
    Firm energy-efficiency investment intensity (paper §3.4 Eq. 6–7,
    gap-closure Stage 2.2).

        x_f = min(x_max, η · max(0, p_E_expected − p_E_reference))

    Returns a non-negative intensity bounded by `x_max`. The reference
    price is the steady-state value captured at end of pre-convergence,
    so `x_f = 0` when the economy is at baseline (no artificial drift
    in energy intensity) and rises only when expected energy prices
    deviate above reference. The saturation at `x_max` reflects the
    physical/organisational limit on how quickly a firm can reduce its
    energy requirement per unit of output.
    """
    deviation = max(0.0, float(p_E_expected) - float(p_E_reference))
    return min(float(x_max), float(eta_xe) * deviation)


def wage_update_by_class(w_by_class,
                         qtrs_since_update,
                         cpi_at_last_update,
                         cpi_new: float,
                         cpi_old: float,
                         u: float,
                         margin: float,
                         margin_reference: float,
                         phi_pi: float,
                         phi_u: float,
                         u_star: float,
                         phi_pm: float,
                         contract_length,
                         max_q_wage_growth: float):
    """
    Income-class wage Phillips curves (paper §3.3, gap-closure Stage 4.1).

    Three classes indexed {L=0, M=1, H=2}. L is a flexible spot market
    that re-prices every quarter and carries an extra firm-profit-margin
    pass-through term (Carlin-Soskice wage curve with bargaining). M and
    H are on rigid multi-quarter contracts and only update when their
    counter reaches `contract_length[q]`; at renewal they apply the
    cumulative inflation since the last renewal.

    A soft per-quarter growth cap (`max_q_wage_growth`) protects against
    CES gross-complements instability under severe shocks. Italian
    2022-2023 wage negotiations delivered ~5% annual growth under
    ~8% inflation, so 10% per quarter is a generous upper bound.

    Returns `(w_by_class_new, qtrs_since_new, cpi_at_last_new)`, each
    a length-3 numpy array.
    """
    import numpy as _np
    w = _np.array(w_by_class, dtype=float, copy=True)
    counters = _np.array(qtrs_since_update, dtype=int, copy=True)
    cpi_at_last = _np.array(cpi_at_last_update, dtype=float, copy=True)
    contract_length = _np.asarray(contract_length, dtype=int)

    # Every class's contract clock advances by 1 this period.
    counters += 1

    # Per-period single-quarter inflation (used for L-class flexible rule).
    pi_q = (cpi_new / cpi_old - 1.0) if cpi_old > 0.0 else 0.0

    for q in range(len(w)):
        c_len = int(contract_length[q])
        if c_len <= 1:
            # L-class flexible Phillips + profit-margin pass-through
            margin_dev = float(margin) - float(margin_reference)
            gross = (1.0
                     + float(phi_pi) * pi_q
                     + float(phi_u)  * (float(u_star) - float(u))
                     + float(phi_pm) * margin_dev)
            gross = min(gross, 1.0 + float(max_q_wage_growth))
            gross = max(gross, 1.0 - float(max_q_wage_growth))
            w[q] = max(1e-6, w[q] * gross)
            # Track baseline cpi for reporting; not used for L updates.
            cpi_at_last[q] = float(cpi_new)
        else:
            # Rigid-contract classes: only update at renewal tick.
            if counters[q] >= c_len:
                # Cumulative inflation since last renewal.
                denom = cpi_at_last[q]
                pi_cum = (float(cpi_new) / denom - 1.0) if denom > 0.0 else 0.0
                gross = (1.0
                         + float(phi_pi) * pi_cum
                         + float(phi_u)  * (float(u_star) - float(u)))
                # Cap growth per contract cycle at the same ceiling as
                # L's per-quarter cap (multiplied by the contract length
                # so rigid contracts can catch up on deferred inflation).
                cap = float(max_q_wage_growth) * float(c_len)
                gross = min(gross, 1.0 + cap)
                gross = max(gross, 1.0 - cap)
                w[q] = max(1e-6, w[q] * gross)
                counters[q] = 0
                cpi_at_last[q] = float(cpi_new)
            # else: wage locked, counter keeps advancing.
    return w, counters, cpi_at_last


def wage_update(w: float,
                cpi_new: float,
                cpi_old: float,
                u: float,
                phi_pi: float,
                phi_u: float,
                u_star: float) -> float:
    """
    Wage Phillips curve:

        w_{t+1} = w_t · (1 + φ_π · π_t + φ_u · (u* - u_t))

    where π_t = cpi_new / cpi_old - 1 is quarter-on-quarter inflation.
    """
    pi_rate = cpi_new / cpi_old - 1.0 if cpi_old > 0.0 else 0.0
    gross = 1.0 + phi_pi * pi_rate + phi_u * (u_star - u)
    # Floor at a tiny positive number: a pathological collapse would
    # make subsequent divisions blow up; leave diagnostic space.
    return max(1e-6, w * gross)
