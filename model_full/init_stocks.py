"""
init_stocks.py — consistent initial-condition generator.

Builds an `Economy` whose opening balance sheet satisfies every SFC
identity exactly:

  - sum of sectoral NW equals total real capital;
  - every cross-sector stock balances (DEP_H+DEP_F+DEP_E = DEP_liab,
    L_F+L_E on the borrower side = L_asset on the bank side,
    GB_out = GB_B+GB_CB);
  - government balanced at t=0 (DEP_CB = 0, net fiscal position = -GB_out).

Unit convention (see docs/units_convention.md)
----------------------------------------------
All monetary quantities are expressed in **euros of 2019-Italian
output**. At t=0 every price is normalised to unity:

  - 1 unit of F-output = €1 of F-goods at 2019 Italian prices
  - 1 unit of E-output = €1 of energy at the 2019 PUN baseline of
    52 €/MWh (so 1 unit_E = 1/52 MWh)
  - wage `w` is in €/quarter; stocks and flows in €
  - carbon tax `τ` is in €/tCO₂ (EU ETS convention)
  - emission factors in tCO₂/unit_E = (tCO₂/MWh) / 52
  - tech marginal costs in €/unit_E = (€/MWh) / 52

This lets the CES taste weight `taste_F` retain its economic meaning
(baseline expenditure share on F) and lets shocks pass through
linearly — e.g., a TTF gas spike 55→250 €/MWh shows up as
mc_gas: 1.058 → 4.808.
"""

from __future__ import annotations

import numpy as np

from agents import (
    BankArray,
    CentralBank,
    Economy,
    EnergyArray,
    FirmArray,
    Government,
    HouseholdArray,
)


# --------------------------------------------------------------------------
# Technology stacks
# --------------------------------------------------------------------------
# Each stack is a list of per-plant records:
#     (tech_label, K_nameplate_eur, mc_nameplate_eur_per_MWh, ef_tCO2_per_MWh)
#
# Capacity share of plant k in total effective dispatch capacity is
#     K_k · cap_factor(tech_k) / Σ_j K_j · cap_factor(tech_j)
# so heterogeneous K lets us match ENTSO-E shares for Italy 2019 without
# having to fiddle with cap_factor values.
#
# STACK_ITALY_2019 — 8-plant representative Italian 2019 generation mix.
# Capacity shares (with cap_factors from cap_factor_by_tech below):
#     gas    ≈ 50%    (2 plants, 3.0 M€ K each, cap_factor 0.40)
#     hydro  ≈ 18%    (2.4 M€ K, cap_factor 0.35)
#     solar  ≈  8%    (2.6 M€ K, cap_factor 0.15)
#     wind   ≈  7%    (1.5 M€ K, cap_factor 0.22)
#     coal   ≈  6%    (0.45 M€ K, cap_factor 0.65 — last Italian coal units)
#     biomass≈  6%    (0.45 M€ K, cap_factor 0.65)
#     oil    ≈  5%    (0.80 M€ K, cap_factor 0.30 — peakers)
# No nuclear (Italy decommissioned its plants in 1990).
# Sources: ENTSO-E Transparency Platform 2019 generation; Terna
# "Dati statistici sull'energia elettrica in Italia 2019".
STACK_ITALY_2019 = [
    # (tech,     K (€),      mc (€/MWh), ef (tCO₂/MWh))
    ("gas",      3_000_000,  55.0,       0.37),
    ("gas",      3_000_000,  57.0,       0.37),  # marginal plant
    ("hydro",    2_400_000,   5.0,       0.00),
    ("solar",    2_600_000,   5.0,       0.00),
    ("wind",     1_500_000,   4.0,       0.00),
    ("coal",       450_000,  40.0,       0.90),
    ("biomass",    450_000,  30.0,       0.05),
    ("oil",        800_000,  80.0,       0.75),  # peaker
]


# COUNTERFACTUALS — diversification-as-resilience scenarios for paper 1.
#
# The paper's thesis is that a diversified supply mix reduces
# fragility to external shocks (e.g., the 2022 TTF gas spike) not
# primarily by capping the clearing price (gas stays marginal in
# realistic cases, because full replacement of Italian gas-fired
# generation is neither economically nor contractually feasible —
# Italy's SONATRACH TransMed contract alone covers ~20 bcm/year, and
# gas peakers provide operational flexibility that baseload
# alternatives can't replicate) but by substituting *volume*.
# Nuclear and renewables each displace a share of gas dispatch; the
# emissions trajectory falls, firm energy bills fall, and the
# Phillips-wage spiral is smaller even when marginal p_E is unchanged.
#
# Stability note: full gas-1 replacement with nuclear (large stack
# change) triggers the θ_div=1 deflation-spiral pathology at the
# model's default parameters — see docs/units_convention.md. The
# scenarios below are calibrated to stay in the stable regime.

# STACK_ITALY_2019_NUCLEAR — one-third of gas-1's capacity (€1 M of
# its €3 M K) swapped for nuclear at equivalent effective capacity.
# Nuclear K = 1 M × 0.40 / 0.88 ≈ 0.455 M. Implied nuclear share of
# effective dispatch ≈ 8% — the low end of Italy's 1980s ENEL nuclear
# plan (original target was ~6 GW, ~10–15% of 2019 generation).
STACK_ITALY_2019_NUCLEAR = [
    ("gas",      2_000_000,  55.0,       0.37),  # gas-1 reduced
    ("gas",      3_000_000,  57.0,       0.37),
    ("hydro",    2_400_000,   5.0,       0.00),
    ("solar",    2_600_000,   5.0,       0.00),
    ("wind",     1_500_000,   4.0,       0.00),
    ("coal",       450_000,  40.0,       0.90),
    ("biomass",    450_000,  30.0,       0.05),
    ("oil",        800_000,  80.0,       0.75),
    ("nuclear",    454_545,  15.0,       0.00),  # ≈ 8% effective share
]

# STACK_ITALY_2019_RENEWABLES — solar and wind capacity doubled,
# nothing else changed. Italian 2019 solar and wind were both below
# EU averages; a "what-if Italy had matched Spanish/German build-outs"
# counterfactual. Effective cheap capacity gain ≈ 0.72 M€, enough to
# shift dispatch volume but not enough to push gas entirely off the
# margin at baseline demand.
STACK_ITALY_2019_RENEWABLES = [
    ("gas",      3_000_000,  55.0,       0.37),
    ("gas",      3_000_000,  57.0,       0.37),
    ("hydro",    2_400_000,   5.0,       0.00),
    ("solar",    5_200_000,   5.0,       0.00),  # 2× baseline
    ("wind",     3_000_000,   4.0,       0.00),  # 2× baseline
    ("coal",       450_000,  40.0,       0.90),
    ("biomass",    450_000,  30.0,       0.05),
    ("oil",        800_000,  80.0,       0.75),
]

# STACK_ITALY_2019_DIVERSIFIED — both levers pulled: moderate nuclear
# AND doubled renewables. This is the maximum-resilience scenario in
# the paper's grid. Empirically shows ~57% emission reduction under a
# moderate TTF shock (gas mc 55 → 120 €/MWh), without triggering the
# θ_div=1 instability.
STACK_ITALY_2019_DIVERSIFIED = [
    ("gas",      2_000_000,  55.0,       0.37),  # gas-1 reduced
    ("gas",      3_000_000,  57.0,       0.37),
    ("hydro",    2_400_000,   5.0,       0.00),
    ("solar",    5_200_000,   5.0,       0.00),  # 2× baseline
    ("wind",     3_000_000,   4.0,       0.00),  # 2× baseline
    ("coal",       450_000,  40.0,       0.90),
    ("biomass",    450_000,  30.0,       0.05),
    ("oil",        800_000,  80.0,       0.75),
    ("nuclear",    454_545,  15.0,       0.00),
]


# --------------------------------------------------------------------------
# Default scale & calibration
# --------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    # Agent counts. N_E is derived from `stack` below — the value here
    # is ignored and kept only for backward compatibility with old code
    # that reads p["N_E"] before make_initial_economy runs.
    N_H=2000,
    N_F=250,
    N_E=8,
    N_B=20,
    # Prices (both normalised to 1 at baseline — Caiani convention)
    p_F=1.0,
    p_E=1.0,
    # Wage & employment. Italian 2019 average gross quarterly wage
    # from Istat "Retribuzioni contrattuali lorde" ≈ €7,500.
    wage_per_q=7500.0,
    # Interest rates (quarterly). Calibrated to Italy 2019-2021 averages
    # from the ECB MFI Interest Rate Statistics and MTS government-bond
    # secondary market. Earlier defaults (r_D=0.005, r_L=0.015,
    # r_GB=0.010) annualised to 2% / 6% / 4% — implausibly high for
    # the Eurozone post-QE era. The current values annualise to
    # ≈ 0.04% / 1.6% / 1.6%, which match the observed spreads.
    #   r_D : household overnight deposits (ECB MFI IRS).
    #   r_L : NFC new loans, all maturities (ECB MFI IRS).
    #   r_GB: 10Y BTP secondary-market yield, MTS quarterly mean.
    #   r_CB: ECB deposit facility rate; was negative 2014–2022 but
    #         is set to zero here so the simple sign convention holds.
    r_D=0.0001,
    r_L=0.004,
    r_GB=0.004,
    r_CB=0.0,
    # ---- Bank-specific loan pricing (paper §3.6 Eq. 14, Stage 3.1) ----
    # Loan rate charged by bank b: r_L_b = r_CB + μ_b + δ̄·(λ_b/λ_max)^γ.
    # `loan_rate_mode = "risk_based"` activates per-bank pricing using the
    # params below; `"scalar"` falls back to the flat r_L above (pre-Stage
    # 3.1 behaviour). With δ̄ = 0 the two modes coincide numerically.
    #
    # Defaults calibrated so the initial-state rate matches the legacy
    # scalar r_L = 0.004:
    #   init leverage λ_init ≈ 6.3 (bank-book asset/NW at the capital-
    #     ratio-target seed; computed on the per-bank arrays, not the
    #     aggregate)
    #   r_CB + μ_b + δ̄·(6.3/20)^2 = 0 + 0.00380 + 0.002·0.099 ≈ 0.004
    # Under leverage stress (λ → λ_max=20) rate saturates at 0.00380 +
    # 0.002 = 0.0058 quarterly ≈ 2.3% annual, a ~180 bps risk-premium
    # widening — in line with Italian 2011-2012 sovereign-risk spreads.
    loan_rate_mode="risk_based",
    mu_b=0.00380,
    delta_bar=0.002,
    gamma_lev=2.0,
    lambda_max=20.0,
    # Concentration-limit coefficient β (paper §3.6 Eq. 13b, Stage 3.2).
    # Applied as: aggregate new lending to sector s ≤ (β · ΣNW_b) −
    # existing_L_s, clipped at zero. β = 0 DEACTIVATES the limit (legacy
    # behaviour). Basel large-exposure rules use β = 0.25 against Tier-1
    # capital for single-counterparty limits; in our sectoral aggregation
    # that interpretation would bind hard at baseline, so we default off
    # and let the user opt in for stress scenarios.
    beta_concentration=0.0,
    # Green preferential risk weight (paper §3.6, Stage 3.3). E-sector
    # loans count at ρ_green ≤ 1 in the bank capital-ratio denominator,
    # so a green supporting factor eases regulatory constraints on E
    # lending. ρ_green = 1.0 (default) deactivates the mechanism.
    # Empirical benchmarks: EU taxonomy discussion papers propose 0.85
    # for green infrastructure; Basel-III "green-supporting-factor"
    # proposals ranged 0.75-0.90. We default off so baseline is
    # unchanged; set ρ_green < 1 to activate for green-finance scenarios.
    rho_green=1.0,
    # Fiscal
    tax_rate=0.20,
    # Phase 0 consumption rule
    mpc=0.90,
    cons_share_F=0.85,   # fraction of H expenditure going to F goods
    cons_share_E=0.15,   # to energy
    # ---- Phase 1 behavioural parameters ----
    alpha1=0.85,         # propensity to consume out of disposable income
    alpha2=0.04,         # propensity to consume out of net wealth
    taste_F=0.85,        # CES taste weight on F-goods (vs energy)
    ces_sigma=0.30,      # CES elasticity of substitution (F ↔ energy)
    # Productivity coefficients — € of output per worker per quarter.
    # Calibrated so that at baseline w=7500, ei_avg≈0.29, markup=0.20,
    # the markup pricing rule p_F = (1+μ)·(w/ξ_F + ei·p_E) returns
    # p_F ≈ 1. Specifically ξ_F = w / (1/(1+μ) − ei_avg) with
    # ei_avg ≈ 0.29 gives ξ_F ≈ 14,000. The same value is used for ξ_E
    # so the CES labour coefficient λ sits in a stable band.
    productivity_F=14000.0,  # € of F-good per worker per quarter
    productivity_E=14000.0,  # € of energy per worker per quarter
    markup=0.20,         # base markup over unit cost (phase 1)
    # Uncertainty-loaded-markup coefficient (paper §3.4 Eq. 8, gap-closure
    # Stage 2.1). Effective markup is ψ_f,t = ψ̅·(1 + χ·σ̂_PE,t), so χ
    # controls how much firms widen their price cushion as electricity-
    # price volatility rises. χ = 0 reproduces the pre-Stage-2.1 fixed
    # markup. Literature values for precautionary price-cost margins
    # sit in 0.3–0.8; 0.5 is a neutral starting point.
    chi=0.5,
    # Caiani-style dividend closure: share of firm + energy operating
    # profit paid out to households each quarter. θ=0 reproduces the
    # original (collapse-prone) Phase-1 formula. θ=1 is the full-
    # distribution closure — operating profit π_F + π_E = C - W is
    # completely recycled, YD = W + Div = C at flow equilibrium, and
    # the consumption fixed point is C = α2·NW/(1-α1). Any θ < 1
    # leaves a residual retention inside firms that drains household
    # NW over the long run and eventually crashes the economy.
    theta_div=1.0,
    phi_pi=1.0,          # wage Phillips curve — inflation pass-through
    phi_u=0.05,          # wage Phillips curve — unemployment slope
    u_star=0.08,         # NAIRU benchmark
    # ---- Income-class wage segmentation (paper §3.3, Stage 4.1) ----
    # Quintile → class mapping {L=0, M=1, H=2}: quintiles 0-1 are L
    # (low-income, flexible spot market), 2-3 are M (middle, rigid),
    # 4 is H (high, rigid). Shares: 40/40/20.
    class_from_quintile=(0, 0, 1, 1, 2),
    # Contract lengths per class, in quarters. L = 1 is effectively
    # re-negotiated every period (spot market); M and H = 4 is one-
    # year indexation, the typical Italian collective-agreement cycle.
    contract_length=(1, 4, 4),
    # L-class firm-profit-margin pass-through. Literature benchmarks
    # (Blanchflower-Oswald wage curves, Carlin-Soskice 2006) bracket
    # 0.2-0.5. We default to 0.3 as a defensible midpoint. Setting
    # phi_pm = 0 recovers the pre-Stage-4.1 Phillips dynamics for L.
    phi_pm=0.3,
    # Soft cap on per-quarter wage growth (any class). Italian 2022
    # wage negotiations delivered ~5% annual growth under ~8% inflation
    # — low-income workers have limited bargaining power regardless of
    # price dynamics. A 10%/quarter ceiling keeps the L class from
    # triggering the CES gross-complements instability under severe
    # shocks while still allowing meaningful pass-through.
    max_q_wage_growth=0.10,
    # ---- Stage 4.2 — DER adoption (Rai & Robinson 2015; ----
    # ----     Monasterolo & Raberto 2018, EIRIN) ----
    # A rooftop residential PV system sized for ~50% of typical Italian
    # household quarterly consumption (≈ 3 kW capacity × 15% capacity
    # factor × 4 quarter = ~1 MWh/yr). Capex is per-unit × capacity =
    # €60 × 50 = €3,000, matching Italian residential 2019-2023 turn-
    # key rates per Rete2 GSE reports. Lifetime 20 years (80 quarters)
    # is the standard IEC 61215 design life.
    der_capacity_per_adopter=50.0,     # units_E per quarter
    der_unit_cost=60.0,                # € per unit-of-capacity
    der_lifetime_quarters=80,          # 20 years
    # Adoption thresholds θ_i (in years) drawn at init. L-class has a
    # wider, higher distribution (more risk aversion, higher imputed
    # discount rate). Calibration to Italian household surveys
    # (Albrizio et al. 2017, OECD; Rai & Sigrin 2013).
    adoption_threshold_L=(5.0, 15.0),
    adoption_threshold_MH=(3.0, 10.0),
    # η_κ: fraction of L-class households with access to consumer
    # credit. 0.4 matches EU-SILC 2019 Italy data on middle- and
    # low-income households reporting difficulty obtaining loans.
    eta_kappa=0.4,
    # Peer-effect stimulus coefficient (Stage 4.3, Bollinger & Gillingham
    # 2012, *Marketing Science*, 31(6): 900-912). The effective adoption
    # threshold is θ_i_effective = θ_i + ρ_peer · adoption_share_class_i.
    # As more same-class peers adopt, the effective threshold rises,
    # bringing marginal households into the adopting mass — this
    # generates the characteristic S-curve of technology diffusion.
    # Bollinger & Gillingham estimate 0.05-0.15 for California rooftop
    # solar; 0.10 is a conservative default. Setting 0 deactivates the
    # peer channel (recovers Stage-4.2 behaviour exactly).
    rho_peer=0.10,
    # Adaptive price-expectation smoothing (paper §3.7 Eq. 15, gap-closure
    # Stage 1.1). p_E_expected_{t+1} = α_p·p_E_realised + (1-α_p)·p_E_expected.
    # α_p = 1 recovers "expectation = last realised" (pre-Stage-1); α_p
    # = 0 freezes expectations at the initial value. Literature values
    # for one-period-ahead price expectations sit in 0.3–0.7; 0.5 is a
    # neutral default.
    alpha_p=0.5,
    # Volatility-tracker window (gap-closure Stage 1.2). Number of past
    # quarters over which σ̂_PE = std(p_E)/mean(p_E) is measured. N = 4
    # gives a 1-year rolling window at the quarterly timestep.
    volatility_window=4,
    # Firm energy-efficiency investment (paper §3.4 Eq. 6–7, gap-closure
    # Stage 2.2). Under the law of motion
    #     e_{f,t+1} = e_{f,t} · (1 − g_e · x_f)
    # the intensity `x_f` is chosen from the deviation rule
    #     x_f = min(x_max, η · max(0, p_E_expected − p_E_reference))
    # with p_E_reference captured at end of pre-convergence so baseline
    # is stationary. g_e sets the maximum per-quarter efficiency gain
    # (g_e · x_max = 0.0025 ⇒ 1% per year under fully saturated shock).
    # eta_xe scales investment intensity per unit of price deviation.
    g_e=0.05,
    eta_xe=0.10,
    x_max=0.05,
    # ---- NPV investment + Wright's Law (paper §3.5 Eq. 11-12, Stage 2.3) ----
    # Mode switch: "replacement" keeps the pre-Stage-2.3 rule I_E_total
    # = δ·K_E_total (pure replacement, allocated across plants by the
    # profitability-weighted reinvestment rule). "npv" activates per-plant
    # NPV-driven investment with Wright's-Law learning. Default "npv"
    # is the post-Stage-2.3 behaviour; "replacement" is retained for
    # backward-compatibility regression checks.
    invest_mode="npv",
    # Quarterly hurdle rates by tech (annualised equivalents in parens).
    # Calibrated to literature estimates for LCOE discount rates: cheap-
    # capital renewables ≈ 5%/yr, nuclear ≈ 8%/yr, thermals ≈ 10-12%/yr.
    r_p_by_tech=dict(
        solar=0.05 / 4,    # 5%/yr → quarterly
        wind=0.05 / 4,
        hydro=0.05 / 4,
        nuclear=0.08 / 4,  # 8%/yr
        coal=0.10 / 4,     # 10%/yr
        gas=0.10 / 4,
        biomass=0.07 / 4,
        oil=0.12 / 4,      # 12%/yr (high risk, peaker)
    ),
    # Wright's-Law learning rates by tech. Positive b_k means each
    # doubling of cumulative output cuts unit cost by a factor (1-2^-b).
    # Solar: ~20%/doubling (IEA, Lafond et al. 2018).
    # Wind:  ~15%/doubling (BNEF).
    # Hydro: ~10%/doubling (mature but site-specific).
    # Nuclear: set to 0 — empirical learning is disputed (Lovering et al.
    # 2016 vs Grubler 2010; Italy's hypothetical nuclear would have been
    # single-digit plants, too few for learning curves to matter).
    # Thermals, oil, biomass: 0.
    b_by_tech=dict(
        solar=0.20,
        wind=0.15,
        hydro=0.10,
        nuclear=0.0,
        coal=0.0,
        gas=0.0,
        biomass=0.0,
        oil=0.0,
    ),
    # Annuity-factor horizon in quarters. 80 q = 20 years is standard for
    # energy-infrastructure NPV (covers most of a plant's useful life).
    horizon_quarters=80,
    # Scale parameter that translates marginal NPV (in € per € of K) into
    # the desired ΔK flow per period. κ·margin·K is the per-quarter
    # expansion flow per plant (above replacement). The margins in this
    # model are large — our baseline clears at 80 €/MWh (oil marginal)
    # while real Italy 2019 cleared at 52 €/MWh (gas marginal), so infra-
    # marginal plants appear to earn supra-realistic gross margins. κ
    # must therefore be calibrated small enough that aggregate expansion
    # stays near aggregate depreciation at baseline. Empirically tuned
    # so baseline aggregate K_E drifts < ~2% per year.
    kappa_invest=1.0e-5,
    # Q_ref multiplier: the cumulative-output anchor is set to
    # Q_ref_multiplier · (initial capacity × 4) = ~Q_ref_multiplier years
    # of operation, so I_unit starts at 1 and learning accumulates
    # smoothly from there. Raising this number slows learning; lowering
    # it accelerates.
    Q_ref_multiplier=4,
    # Initial wage & price (phase 1; ignored in phase 0)
    w_initial=7500.0,
    u_initial=0.08,      # set to u_star so Phillips curve is neutral at t=0
    # Initial stock scales — all in € (Caiani convention).
    # DEP_H per household = 5 quarters of average wage.
    # Firm and household-sector stocks are still per-agent scalars; the
    # energy-sector stocks are now derived from the `stack` (see below)
    # via constant ratios DEP_to_K_ratio_E and L_to_K_ratio_E so larger
    # plants carry proportionally larger deposits and loan books.
    DEP_H_per_hh=37500.0,
    DEP_F_per_firm=15000.0,
    K_F_per_firm=375000.0,
    L_F_per_firm=150000.0,
    # Energy-sector balance-sheet ratios (applied per plant).
    #   DEP_E_i = DEP_to_K_ratio_E · K_E_i
    #   L_E_i   = L_to_K_ratio_E   · K_E_i
    # Chosen so the pre-item-4 uniform-stack values (DEP=45k, L=450k,
    # K=1.125M) are recovered as the 0.04 / 0.40 ratios.
    DEP_to_K_ratio_E=0.04,
    L_to_K_ratio_E=0.40,
    # ---- Technology stack ----
    # `stack` is a list of (tech, K_eur, mc_nameplate, ef_nameplate)
    # per-plant records — see STACK_ITALY_2019 at the top of this file
    # for the default. Swap in STACK_ITALY_2019_NUCLEAR (or any custom
    # list) to run counterfactual tech-mix scenarios.
    stack=STACK_ITALY_2019,
    # Per-tech capacity factors (dimensionless). Oil peakers ≈ 30% per
    # ENTSO-E Italy 2019; other values from EU/IEA 2019–2021 averages
    # (solar PV 13–20% IT, wind onshore 20–25% IT, gas CCGT 30–55% EU,
    # coal 60–75% EU, biomass 60–80% EU, hydro 30–40% IT, nuclear
    # 85–92% EU).
    cap_factor_by_tech=dict(
        coal=0.65,
        gas=0.40,
        nuclear=0.88,
        solar=0.15,
        wind=0.22,
        biomass=0.65,
        hydro=0.35,
        oil=0.30,     # Italian peakers, ENTSO-E 2019
    ),
    # PUN 2019 reference price (€/MWh) used to convert the nameplate
    # mc and ef values in `stack` into the Caiani-convention model
    # units (€/unit_E and tCO₂/unit_E respectively). Under the Caiani
    # convention, 1 unit_E = €1 of energy at 2019 prices ≡ 1/52 MWh.
    pun_2019_eur_per_mwh=52.0,
    # ---- Phase-2b credit & investment parameters ----
    # Quarterly depreciation rates on real capital.
    depreciation_rate_F=0.010,
    depreciation_rate_E=0.008,
    # Quarterly amortisation rates on outstanding loan principal.
    # amort_rate × L_total each period reduces L and transfers cash
    # from borrower to bank.
    amort_rate_F=0.025,
    amort_rate_E=0.020,
    # Firm investment propensity: fraction of nominal F-output reinvested.
    i_rate_F=0.05,
    # Working-capital coverage: fraction of the quarter's wage bill that
    # firms/energy producers request new loans to cover.
    working_capital_coverage=0.25,
    # Bank capital-ratio target (Basel-style). 0.08 matches the value
    # used to set initial bank GB holdings, so the ratio is non-binding
    # at t=0 and only begins to bite if bank NW later deteriorates.
    bank_capital_target=0.08,
    # ---- Phase-2c policy parameters ----
    # Carbon tax in €/tCO₂ (EU ETS convention under the Caiani
    # convention). Default 0 leaves Phase-2a/2b dispatch untouched.
    # Reference EU ETS averages: ~25 €/tCO₂ (2019), ~85 (2022), ~83
    # (2023). A coal plant (ef≈0.01731 tCO₂/unit_E, post-conversion)
    # with τ=25 sees its effective mc rise by 25·0.01731 ≈ 0.433 —
    # comparable to its own mc of 0.77, so the dispatch tilt is
    # substantial.
    carbon_tax=0.0,
    # ---- Stage 5.1 — ETS permit market ----
    # With `ets_mode = True` the carbon_tax becomes endogenous: each
    # period the dispatch is cleared against an emissions cap (paper
    # §5 Eq. 20). The period-τ is solved via bisection such that
    # realised emissions = ets_cap. ets_cap is in tCO₂ per quarter.
    # ets_mode = False keeps the legacy exogenous-τ behaviour.
    ets_mode=False,
    ets_cap=10000.0,          # tCO₂ per quarter; 10k is roughly baseline
    ets_tau_max=500.0,         # €/tCO₂ ceiling for bisection
    ets_tolerance=0.01,        # 1% cap tolerance
    # ---- Stage 5.2 — policy credibility κ_G (paper §3.7) ----
    # `announced_carbon_tax` is the level the government commits to.
    # Credibility compares realised τ (possibly set endogenously by
    # ETS clearing) to this announcement. Deviation erodes κ_G
    # exponentially at rate `credibility_update_rate`. Low κ_G shrinks
    # the effective DER-adoption threshold, reducing adoption even
    # when payback is short — the "policy-volatility penalty".
    # Default: announcement matches τ so κ_G stays 1 (fully credible).
    announced_carbon_tax=None,        # None ⇒ mirrors `carbon_tax`
    credibility_update_rate=0.10,     # β; 0.10 ⇒ 10q half-life
    credibility_min=0.05,              # floor so adoption doesn't vanish
    # ---- Stage 5.3 — green-bond issuance (paper §3.10 Eq. 32) ----
    # Direct government transfer to the E sector per quarter, in € per
    # period. Financed by new GB issuance so the government's budget
    # identity closes: GB_issued = green_subsidy_E (any shortfall is
    # the period fiscal deficit). Default 0 leaves the balanced-budget
    # regime intact; set > 0 in policy-experiment scenarios to model a
    # green stimulus programme.
    green_subsidy_E=0.0,
    # ---- Stage 5.4 — price-support instruments (paper §5) ----
    # Feed-in tariff (FiT): G pays supported renewable plants a fixed
    # per-unit premium on top of the merit-order clearing price. Fiscal
    # cost financed by additional GB issuance. `fit_price` is the top-up
    # in €/unit_E (model units), `fit_supported_techs` lists the plant
    # technologies eligible. Default 0 / empty leaves FiT inactive.
    fit_price=0.0,
    fit_supported_techs=("solar", "wind", "hydro"),
    # Contract for Difference (CfD): plants in supported_techs settle
    # against strike price each period. Producer receives (strike − p_E)
    # per unit output if p_E < strike, pays (p_E − strike) otherwise —
    # a symmetric price hedge. Net fiscal position is a deficit when
    # p_E < strike on average. strike = 0 (default) deactivates CfD.
    cfd_strike=0.0,
    cfd_supported_techs=("solar", "wind", "hydro"),
    # Profitability exponent in the reinvestment rule w_k ∝ π_k^α.
    # α=1 is proportional; α=2 sharpens the preference for profitable
    # plants without making the rule degenerate.
    churn_alpha=2.0,
)


def make_initial_economy(params: dict | None = None,
                         seed: int = 0,
                         pre_converge: int = 500) -> Economy:
    """
    Build an `Economy` whose opening stocks satisfy every SFC identity
    exactly.

    If `pre_converge > 0`, the freshly-built economy is silently run
    forward for that many quarters under Phase 2c at τ=0 (the "full
    model at zero policy") so that the returned state represents the
    converged steady-state baseline rather than an arbitrary initial
    condition. This matters for scenario analysis: applying a shock
    at t=1 from the un-equilibrated state layers shock on transient
    and the two are hard to disentangle.

    The pre-converge pass:
      * advances the economy through the Phillips-curve transient
        until u ≈ u_star,
      * leaves all stocks, prices, and behavioural state at their
        steady-state values,
      * resets the simulation counter `eco.t` back to 0 so downstream
        code sees a fresh run,
      * zeroes the cumulative `emissions_stock` so paper-reported
        emissions count only the post-bootstrap horizon.

    Set `pre_converge = 0` to recover the legacy un-equilibrated
    behaviour (useful for debugging the warm-start itself or for
    closure tests that reason about t=0 flows from first-principles
    initial conditions).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    # ---- Households ----
    N_H = p["N_H"]
    # Stage 6.1: per-quintile initial deposit distribution (Italian
    # HFCS 2017 calibration). If `DEP_H_by_quintile` is provided,
    # households in quintile q get DEP_H_by_quintile[q]. Falls back to
    # the scalar `DEP_H_per_hh` for backward compatibility.
    dep_by_q = p.get("DEP_H_by_quintile", None)
    # Quintile assignment — 20% of households per quintile (0..4).
    quintile = np.tile(np.arange(5), int(np.ceil(N_H / 5)))[:N_H]
    if dep_by_q is not None:
        dep_by_q_arr = np.asarray(dep_by_q, dtype=float)
        DEP_H = dep_by_q_arr[quintile]
    else:
        DEP_H = np.full(N_H, p["DEP_H_per_hh"], dtype=float)
    # Stage 4.2 DER state — per-household initialisation.
    # Class map: quintiles 0-1 → L, 2-3 → M, 4 → H (same as Stage 4.1).
    class_from_q = np.asarray(p.get("class_from_quintile", (0, 0, 1, 1, 2)),
                              dtype=int)
    class_of_hh = class_from_q[quintile]
    is_L = (class_of_hh == 0)
    # Stage 6.2: credit access by quintile. When `eta_kappa_by_quintile`
    # is provided, each household gets a Bernoulli draw against its own
    # quintile's probability. Falls back to the scalar `eta_kappa` applied
    # only to the L-class (quintiles 0-1), with M/H = 1.0 — the pre-
    # Stage-6 behaviour.
    credit_access = np.ones(N_H, dtype=bool)
    eta_by_q = p.get("eta_kappa_by_quintile", None)
    if eta_by_q is not None:
        eta_arr = np.asarray(eta_by_q, dtype=float)
        # Per-household threshold, one random draw each.
        draws = rng.random(N_H)
        credit_access = draws < eta_arr[quintile]
    else:
        eta_kappa = float(p.get("eta_kappa", 0.4))
        credit_access[is_L] = rng.random(int(is_L.sum())) < eta_kappa
    # Adoption thresholds θ_i: class-specific uniform distributions.
    thresh_L_lo, thresh_L_hi = p.get("adoption_threshold_L", (5.0, 15.0))
    thresh_MH_lo, thresh_MH_hi = p.get("adoption_threshold_MH", (3.0, 10.0))
    adoption_threshold = np.zeros(N_H, dtype=float)
    adoption_threshold[is_L]  = rng.uniform(
        thresh_L_lo, thresh_L_hi, size=int(is_L.sum()))
    adoption_threshold[~is_L] = rng.uniform(
        thresh_MH_lo, thresh_MH_hi, size=int((~is_L).sum()))
    H = HouseholdArray(
        N=N_H, DEP=DEP_H, quintile=quintile,
        has_DER=np.zeros(N_H, dtype=bool),
        DER_capacity=np.zeros(N_H, dtype=float),
        DER_age=np.zeros(N_H, dtype=int),
        credit_access=credit_access,
        adoption_threshold=adoption_threshold,
    )

    # ---- Firms ----
    N_F = p["N_F"]
    K_F   = np.full(N_F, p["K_F_per_firm"],   dtype=float)
    DEP_F = np.full(N_F, p["DEP_F_per_firm"], dtype=float)
    L_F   = np.full(N_F, p["L_F_per_firm"],   dtype=float)
    # Energy intensity = units of E (€-of-energy) consumed per unit of
    # F-output (€-of-F). Under Caiani-convention prices ~ 1, a mean of
    # ≈ 0.29 gives an energy cost share in F-production of ei/UC ≈
    # 0.29/0.833 ≈ 35%, consistent with economy-wide Italian 2019 gross
    # energy-cost exposure (energy-intensive manufacturing lifts the
    # average above narrow-sector numbers). Uniform ±15% heterogeneity.
    energy_intensity = rng.uniform(0.25, 0.33, size=N_F)
    F = FirmArray(N=N_F, K=K_F, DEP=DEP_F, L=L_F, energy_intensity=energy_intensity)

    # ---- Energy producers ----
    # Build per-plant arrays from `stack`. Each record is
    #     (tech_label, K_eur, mc_nameplate_eur_per_MWh, ef_tCO2_per_MWh)
    # and `stack` length determines N_E. Nameplate mc and ef are
    # converted to Caiani model units (€/unit_E and tCO₂/unit_E) by
    # dividing by PUN 2019 (52 €/MWh).
    stack = list(p["stack"])
    if not stack:
        raise ValueError("init_stocks: `stack` is empty — at least one "
                         "generation plant is required.")
    N_E = len(stack)
    p["N_E"] = N_E  # keep params dict consistent with the realised stack
    pun = float(p["pun_2019_eur_per_mwh"])
    tech = np.array([rec[0] for rec in stack])
    K_E  = np.array([float(rec[1]) for rec in stack], dtype=float)
    mc   = np.array([float(rec[2]) / pun for rec in stack], dtype=float)
    ef   = np.array([float(rec[3]) / pun for rec in stack], dtype=float)
    # Balance-sheet scaling: deposits and loans proportional to per-
    # plant capital so bigger plants carry bigger books.
    DEP_E = p["DEP_to_K_ratio_E"] * K_E
    L_E   = p["L_to_K_ratio_E"]   * K_E
    # Per-plant capacity factor from the tech → factor dict. Any tech
    # label not covered by the dict falls back to 0.15 (the pre-item-3
    # flat default) so unit tests and unknown plant types still run.
    cap_factor_by_tech = p["cap_factor_by_tech"]
    cap_factor_E = np.array(
        [cap_factor_by_tech.get(str(t), 0.15) for t in tech],
        dtype=float,
    )
    # Cumulative-output anchor for Wright's Law (Stage 2.3). Seed each
    # plant's cum_Q with Q_ref_multiplier years × expected baseline output
    # at Q_plant = cap_factor_k · K_k, so I_unit = 1 at t=0 regardless of
    # the learning rate. Subsequent dispatches add real output to this
    # counter, and I_unit decays per Wright's Law.
    Q_ref_mult = int(p.get("Q_ref_multiplier", 4))
    cum_Q0 = Q_ref_mult * 4.0 * (cap_factor_E * K_E)   # 4 quarters/year
    I_unit0 = np.ones(N_E, dtype=float)
    E = EnergyArray(N=N_E, tech=tech, K=K_E, DEP=DEP_E, L=L_E,
                    mc=mc, emission_factor=ef,
                    cap_factor=cap_factor_E,
                    cumulative_output=cum_Q0,
                    I_unit=I_unit0)

    # ---- Banks ----
    N_B = p["N_B"]
    # Aggregate loan and deposit books
    total_L_F = float(L_F.sum())
    total_L_E = float(L_E.sum())
    total_DEP = float(DEP_H.sum() + DEP_F.sum() + DEP_E.sum())

    # Spread loans and deposits uniformly across banks
    L_F_per_bank = np.full(N_B, total_L_F / N_B, dtype=float)
    L_E_per_bank = np.full(N_B, total_L_E / N_B, dtype=float)
    DEP_liab_per_bank = np.full(N_B, total_DEP / N_B, dtype=float)

    # Bank capital ratio target: NW_B = 8% * assets
    # Use a stock of government bonds to make NW_B positive while preserving
    # balance-sheet closure.  Solve for GB such that
    #   NW_B = L_F + L_E + GB + Res - DEP_liab = 0.08 * (L_F + L_E + GB + Res)
    # with Res = 0 for simplicity.
    target_ratio = 0.08
    assets_ex_GB = total_L_F + total_L_E
    # target: NW = ratio * (assets_ex_GB + GB)
    # NW = assets_ex_GB + GB - total_DEP
    # so assets_ex_GB + GB - total_DEP = ratio * (assets_ex_GB + GB)
    # (1 - ratio) * (assets_ex_GB + GB) = total_DEP
    # GB = total_DEP / (1 - ratio) - assets_ex_GB
    total_GB_B = total_DEP / (1.0 - target_ratio) - assets_ex_GB
    if total_GB_B < 0:
        total_GB_B = 0.0  # fall back if calibration would imply negative bonds

    GB_per_bank = np.full(N_B, total_GB_B / N_B, dtype=float)
    Res_per_bank = np.zeros(N_B, dtype=float)

    B = BankArray(N=N_B,
                  L_F=L_F_per_bank,
                  L_E=L_E_per_bank,
                  GB=GB_per_bank,
                  Res=Res_per_bank,
                  DEP_liab=DEP_liab_per_bank)

    # ---- Government & Central Bank ----
    G = Government(GB_out=total_GB_B, DEP_CB=0.0)
    CB = CentralBank(GB_held=0.0, Res_liab=0.0, DEP_G=0.0)

    # ---- Phase-1 state scalars ----
    # Derive initial representative F-price from the pricing rule so that
    # phase-1 dynamics start on-manifold (price consistent with current
    # wage and unit cost).
    ei_avg = float(energy_intensity.mean())
    w0     = p["w_initial"]
    p_E    = p["p_E"]
    UC0    = w0 / p["productivity_F"] + ei_avg * p_E
    p_F0   = (1.0 + p["markup"]) * UC0
    cpi0   = p["taste_F"] * p_F0 + (1.0 - p["taste_F"]) * p_E

    eco = Economy(
        H=H, F=F, E=E, B=B, G=G, CB=CB, params=p, t=0,
        w=w0, p_F=p_F0, cpi=cpi0,
        u=p["u_initial"], Y_F=0.0, Y_E=0.0,
        # Adaptive expectations: both initialised to the baseline
        # p_E so Phase 1 decisions at t=0 see the same price the
        # legacy code did; divergence only begins once the first
        # dispatch produces a realised price that differs.
        p_E_expected=float(p_E),
        p_E_realised=float(p_E),
        # Stage-4.1 income-class wage state. All three classes start
        # at the initial wage; Phillips dynamics diverge them. Counters
        # initialised to 0. `margin_reference = markup` so at baseline
        # the L-class φ_pm pass-through is zero.
        w_by_class=np.full(3, float(p["w_initial"]), dtype=float),
        qtrs_since_update=np.zeros(3, dtype=int),
        cpi_at_last_update=np.full(3, float(cpi0), dtype=float),
        margin_reference=float(p["markup"]),
        # Volatility-tracker window: history seeded with the baseline
        # price, so σ̂_PE = 0 at t=0 (no variance, no history of shocks).
        p_E_history=np.full(
            int(p.get("volatility_window", 4)), float(p_E), dtype=float),
        p_E_volatility=0.0,
        # Phase-2a diagnostics: all zero at t=0.
        Y_per_plant=np.zeros(N_E, dtype=float),
        emissions_stock=0.0,
        # Phase-2c: policy lever and profit memory. τ=0 reproduces the
        # Phase-2b trajectory bit-for-bit; pull the lever to switch on
        # carbon pricing.
        carbon_tax=p["carbon_tax"],
        profit_per_plant=np.zeros(N_E, dtype=float),
    )

    # --- Warm-start the Caiani dividend lag ---
    # Without this, at t=1 the predetermined dividend is zero and
    # make_phase1_flows falls back to the no-dividend consumption rule,
    # which starts the economy in the death-spiral regime. Warm-start
    # by solving the θ-equilibrium consumption fixed point analytically
    # at the initial wage, then imputing the sector-wise operating
    # profits that sustain it.
    theta = p.get("theta_div", 0.0)
    if theta > 0.0:
        NW_H0 = float(H.NW.sum())
        r_D_H_0 = p["r_D"] * NW_H0
        r_GB_B_0 = p["r_GB"] * float(B.GB.sum())
        # CES shares at initial prices (taste_F = phase-1 weight)
        num_F = p["taste_F"] * p_F0 ** (1.0 - p["ces_sigma"])
        s_F0  = num_F / (num_F + (1.0 - p["taste_F"])
                         * p_E ** (1.0 - p["ces_sigma"]))
        lam0 = (s_F0 / (p_F0 * p["productivity_F"])
                + (1.0 - s_F0) / (p_E * p["productivity_E"])
                + ei_avg * s_F0 / (p_F0 * p["productivity_E"]))
        # θ-equilibrium consumption:
        #   C · (1 - α1·[θ + (1-θ)·wλ]) = α1·(r_D_H - r_GB_B) + α2·NW
        denom_ss = 1.0 - p["alpha1"] * (theta + (1.0 - theta)
                                         * w0 * lam0)
        num_ss = (p["alpha1"] * (r_D_H_0 - r_GB_B_0)
                  + p["alpha2"] * NW_H0)
        if denom_ss > 0:
            C_ss = max(0.0, num_ss / denom_ss)
            # Impute steady-state sectoral flows at initial prices.
            C_H_F_ss = s_F0 * C_ss
            EB_H_ss  = (1.0 - s_F0) * C_ss
            Y_F_ss = C_H_F_ss / p_F0
            EB_F_ss = ei_avg * Y_F_ss * p_E
            Y_E_ss = (EB_H_ss + EB_F_ss) / p_E
            N_F_ss = Y_F_ss / p["productivity_F"]
            N_E_ss = Y_E_ss / p["productivity_E"]
            W_F_ss = w0 * N_F_ss
            W_E_ss = w0 * N_E_ss
            # Operating profits (no loans/tax flows at t=0 yet).
            eco.last_profit_F = (C_H_F_ss - W_F_ss - EB_F_ss
                                 + p["r_D"] * float(F.DEP.sum())
                                 - p["r_L"] * float(F.L.sum()))
            eco.last_profit_E = (EB_H_ss + EB_F_ss - W_E_ss
                                 + p["r_D"] * float(E.DEP.sum())
                                 - p["r_L"] * float(E.L.sum()))
            # Bank net interest margin. At t=0 there are no amort,
            # new_loans or writeoff flows, so π_B reduces to the
            # interest spread alone.
            eco.last_profit_B = (p["r_L"] * float(F.L.sum())
                                 + p["r_L"] * float(E.L.sum())
                                 + p["r_GB"] * float(B.GB.sum())
                                 - p["r_D"] * float(H.DEP.sum())
                                 - p["r_D"] * float(F.DEP.sum())
                                 - p["r_D"] * float(E.DEP.sum()))

    # ---- Pre-converge to steady state (Option B from scoping pass) ----
    # Silently advance the economy through the Phillips-curve transient
    # so the returned state has u ≈ u_star. Without this the first
    # ~400 quarters are consumed by the wage-price adjustment and any
    # shock applied earlier gets tangled up with it. Local import to
    # avoid a circular `dynamics -> init_stocks` reference.
    if pre_converge > 0:
        from dynamics import step as _step
        # Suppress deviation-driven behavioural rules during the bootstrap
        # period. Before the first dispatch runs, `p_E_reference` sits at
        # its default (1.0) while `p_E_expected` drifts toward the true
        # steady-state clearing price. Any rule that reacts to
        # `(p_E_expected - p_E_reference)` would fire on that transient
        # gap and corrupt the initial conditions (e.g., firm efficiency
        # investment would drive `energy_intensity` down by ~50% during
        # the warm-up). We temporarily neutralise the relevant sensitivities
        # here, then restore them once the reference has been calibrated.
        # Stage 2.3 NPV investment is also suppressed: pre-convergence
        # uses pure replacement (I_E_total = δ·K_E_total) so the warm-up
        # reaches a clean stationary point, from which the post-bootstrap
        # trend-stationary trajectory emerges organically once NPV kicks in.
        _saved_g_e           = p.get("g_e", 0.0)
        _saved_mode          = p.get("invest_mode", "npv")
        _saved_phi_pm        = p.get("phi_pm", 0.0)
        _saved_contract_len  = p.get("contract_length", (1, 4, 4))
        p["g_e"] = 0.0
        p["invest_mode"] = "replacement"
        p["phi_pm"] = 0.0  # Stage-4.1 deviation term off during bootstrap
        # Stage 4.1: neutralise contract rigidity during pre-convergence
        # so all three classes update at the same rhythm and converge to
        # identical baseline wages. Without this, the L class (quarterly
        # updates) drifts 4× as far as M/H (4-quarter contracts) over
        # the 500q warm-up — an artefact of update frequency, not
        # economics. Restored to the real (1, 4, 4) cycle afterward.
        p["contract_length"] = (1, 1, 1)
        # Stage 4.2: suppress DER adoption during the warm-up. Without
        # this, any household whose payback threshold is below baseline
        # p_E's payback (which is ~10y at p_E ≈ 1.54) would adopt
        # during pre-converge, distorting the baseline household state.
        p["_der_suppressed"] = True
        # Stage 5.2: suppress credibility drift during bootstrap so κ_G
        # stays at its initial value. Lets scenario drivers start from a
        # "fully credible" baseline and introduce policy breaks mid-run.
        p["_cred_suppressed"] = True
        for _ in range(pre_converge):
            _step(eco, phase=4, tol=1e-10)
        p["g_e"] = _saved_g_e
        p["invest_mode"] = _saved_mode
        p["phi_pm"] = _saved_phi_pm
        p["contract_length"] = _saved_contract_len
        p["_der_suppressed"] = False
        p["_cred_suppressed"] = False
        # Reset simulation counter and cumulative-flow diagnostics so
        # downstream code treats the converged state as a fresh t=0.
        eco.t = 0
        eco.emissions_stock = 0.0
        if eco.E.emissions_flow is not None:
            eco.E.emissions_flow.fill(0.0)
        # Capture the converged p_E as the reference for deviation-based
        # firm decisions (Stage 2.2). This ensures the efficiency-
        # investment rule returns zero at baseline, keeping energy
        # intensity stationary absent shocks.
        eco.p_E_reference = float(eco.p_E_expected)
        # Stage 4.1: capture the converged firm profit margin as the
        # reference for the L-class wage pass-through. At baseline the
        # margin_dev term is zero, so L-wages follow the standard
        # Phillips rule absent shock; shocks to firm margins (e.g. via
        # Stage-2.1 uncertainty markup) now bid up L-wages directly.
        if eco.p_F > 0.0:
            # markup proxy: (p_F − UC) / UC where UC = p_F / (1+μ)
            # under the calibrated markup rule. The realised effective
            # markup is (p_F × (1+μ) − p_F) / p_F = μ — so the
            # baseline reference just reads the params markup.
            eco.margin_reference = float(p.get("markup", 0.20))
    return eco
