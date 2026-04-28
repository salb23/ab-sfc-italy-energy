"""
agents.py — sector dataclasses for the full AB-SFC framework.

Phase 0 uses state attributes only; behavioural methods are added in
later phases. Stocks are stored as numpy arrays so the sector aggregates
are simply `arr.sum()`.

Sign convention (consistent with v2.1 Table 2, BSM):
  - all stored stocks are non-negative;
  - the role (asset vs liability) of each stock is implicit in the
    sector that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Households
# --------------------------------------------------------------------------
@dataclass
class HouseholdArray:
    """
    Households hold deposits at banks as their only financial asset.
    Heterogeneity is indexed by `quintile` (0..4).

    Stage-4.2 DER state (distributed-energy-resources adoption):
    heterogeneous payback-threshold rule from Rai & Robinson (2015,
    *Ecological Economics*); SFC integration follows Monasterolo &
    Raberto (2018, *Ecological Economics*, EIRIN model).

    The DER capex is treated as an F-sector purchase (households
    buy solar-panel hardware from the real economy), which keeps
    the TFM closure trivial without introducing a new sector or a
    new loan type. Subsequent-period energy savings (DER output ×
    p_E) reduce the household-to-E cash flow and implicitly boost
    H deposits.
    """
    N: int
    DEP: np.ndarray       # deposits held at banks, shape (N,)
    quintile: np.ndarray  # int in {0,1,2,3,4}, shape (N,)

    # ---- Stage-4.2 DER state ----
    # has_DER: True iff household has installed DER (solar rooftop).
    # DER_capacity: installed generation capacity in units of E per
    #     quarter (under Caiani convention, 1 unit_E ≈ 1/52 MWh).
    # DER_age: quarters since adoption; drives end-of-life retirement.
    # credit_access: one-time Bernoulli flag set at init; L-class (low-
    #     income quintiles) receive this with probability η_κ, M/H
    #     class always receive it. Households without credit access
    #     cannot finance DER even when payback is attractive.
    # adoption_threshold: θ_i ∈ years; household adopts when payback <
    #     θ_i. Heterogeneous, drawn at init. L-class uses a wider
    #     distribution (higher risk aversion, higher imputed discount
    #     rate) than M/H.
    has_DER:             np.ndarray = None  # type: ignore[assignment]
    DER_capacity:        np.ndarray = None  # type: ignore[assignment]
    DER_age:             np.ndarray = None  # type: ignore[assignment]
    credit_access:       np.ndarray = None  # type: ignore[assignment]
    adoption_threshold:  np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.has_DER is None:
            self.has_DER = np.zeros(self.N, dtype=bool)
        if self.DER_capacity is None:
            self.DER_capacity = np.zeros(self.N, dtype=float)
        if self.DER_age is None:
            self.DER_age = np.zeros(self.N, dtype=int)
        if self.credit_access is None:
            self.credit_access = np.ones(self.N, dtype=bool)
        if self.adoption_threshold is None:
            self.adoption_threshold = np.full(self.N, 10.0, dtype=float)

    @property
    def DEP_total(self) -> float:
        return float(self.DEP.sum())

    @property
    def NW(self) -> np.ndarray:
        # Households hold only deposits. DER is treated as an F-sector
        # purchase (see class docstring) so it does NOT appear on the
        # household balance sheet as an asset. Households are "poorer"
        # just after adoption (DEP down) but recoup through cheaper
        # energy bills in subsequent periods.
        return self.DEP.copy()

    @property
    def NW_total(self) -> float:
        return float(self.NW.sum())


# --------------------------------------------------------------------------
# Non-energy firms
# --------------------------------------------------------------------------
@dataclass
class FirmArray:
    """
    Firms hold real capital K and bank deposits (retained earnings)
    as assets, bank loans L as liability.
    """
    N: int
    K:   np.ndarray       # real capital stock, shape (N,)
    DEP: np.ndarray       # firm bank deposits, shape (N,)
    L:   np.ndarray       # outstanding bank loans (liability), shape (N,)
    energy_intensity: np.ndarray  # units of E per unit of F-output, shape (N,)
    # Energy-efficiency investment intensity (paper §3.4 Eq. 6–7,
    # gap-closure Stage 2.2). Each quarter firms devote a fraction
    # `x_f ∈ [0, x_max]` of their attention to efficiency upgrades,
    # which drives the law of motion
    #     energy_intensity_{f,t+1} = energy_intensity_{f,t} · (1 − g_e · x_f)
    # The decision rule `x_f = min(x_max, η · max(0, p_E_expected − p_E_ref))`
    # returns 0 at the steady-state reference price (baseline ei stationary)
    # and saturates to x_max during severe shocks.
    x: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.x is None:
            self.x = np.zeros(self.N, dtype=float)

    @property
    def NW(self) -> np.ndarray:
        return self.K + self.DEP - self.L

    @property
    def NW_total(self) -> float:
        return float(self.NW.sum())


# --------------------------------------------------------------------------
# Energy producers
# --------------------------------------------------------------------------
@dataclass
class EnergyArray:
    """
    Energy producers have a capital stock per technology class and a
    stock of outstanding green loans.
    """
    N: int
    tech: np.ndarray         # str labels: 'coal','gas','nuclear','solar','wind','biomass'
    K:   np.ndarray          # installed capacity, shape (N,)
    DEP: np.ndarray          # bank deposits (operating cash), shape (N,)
    L:   np.ndarray          # green-loan liability, shape (N,)
    mc:  np.ndarray          # short-run marginal cost (model units), shape (N,)
    emission_factor: np.ndarray  # tCO2 per unit of output, shape (N,)
    # Per-tech capacity factor: share of installed K actually available
    # each quarter. Real values differ by orders of magnitude across
    # technologies (solar ≈ 0.15, wind ≈ 0.22, gas CCGT ≈ 0.40,
    # coal ≈ 0.65, biomass ≈ 0.65, nuclear ≈ 0.88). The merit-order
    # dispatch reads capacity = cap_factor * K per plant.
    cap_factor: np.ndarray = None  # type: ignore[assignment]
    # ---- Phase-2a diagnostic field ----
    # Per-plant emissions flow produced in the most recent period.
    # Not part of any BSM/TFM identity — a trace for post-run analysis.
    emissions_flow: np.ndarray = None  # type: ignore[assignment]
    # ---- Stage-2.3 state (NPV investment + Wright's-Law learning) ----
    # cumulative_output: running sum of per-plant real output since the
    # economy started (reset to an initial anchor at pre-convergence so
    # learning curves start near their calibrated baseline cost). Drives
    # Wright's-Law unit-cost decay — more cumulative experience lowers
    # I_unit for learning techs (solar, wind, hydro); zero learning rate
    # for thermals leaves their I_unit constant.
    # I_unit: current per-plant unit cost of new capacity (€ of outlay
    # per € of K added). Anchored to 1 at baseline; Wright's Law pulls
    # it below 1 for learning techs as cumulative output grows.
    cumulative_output: np.ndarray = None  # type: ignore[assignment]
    I_unit:            np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.emissions_flow is None:
            self.emissions_flow = np.zeros(self.N, dtype=float)
        if self.cap_factor is None:
            # Backwards-compatible fallback for callers that still use
            # the old scalar `cap_factor`. A uniform 0.15 recovers the
            # pre-item-3 behaviour exactly.
            self.cap_factor = np.full(self.N, 0.15, dtype=float)
        if self.cumulative_output is None:
            self.cumulative_output = np.zeros(self.N, dtype=float)
        if self.I_unit is None:
            self.I_unit = np.ones(self.N, dtype=float)

    @property
    def NW(self) -> np.ndarray:
        return self.K + self.DEP - self.L

    @property
    def NW_total(self) -> float:
        return float(self.NW.sum())


# --------------------------------------------------------------------------
# Commercial banks
# --------------------------------------------------------------------------
@dataclass
class BankArray:
    """
    Banks hold loans to F and E and a portfolio of government bonds GB
    as assets; customer deposits as liability. Reserves at the central
    bank are held as the residual liquidity buffer.
    """
    N: int
    L_F:   np.ndarray        # loans extended to F (asset), shape (N,)
    L_E:   np.ndarray        # loans extended to E (asset), shape (N,)
    GB:    np.ndarray        # government bonds held (asset), shape (N,)
    Res:   np.ndarray        # reserves at CB (asset), shape (N,)
    DEP_liab: np.ndarray     # sum of all customer deposits (liability), shape (N,)

    @property
    def NW(self) -> np.ndarray:
        return self.L_F + self.L_E + self.GB + self.Res - self.DEP_liab

    @property
    def NW_total(self) -> float:
        return float(self.NW.sum())


# --------------------------------------------------------------------------
# Government (singleton)
# --------------------------------------------------------------------------
@dataclass
class Government:
    GB_out: float = 0.0      # total bonds outstanding (liability)
    DEP_CB: float = 0.0      # government account at CB (asset, usually 0)

    @property
    def NW(self) -> float:
        return self.DEP_CB - self.GB_out


# --------------------------------------------------------------------------
# Central bank (singleton)
# --------------------------------------------------------------------------
@dataclass
class CentralBank:
    GB_held: float = 0.0     # bonds held by CB (asset)
    Res_liab: float = 0.0    # reserves issued to banks (liability)
    DEP_G:    float = 0.0    # government's deposit at CB (liability, mirror of Gov.DEP_CB)

    @property
    def NW(self) -> float:
        return self.GB_held - self.Res_liab - self.DEP_G


# --------------------------------------------------------------------------
# Economy — top-level container
# --------------------------------------------------------------------------
@dataclass
class Economy:
    """
    Top-level container. All sectors and scalar parameters live here.
    The `t` field is the integer quarter counter.

    Phase-1 state scalars (w, p_F, cpi, u, Y_F, Y_E) are unused in
    phase 0 but are carried on the dataclass so the same `Economy`
    type serves every phase. In phase 2 they become per-agent fields
    on FirmArray / HouseholdArray.
    """
    H:  HouseholdArray
    F:  FirmArray
    E:  EnergyArray
    B:  BankArray
    G:  Government
    CB: CentralBank
    params: dict = field(default_factory=dict)
    t: int = 0

    # ---- Phase-1 state scalars (unused in phase 0) ----
    w:    float = 0.0   # quarterly wage per employed worker
    p_F:  float = 0.0   # representative F-goods price
    cpi:  float = 1.0   # aggregate price index, previous period
    u:    float = 0.0   # unemployment rate, previous period
    Y_F:  float = 0.0   # real output of F sector, previous period
    Y_E:  float = 0.0   # real output of E sector, previous period

    # ---- Adaptive price expectations (paper §3.7, Eq. 15) ----
    # Stage-1 of the gap-closure plan: firms and households now make
    # decisions against an expectation rather than last period's realised
    # price. The expectation is an exponential moving average:
    #   p_E_expected_{t+1} = α_p · p_E_realised_t + (1 − α_p) · p_E_expected_t
    # α_p ∈ [0, 1] comes from params["alpha_p"] (default 0.5). α_p = 1
    # reproduces the pre-Stage-1 "expectation = last realised"
    # behaviour. Smaller α_p damps expectation revisions and builds
    # inertia against transient price shocks — important for the
    # paper's fragility channel.
    #
    # p_E_realised holds the most recent merit-order clearing price,
    # stored for diagnostics and for the expectation update at commit.
    p_E_expected: float = 1.0
    p_E_realised: float = 1.0
    # Steady-state reference price for deviation-based decisions such
    # as firm efficiency investment (Stage 2.2). Captured at the end
    # of pre-convergence so firms treat the converged baseline as the
    # null hypothesis for "normal times" and only respond to deviations.
    p_E_reference: float = 1.0

    # ---- Income-class wage segmentation (paper §3.3, Stage 4.1) ----
    # The quintile tag on HouseholdArray maps to three classes:
    #   L (quintiles 0-1): flexible-contract spot-market wages that
    #       respond every period to inflation, unemployment, AND firm
    #       profit margin (φ_pm pass-through — Stage-4.1 novelty).
    #   M (quintiles 2-3): rigid 4-quarter contracts; wage locked
    #       between renewals, updated on cumulative inflation since
    #       last renewal.
    #   H (quintile 4):    same rigidity as M.
    # The scalar `w` above stays as the employment-weighted average
    # (backward-compat: solve_consumption and the λ coefficient still
    # read it). `w_by_class` is the primary state; Phillips dynamics
    # operate per-class.
    w_by_class:         np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float))
    qtrs_since_update:  np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=int))
    cpi_at_last_update: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=float))
    # Reference firm profit margin for the L-class pass-through rule.
    # Captured at end of pre-convergence (same pattern as
    # `p_E_reference`) so baseline φ_pm contribution is zero and
    # L-wages only respond to inflation / unemployment in steady state.
    margin_reference: float = 0.0

    # ---- Price-volatility tracker σ̂_PE (gap-closure Stage 1.2) ----
    # Rolling record of the last `volatility_window` realised prices
    # (default window = 4 quarters = 1 year). After each dispatch the
    # oldest entry is dropped and the newest realised price appended.
    # σ̂_PE is the coefficient of variation over the window,
    #   σ̂_PE = std(p_E_history) / mean(p_E_history),
    # i.e. dimensionless price volatility. Used by Stage-2.1 (uncertainty-
    # loaded markup) and downstream by the NPV / investment rules.
    # Starts at 0 (history full of baseline p_E ⇒ zero variance).
    p_E_history:    np.ndarray = field(
        default_factory=lambda: np.ones(4, dtype=float))
    p_E_volatility: float = 0.0

    # ---- Phase-2a state (unused in phases 0-1) ----
    # Y_per_plant: real output of each energy plant in the most recent
    # dispatch clearing. Diagnostic only — does not appear in any
    # BSM/TFM identity.
    Y_per_plant:     np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    # emissions_stock: cumulative tCO2 emitted since t=0 (flow stock;
    # not part of NW identity, tracked for policy diagnostics).
    emissions_stock: float = 0.0

    # ---- Phase-2c state (unused in phases 0-2b) ----
    # carbon_tax: policy lever added to each plant's effective marginal
    # cost, τ · ef_k. In model-price units per unit of emission factor.
    # Default 0 leaves the Phase-2a/2b dispatch unchanged.
    carbon_tax: float = 0.0
    # Policy credibility κ_G ∈ [0, 1] (paper §3.7, gap-closure Stage 5.2).
    # Exponential moving average of |realised − announced| policy
    # distance, rescaled to [0, 1]. Low credibility shrinks the
    # effective DER-adoption threshold: households discount future
    # savings when the policy signal is volatile. Initialised at 1.0
    # (full trust); updated at commit time from the realised vs.
    # announced carbon tax each period.
    credibility: float = 1.0
    # profit_per_plant: last period's producer surplus per plant
    # (y_k · (p_E - mc_eff_k)). Consumed by the reinvestment rule in
    # the following period. Not part of any BSM/TFM identity.
    profit_per_plant: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))

    # ---- Phase-1 dividend closure (unused when theta_div=0) ----
    # last_profit_F, last_profit_E: aggregate cash-flow profits for F
    # and E in the most recently-committed period. Consumed by the
    # Caiani dividend rule at the START of the next period —
    #     Div_k = theta_div * max(0, last_profit_k)
    # — which is added to YD and recycles value-added so that a flow
    # equilibrium exists. Predetermined (1-period lag), so the
    # consumption fixed point in make_phase1_flows stays closed-form.
    last_profit_F: float = 0.0
    last_profit_E: float = 0.0
    last_profit_B: float = 0.0

    # ---- Sector aggregates used by the SFC checks ----
    @property
    def DEP_H(self) -> float: return float(self.H.DEP.sum())
    @property
    def DEP_F(self) -> float: return float(self.F.DEP.sum())
    @property
    def DEP_E(self) -> float: return float(self.E.DEP.sum())
    @property
    def DEP_liab(self) -> float: return float(self.B.DEP_liab.sum())

    @property
    def L_F_total(self) -> float: return float(self.F.L.sum())
    @property
    def L_E_total(self) -> float: return float(self.E.L.sum())
    @property
    def L_asset(self) -> float: return float(self.B.L_F.sum() + self.B.L_E.sum())

    @property
    def GB_B(self)  -> float: return float(self.B.GB.sum())
    @property
    def GB_CB(self) -> float: return self.CB.GB_held

    @property
    def K_F_total(self) -> float: return float(self.F.K.sum())
    @property
    def K_E_total(self) -> float: return float(self.E.K.sum())
    @property
    def K_total(self)   -> float: return self.K_F_total + self.K_E_total

    # ---- Nominal GDP proxy (used for residual tolerance only) ----
    @property
    def GDP_proxy(self) -> float:
        """
        Scale used as the denominator for residual tolerance.
        In phase 0 we use total real capital as a stable proxy; from
        phase 1 we will use realised nominal output.
        """
        return max(1.0, self.K_total)
