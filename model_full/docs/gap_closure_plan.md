# Gap-Closure Plan — Paper v2.1 ↔ code alignment

**Status:** Path B (extend code to match paper) chosen 2026-04-23.
**Audit basis:** 47-finding coherence audit of `tmp/v21.md` vs `model_full/*`.
**Principle:** correctness and scientific validity over schedule.
**Invariant at each stage:** full 17-test SFC closure suite plus `tmp/check_all_phases.py` smoke must pass without regression.

---

## Stage 1 — Foundations

Everything else reads from these two primitives.

### 1.1 Adaptive price expectations (§3.7, Eq. 15)
- Add `p_E_expected: float` and `p_E_realised: float` to `Economy`.
- Initialise both to `params["p_E"]`.
- In Phase-1 decisions (`make_phase1_flows`), read `eco.p_E_expected` where the code currently reads `eco.params["p_E"]`.
- After dispatch in Phase 2a commit, store `p_E_realised = _pending_p_E`, then update
  `p_E_expected ← α_p · p_E_realised + (1−α_p) · p_E_expected`.
- New param: `alpha_p = 0.5` (literature 0.3–0.7).
- Test: starting from a shock, `p_E_expected` converges to realised price geometrically. Steady-state unaffected.
- Effort: ~1 day.

### 1.2 Price-volatility tracker σ̂_PE
- Keep a rolling window of the last N=4 `p_E_realised` values.
- Compute `σ̂_PE = std(window) / mean(window)` each period.
- Store on `Economy`.
- Needed by 2.1 (uncertainty-loaded markup) and other stages.
- Test: under a flat baseline σ̂_PE → 0; under a shock it spikes then decays.
- Effort: ~half day.

---

## Stage 2 — Fragility amplifiers (paper Story 1)

### 2.1 Uncertainty-loaded markup (§3.4, Eq. 8)
- `p_F = (1 + ψ̄·(1 + χ·σ̂_PE)) · UC` replacing the fixed markup.
- New param: `chi = 0.5`.
- Likely stabilises the CES instability observed at 4.5× shocks — re-test after implementing.
- Test: under a volatile-shock scenario, p_F is higher than under a flat-shock scenario of the same mean.
- Effort: ~half day.

### 2.2 Firm energy-intensity efficiency investment (§3.4, Eq. 6–7)
- Add `x_f` field on `FirmArray` (fraction of revenue spent on efficiency).
- Update rule: `e_{f,t+1} = e_{f,t} · (1 − g_e · x_{f,t})`.
- Firms choose `x_f` increasing in expected p_E: `x_f = min(x_max, η · p_E_expected)`.
- New params: `g_e = 0.05`, `eta_xe = 0.1`, `x_max = 0.05`.
- Test: after 20q of sustained high p_E, aggregate `energy_intensity.mean()` falls monotonically.
- Effort: ~1 day.

### 2.3 NPV investment + Wright's-Law learning for energy (§3.5, Eq. 11–12)
- Replace `reinvestment_weights` rule in `policy.py` with NPV-based capacity decisions per plant:
  - `NPV_k = Σ_t (p_E_expected − MC_k) · Q_k_expected · (1+r_p)^-t − I_unit · ΔK_k`
  - `I_k = max(0, NPV_k / (r_p · horizon))`.
- Wright's Law: unit cost `I_unit_k = I_unit_{k,0} · (cum_Q_k / cum_Q_{k,0})^(−b_k)`.
- New state on `EnergyArray`: `cumulative_output: np.ndarray`.
- New params: `r_p_by_tech` (hurdle rates), `horizon_years = 20`, `b_by_tech` (learning rates; solar ≈ 0.20, wind ≈ 0.15, nuclear ≈ 0.05, others ≈ 0).
- Test: under rising p_E, cheap-clean plants' K grows; under carbon tax, coal K shrinks to zero over 40q.
- Effort: ~2–3 days.

---

## Stage 3 — Credit sophistication (Story 1 supporting)

### 3.1 Bank-specific interest-rate pricing (§3.6, Eq. 14)
- Replace scalar `r_L` with per-bank per-borrower `r_L_{b,j} = r_CB + μ_b + δ̄·(λ_j/λ_max)^γ`.
- Files: `credit.py`, `dynamics.py` (interest flow assembly).
- New params: `mu_b = 0.015`, `delta_bar = 0.02`, `gamma = 2`, `lambda_max = 12`.
- Test: banks with higher λ charge higher rates; credit spreads widen under stress.
- Effort: ~1 day.

### 3.2 Concentration limit β (§3.6, Eq. 13b)
- In `allocate_new_loans`, cap single-borrower exposure at `β · NW_b`.
- New param: `beta_concentration = 0.25`.
- Test: large borrowers rationed as exposure approaches β; small borrowers unaffected.
- Effort: ~half day.

### 3.3 Green preferential risk weight ρ_green (§3.6)
- Multiply the leverage-constraint weight for E-sector loans by `ρ_green < 1`.
- New param: `rho_green = 0.7`.
- Test: E-sector loans grow faster than F-sector under identical demand conditions.
- Effort: ~half day.

---

## Stage 4 — Distributional layer (paper Story 2)

### 4.1 Income-class wage segmentation (§3.3, lines 749–750)
- Replace scalar `w` with `w_by_class[q]` for q ∈ {L, M, H}.
- Map quintiles: 0–1 → L, 2–3 → M, 4 → H.
- L: flexible Phillips responding to inflation and firm profit margin each period.
- M, H: rigid 4-quarter contracts; wage updates only every 4 quarters.
- Files: `agents.py`, `behaviour.py`, `dynamics.py`.
- New params: `contract_length_M = 4`, `contract_length_H = 4`, `phi_pm = 0.3` (profit-margin pass-through for L).
- Test: under an energy shock, L-class real wage falls more than H-class; quintile-level deposits diverge.
- Effort: ~1.5 days.

### 4.2 Household DER adoption with credit gate (§3.3, Eq. 3–4)
- Extend `HouseholdArray` with `DER_i: np.ndarray` (kWh/q of own-generation) and `has_DER: np.ndarray` (bool).
- Adoption rule each period: `adopt` if `payback_period_i < theta_i · κ_G` (credibility-weighted) AND `has_credit_access_i`.
- `has_credit_access_i = True` for M/H; for L, True with probability `eta_kappa`.
- Adoption cost enters as a one-shot `I_DER_i` flow, paid from DEP or from new household loan (adds a small L_H stock).
- DER output deducts from household energy purchases in the CES layer.
- Files: `agents.py` (major extension), `behaviour.py` (new `adopt_der` function), `dynamics.py` (new flow block), `accounting.py` (new flow type: household investment).
- New params: `theta_dist` (distribution of adoption thresholds by class), `eta_kappa = 0.4`, `der_unit_cost = 2500` (€ per kWh/q of capacity), `der_payback_horizon = 8`.
- Test: under rising p_E, adoption share rises; L-class adoption lags M/H.
- Effort: ~3 days.

### 4.3 Peer-effect stimulus S_{i,t} (§3.3, Eq. 4)
- Mean-field: `S = ρ_peer · adoption_share_q`.
- Enters the adoption rule as an additive boost to the effective threshold.
- New param: `rho_peer = 0.1`.
- Test: adoption curve is S-shaped over time (diffusion).
- Effort: ~half day.

---

## Stage 5 — Policy layer (paper Story 3 + final instruments)

### 5.1 ETS regime with endogenous permit price (§5, Eq. 20)
- Introduce `Policy` class hierarchy:
  - `CarbonTaxPolicy(tau)`: current scalar behaviour.
  - `ETSPolicy(cap_t)`: endogenously solves `Σ_k ε_k · Q_k^disp = cap_t` for τ.
- Store as `eco.policy`.
- ETS clearing: each dispatch period, binary-search τ such that emissions ≤ cap within tolerance.
- Files: new classes in `policy.py`, hooks in `dynamics.py` (dispatch loop calls `policy.effective_tau_and_dispatch(...)`).
- Test: `CarbonTaxPolicy(τ=0)` reproduces current Phase 2c baseline; tight `ETSPolicy(cap)` produces τ > 0 that clamps emissions to cap.
- Effort: ~1.5 days.

### 5.2 Policy credibility κ_G (§3.7)
- Add `eco.credibility ∈ [0,1]`, initialised to 1.0.
- Each period, compare announced policy (stored as `eco.policy_announced`) vs. realised policy.
- Update `κ_G ← (1−β_cr)·κ_G + β_cr · (1 − |announced − realised|/announced)`.
- Credibility enters DER adoption via adjusted effective payback period.
- New param: `beta_cr = 0.1`.
- Test: under a policy U-turn, credibility drops; adoption stalls.
- Effort: ~1 day.

### 5.3 Green-bond issuance dynamics ΔGB_G (§3.10, Eq. 32)
- Government runs a fiscal deficit when transfers + subsidies > tax revenue.
- Deficit covered by new green-bond issuance; `ΔGB_out = DEF_G`.
- Files: `accounting.py`, `dynamics.py`.
- Test: under green-subsidy policy with insufficient carbon revenue, `GB_out` grows; SFC still closes.
- Effort: ~1 day.

### 5.4 Feed-in tariffs (FiT) and Contracts for Difference (CfD) (§5, lines 1352–1359)
- FiT: fixed `p_E^FiT` paid to renewable plants per unit output, financed from general budget.
- CfD: strike price `p_strike`; plant receives `p_strike − p_E` (if `p_E < p_strike`) or pays `p_E − p_strike` (if above).
- Both implemented as policy objects alongside Carbon Tax / ETS; multiple can be active simultaneously.
- Test: under FiT, renewable profit floor holds; under CfD with `p_strike = 60`, renewables insulated from price volatility.
- Effort: ~1 day.

---

## Sequencing rationale

- **Stage 1 is strict prerequisite** for every subsequent behaviour (expectations enter the markup rule, investment NPV, consumption, adoption).
- **Stage 2 before Stage 4** because Story 1 (fragility channel) is the paper's primary contribution; Story 2 (distributional) is an extension.
- **Stage 3 before Stage 5** because ETS and credibility ride on top of working credit mechanics.
- If priorities compress, Stages 2.1 and 2.3 deliver the largest per-day scientific value; Stages 4.2 and 5.1 deliver the largest narrative value.

## Totals

| Stage | Sub-steps | Effort (days) |
|---|---|---|
| 1. Foundations | 2 | 1.5 |
| 2. Fragility amplifiers | 3 | 4 |
| 3. Credit sophistication | 3 | 2 |
| 4. Distributional layer | 3 | 5 |
| 5. Policy layer | 4 | 4.5 |
| **Total** | **15** | **~17** |

Each sub-step is completed when (a) its own test passes, (b) the full 17-test suite passes, and (c) `tmp/check_all_phases.py` reports all five phases surviving 300q with closure residuals within tolerance.

## What stays out of scope

- Spatial / geographic structure (the model is a single region; Italy aggregate).
- Hourly dispatch resolution (quarterly steps retained).
- International trade and imports (Italian net imports ~15% of electricity in 2019 are implicit in the stack calibration, not modelled separately).
- Capex financing for new capacity at plant level (investment.py treats capex as a bank loan; no bond-market issuance for firms).

These are flagged as "paper 2 / future work" — not divergences, just limitations.
