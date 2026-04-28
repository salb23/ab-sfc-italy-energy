# Unit convention

**Status:** Adopted for SIE 2026 paper 1 (scenario analysis).
**Predecessor:** Pre-rescaling "dimensionless model units" with log-transformed merit-order dispatch.
**Reference in literature:** Caiani et al. (2016, JEDC), Poledna et al. (2020, JEBO), Dawid et al. (2018, EURACE@Unibi).

---

## Core principle

All monetary quantities in the model are expressed in **euros of 2019-Italian output**.
All physical quantities keep their natural units (workers, MWh, tCO₂).
At `t=0` every price is normalised to unity.

Concretely:

- **1 unit of F-output** = €1 worth of F-sector goods at 2019 Italian prices.
- **1 unit of E-output** = €1 worth of energy at the 2019 Italian PUN baseline of 52 €/MWh. Equivalently, 1 unit_E = 1/52 MWh ≈ 19.2 kWh.
- **1 unit of K** (real capital) = €1 worth of capital at 2019 prices.
- **Wages** `w` are in €/quarter.
- **Stocks** (`DEP_H`, `L_F`, `GB`, …) are in €.
- **Flows** (`C`, `YD`, `I`, dividends, interest) are in €/quarter.
- **Carbon tax** `τ` is in €/tCO₂ (EU ETS convention).
- **Emission factors** `ef_k` are in tCO₂/unit_E (i.e. tCO₂ per euro-of-energy at 2019 prices).
- **Interest rates** are quarterly fractions (dimensionless).

At `t=0`:

- `p_F ≈ 1` (initial value 1.0; after the first step the markup rule delivers `p_F = (1+μ)·(w/ξ_F + ei·p_E) ≈ 1.000` given the baseline calibration ξ_F=14,000, ei_avg≈0.29)
- `p_E` *initial parameter* = 1.0. After the first merit-order dispatch `p_E` clears at the **marginal plant's mc**, not at the PUN average. In the default Italian 2019 stack the marginal plant is gas-2 with nameplate mc = 57 €/MWh, so steady-state `p_E_model = 57/52 ≈ 1.096` (paper-reporting value 57 €/MWh). The PUN baseline of 52 €/MWh applies to the *unit conversion* (definition of 1 unit_E), not to the clearing price — the ~10% gap reflects solar/wind setting the price in off-peak hours, which is not captured by a static merit order.
- `cpi_initial = taste_F · p_F + (1−taste_F) · p_E_initial = 1.0`; steady-state cpi is slightly below 1 (≈0.95) because the dispatch clears above the initial `p_E` guess and the Phillips curve adjusts w downward over the bootstrap period.
- `w_initial = 7,500` (Italian 2019 gross quarterly wage, Istat "Retribuzioni contrattuali lorde"). Steady-state `w ≈ 6,280` once Phillips equilibrates on u = u_star.
- `DEP_H_per_hh = 37,500` (= 5 quarters of wage). Paper-reporting: €37,500 per representative household.

---

## Why this convention

Two competing requirements:

1. **CES taste weight must retain its economic meaning.** `taste_F` is the baseline expenditure share on F, which is ~0.85 in the Italian 2019 household budget survey. Setting `p_F = p_E = 1` at baseline keeps `s_E = 1 − taste_F = 0.15` by inspection of the CES share formula, regardless of the elasticity σ.
2. **Shocks must pass through linearly.** A TTF gas spike from 55 → 250 €/MWh must appear in the model as `mc_gas × 4.5`, not as a log-compressed nudge. The pre-rescaling log-transform in `init_stocks.py` conflicted with this, and is retired.

Setting `1 unit = €1 at 2019 prices` satisfies both: prices start at 1, shocks scale linearly, taste weight equals the observed share.

---

## Conversion table

Whenever a model quantity is reported in the paper, apply the following transforms.

| Model quantity | Paper quantity | Conversion |
|---|---|---|
| `p_F_model` | GDP deflator (2019 = 1) | identity |
| `p_E_model` | Wholesale electricity €/MWh | × 52 |
| `w_model` | Quarterly gross wage, € | identity |
| `Y_F_model` | F-sector nominal output, € | identity |
| `Y_E_model` | Energy nominal output, € | identity |
| `Y_E_model / p_E_model` | Energy in MWh | ÷ 52 |
| `C_model` | Nominal consumption, € | identity |
| `DEP_H_model` | Household deposits, € | identity |
| `emissions_stock` | Cumulative tCO₂ | identity (already physical) |
| `carbon_tax` | EU ETS price €/tCO₂ | identity |
| `ef_model` (tCO₂/unit_E) | ef_nameplate (tCO₂/MWh) | × 52 |
| `mc_model` (€/unit_E) | mc_nameplate (€/MWh) | × 52 |

---

## Baseline technology block (Italian 2019, normalised)

Nameplate marginal cost in €/MWh divided by 52 to get model units (€/unit_E).
Emission factor in tCO₂/MWh divided by 52.

| Tech | mc nameplate (€/MWh) | mc model (€/unit_E) | ef nameplate (tCO₂/MWh) | ef model (tCO₂/unit_E) |
|---|---|---|---|---|
| coal-1 | 40 | 0.769 | 0.90 | 0.01731 |
| coal-2 | 42 | 0.808 | 0.90 | 0.01731 |
| gas-1 | 55 | 1.058 | 0.37 | 0.00712 |
| gas-2 | 57 | 1.096 | 0.37 | 0.00712 |
| nuclear | 15 | 0.288 | 0.00 | 0.00000 |
| solar | 5 | 0.096 | 0.00 | 0.00000 |
| wind | 4 | 0.077 | 0.00 | 0.00000 |
| biomass | 30 | 0.577 | 0.05 | 0.00096 |

Note: nuclear is retained in the stack for now. Italy has no nuclear generation; this is flagged for Tier-1 item 4 ("retire nuclear, add coal weight, add hydro") and will be revised before data-bridge construction.

---

## What is invariant, what scales with α

With α = 7,500 (wage-anchored), the following rescale linearly:

- `w`, `p_F` (initial), `p_E` (initial), `cpi` (initial)
- All deposit, loan, capital, and bond stocks at t=0
- `tech_mc` (but nonlinearly — we apply the ÷52 conversion, not ×α)

The following are **unchanged**:

- `alpha1`, `alpha2` (propensities to consume out of income / wealth)
- `taste_F`, `ces_sigma` (CES structure)
- `markup` (price rule)
- `phi_pi`, `phi_u`, `u_star` (Phillips curve)
- `r_D`, `r_L`, `r_GB`, `r_CB` (interest rates, already Italian 2019–21)
- `tax_rate` (fiscal)
- `depreciation_rate_F/E`, `amort_rate_F/E`
- `i_rate_F`, `working_capital_coverage`, `bank_capital_target`
- `cap_factor_by_tech` (per-tech capacity factors, already literature-calibrated)
- `tech_ef` (converted ÷52 to unit_E, but structurally unchanged)
- `theta_div`, `churn_alpha`
- `productivity_F`, `productivity_E`

**Consequence:** every law of motion is homogeneous of degree 1 in €, so the rescaling is exact — no behavioural re-tuning is required. This follows the argument in Godley & Lavoie (2007, §3.2) that SFC models are scale-invariant.

---

## Shock semantics

Under the new convention, shocks have a direct economic interpretation:

- **TTF gas shock (2022):** gas nameplate marginal cost rises 55 → 250 €/MWh. Apply to model by setting `E.mc[gas_plants] = 250/52 = 4.81` during the shock window. Merit-order dispatch responds directly; `p_E` spikes toward the new marginal plant's cost.
- **EU ETS ratchet:** set `carbon_tax` directly in €/tCO₂. Observed EU ETS averages: ~25 (2019), ~25 (2020), ~53 (2021), ~85 (2022), ~83 (2023).
- **Wage shock / cost-push inflation:** modify `w` directly (e.g. `w = 7,500 × 1.08` for an 8% nominal wage bump).
- **Demand shock:** perturb `alpha1` or `alpha2` directly — unit-invariant.

---

## Tolerance thresholds

SFC closure is tested by residual absolute values. Under the old dimensionless units residuals of 1e-10 were the target. Under the new convention absolute residuals will scale by ~α = 7,500, so the relevant test is **relative residual** `max_abs / GDP_proxy`. The test suite should be updated to use a relative tolerance of 1e-11 (which corresponds to 1e-7 absolute at the new scale).

---

## Open items (not in this convention, flagged for the next pass)

1. **Tech-stack realism** (Tier-1 item 4): retire nuclear; consider adding a second coal unit, a hydro plant (~17% of Italian 2019 generation), and tilting capacity shares toward 2019 ENTSO-E Italy. Requires matching `cap_factor_by_tech` keys and allowed `tech_labels`.
2. **Decision on whether `DEP_H_per_hh = 7,500` is the right stock anchor.** Wage-anchored flow calibration may leave balance sheets slimmer than Italian HFCS (median €11k). That's an SFC-model-finding worth documenting, but we may want to relax `w = 7,500` if the deposit realism matters more for the fragility story.
3. **Relative-tolerance refactor of tests/test_sfc_closure.py.**

---

## Citation anchors (2019 Italian baseline)

| Quantity | Value | Source |
|---|---|---|
| Average gross quarterly wage | ~€7,500 | Istat, Retribuzioni contrattuali lorde 2019 |
| PUN electricity wholesale avg | ~52 €/MWh | GME MGP Annual Report 2019 |
| Household deposits, median | ~€11,000 | HFCS 2019 wave, Bank of Italy |
| EU ETS avg allowance price | ~€25/tCO₂ | EEX EUA front-month 2019 mean |
| 10Y BTP secondary-market yield | ~2.0% annual (0.5%/q) | MTS quarterly mean |
| ECB MFI household overnight rate | ~0.04% annual | ECB SDW, MIR.M.IT |
| ECB MFI NFC new-loan rate | ~1.6% annual | ECB SDW, MIR.M.IT |
