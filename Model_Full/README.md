# model_full — Full AB-SFC framework for the SIE 2026 paper

This directory contains the **framework-exercising** simulation that accompanies
the paper *Agent-Based Stock-Flow Consistent Framework for Energy-Policy
Design*. The code is a stylised implementation — calibration is by
order-of-magnitude plausibility rather than by SMM on real data. The point
of this codebase is not to reproduce Italian macro series; it is to
demonstrate that the framework described in the paper

1. actually runs,
2. satisfies stock-flow consistency period by period, and
3. produces qualitatively interpretable responses to the policy regimes
   defined in §5.

A separate directory, `../model_mvp/`, contains the data-calibrated MVP
used for the quantitative experiments.

## Agent classes and counts

| Symbol | Sector                  | Count  |
|--------|-------------------------|--------|
| H      | Households              | 2,000  |
| F      | Non-energy firms        | 250    |
| E      | Energy producers        | 8 (2 coal, 2 gas, 1 nuclear, 1 solar, 1 wind, 1 biomass) |
| B      | Commercial banks        | 20     |
| G      | Government              | 1      |
| CB     | Central bank            | 1      |

Time step is quarterly, following Caiani (2016).

## Phase structure

The model is built in numbered phases, each of which extends the previous
one. The SFC residual check (see `accounting.py`) is run after every
period of every phase; a phase is accepted only if the residual stays
below `1e-10 * |GDP|` for the entire simulation window.

| Phase | Adds                                                                    |
|-------|-------------------------------------------------------------------------|
| 0     | Pure accounting core: fixed flows, no behaviour. Proves closure.         |
| 1     | Household consumption rule, firm pricing and production, labour market.  |
| 2     | Bank credit block, energy merit-order dispatch, Wright's-law learning.   |
| 3     | Policy regimes (carbon tax, ETS, ETS+floor, green subsidy, nuclear).     |
| 4     | Monte Carlo protocol, bootstrap CIs, Wilcoxon tests, sensitivity.        |

This README covers **phase 0 only**. Later phases will be added after
the closure test passes and Sal has signed off.

## SFC invariant

At every period `t` the following must hold up to floating-point
tolerance:

1. Cross-sector cancellation of every financial stock:
   `DEP_H + DEP_F + DEP_E = DEP_liab` (bank side),
   `L_F + L_E = L_asset` (bank side),
   `GB_B + GB_CB = GB_out` (government side).
2. Net-wealth identity: `sum over sectors of NW = K_F + K_E`
   (total real capital).
3. Redundant-equation check: any one of the six stock laws of motion
   in v2.1 §3.10 is computed as the sum of the TFM flows and compared
   against the direct stock update; the difference must be
   `< 1e-10 * |GDP|`.

If any of these fails, `accounting.residual_check()` raises
`SFCClosureError`.

## Policy regimes (phases 3+)

Six regimes, all sharing the same seed sequence (common random numbers):

1. Baseline (no carbon price, no state-backed nuclear).
2. Carbon Tax at constant τ.
3. ETS with linearly declining cap.
4. ETS + price floor.
5. ETS + revenue recycling into renewable investment subsidies.
6. ETS + state-financed nuclear capacity (capex funded via green bonds,
   Eq. 32).

Regime 6 is the one Sal added during scoping: it isolates the effect of
state-backed low-marginal-cost dispatchable generation on merit-order
prices, emissions, and public-debt dynamics, on top of an ETS signal.

## KPI list (phases 3+)

Every KPI is reported as median across 100 replications with 95 %
bootstrap CI and a Wilcoxon signed-rank test against Baseline on
paired replication differences.

- **RESI(t)** and its four diagnostics: pre-shock level R₀, minimum
  R_min, recovery rate k_rec, new steady state R_∞. *(focal KPI)*
- Real GDP deviation from baseline
- Unemployment rate
- CPI inflation
- Investment-to-GDP ratio
- Green capacity share K_{E,renew} / K_E
- Cumulative CO₂ emissions
- Wholesale electricity price: mean and volatility
- Firm default rate
- Bank capital ratio
- Government debt-to-GDP
- Quintile-specific real consumption (inequality channel)

## File layout

```
model_full/
├── README.md                    # this file
├── accounting.py                # TFM, BSM, laws of motion, residual check
├── agents.py                    # sector dataclasses (state only, no behaviour)
├── init_stocks.py               # consistent initial-condition generator
├── dynamics.py                  # period step, phase-gated
├── run_phase0.py                # driver for phase-0 closure test
└── tests/
    └── test_sfc_closure.py      # pytest closure + stationarity test
```

## Running phase 0

```bash
cd "model_full"
python run_phase0.py            # prints residual statistics, saves residual_phase0.png
pytest tests/test_sfc_closure.py
```

The script exits with status 0 only if `max|residual| < 1e-10 * |GDP|`
over all 500 quarters.
