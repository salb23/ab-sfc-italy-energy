"""
energy_market.py — Phase-2a merit-order dispatch.

Pure function: takes aggregate real energy demand and plant
characteristics, returns per-plant output, market-clearing price
(= marginal plant's MC), and total emissions.

Merit-order logic (stylised wholesale market):
  1. Plants are sorted by marginal cost, cheapest first.
  2. Each plant is dispatched up to its capacity until cumulative
     output meets demand.
  3. The clearing price is the MC of the marginal (last-called) plant.
  4. Inframarginal plants receive the clearing price as revenue;
     the wedge (clearing - MC) is producer surplus. Phase-2a does
     not yet track that surplus separately on the SFC books — the
     period's cash flows use the endogenous clearing price
     uniformly, so closure is preserved.

All quantities are in model units: MC is rescaled from nameplate
€/MWh via `mc_scale` at the EnergyArray build step (init_stocks),
capacity is in output-units per quarter.
"""

from __future__ import annotations

import numpy as np


def dispatch(demand: float,
             mc: np.ndarray,
             capacity: np.ndarray,
             emission_factor: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """
    Merit-order dispatch.

    Parameters
    ----------
    demand : float
        Real energy demand for the period (units of output).
    mc : np.ndarray, shape (N_E,)
        Marginal cost of each plant (in model price units).
    capacity : np.ndarray, shape (N_E,)
        Maximum output per plant per quarter.
    emission_factor : np.ndarray, shape (N_E,)
        tCO2 per unit of output.

    Returns
    -------
    y : np.ndarray, shape (N_E,)
        Realised output per plant (original input order).
    clearing_price : float
        Marginal plant's MC (if demand=0, defaults to cheapest plant's MC
        so the price series is well-defined even at the boundary).
    emissions : float
        Σ_k y_k · ef_k over active plants.
    rationed : float
        Unserved demand (0 unless demand > Σ capacity). Zero in
        well-calibrated runs; non-zero values are a calibration signal,
        not an SFC violation — cash flows still close.
    """
    N = len(mc)
    y = np.zeros(N, dtype=float)
    order = np.argsort(mc)  # indices cheapest → most expensive

    remaining = max(0.0, float(demand))
    marginal_idx = int(order[0])
    for k in order:
        if remaining <= 0.0:
            break
        take = float(min(capacity[k], remaining))
        y[k] = take
        remaining -= take
        marginal_idx = int(k)

    clearing_price = float(mc[marginal_idx])
    emissions = float((y * emission_factor).sum())
    rationed = float(max(0.0, remaining))
    return y, clearing_price, emissions, rationed


def per_plant_emissions(y: np.ndarray, emission_factor: np.ndarray) -> np.ndarray:
    """Convenience: per-plant emissions vector for diagnostics."""
    return y * emission_factor
