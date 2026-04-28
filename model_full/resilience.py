"""
resilience.py — Composite Resilience Index (RESI) module.

Stage 7 of the gap-closure plan. Implements the four-dimensional
resilience composite described in paper §5.4 and §6, based on the
engineering-resilience framework of Hosseini, Barker & Ramirez-Marquez
(2016). Four sub-indicators:

  * Absorptive capacity (A): the system's ability to withstand a
    shock without deviating from baseline. Measured as one minus the
    normalised peak KPI deviation during the shock window.

  * Adaptive capacity (Ad): the speed at which the system begins to
    arrest further deterioration during the shock. Measured as one
    minus the time-to-peak deviation relative to shock duration.

  * Restorative capacity (R): the extent to which the system returns
    to baseline after the shock ends. Measured as one minus the
    post-shock deviation at a specified recovery horizon.

  * Resourcefulness (Rs): the magnitude of resources mobilised to
    counter the shock (policy revenue, subsidies, new investment,
    DER capex, ...), normalised by pre-shock nominal output.

Each sub-indicator is in [0, 1]; 1 indicates perfect resilience on
that dimension. The composite RESI is the weighted mean of the four
sub-scores (equal weights by default; the paper's policy-experiment
design may override this with Hosseini-style literature-justified
weights).

The module is pure: no side effects, no simulation state. Users
(scenario drivers, post-run analytics) supply:

  - a KPI time series (np.ndarray)
  - the pre-shock baseline value (scalar)
  - the shock window (t_start, t_end)
  - optionally: resources mobilised and baseline GDP for Rs

References
----------
- Hosseini, S., Barker, K., & Ramirez-Marquez, J. E. (2016). A review
  of definitions and measures of system resilience. Reliability
  Engineering & System Safety, 145, 47-61.
- Schill, W. P., Mier, M., Göke, L., Schmidt, F., & Kemfert, C.
  (2025). Design of the electricity market and its role in the energy
  policy trilemma (paper §2.6 reference on resilience in energy
  systems).
- Bruneau, M., et al. (2003). A framework to quantitatively assess and
  enhance the seismic resilience of communities. Earthquake Spectra,
  19(4), 733-752 (origin of the 4-dimensional A/Ad/R/Rs decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Sub-indicator calculators
# --------------------------------------------------------------------------

def compute_absorptive(kpi_trajectory: np.ndarray,
                       baseline_value: float,
                       shock_window: Tuple[int, int],
                       orientation: str = "deviation") -> float:
    """
    Absorptive capacity: 1 − peak_normalised_deviation during shock.

    `orientation` specifies whether higher KPI values are good or bad:
      - "deviation" (default): absolute distance from baseline is
        penalised, regardless of sign. Suitable for cpi, p_E, u.
      - "increase_bad": positive deviations only are penalised (e.g.
        emissions rising bad, falling fine). Zero for negative Δ.
      - "decrease_bad": negative deviations penalised (e.g. Y_F
        falling is bad, rising is fine).

    Returns a score in [0, 1]; 1 = no deviation, 0 = deviation ≥ 100%
    of baseline magnitude.
    """
    traj = np.asarray(kpi_trajectory, dtype=float)
    t_start, t_end = shock_window
    if t_end <= t_start or t_start < 0 or t_end > len(traj):
        return 1.0
    window = traj[t_start:t_end]
    denom = max(abs(float(baseline_value)), 1e-12)
    deltas = window - float(baseline_value)
    if orientation == "deviation":
        rel = np.abs(deltas) / denom
    elif orientation == "increase_bad":
        rel = np.maximum(0.0, deltas) / denom
    elif orientation == "decrease_bad":
        rel = np.maximum(0.0, -deltas) / denom
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")
    peak_rel = float(rel.max()) if rel.size > 0 else 0.0
    return max(0.0, 1.0 - min(1.0, peak_rel))


def compute_adaptive(kpi_trajectory: np.ndarray,
                     baseline_value: float,
                     shock_window: Tuple[int, int],
                     orientation: str = "deviation") -> float:
    """
    Adaptive capacity: 1 − (time-to-peak / shock-duration).

    If the peak deviation occurs early in the shock (system quickly
    finds a new equilibrium), Ad → 1. If the peak is at the very end
    (still deteriorating throughout), Ad → 0.
    """
    traj = np.asarray(kpi_trajectory, dtype=float)
    t_start, t_end = shock_window
    if t_end <= t_start + 1:
        return 1.0
    window = traj[t_start:t_end]
    duration = float(t_end - t_start)
    deltas = window - float(baseline_value)
    if orientation == "deviation":
        rel = np.abs(deltas)
    elif orientation == "increase_bad":
        rel = np.maximum(0.0, deltas)
    elif orientation == "decrease_bad":
        rel = np.maximum(0.0, -deltas)
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")
    if rel.size == 0 or rel.max() <= 1e-12:
        return 1.0
    t_peak = float(np.argmax(rel))
    return max(0.0, 1.0 - t_peak / max(duration - 1.0, 1.0))


def compute_restorative(kpi_trajectory: np.ndarray,
                        baseline_value: float,
                        shock_window: Tuple[int, int],
                        recovery_window: int = 8,
                        orientation: str = "deviation") -> float:
    """
    Restorative capacity: 1 − |deviation at t_end + recovery_window|.

    Measures how well the system returns to baseline at a specified
    horizon after the shock ends. `recovery_window` is in the same
    time units as the trajectory (quarters).
    """
    traj = np.asarray(kpi_trajectory, dtype=float)
    t_end = shock_window[1]
    t_probe = min(t_end + recovery_window, len(traj) - 1)
    if t_probe < 0 or t_probe >= len(traj):
        return 1.0
    denom = max(abs(float(baseline_value)), 1e-12)
    delta = float(traj[t_probe]) - float(baseline_value)
    if orientation == "deviation":
        rel = abs(delta) / denom
    elif orientation == "increase_bad":
        rel = max(0.0, delta) / denom
    elif orientation == "decrease_bad":
        rel = max(0.0, -delta) / denom
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")
    return max(0.0, 1.0 - min(1.0, rel))


def compute_resourcefulness(resources_mobilised: float,
                             baseline_gdp: float,
                             saturation: float = 0.25) -> float:
    """
    Resourcefulness: total resources mobilised during the shock,
    normalised by pre-shock quarterly output and saturated at
    `saturation`. Interpretable as "fraction of GDP mobilised for
    response, capped at the `saturation` fraction which is treated as
    the ceiling for a fully-mobilised economy".

    With saturation = 0.25 and resources = 0.05 × GDP, Rs = 0.20.
    With resources ≥ 0.25 × GDP, Rs = 1.0 (fully resourceful).
    """
    if baseline_gdp <= 0.0 or saturation <= 0.0:
        return 0.0
    ratio = float(resources_mobilised) / (float(baseline_gdp) * float(saturation))
    return max(0.0, min(1.0, ratio))


# --------------------------------------------------------------------------
# Composite RESI
# --------------------------------------------------------------------------

def compute_resi(kpi_trajectory: np.ndarray,
                 baseline_value: float,
                 shock_window: Tuple[int, int],
                 resources_mobilised: float = 0.0,
                 baseline_gdp: float = 1.0,
                 recovery_window: int = 8,
                 orientation: str = "deviation",
                 weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
                 saturation: float = 0.25
                 ) -> Dict[str, float]:
    """
    Full RESI calculation returning the composite and its four
    sub-indicators.

    Inputs
    ------
    kpi_trajectory : np.ndarray
        Time series of the key performance indicator (e.g. u, cpi,
        emissions/q, Y_F).
    baseline_value : float
        Pre-shock baseline of the KPI (typically the mean over a
        window before t_start).
    shock_window : (t_start, t_end)
        Inclusive-exclusive indices of the shock period.
    resources_mobilised : float
        Total nominal resources deployed during the shock (policy
        revenue, subsidies, new investment, DER capex …).
    baseline_gdp : float
        Pre-shock quarterly nominal output, for Rs normalisation.
    recovery_window : int
        Quarters after shock end at which R is probed.
    orientation : str
        See `compute_absorptive`.
    weights : 4-tuple
        Weights for (A, Ad, R, Rs). Default equal weights.
    saturation : float
        Rs saturation point (fraction of GDP).

    Returns
    -------
    dict with keys 'A', 'Ad', 'R', 'Rs', and 'RESI' (composite).
    """
    A  = compute_absorptive(kpi_trajectory, baseline_value, shock_window,
                            orientation=orientation)
    Ad = compute_adaptive(kpi_trajectory, baseline_value, shock_window,
                          orientation=orientation)
    R  = compute_restorative(kpi_trajectory, baseline_value, shock_window,
                              recovery_window=recovery_window,
                              orientation=orientation)
    Rs = compute_resourcefulness(resources_mobilised, baseline_gdp,
                                  saturation=saturation)
    w_sum = sum(weights) if sum(weights) > 0 else 1.0
    wA, wAd, wR, wRs = [w / w_sum for w in weights]
    RESI = wA * A + wAd * Ad + wR * R + wRs * Rs
    return {"A": A, "Ad": Ad, "R": R, "Rs": Rs, "RESI": float(RESI)}


# --------------------------------------------------------------------------
# Trajectory recorder
# --------------------------------------------------------------------------

@dataclass
class ResilienceRecorder:
    """
    Lightweight trajectory buffer for scenario drivers. Append KPIs
    each period; at end-of-run compute RESI against a chosen shock
    window and baseline.

    Example
    -------
    >>> rec = ResilienceRecorder()
    >>> for t in range(1, 101):
    ...     step(eco, phase=4, tol=1e-10)
    ...     rec.record(t, u=eco.u, cpi=eco.cpi,
    ...                emissions=eco.emissions_stock, w=eco.w)
    >>> baseline = rec.baseline_mean("cpi", window=(10, 30))
    >>> resi = compute_resi(
    ...     kpi_trajectory=rec.series("cpi"),
    ...     baseline_value=baseline,
    ...     shock_window=(50, 58),
    ... )
    """
    t_log:  List[int] = field(default_factory=list)
    kpis:   Dict[str, List[float]] = field(default_factory=dict)

    def record(self, t: int, **kwargs: float) -> None:
        self.t_log.append(int(t))
        for k, v in kwargs.items():
            self.kpis.setdefault(k, []).append(float(v))

    def series(self, name: str) -> np.ndarray:
        return np.asarray(self.kpis.get(name, []), dtype=float)

    def baseline_mean(self, name: str, window: Tuple[int, int]) -> float:
        """Mean of KPI over `window` (pre-shock baseline estimate)."""
        s = self.series(name)
        t0, t1 = window
        t0 = max(0, t0); t1 = min(len(s), t1)
        if t1 <= t0:
            return float("nan")
        return float(s[t0:t1].mean())

    def n(self) -> int:
        return len(self.t_log)
