"""Monte Carlo summary statistics: bias, RMSE, and normality (Section 4.1,
Table 1).

Custom implementation (paper Section 4.1): bias and RMSE are plain estimator
summaries; the Jarque-Bera normality test itself is taken from
`scipy.stats.jarque_bera`, used exactly as a library primitive.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import jarque_bera


def bias(estimates: np.ndarray, true_value: float) -> float:
    """Monte Carlo bias: mean(estimate) - true_value."""
    return float(np.mean(estimates) - true_value)


def rmse(estimates: np.ndarray, true_value: float) -> float:
    """Monte Carlo root mean squared error around the true value."""
    return float(np.sqrt(np.mean((estimates - true_value) ** 2)))


def jarque_bera_pvalue(estimates: np.ndarray) -> float:
    """p-value of the Jarque-Bera normality test on the replicated estimator
    values (Section 4.1: normality assessed at 1% significance)."""
    if len(estimates) < 8 or np.allclose(estimates, estimates[0]):
        return 0.0
    stat, p_value = jarque_bera(estimates)
    return float(p_value)


def smallest_normal_T(estimates_by_T: dict, significance: float) -> int | None:
    """Smallest sample size T, among the keys of ``estimates_by_T`` (sorted
    ascending), for which the Jarque-Bera test does not reject normality at
    ``significance`` (Section 4.1, "last column" of Table 1). Returns None if
    normality is never achieved for the provided T values."""
    for t in sorted(estimates_by_T):
        if jarque_bera_pvalue(estimates_by_T[t]) >= significance:
            return t
    return None


def summarize_replications(estimates: np.ndarray, true_value: float) -> dict:
    """Bias, RMSE and Jarque-Bera p-value for one (model, parameter, T) cell."""
    return {
        "bias": bias(estimates, true_value),
        "rmse": rmse(estimates, true_value),
        "jb_pvalue": jarque_bera_pvalue(estimates),
        "n_replications": int(len(estimates)),
    }
