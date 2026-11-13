"""Monte Carlo data-generating process for the realized covariance models
(Section 4.1 "Monte Carlo Evidence").

The DGP is model (2.3):

    S_t = kappa * Gamma_bar + A_t (Hadamard) Gamma_{t-1} + b * S_{t-1}

with Gamma_t ~ Wishart(S_t / nu, nu), nu = 3n = 90 (Section 4.1, footnote 5).
The conditional expectation of the Wishart draw is S_t (E[Gamma_t | F_{t-1}]
= S_t for Gamma_t ~ W(S_t/nu, nu)), matching assumption 2 of Section 4,
equation (4.1).

Custom implementation (paper Section 4.1): the recursive construction of the
covariance path is specific to this paper; `scipy.stats.wishart` supplies the
underlying matrix-variate random-number generator.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wishart

from src.models import compute_A, unpack_params


def simulate_series(model: str, theta: np.ndarray, n: int, nu: int, t_len: int,
                     s0: np.ndarray, gamma_bar: np.ndarray, burn_in: int = 0,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``t_len`` realized covariance matrices Gamma_1, ..., Gamma_T
    (and the implied conditional path S_1, ..., S_T) from model (2.3).

    Returns (gamma_series, s_series), each of shape (t_len, n, n).
    ``burn_in`` extra periods are simulated and discarded before recording,
    so that the recorded path does not depend on the arbitrary choice of
    ``s0`` (Section 4.1 draws S_0 "from the realized covariance series of the
    application in Section 5"; the burn-in removes residual sensitivity to
    that particular starting value).
    """
    if rng is None:
        rng = np.random.default_rng()
    p = unpack_params(model, theta)

    total = burn_in + t_len
    gamma_full = np.empty((total, n, n))
    s_full = np.empty((total, n, n))

    s_prev = s0
    gamma_prev = s0  # Gamma_0 initialized at S_0, consistent with E[Gamma_t|F_{t-1}] = S_t
    for t in range(total):
        a_t = compute_A(model, theta, gamma_prev, n)
        s_t = p["kappa"] * gamma_bar + a_t * gamma_prev + p["b"] * s_prev
        s_t = 0.5 * (s_t + s_t.T)  # enforce exact symmetry against floating-point drift
        gamma_t = wishart.rvs(df=nu, scale=s_t / nu, random_state=rng)
        gamma_full[t] = gamma_t
        s_full[t] = s_t
        s_prev = s_t
        gamma_prev = gamma_t

    return gamma_full[burn_in:], s_full[burn_in:]


def realized_covariance_s0(gamma_series: np.ndarray) -> np.ndarray:
    """A starting value S_0 extracted from an observed realized covariance
    series, used by the Monte Carlo DGP (Section 4.1: "the initial value of
    S_0 was extracted from the realized covariance series of the application
    in Section 5"). The sample mean of the series is a natural, stationary
    choice of initial conditional covariance."""
    return gamma_series.mean(axis=0)
