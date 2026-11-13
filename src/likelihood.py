"""Quasi maximum likelihood objective for model (2.3).

Equation (4.2): sum_{t=2}^{T} [ delta_t - nu/2 (ln|S_t| + tr(S_t^{-1} Gamma_t)) ].
The parameter estimates do not depend on nu or on delta_t (Section 4, after
eq. 4.2), so the second-step estimation maximizes

    sum_{t=2}^{T} [ -(ln|S_t| + tr(S_t^{-1} Gamma_t)) ]

which is proportional to (4.2) up to the additive, parameter-free delta_t
terms and the positive scale nu/2.

Custom implementation (paper Section 4, equation 4.2): this quasi
log-likelihood is specific to the paper's CAW-BEKK family and has no
off-the-shelf implementation in standard covariance-model libraries (`arch`
targets univariate/DCC GARCH, not Wishart-based realized-covariance BEKK).
"""
from __future__ import annotations

import numpy as np

from src.models import compute_A, unpack_params


def neg_log_likelihood(theta: np.ndarray, model: str, gamma_series: np.ndarray,
                        s0: np.ndarray, gamma_bar: np.ndarray) -> float:
    """Negative of the (nu-free) quasi log-likelihood (4.2), summed for
    t = 2, ..., T (``gamma_series`` indexed 0..T-1 in code, i.e. t=1..T)."""
    p = unpack_params(model, theta)
    t_len, n, _ = gamma_series.shape
    s_prev = s0
    gamma_prev = gamma_series[0]
    total = 0.0
    for t in range(1, t_len):
        try:
            a_t = compute_A(model, theta, gamma_prev, n)
        except np.linalg.LinAlgError:
            return 1e10
        s_t = p["kappa"] * gamma_bar + a_t * gamma_prev + p["b"] * s_prev
        sign, logdet = np.linalg.slogdet(s_t)
        if sign <= 0:
            return 1e10
        try:
            trace_term = np.trace(np.linalg.solve(s_t, gamma_series[t]))
        except np.linalg.LinAlgError:
            return 1e10
        total += logdet + trace_term
        s_prev = s_t
        gamma_prev = gamma_series[t]
    if not np.isfinite(total):
        return 1e10
    return total  # negative log-likelihood up to the positive scale nu/2


def score_contributions(theta: np.ndarray, model: str, gamma_series: np.ndarray,
                         s0: np.ndarray, gamma_bar: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-parameter numerical gradient of the negative log-likelihood,
    used only to build the robust sandwich covariance (White, 1982)."""
    grads = np.zeros_like(theta)
    base = neg_log_likelihood(theta, model, gamma_series, s0, gamma_bar)
    for i in range(len(theta)):
        bump = np.array(theta, dtype=float)
        bump[i] += eps
        f_plus = neg_log_likelihood(bump, model, gamma_series, s0, gamma_bar)
        grads[i] = (f_plus - base) / eps
    return grads
