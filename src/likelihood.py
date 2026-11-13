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
from scipy.special import gammaln

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
            return _infeasible_penalty(s_prev)
        s_t = p["kappa"] * gamma_bar + a_t * gamma_prev + p["b"] * s_prev
        eigvals = np.linalg.eigvalsh(0.5 * (s_t + s_t.T))
        if eigvals.min() <= 1e-10:
            # S_t left the positive-definite cone: the plain parameter-free
            # 1e10 constant used previously gives SLSQP zero gradient and
            # traps the optimizer at any infeasible starting point (observed
            # for Log-BEKK/Log-HE-BEKK, whose logistic A_t need not be PSD).
            # A penalty growing with the eigenvalue violation restores a
            # usable gradient back toward the feasible region.
            return 1e8 + 1e8 * float(-eigvals.min())
        sign, logdet = np.linalg.slogdet(s_t)
        try:
            trace_term = np.trace(np.linalg.solve(s_t, gamma_series[t]))
        except np.linalg.LinAlgError:
            return _infeasible_penalty(s_prev)
        total += logdet + trace_term
        s_prev = s_t
        gamma_prev = gamma_series[t]
    if not np.isfinite(total):
        return 1e10
    return total  # negative log-likelihood up to the positive scale nu/2


def _infeasible_penalty(s_prev: np.ndarray) -> float:
    """Fallback penalty when ``compute_A``/``solve`` itself raises, scaled by
    how far ``s_prev`` already is from well-conditioned (keeps a gradient)."""
    try:
        cond = float(np.linalg.cond(s_prev))
    except np.linalg.LinAlgError:
        cond = 1e12
    return 1e8 + min(cond, 1e8)


def delta_t_sum(gamma_series: np.ndarray, nu: float) -> float:
    """sum_{t=2}^{T} delta_t (eq. 4.2, Section 4), the additive constant that
    does not involve the model parameters:

        delta_t = (n*nu/2) ln(nu/2) - (n(n-1)/4) ln(pi)
                  + ((nu-n-1)/2) ln|Gamma_t| - sum_{i=1}^{n} ln Gamma((nu+1-i)/2)

    Included only to report a log-likelihood on the same scale as the
    paper's Table 3; it is identical across the six models for fixed nu and
    therefore irrelevant to AIC/BIC differences and MCS rankings."""
    t_len, n, _ = gamma_series.shape
    const = (n * nu / 2.0) * np.log(nu / 2.0) - (n * (n - 1) / 4.0) * np.log(np.pi)
    const -= gammaln((nu + 1 - np.arange(1, n + 1)) / 2.0).sum()
    total = 0.0
    for t in range(1, t_len):
        sign, logdet = np.linalg.slogdet(gamma_series[t])
        total += const + ((nu - n - 1) / 2.0) * logdet
    return total


def full_log_likelihood(theta: np.ndarray, model: str, gamma_series: np.ndarray,
                         s0: np.ndarray, gamma_bar: np.ndarray, nu: float) -> float:
    """Full quasi log-likelihood (4.2) including delta_t and the nu/2 scale,
    evaluated at the two-step QML estimate theta_hat."""
    neg_ll = neg_log_likelihood(theta, model, gamma_series, s0, gamma_bar)
    return delta_t_sum(gamma_series, nu) - (nu / 2.0) * neg_ll


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
