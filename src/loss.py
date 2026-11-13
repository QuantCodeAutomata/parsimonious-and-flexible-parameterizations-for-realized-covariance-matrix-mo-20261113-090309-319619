"""Model-selection and forecast-loss functions for Section 5 ("Empirical
Application").

Custom implementation (paper Section 5): the Quasi-Likelihood (QL) and Global
Minimum-Variance (GMV) loss functions, and the EWMA/vecHAR benchmark
forecasts, are defined directly in the paper; no portfolio or covariance
library reproduces exactly this QL/GMV pair, so they are implemented from the
stated formulas rather than drawn from a generic risk-model package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import VECHAR_DAILY_LAG, VECHAR_MONTHLY_LAG, VECHAR_WEEKLY_LAG


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion: -2*logL + 2*k (Section 5, Table 3)."""
    return -2.0 * log_likelihood + 2.0 * n_params


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian information criterion: -2*logL + k*ln(T) (Section 5, Table 3)."""
    return -2.0 * log_likelihood + n_params * np.log(n_obs)


def quasi_likelihood_loss(s_hat: np.ndarray, gamma: np.ndarray) -> float:
    """Quasi-Likelihood loss for one period: ln|S_hat| + tr(S_hat^{-1} Gamma)
    (Section 5: QL loss used for in-sample and out-of-sample comparison)."""
    try:
        np.linalg.cholesky(s_hat)  # positive-definiteness check (sign of slogdet
    except np.linalg.LinAlgError:  # alone misses negative-definite matrices of even order)
        return np.inf
    _, logdet = np.linalg.slogdet(s_hat)
    return float(logdet + np.trace(np.linalg.solve(s_hat, gamma)))


def global_minimum_variance_loss(s_hat: np.ndarray, gamma: np.ndarray) -> float:
    """GMV loss for one period: w_hat' Gamma w_hat, where the minimum-variance
    weights w_hat = sqrt(n) S_hat^{-1} j_n / (j_n' S_hat^{-1} j_n) are built
    from the *forecast* S_hat and evaluated against the realized Gamma
    (Section 5: GMV loss, "w_hat_t' S_hat_t w_hat_t" formula in the
    specification, evaluated here on the realized covariance to measure
    achieved portfolio risk out of sample)."""
    n = s_hat.shape[0]
    j_n = np.ones(n)
    s_inv_j = np.linalg.solve(s_hat, j_n)
    denom = j_n @ s_inv_j
    w_hat = np.sqrt(n) * s_inv_j / denom
    return float(w_hat @ gamma @ w_hat)


def average_loss(loss_fn, s_hat_series: np.ndarray, gamma_series: np.ndarray) -> float:
    """Average of ``loss_fn`` over all periods: (1/T*) * sum_t loss_fn(...)."""
    losses = [loss_fn(s_hat_series[t], gamma_series[t]) for t in range(len(gamma_series))]
    return float(np.mean(losses))


def loss_series(loss_fn, s_hat_series: np.ndarray, gamma_series: np.ndarray) -> np.ndarray:
    """Per-period loss series, used as the input to the Model Confidence Set."""
    return np.array([loss_fn(s_hat_series[t], gamma_series[t]) for t in range(len(gamma_series))])


def ewma_forecast(gamma_series: np.ndarray, s0: np.ndarray, decay: float) -> np.ndarray:
    """EWMA benchmark: S_t = (1-decay)*Gamma_{t-1} + decay*S_{t-1}
    (Section 5: "decay parameter equal to 0.96")."""
    t_len = gamma_series.shape[0]
    s_path = np.empty_like(gamma_series)
    s_prev = s0
    for t in range(t_len):
        gamma_prev = gamma_series[t - 1] if t > 0 else s0
        s_t = (1 - decay) * gamma_prev + decay * s_prev
        s_path[t] = s_t
        s_prev = s_t
    return s_path


def _vech(mat: np.ndarray) -> np.ndarray:
    """Half-vectorization vech(mat): stacked lower-triangular entries."""
    idx = np.tril_indices(mat.shape[0])
    return mat[idx]


def _unvech(vec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`_vech`: rebuild the symmetric n x n matrix."""
    mat = np.zeros((n, n))
    idx = np.tril_indices(n)
    mat[idx] = vec
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def fit_vechar(gamma_series: np.ndarray) -> np.ndarray:
    """OLS coefficients of the vecHAR benchmark (Corsi, 2009 HAR structure,
    applied to vech(Gamma_t) as in Section 5): regress vech(Gamma_t) on an
    intercept and the daily, weekly and monthly rolling means of
    vech(Gamma_{t-1}). Returns the (4, n*) coefficient matrix [intercept,
    daily, weekly, monthly]."""
    t_len, n, _ = gamma_series.shape
    vech_series = np.array([_vech(gamma_series[t]) for t in range(t_len)])
    daily = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_DAILY_LAG).mean().to_numpy()
    weekly = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_WEEKLY_LAG).mean().to_numpy()
    monthly = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_MONTHLY_LAG).mean().to_numpy()
    start = VECHAR_MONTHLY_LAG  # first index with all three rolling means available
    y = vech_series[start:]
    # Separate OLS per vech coordinate (each coordinate uses the same HAR
    # regressors, built from that same coordinate's own daily/weekly/monthly
    # rolling averages of vech(Gamma_{t-1}), following Corsi (2009)).
    n_star = vech_series.shape[1]
    coefs = np.zeros((4, n_star))
    for k in range(n_star):
        xk = np.column_stack([np.ones(t_len - start), daily[start:, k], weekly[start:, k], monthly[start:, k]])
        beta, *_ = np.linalg.lstsq(xk, y[:, k], rcond=None)
        coefs[:, k] = beta
    return coefs


def vechar_forecast(gamma_series: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """One-step-ahead vecHAR forecasts S_hat_t for every t, using the daily,
    weekly and monthly rolling means of vech(Gamma_{t-1}) with the fitted
    ``coefs`` (Section 5 benchmark; Corsi, 2009)."""
    t_len, n, _ = gamma_series.shape
    vech_series = np.array([_vech(gamma_series[t]) for t in range(t_len)])
    n_star = vech_series.shape[1]
    daily = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_DAILY_LAG).mean().to_numpy()
    weekly = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_WEEKLY_LAG).mean().to_numpy()
    monthly = pd.DataFrame(vech_series).shift(1).rolling(VECHAR_MONTHLY_LAG).mean().to_numpy()
    daily = np.nan_to_num(daily, nan=0.0)
    weekly = np.nan_to_num(weekly, nan=0.0)
    monthly = np.nan_to_num(monthly, nan=0.0)
    s_path = np.empty_like(gamma_series)
    for t in range(t_len):
        vech_hat = coefs[0] + coefs[1] * daily[t] + coefs[2] * weekly[t] + coefs[3] * monthly[t]
        s_path[t] = _unvech(vech_hat, n)
    return s_path
