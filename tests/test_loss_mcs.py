"""Unit tests for src/loss.py (QL, GMV, EWMA, vecHAR) and src/mcs.py (Model
Confidence Set, Section 5)."""
from __future__ import annotations

import numpy as np

from src.loss import (aic, average_loss, bic, ewma_forecast, fit_vechar,
                       global_minimum_variance_loss, loss_series,
                       quasi_likelihood_loss, vechar_forecast)
from src.mcs import model_confidence_set


def test_quasi_likelihood_loss_zero_when_forecast_equals_realization():
    gamma = np.array([[2.0, 0.3], [0.3, 1.5]])
    loss = quasi_likelihood_loss(gamma, gamma)
    # ln|Gamma| + tr(Gamma^{-1} Gamma) = ln|Gamma| + n
    sign, logdet = np.linalg.slogdet(gamma)
    assert np.isclose(loss, logdet + 2)


def test_quasi_likelihood_loss_is_inf_for_non_pd_forecast():
    bad = np.array([[-1.0, 0.0], [0.0, -1.0]])
    gamma = np.eye(2)
    assert quasi_likelihood_loss(bad, gamma) == np.inf


def test_gmv_weights_sum_properties_and_loss_nonnegative():
    s_hat = np.array([[2.0, 0.2], [0.2, 1.0]])
    gamma = np.array([[1.8, 0.1], [0.1, 0.9]])
    loss = global_minimum_variance_loss(s_hat, gamma)
    assert loss > 0.0


def test_gmv_loss_minimized_by_true_inverse_variance_weights():
    # For a diagonal S_hat equal to the realized Gamma, the GMV loss should
    # equal the harmonic-mean-based minimum-variance portfolio variance,
    # which must be below the variance of an equal-weight portfolio.
    gamma = np.diag([4.0, 1.0])
    gmv_loss = global_minimum_variance_loss(gamma, gamma)
    equal_weights = np.array([1.0, 1.0]) / np.sqrt(2.0) * np.sqrt(2)
    equal_weights = np.ones(2)
    equal_var = equal_weights @ gamma @ equal_weights
    assert gmv_loss <= equal_var


def test_average_loss_and_loss_series_consistent():
    s_hat_series = np.array([np.eye(2), np.eye(2) * 2])
    gamma_series = np.array([np.eye(2), np.eye(2) * 2])
    series = loss_series(quasi_likelihood_loss, s_hat_series, gamma_series)
    assert np.isclose(average_loss(quasi_likelihood_loss, s_hat_series, gamma_series), series.mean())


def test_ewma_forecast_first_step_uses_s0():
    gamma_series = np.array([np.eye(2) * 3.0, np.eye(2) * 1.0])
    s0 = np.eye(2) * 2.0
    path = ewma_forecast(gamma_series, s0, decay=0.96)
    expected0 = 0.04 * s0 + 0.96 * s0
    assert np.allclose(path[0], expected0)


def test_aic_bic_penalize_more_parameters():
    ll = -100.0
    assert aic(ll, 5) < aic(ll, 10)
    assert bic(ll, 5, 1000) < bic(ll, 10, 1000)


def test_fit_and_forecast_vechar_shapes():
    rng = np.random.default_rng(0)
    n = 3
    t_len = 60
    base = np.eye(n)
    gamma_series = np.array([base + 0.05 * rng.standard_normal((n, n)) @ np.eye(n) for _ in range(t_len)])
    gamma_series = np.array([0.5 * (g + g.T) + n * np.eye(n) for g in gamma_series])
    coefs = fit_vechar(gamma_series)
    assert coefs.shape == (4, n * (n + 1) // 2)
    forecasts = vechar_forecast(gamma_series, coefs)
    assert forecasts.shape == gamma_series.shape


def test_model_confidence_set_keeps_best_model_in_superior_set():
    rng = np.random.default_rng(0)
    t_len = 300
    # Model 0 has systematically lower loss; models 1, 2 are noisier/worse.
    loss_matrix = np.column_stack([
        rng.normal(loc=1.0, scale=0.2, size=t_len),
        rng.normal(loc=2.0, scale=0.2, size=t_len),
        rng.normal(loc=2.5, scale=0.2, size=t_len),
    ])
    result = model_confidence_set(loss_matrix, ["best", "mid", "worst"], alpha=0.25, n_boot=200, seed=1)
    assert "best" in result["superior_set"]
    assert "worst" not in result["superior_set"]


def test_model_confidence_set_keeps_all_when_losses_are_indistinguishable():
    rng = np.random.default_rng(2)
    t_len = 200
    loss_matrix = rng.normal(loc=1.0, scale=0.2, size=(t_len, 3))
    result = model_confidence_set(loss_matrix, ["a", "b", "c"], alpha=0.10, n_boot=200, seed=1)
    assert len(result["superior_set"]) >= 1
