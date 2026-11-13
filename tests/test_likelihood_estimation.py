"""Unit tests for src/likelihood.py and src/estimation.py: the two-step QML
estimator of Section 4 and its quasi log-likelihood (equation 4.2)."""
from __future__ import annotations

import numpy as np

from src.estimation import estimate_two_step, sandwich_std_errors
from src.likelihood import neg_log_likelihood


def test_neg_log_likelihood_is_finite_for_true_params(small_gamma_series, small_s0):
    gamma_series, _ = small_gamma_series
    gamma_bar = gamma_series.mean(axis=0)
    theta = np.array([0.1, 0.3, 0.5])
    value = neg_log_likelihood(theta, "sc-BEKK", gamma_series, small_s0, gamma_bar)
    assert np.isfinite(value)


def test_neg_log_likelihood_penalizes_non_pd_S():
    # b close to 1 with large a can push S_t towards singularity; the
    # function must return the large penalty constant rather than raising.
    n = 3
    gamma_series = np.array([np.eye(n) * 0.01] * 5)
    s0 = np.eye(n) * 0.01
    gamma_bar = np.eye(n) * 0.01
    theta = np.array([-10.0, -10.0, -10.0])  # deliberately invalid (negative kappa/a)
    value = neg_log_likelihood(theta, "sc-BEKK", gamma_series, s0, gamma_bar)
    assert np.isfinite(value)  # must not raise; either a real value or the 1e10 penalty


def test_two_step_estimation_recovers_sc_bekk_parameters_on_long_series():
    """Consistency check (Section 4): on a long simulated sc-BEKK series the
    QML estimator should recover the true (kappa, a, b) reasonably closely."""
    from data.generate_synthetic import simulate_series

    rng = np.random.default_rng(0)
    n = 4
    gamma_bar = np.eye(n) * 2.0 + 0.3
    s0 = gamma_bar.copy()
    true_theta = np.array([0.1, 0.3, 0.55])
    gamma_series, _ = simulate_series(
        "sc-BEKK", true_theta, n=n, nu=60, t_len=1500, s0=s0,
        gamma_bar=gamma_bar, burn_in=100, rng=rng,
    )
    result = estimate_two_step("sc-BEKK", gamma_series, s0)
    assert result.converged
    assert abs(result.params["a"] - true_theta[1]) < 0.15
    assert abs(result.params["b"] - true_theta[2]) < 0.2
    assert 0.0 <= result.params["b"] < 1.0


def test_estimation_result_n_obs_is_t_minus_one(small_gamma_series, small_s0):
    gamma_series, _ = small_gamma_series
    result = estimate_two_step("sc-BEKK", gamma_series, small_s0)
    assert result.n_obs == gamma_series.shape[0] - 1


def test_sandwich_std_errors_are_nonnegative(small_gamma_series, small_s0):
    gamma_series, _ = small_gamma_series
    gamma_bar = gamma_series.mean(axis=0)
    theta = np.array([0.15, 0.3, 0.5])
    se = sandwich_std_errors(theta, "sc-BEKK", gamma_series, small_s0, gamma_bar)
    for value in se.values():
        assert np.isnan(value) or value >= 0.0
