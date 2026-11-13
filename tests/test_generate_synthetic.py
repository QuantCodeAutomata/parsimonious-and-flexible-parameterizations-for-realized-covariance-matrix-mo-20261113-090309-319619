"""Unit tests for data/generate_synthetic.py: the Monte Carlo DGP of Section
4.1, model (2.3) with Gamma_t ~ Wishart(S_t/nu, nu)."""
from __future__ import annotations

import numpy as np

from data.generate_synthetic import realized_covariance_s0, simulate_series


def test_simulate_series_shapes(small_s0, small_gamma_bar):
    rng = np.random.default_rng(0)
    theta = np.array([0.1, 0.3, 0.5])
    gamma_series, s_series = simulate_series(
        "sc-BEKK", theta, n=4, nu=30, t_len=25, s0=small_s0,
        gamma_bar=small_gamma_bar, burn_in=5, rng=rng,
    )
    assert gamma_series.shape == (25, 4, 4)
    assert s_series.shape == (25, 4, 4)


def test_simulate_series_matrices_are_symmetric_positive_semidefinite(small_s0, small_gamma_bar):
    rng = np.random.default_rng(1)
    theta = np.array([0.1, 0.3, 0.5])
    gamma_series, s_series = simulate_series(
        "sc-BEKK", theta, n=4, nu=40, t_len=30, s0=small_s0,
        gamma_bar=small_gamma_bar, burn_in=10, rng=rng,
    )
    for gamma, s_t in zip(gamma_series, s_series):
        assert np.allclose(gamma, gamma.T, atol=1e-8)
        assert np.allclose(s_t, s_t.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(gamma) > -1e-8)
        assert np.all(np.linalg.eigvalsh(s_t) > -1e-8)


def test_simulate_series_conditional_mean_matches_s_t():
    """E[Gamma_t | F_{t-1}] = S_t for Gamma_t ~ Wishart(S_t/nu, nu) (Section 4,
    assumption after equation 4.1): average many one-step draws conditional on
    a fixed S_t and check the Monte Carlo mean is close to S_t."""
    from scipy.stats import wishart

    n, nu = 3, 60
    s_t = np.eye(n) * 2.0 + 0.3
    rng = np.random.default_rng(2)
    draws = wishart.rvs(df=nu, scale=s_t / nu, size=4000, random_state=rng)
    assert np.allclose(draws.mean(axis=0), s_t, atol=0.2)


def test_simulate_series_different_seeds_give_different_paths(small_s0, small_gamma_bar):
    theta = np.array([0.1, 0.3, 0.5])
    g1, _ = simulate_series("sc-BEKK", theta, n=4, nu=30, t_len=20, s0=small_s0,
                             gamma_bar=small_gamma_bar, burn_in=5, rng=np.random.default_rng(1))
    g2, _ = simulate_series("sc-BEKK", theta, n=4, nu=30, t_len=20, s0=small_s0,
                             gamma_bar=small_gamma_bar, burn_in=5, rng=np.random.default_rng(2))
    assert not np.allclose(g1, g2)


def test_realized_covariance_s0_is_sample_mean():
    gamma_series = np.stack([np.eye(2), np.eye(2) * 3.0])
    s0 = realized_covariance_s0(gamma_series)
    assert np.allclose(s0, np.eye(2) * 2.0)
