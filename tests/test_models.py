"""Unit tests for src/models.py: coefficient-matrix parameterizations (2.3),
Section 3."""
from __future__ import annotations

import numpy as np

from src.config import PARAM_NAMES, TRUE_PARAMS
from src.models import (compute_A, correlation_matrix, max_xi_lin_bekk,
                         min_xi_star_lin_he_bekk, simulate_S_path,
                         unpack_params, variance_diag)


def _theta(model: str) -> np.ndarray:
    return np.array([TRUE_PARAMS[model][name] for name in PARAM_NAMES[model]])


def test_unpack_params_matches_param_names():
    theta = np.arange(3.0)
    params = unpack_params("sc-BEKK", theta)
    assert params == {"kappa": 0.0, "a": 1.0, "b": 2.0}


def test_unpack_params_wrong_length_raises():
    try:
        unpack_params("sc-BEKK", np.arange(2.0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_correlation_matrix_diagonal_is_one():
    gamma = np.array([[4.0, 1.0], [1.0, 9.0]])
    corr = correlation_matrix(gamma)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.isclose(corr[0, 1], 1.0 / 6.0)


def test_variance_diag_is_diagonal_of_input():
    gamma = np.array([[4.0, 1.0], [1.0, 9.0]])
    v = variance_diag(gamma)
    assert np.allclose(v, np.diag([4.0, 9.0]))
    assert np.allclose(v - np.diag(np.diag(v)), 0.0)


def test_sc_bekk_a_is_constant_scalar_matrix():
    n = 5
    gamma_prev = np.eye(n) * 2.0
    a_t = compute_A("sc-BEKK", _theta("sc-BEKK"), gamma_prev, n)
    assert np.allclose(a_t, TRUE_PARAMS["sc-BEKK"]["a"])


def test_all_models_produce_finite_square_A(small_gamma_bar):
    n = small_gamma_bar.shape[0]
    gamma_prev = small_gamma_bar
    for model in PARAM_NAMES:
        if model in ("Lin-BEKK", "Lin-HE-BEKK"):
            continue  # need sample-level xi constants, tested separately
        a_t = compute_A(model, _theta(model), gamma_prev, n)
        assert a_t.shape == (n, n)
        assert np.all(np.isfinite(a_t))


def test_s_path_equals_s0_at_first_step(small_gamma_series, small_s0):
    gamma_series, _ = small_gamma_series
    theta = np.array([0.1, 0.3, 0.5])
    gamma_bar = gamma_series.mean(axis=0)
    s_path = simulate_S_path("sc-BEKK", theta, gamma_series, small_s0, gamma_bar)
    assert np.allclose(s_path[0], small_s0)
    assert s_path.shape == gamma_series.shape


def test_s_path_matches_manual_recursion(small_gamma_series, small_s0):
    gamma_series, _ = small_gamma_series
    theta = np.array([0.1, 0.3, 0.5])
    gamma_bar = gamma_series.mean(axis=0)
    s_path = simulate_S_path("sc-BEKK", theta, gamma_series, small_s0, gamma_bar)
    manual = theta[0] * gamma_bar + theta[1] * gamma_series[0] + theta[2] * small_s0
    assert np.allclose(s_path[1], manual)


def test_max_xi_lin_bekk_is_max_eigenvalue_over_series():
    gamma_series = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 0.5])])
    assert np.isclose(max_xi_lin_bekk(gamma_series), 3.0)


def test_min_xi_star_lin_he_bekk_matches_definition():
    gamma_series = np.array([np.diag([4.0, 9.0])])
    value = min_xi_star_lin_he_bekk(gamma_series)
    # For a diagonal Gamma, the correlation matrix is the identity, so
    # V^{-1/2} P V^{-1/2} = V^{-1} and its eigenvalues are 1/variance.
    assert np.isclose(value, 1.0 / 9.0)
