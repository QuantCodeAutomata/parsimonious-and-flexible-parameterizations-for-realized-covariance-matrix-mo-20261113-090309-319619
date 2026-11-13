"""Coefficient-matrix parameterizations A_t of model (2.3).

S_t = kappa * Gamma_bar + A_t (Hadamard) Gamma_{t-1} + b * S_{t-1}       (2.3)

Each parameterization below reproduces one of the six specifications of
Section 3 ("Feasible Parameterizations"): sc-BEKK (benchmark scalar case),
Log-BEKK (3.1)-(3.2), Lin-BEKK (3.3)-(3.4), HE-BEKK (3.6), Log-HE-BEKK (3.7),
Lin-HE-BEKK (3.8).

Custom implementation (paper Section 3): no library provides these
positive-semidefinite, parsimonious coefficient-matrix parameterizations, so
they are implemented directly from the equations above.
"""
from __future__ import annotations

import numpy as np

from src.config import PARAM_NAMES


def unpack_params(model: str, theta: np.ndarray) -> dict:
    """Map a flat parameter vector to the named parameters of ``model``."""
    names = PARAM_NAMES[model]
    if len(theta) != len(names):
        raise ValueError(f"{model} expects {len(names)} parameters, got {len(theta)}")
    return dict(zip(names, theta))


def correlation_matrix(gamma: np.ndarray) -> np.ndarray:
    """Correlation matrix P implied by a covariance matrix Gamma (Section 3.3):
    P_ij = Gamma_ij / sqrt(Gamma_ii * Gamma_jj)."""
    d = np.sqrt(np.diag(gamma))
    outer = np.outer(d, d)
    return gamma / outer


def variance_diag(gamma: np.ndarray) -> np.ndarray:
    """Diagonal matrix V_t of realized variances (Section 3.3, Log-HE/Lin-HE)."""
    return np.diag(np.diag(gamma))


def _matrix_logistic(sym: np.ndarray, phi: float, c: float) -> np.ndarray:
    """Logistic function G_t = (I_n + exp(-phi*(sym - c)))^{-1} (eq. 3.2).

    Per the paper's implementation notes (Section 3.1): the exponential
    inside is an elementwise (Hadamard) exponential of the matrix
    ``-phi*(sym - c)`` (``c`` is a scalar subtracted from every entry), and
    the outer ``^{-1}`` is an ordinary matrix inverse of the resulting
    I_n + elementwise-exponential matrix.
    """
    exponent = np.clip(-phi * (sym - c), -700.0, 700.0)  # avoid float overflow in exp
    inner = np.eye(sym.shape[0]) + np.exp(exponent)
    inner += 1e-10 * np.eye(sym.shape[0])  # ridge: guards against near-singular inner matrix
    return np.linalg.inv(inner)


def compute_A(model: str, theta: np.ndarray, gamma_prev: np.ndarray, n: int,
              extra: dict | None = None) -> np.ndarray:
    """Coefficient matrix A_t for ``model`` given lagged realized covariance
    ``gamma_prev`` (n x n) and parameter vector ``theta``.

    ``extra`` carries sample-dependent constants needed by some
    parameterizations (not estimated themselves): ``max_xi`` (maximum
    eigenvalue of Gamma_t over the sample, Lin-BEKK, Proposition 6) and
    ``min_xi_star`` (minimum eigenvalue of V_t^{-1/2} P_t V_t^{-1/2} used by
    Lin-HE-BEKK, Proposition 8); they are only read, never estimated.
    """
    p = unpack_params(model, theta)
    j_n = np.ones((n, n))

    if model == "sc-BEKK":
        return p["a"] * j_n

    if model == "Log-BEKK":
        g_t = _matrix_logistic(gamma_prev, p["phi"], p["c"])
        return p["a0"] * j_n + p["a1"] * g_t

    if model == "Lin-BEKK":
        c_mat = (p["alpha_v"] - p["alpha_c"]) * np.eye(n) + p["alpha_c"] * j_n
        return c_mat - p["alpha_a"] * gamma_prev

    if model == "HE-BEKK":
        corr = correlation_matrix(gamma_prev)
        return p["a"] * np.exp(p["varphi"] * (corr - j_n))

    if model == "Log-HE-BEKK":
        corr = correlation_matrix(gamma_prev)
        w_t = p["a"] * np.exp(p["varphi"] * (corr - j_n))
        v_prev = variance_diag(gamma_prev)
        g_star = _matrix_logistic(v_prev, p["phi"], p["c"])
        return w_t + p["a1"] * g_star

    if model == "Lin-HE-BEKK":
        corr = correlation_matrix(gamma_prev)
        w_t = p["a"] * np.exp(p["varphi"] * (corr - j_n))
        v_prev = variance_diag(gamma_prev)
        return w_t - p["alpha_a"] * v_prev

    raise ValueError(f"Unknown model {model!r}")


def simulate_S_path(model: str, theta: np.ndarray, gamma_series: np.ndarray,
                     s0: np.ndarray, gamma_bar: np.ndarray) -> np.ndarray:
    """Recursively build the conditional covariance path S_1, ..., S_T from
    an observed/simulated realized covariance series ``gamma_series``
    (shape (T, n, n)) using recursion (2.3).

    S_1 is set to ``s0`` (the recursion needs a lagged S and a lagged
    Gamma; the first return corresponds to t=1 so S_t[0] == s0).
    """
    p = unpack_params(model, theta)
    t_len, n, _ = gamma_series.shape
    s_path = np.empty_like(gamma_series)
    s_path[0] = s0
    gamma_prev = gamma_series[0] if t_len > 0 else s0
    for t in range(1, t_len):
        gamma_prev = gamma_series[t - 1]
        s_prev = s_path[t - 1]
        a_t = compute_A(model, theta, gamma_prev, n)
        s_path[t] = p["kappa"] * gamma_bar + a_t * gamma_prev + p["b"] * s_prev
    return s_path


def max_xi_lin_bekk(gamma_series: np.ndarray) -> float:
    """max(xi): largest value, over the sample, of the maximum eigenvalue of
    each realized covariance matrix Gamma_t (Lin-BEKK, Proposition 6)."""
    return max(np.linalg.eigvalsh(g).max() for g in gamma_series)


def min_xi_star_lin_he_bekk(gamma_series: np.ndarray) -> float:
    """min(xi**): minimum, over the sample, of the eigenvalues of
    V_{t-1}^{-1/2} P_{t-1} V_{t-1}^{-1/2} (Lin-HE-BEKK, Proposition 8 proof).

    This quantity depends only on the data, not on the parameters being
    estimated. The constraint itself, alpha_a <= a*varphi*min(xi**)/exp(varphi),
    is evaluated at each candidate (a, varphi) during optimization.
    """
    min_eig = np.inf
    for gamma in gamma_series:
        corr = correlation_matrix(gamma)
        v = variance_diag(gamma)
        v_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(v)))
        mat = v_inv_sqrt @ corr @ v_inv_sqrt
        min_eig = min(min_eig, np.linalg.eigvalsh(mat).min())
    return min_eig
