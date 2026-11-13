"""Two-step QML estimation of model (2.3) (Section 4 "Estimator Properties").

Step 1: Gamma_bar is estimated by the sample mean of the realized covariance
matrices (consistent estimator, Section 4).
Step 2: kappa, b and the parameters defining A_t are estimated by maximizing
the quasi log-likelihood (4.2), implemented in src.likelihood.

Custom implementation (paper Section 4): the two-step M-estimator and its
per-model inequality constraints are specific to this paper and are not
provided by a general optimization or econometrics library; `scipy.optimize`
supplies only the generic constrained solver (SLSQP).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.config import OPTIMIZER_FTOL, OPTIMIZER_MAXITER, OPTIMIZER_METHOD, PARAM_NAMES
from src.likelihood import neg_log_likelihood, score_contributions
from src.models import max_xi_lin_bekk, min_xi_star_lin_he_bekk


@dataclass
class EstimationResult:
    model: str
    params: dict
    theta: np.ndarray
    neg_loglik: float
    log_likelihood: float
    n_obs: int
    converged: bool
    std_errors: dict = field(default_factory=dict)


def _initial_guess(model: str, gamma_bar: np.ndarray) -> np.ndarray:
    """Starting values (not specified in the paper): small persistence and
    news coefficients, kappa scaled so that kappa*Gamma_bar ~ unconditional
    mean under modest persistence."""
    scale = 0.1
    defaults = {
        "kappa": scale, "a": 0.3, "b": 0.5, "a0": 0.2, "a1": 0.1, "phi": 1.0,
        "c": float(np.median(np.diag(gamma_bar))), "alpha_v": 0.3, "alpha_c": 0.1,
        "alpha_a": 0.05, "varphi": 0.5,
    }
    return np.array([defaults[name] for name in PARAM_NAMES[model]], dtype=float)


def _bounds_and_constraints(model: str, gamma_series: np.ndarray):
    """Bounds and nonlinear inequality constraints enforcing the stability and
    positive-semidefiniteness conditions of Section 3 for ``model``."""
    names = PARAM_NAMES[model]
    idx = {name: i for i, name in enumerate(names)}
    # kappa*Gamma_bar is one additive component of a matrix whose stationary
    # scale is Gamma_bar itself (Section 2), so kappa cannot stray far above
    # unity without violating (2.3); an explicit finite bound keeps SLSQP from
    # diverging to numerically meaningless values in the (data-independent)
    # unbounded region above the inequality constraints below.
    bounds = [(1e-6, 5.0)] * len(names)
    cons = []

    if model == "sc-BEKK":
        bounds[idx["a"]] = (0.0, 5.0)
        bounds[idx["b"]] = (0.0, 0.999)

    elif model == "Log-BEKK":
        bounds[idx["a0"]] = (1e-6, 5.0)
        bounds[idx["a1"]] = (0.0, 5.0)
        bounds[idx["phi"]] = (1e-6, 50.0)
        bounds[idx["c"]] = (None, None)
        bounds[idx["b"]] = (0.0, 0.999)
        cons.append({"type": "ineq", "fun": lambda t: 1 - (t[idx["a0"]] + t[idx["a1"]] + t[idx["b"]]) - 1e-6})

    elif model == "Lin-BEKK":
        max_xi = max_xi_lin_bekk(gamma_series)
        bounds[idx["alpha_v"]] = (1e-6, 10.0)
        bounds[idx["alpha_c"]] = (0.0, 10.0)
        bounds[idx["alpha_a"]] = (0.0, 20.0)
        bounds[idx["b"]] = (0.0, 0.999)
        cons.append({"type": "ineq", "fun": lambda t: t[idx["alpha_v"]] - t[idx["alpha_c"]] - 1e-6})
        cons.append({"type": "ineq",
                     "fun": lambda t: (t[idx["alpha_v"]] - t[idx["alpha_c"]]) / max_xi - t[idx["alpha_a"]]})
        cons.append({"type": "ineq", "fun": lambda t: 1 - (t[idx["alpha_v"]] + t[idx["b"]]) - 1e-6})

    elif model == "HE-BEKK":
        bounds[idx["a"]] = (1e-6, 5.0)
        bounds[idx["varphi"]] = (0.0, 20.0)
        bounds[idx["b"]] = (0.0, 0.999)
        cons.append({"type": "ineq", "fun": lambda t: 1 - (t[idx["a"]] + t[idx["b"]]) - 1e-6})

    elif model == "Log-HE-BEKK":
        bounds[idx["a"]] = (1e-6, 5.0)
        bounds[idx["a1"]] = (0.0, 5.0)
        bounds[idx["phi"]] = (1e-6, 50.0)
        bounds[idx["c"]] = (None, None)
        bounds[idx["varphi"]] = (0.0, 20.0)
        bounds[idx["b"]] = (0.0, 0.999)
        cons.append({"type": "ineq",
                     "fun": lambda t: 1 - (t[idx["a"]] + t[idx["a1"]] + t[idx["b"]]) - 1e-6})

    elif model == "Lin-HE-BEKK":
        min_xi_star = min_xi_star_lin_he_bekk(gamma_series)
        bounds[idx["a"]] = (1e-6, 5.0)
        bounds[idx["varphi"]] = (0.0, 20.0)
        bounds[idx["alpha_a"]] = (0.0, 20.0)
        bounds[idx["b"]] = (0.0, 0.999)
        cons.append({"type": "ineq", "fun": lambda t: 1 - (t[idx["a"]] + t[idx["b"]]) - 1e-6})
        cons.append({"type": "ineq",
                     "fun": lambda t: t[idx["a"]] * t[idx["varphi"]] * min_xi_star / np.exp(t[idx["varphi"]])
                     - t[idx["alpha_a"]]})
    else:
        raise ValueError(f"Unknown model {model!r}")

    return bounds, cons


def estimate_two_step(model: str, gamma_series: np.ndarray, s0: np.ndarray,
                       theta0: np.ndarray | None = None,
                       compute_se: bool = False) -> EstimationResult:
    """Two-step QML estimation of ``model`` on an observed/simulated realized
    covariance series ``gamma_series`` (shape (T, n, n)).

    Step 1 (Section 4): Gamma_bar = sample mean of ``gamma_series``.
    Step 2: maximize (4.2) over kappa, b and the parameters of A_t.
    """
    gamma_bar = gamma_series.mean(axis=0)
    if theta0 is None:
        theta0 = _initial_guess(model, gamma_bar)
    bounds, cons = _bounds_and_constraints(model, gamma_series)

    result = minimize(
        neg_log_likelihood, theta0, args=(model, gamma_series, s0, gamma_bar),
        method=OPTIMIZER_METHOD, bounds=bounds, constraints=cons,
        options={"maxiter": OPTIMIZER_MAXITER, "ftol": OPTIMIZER_FTOL},
    )

    std_errors = {}
    if compute_se and result.success:
        std_errors = sandwich_std_errors(result.x, model, gamma_series, s0, gamma_bar)

    names = PARAM_NAMES[model]
    return EstimationResult(
        model=model,
        params=dict(zip(names, result.x)),
        theta=result.x,
        neg_loglik=result.fun,
        log_likelihood=-result.fun,
        n_obs=gamma_series.shape[0] - 1,
        converged=bool(result.success),
        std_errors=std_errors,
    )


def sandwich_std_errors(theta: np.ndarray, model: str, gamma_series: np.ndarray,
                         s0: np.ndarray, gamma_bar: np.ndarray, eps: float = 1e-4) -> dict:
    """Robust sandwich standard errors (White, 1982), as used for Table 2
    (Section 5): V = H^{-1} Omega H^{-1}, with H the numerical Hessian of the
    negative log-likelihood and Omega the outer product of per-period score
    contributions. Scores are computed by finite differences of the
    per-period likelihood contributions."""
    k = len(theta)
    names = PARAM_NAMES[model]

    def total_nll(th):
        return neg_log_likelihood(th, model, gamma_series, s0, gamma_bar)

    hessian = np.zeros((k, k))
    f0 = total_nll(theta)
    for i in range(k):
        for j in range(k):
            ti, tj = theta.copy(), theta.copy()
            ti[i] += eps
            tj[j] += eps
            tij = theta.copy()
            tij[i] += eps
            tij[j] += eps
            f_ij = total_nll(tij)
            f_i = total_nll(ti)
            f_j = total_nll(tj)
            hessian[i, j] = (f_ij - f_i - f_j + f0) / (eps ** 2)

    try:
        h_inv = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return {name: float("nan") for name in names}

    scores = score_contributions(theta, model, gamma_series, s0, gamma_bar)
    omega = np.outer(scores, scores)
    cov = h_inv @ omega @ h_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(zip(names, se))
