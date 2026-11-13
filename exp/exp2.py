"""Experiment 2: estimation and forecasting comparison of covariance matrix
models on DJIA realized-covariance data (Section 5, "Empirical Application").

Six BEKK-type models (sc-BEKK, Log-BEKK, Lin-BEKK, HE-BEKK, Log-HE-BEKK,
Lin-HE-BEKK) and two benchmarks (EWMA, vecHAR) are estimated on daily
annualized realized covariance matrices of DJIA constituents, compared by
log-likelihood/AIC/BIC (Table 3) and by in-sample/out-of-sample
Quasi-Likelihood (QL) and Global Minimum-Variance (GMV) loss (Section 5),
with the Model Confidence Set (Hansen, Lunde and Nason, 2011) used to select
the superior set of models at the 10% and 25% significance levels.

Run as ``python -m exp.exp2`` (executed configuration: 10 DJIA tickers,
2024-01-02 to 2026-05-29) or ``python -m exp.exp2 --full`` for the paper's
30-ticker, 2015-2026 configuration (see results/RESULTS.md for the measured
download time behind this reduction).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_djia_realized_covariance
from data.generate_synthetic import realized_covariance_s0
from src.config import (
    DJIA_END_DATE_PAPER, DJIA_END_DATE_RUN, DJIA_IDENTITY_DATE,
    DJIA_IN_SAMPLE_END_PAPER, DJIA_IN_SAMPLE_END_RUN, DJIA_START_DATE_PAPER,
    DJIA_START_DATE_RUN, DJIA_TICKERS_PAPER, DJIA_TICKERS_RUN, EWMA_DECAY,
    MCS_ALPHA_LEVELS, MCS_BOOTSTRAP_REPS_PAPER, MCS_BOOTSTRAP_REPS_RUN,
    MODEL_NAMES, NU, PARAM_NAMES, RC_BAR_TIMEFRAME,
)
from src.estimation import estimate_two_step
from src.likelihood import full_log_likelihood
from src.loss import (
    aic, average_loss, bic, ewma_forecast, fit_vechar,
    global_minimum_variance_loss, loss_series, quasi_likelihood_loss,
    vechar_forecast,
)
from src.mcs import model_confidence_set
from src.models import simulate_S_path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BENCHMARK_NAMES = ["EWMA", "vecHAR"]
ALL_MODEL_NAMES = MODEL_NAMES + BENCHMARK_NAMES


def _split(dates: pd.DatetimeIndex, gamma_series: np.ndarray, in_sample_end: str):
    """Split the realized-covariance series into in-sample (<= in_sample_end)
    and out-of-sample (> in_sample_end) segments (Section 5)."""
    cutoff = pd.Timestamp(in_sample_end)
    in_mask = dates <= cutoff
    return (
        dates[in_mask], gamma_series[in_mask],
        dates[~in_mask], gamma_series[~in_mask],
    )


def _fit_bekk_models(gamma_in: np.ndarray, s0: np.ndarray) -> dict:
    """Two-step QML estimation of the six BEKK models on the in-sample
    realized-covariance series (Section 4, equation 4.2)."""
    gamma_bar = gamma_in.mean(axis=0)
    fits = {}
    for model in MODEL_NAMES:
        t0 = time.time()
        result = estimate_two_step(model, gamma_in, s0, compute_se=True)
        elapsed = time.time() - t0
        full_ll = full_log_likelihood(result.theta, model, gamma_in, s0, gamma_bar, NU)
        k = len(result.theta)
        fits[model] = {
            "result": result,
            "gamma_bar": gamma_bar,
            "full_log_likelihood": full_ll,
            "aic": aic(full_ll, k),
            "bic": bic(full_ll, k, result.n_obs),
            "elapsed_seconds": elapsed,
        }
        print(f"[exp2] fitted {model} in {elapsed:.1f}s, "
              f"converged={result.converged}, logL={full_ll:.1f}")
    return fits


def _forecast_paths(fits: dict, gamma_in: np.ndarray, gamma_out: np.ndarray, s0: np.ndarray):
    """One-step-ahead S_hat paths for every BEKK model and benchmark, split
    into in-sample fitted values and out-of-sample forecasts built by
    continuing the recursion with parameters fixed at the in-sample estimate
    (Section 5: "parameters fixed at in-sample estimates")."""
    full_gamma = np.concatenate([gamma_in, gamma_out], axis=0)
    t_in = gamma_in.shape[0]
    paths = {}
    for model, fit in fits.items():
        s_path = simulate_S_path(model, fit["result"].theta, full_gamma, s0, fit["gamma_bar"])
        paths[model] = {"in_sample": s_path[:t_in], "out_of_sample": s_path[t_in:]}

    ewma_path = ewma_forecast(full_gamma, s0, EWMA_DECAY)
    paths["EWMA"] = {"in_sample": ewma_path[:t_in], "out_of_sample": ewma_path[t_in:]}

    vechar_coefs = fit_vechar(gamma_in)
    vechar_path = vechar_forecast(full_gamma, vechar_coefs)
    paths["vecHAR"] = {"in_sample": vechar_path[:t_in], "out_of_sample": vechar_path[t_in:]}
    return paths


def _loss_table(paths: dict, gamma_in: np.ndarray, gamma_out: np.ndarray) -> pd.DataFrame:
    """Average QL and GMV loss in-sample and out-of-sample for every model
    and benchmark (Section 5)."""
    rows = []
    for name in ALL_MODEL_NAMES:
        s_in, s_out = paths[name]["in_sample"], paths[name]["out_of_sample"]
        rows.append({
            "model": name,
            "ql_in_sample": average_loss(quasi_likelihood_loss, s_in, gamma_in),
            "ql_out_of_sample": average_loss(quasi_likelihood_loss, s_out, gamma_out),
            "gmv_in_sample": average_loss(global_minimum_variance_loss, s_in, gamma_in),
            "gmv_out_of_sample": average_loss(global_minimum_variance_loss, s_out, gamma_out),
        })
    return pd.DataFrame(rows)


def _run_mcs(paths: dict, gamma_in: np.ndarray, gamma_out: np.ndarray, n_boot: int) -> dict:
    """Model Confidence Set at 10% and 25% significance, for QL and GMV loss,
    in-sample and out-of-sample (Section 5)."""
    segments = {"in_sample": gamma_in, "out_of_sample": gamma_out}
    loss_fns = {"QL": quasi_likelihood_loss, "GMV": global_minimum_variance_loss}
    mcs_results = {}
    for seg_name, gamma in segments.items():
        for loss_name, loss_fn in loss_fns.items():
            loss_matrix = np.column_stack([
                loss_series(loss_fn, paths[name][seg_name], gamma) for name in ALL_MODEL_NAMES
            ])
            for alpha in MCS_ALPHA_LEVELS:
                key = f"{loss_name}_{seg_name}_alpha{int(alpha * 100)}"
                mcs_results[key] = model_confidence_set(
                    loss_matrix, ALL_MODEL_NAMES, alpha=alpha, n_boot=n_boot, seed=0,
                )
    return mcs_results


def _plot_coefficients(fits: dict, dates_in: pd.DatetimeIndex, gamma_in: np.ndarray,
                        s0: np.ndarray, path: Path) -> None:
    """Time-varying diagonal (variance) coefficients A_t[0, 0] for the
    flexible parameterizations, to visualize the dynamics referenced in
    Section 5 around market-stress episodes."""
    from src.models import compute_A

    fig, ax = plt.subplots(figsize=(10, 4))
    for model in ["Log-BEKK", "HE-BEKK", "Log-HE-BEKK"]:
        theta = fits[model]["result"].theta
        coefs = [compute_A(model, theta, gamma_in[t - 1], gamma_in.shape[1])[0, 0]
                 for t in range(1, gamma_in.shape[0])]
        ax.plot(dates_in[1:], coefs, label=model, linewidth=0.8)
    ax.set_title("Time-varying diagonal coefficient of A_t (first asset)")
    ax.set_xlabel("date")
    ax.set_ylabel("A_t[0, 0]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's 30-ticker, 2015-2026 configuration "
                              "instead of the reduced download")
    args = parser.parse_args()

    tickers = DJIA_TICKERS_PAPER if args.full else DJIA_TICKERS_RUN
    start_date = DJIA_START_DATE_PAPER if args.full else DJIA_START_DATE_RUN
    end_date = DJIA_END_DATE_PAPER if args.full else DJIA_END_DATE_RUN
    in_sample_end = DJIA_IN_SAMPLE_END_PAPER if args.full else DJIA_IN_SAMPLE_END_RUN
    n_boot = MCS_BOOTSTRAP_REPS_PAPER if args.full else MCS_BOOTSTRAP_REPS_RUN

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dates, gamma_series, meta = load_djia_realized_covariance(
        tickers, start_date, end_date, timeframe=RC_BAR_TIMEFRAME,
        identity_date=DJIA_IDENTITY_DATE,
    )
    load_elapsed = time.time() - t0
    print(f"[exp2] loaded {len(dates)} realized-covariance days for "
          f"{len(tickers)} tickers in {load_elapsed:.1f}s "
          f"({dates.min()} to {dates.max()})")

    dates_in, gamma_in, dates_out, gamma_out = _split(dates, gamma_series, in_sample_end)
    assert gamma_in.shape[0] > 0 and gamma_out.shape[0] > 0, "empty in-sample or out-of-sample segment"
    s0 = realized_covariance_s0(gamma_in)

    t0 = time.time()
    fits = _fit_bekk_models(gamma_in, s0)
    fit_elapsed = time.time() - t0

    paths = _forecast_paths(fits, gamma_in, gamma_out, s0)
    loss_table = _loss_table(paths, gamma_in, gamma_out)

    t0 = time.time()
    mcs_results = _run_mcs(paths, gamma_in, gamma_out, n_boot)
    mcs_elapsed = time.time() - t0

    fit_table = pd.DataFrame([
        {
            "model": model,
            "log_likelihood": fit["full_log_likelihood"],
            "aic": fit["aic"],
            "bic": fit["bic"],
            "converged": fit["result"].converged,
            "n_params": len(fit["result"].theta),
            **{f"param_{k}": v for k, v in fit["result"].params.items()},
            **{f"se_{k}": v for k, v in fit["result"].std_errors.items()},
        }
        for model, fit in fits.items()
    ])

    fit_table.to_csv(RESULTS_DIR / "exp2_model_fit.csv", index=False)
    loss_table.to_csv(RESULTS_DIR / "exp2_loss_table.csv", index=False)
    (RESULTS_DIR / "exp2_mcs.json").write_text(json.dumps(mcs_results, indent=2, default=str))
    _plot_coefficients(fits, dates_in, gamma_in, s0, RESULTS_DIR / "exp2_coefficients.png")

    log = {
        "full_run": args.full,
        "tickers": tickers,
        "n_tickers": len(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "in_sample_end": in_sample_end,
        "n_in_sample": int(gamma_in.shape[0]),
        "n_out_of_sample": int(gamma_out.shape[0]),
        "mcs_bootstrap_reps": n_boot,
        "load_elapsed_seconds": load_elapsed,
        "fit_elapsed_seconds": fit_elapsed,
        "mcs_elapsed_seconds": mcs_elapsed,
        "failed_tickers": meta.get("failed"),
        "non_massive_source": meta.get("non_massive_source"),
        "date_range": [str(dates.min()), str(dates.max())],
    }
    (RESULTS_DIR / "exp2_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"[exp2] done: fit {fit_elapsed:.1f}s, MCS {mcs_elapsed:.1f}s, results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
