"""Experiment 1: Monte Carlo evidence on the finite-sample properties of the
two-step QML estimators of the six realized-covariance BEKK parameterizations
(Section 4.1 "Monte Carlo Evidence", Table 1).

For each of sc-BEKK, Log-BEKK, Lin-BEKK, HE-BEKK, Log-HE-BEKK and Lin-HE-BEKK,
and for each sample size T in {100, 500, 1000, 2000}, this script simulates
replicated realized-covariance paths from the data-generating process (2.3)
with the true parameters of Table 2, re-estimates every path by two-step QML,
and reports the Monte Carlo bias, RMSE and Jarque-Bera normality p-value of
every parameter (Section 4.1).

Run: python -m exp.exp1 [--full]
``--full`` runs the paper's 500 replications per cell (MC_REPLICATIONS_PAPER)
instead of the reduced, time-budgeted MC_REPLICATIONS_RUN (see
results/RESULTS.md, "Deviations from the paper").
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import load_djia_realized_covariance
from data.generate_synthetic import realized_covariance_s0, simulate_series
from src.config import (
    DJIA_END_DATE_RUN, DJIA_IDENTITY_DATE, DJIA_START_DATE_RUN, DJIA_TICKERS,
    MC_BURNIN, MC_JB_SIGNIFICANCE, MC_REPLICATIONS_PAPER, MC_REPLICATIONS_RUN,
    MC_SAMPLE_SIZES, MC_SEED, MC_WORKERS, MODEL_NAMES, N_ASSETS, NU,
    PARAM_NAMES, RC_BAR_TIMEFRAME, TRUE_PARAMS,
)
from src.estimation import estimate_two_step
from src.metrics import smallest_normal_T, summarize_replications

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _theta_true(model: str) -> np.ndarray:
    names = PARAM_NAMES[model]
    return np.array([TRUE_PARAMS[model][name] for name in names], dtype=float)


def _one_replication(args: tuple) -> np.ndarray | None:
    """Simulate one path of length ``t_len`` from ``model`` and re-estimate
    it by two-step QML (Section 4.1). Returns the estimated parameter vector,
    or None if the optimizer failed to produce a finite result."""
    model, theta_true, t_len, s0, gamma_bar, seed = args
    rng = np.random.default_rng(seed)
    gamma_series, _ = simulate_series(
        model, theta_true, N_ASSETS, NU, t_len, s0, gamma_bar,
        burn_in=MC_BURNIN, rng=rng,
    )
    result = estimate_two_step(model, gamma_series, s0=s0)
    if not np.all(np.isfinite(result.theta)):
        return None
    return result.theta


def _gamma_bar_and_s0() -> tuple[np.ndarray, np.ndarray, dict]:
    """Unconditional mean Gamma_bar and starting value S_0 used by the Monte
    Carlo DGP, both "extracted from the realized covariance series of the
    application in Section 5" (Section 4.1, footnote 5)."""
    dates, gamma_series, meta = load_djia_realized_covariance(
        DJIA_TICKERS, DJIA_START_DATE_RUN, DJIA_END_DATE_RUN,
        timeframe=RC_BAR_TIMEFRAME, identity_date=DJIA_IDENTITY_DATE,
    )
    gamma_bar = gamma_series.mean(axis=0)
    s0 = realized_covariance_s0(gamma_series)
    info = {
        "n_days": int(len(dates)),
        "date_range": [str(dates.min()), str(dates.max())],
        "universe_size": len(DJIA_TICKERS),
    }
    return gamma_bar, s0, info


def run_monte_carlo(n_replications: int) -> tuple[pd.DataFrame, dict]:
    gamma_bar, s0, data_info = _gamma_bar_and_s0()
    print(f"[exp1] Gamma_bar/S0 from {data_info['n_days']} trading days "
          f"({data_info['date_range'][0]} to {data_info['date_range'][1]}), "
          f"{data_info['universe_size']} tickers")

    rows = []
    estimates_by_cell: dict = {}
    rng_master = np.random.default_rng(MC_SEED)

    with ProcessPoolExecutor(max_workers=MC_WORKERS) as pool:
        for model in MODEL_NAMES:
            theta_true = _theta_true(model)
            names = PARAM_NAMES[model]
            for t_len in MC_SAMPLE_SIZES:
                t0 = time.time()
                seeds = rng_master.integers(0, 2**31 - 1, size=n_replications)
                jobs = [(model, theta_true, t_len, s0, gamma_bar, int(seed)) for seed in seeds]
                thetas = list(pool.map(_one_replication, jobs, chunksize=1))
                thetas = [t for t in thetas if t is not None]
                elapsed = time.time() - t0
                print(f"[exp1] {model} T={t_len}: {len(thetas)}/{n_replications} "
                      f"converged replications in {elapsed:.1f}s")
                if not thetas:
                    continue
                theta_mat = np.array(thetas)
                estimates_by_cell[(model, t_len)] = theta_mat
                for i, name in enumerate(names):
                    summary = summarize_replications(theta_mat[:, i], theta_true[i])
                    rows.append({
                        "model": model, "parameter": name, "T": t_len,
                        "true_value": theta_true[i], **summary,
                    })

    table = pd.DataFrame(rows)

    smallest_t = {}
    for model in MODEL_NAMES:
        for name in PARAM_NAMES[model]:
            by_t = {t_len: estimates_by_cell[(model, t_len)][:, PARAM_NAMES[model].index(name)]
                    for t_len in MC_SAMPLE_SIZES if (model, t_len) in estimates_by_cell}
            smallest_t[f"{model}:{name}"] = smallest_normal_T(by_t, MC_JB_SIGNIFICANCE)
    return table, {"smallest_normal_T": smallest_t, "data_info": data_info}


def _plot(table: pd.DataFrame, path: Path) -> None:
    models = table["model"].unique()
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3 * len(models)))
    for row, model in enumerate(models):
        sub = table[table["model"] == model]
        for name, grp in sub.groupby("parameter"):
            grp = grp.sort_values("T")
            axes[row, 0].plot(grp["T"], grp["bias"], marker="o", label=name)
            axes[row, 1].plot(grp["T"], grp["rmse"], marker="o", label=name)
        axes[row, 0].axhline(0.0, color="black", linewidth=0.5)
        axes[row, 0].set_title(f"{model}: bias")
        axes[row, 1].set_title(f"{model}: RMSE")
        for ax in axes[row]:
            ax.set_xlabel("T")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's 500 replications per cell instead of the reduced run")
    args = parser.parse_args()
    n_replications = MC_REPLICATIONS_PAPER if args.full else MC_REPLICATIONS_RUN

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table, extra = run_monte_carlo(n_replications)
    elapsed = time.time() - t0

    table.to_csv(RESULTS_DIR / "exp1_monte_carlo.csv", index=False)
    _plot(table, RESULTS_DIR / "exp1_bias_rmse.png")

    log = {
        "n_replications_used": n_replications,
        "n_replications_paper": MC_REPLICATIONS_PAPER,
        "sample_sizes": MC_SAMPLE_SIZES,
        "seed": MC_SEED,
        "elapsed_seconds": elapsed,
        **extra,
    }
    (RESULTS_DIR / "exp1_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"[exp1] done in {elapsed:.1f}s, results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
