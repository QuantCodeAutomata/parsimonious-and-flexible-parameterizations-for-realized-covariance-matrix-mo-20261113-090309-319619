"""Unit tests for src/metrics.py: Monte Carlo bias, RMSE and normality
(Section 4.1, Table 1)."""
from __future__ import annotations

import numpy as np

from src.metrics import (bias, jarque_bera_pvalue, rmse, smallest_normal_T,
                          summarize_replications)


def test_bias_is_zero_for_unbiased_estimates():
    rng = np.random.default_rng(0)
    estimates = 0.5 + rng.normal(scale=0.01, size=5000)
    assert abs(bias(estimates, 0.5)) < 0.01


def test_rmse_nonnegative_and_zero_for_exact_estimates():
    estimates = np.full(10, 0.5)
    assert rmse(estimates, 0.5) == 0.0
    assert rmse(np.array([0.4, 0.6]), 0.5) > 0.0


def test_rmse_decreases_with_more_precise_estimates():
    rng = np.random.default_rng(1)
    wide = 0.5 + rng.normal(scale=0.5, size=2000)
    narrow = 0.5 + rng.normal(scale=0.05, size=2000)
    assert rmse(narrow, 0.5) < rmse(wide, 0.5)


def test_jarque_bera_pvalue_high_for_normal_sample():
    rng = np.random.default_rng(2)
    normal_sample = rng.normal(size=2000)
    assert jarque_bera_pvalue(normal_sample) > 0.01


def test_jarque_bera_pvalue_low_for_skewed_sample():
    rng = np.random.default_rng(3)
    skewed_sample = rng.exponential(size=2000)
    assert jarque_bera_pvalue(skewed_sample) < 0.01


def test_jarque_bera_pvalue_handles_degenerate_sample():
    assert jarque_bera_pvalue(np.full(10, 1.0)) == 0.0


def test_smallest_normal_T_picks_first_passing_sample_size():
    rng = np.random.default_rng(4)
    estimates_by_T = {
        100: rng.exponential(size=500),
        500: rng.normal(size=500),
        1000: rng.normal(size=500),
    }
    assert smallest_normal_T(estimates_by_T, 0.01) == 500


def test_smallest_normal_T_returns_none_when_never_normal():
    estimates_by_T = {100: np.random.default_rng(5).exponential(size=500)}
    assert smallest_normal_T(estimates_by_T, 0.01) is None


def test_summarize_replications_keys():
    summary = summarize_replications(np.array([0.1, 0.2, 0.3]), 0.2)
    assert set(summary) == {"bias", "rmse", "jb_pvalue", "n_replications"}
    assert summary["n_replications"] == 3
