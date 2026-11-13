"""Shared pytest fixtures.

Synthetic covariance data for unit tests is generated here (small, fixed
dimension) rather than in ``data/``, since no experiment of this repository
uses synthetic data as its reported result -- synthetic series are only a
Monte Carlo data-generating process consumed directly by exp.exp1 through
``data/generate_synthetic.py``.

Market-data tests load one reduced slice of DJIA intraday bars through
``data/data_loader.py`` and cache it across test runs via pytest's cache.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load real market data (skipped without API keys)"
    )


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def small_gamma_bar(rng):
    """A small (n=4) positive-definite target covariance used as Gamma_bar."""
    n = 4
    a = rng.normal(size=(n, n))
    return (a @ a.T) / n + np.eye(n) * 0.5


@pytest.fixture(scope="session")
def small_s0(small_gamma_bar):
    return small_gamma_bar.copy()


@pytest.fixture(scope="session")
def small_gamma_series(small_gamma_bar, small_s0, rng):
    """A short synthetic sc-BEKK realized-covariance series (n=4, T=40) for
    fast unit tests of the shared estimation/likelihood pipeline."""
    from data.generate_synthetic import simulate_series

    theta = np.array([0.1, 0.3, 0.5])  # kappa, a, b (sc-BEKK order)
    gamma_series, s_series = simulate_series(
        "sc-BEKK", theta, n=4, nu=30, t_len=40, s0=small_s0,
        gamma_bar=small_gamma_bar, burn_in=20, rng=rng,
    )
    return gamma_series, s_series


@pytest.fixture(scope="session")
def djia_slice(request):
    """A short, cached real-data slice: 3 DJIA tickers, 2 months of 5-minute
    bars, used only by tests marked ``market_data``. Skips when
    DATALOADER_API_KEY is absent."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    cache_dir = request.config.cache.mkdir("djia_slice")
    bars_path = cache_dir / "bars.csv"
    from data.data_loader import load_ohlcv

    import pandas as pd

    if bars_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    else:
        bars, _meta = load_ohlcv(
            ["AAPL", "MSFT", "JNJ"], "2024-01-02", "2024-02-29", timeframe="5m",
            identity_date="2024-02-29",
        )
        bars.to_csv(bars_path, index=False)
    return bars
