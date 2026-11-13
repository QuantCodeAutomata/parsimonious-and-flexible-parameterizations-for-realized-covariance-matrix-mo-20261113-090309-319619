"""Model Confidence Set (Hansen, Lunde and Nason, 2011), used in Section 5 to
select the set of models whose forecast losses are not significantly
dominated at the 10% and 25% significance levels.

Custom implementation (paper Section 5): the MCS elimination procedure with
the semi-quadratic (``T_SQ``) statistic and a block bootstrap is implemented
directly from Hansen, Lunde and Nason (2011); no installed package provides
the Hansen-Lunde-Nason MCS test (the `arch` package's bootstrap utilities are
used only for the underlying stationary block resampling).
"""
from __future__ import annotations

import numpy as np
from arch.bootstrap import StationaryBootstrap

from src.config import MCS_BLOCK_LENGTH


def _stationary_bootstrap_indices(t_len: int, n_boot: int, block_length: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """``n_boot`` resampled index paths of length ``t_len`` drawn with the
    stationary bootstrap (Politis and Romano, 1994), via `arch`'s
    `StationaryBootstrap` resampler."""
    base = np.arange(t_len)
    bootstrap = StationaryBootstrap(block_length, base, seed=rng)
    paths = np.empty((n_boot, t_len), dtype=int)
    for i, data in enumerate(bootstrap.bootstrap(n_boot)):
        paths[i] = data[0][0]
    return paths


def model_confidence_set(loss_matrix: np.ndarray, model_names: list[str],
                          alpha: float, n_boot: int, block_length: float = MCS_BLOCK_LENGTH,
                          seed: int = 0) -> dict:
    """Model Confidence Set procedure of Hansen, Lunde and Nason (2011).

    Parameters
    ----------
    loss_matrix: array (T, M) of per-period losses for the M candidate models.
    alpha: significance level (Section 5 uses 10% and 25%).
    n_boot: number of bootstrap replications (Section 5, footnote 9: "10,000
        bootstrap replications"; ``MCS_BOOTSTRAP_REPS_RUN`` in src.config may
        reduce this for the executed run, see results/RESULTS.md).

    Returns a dict with ``superior_set`` (model names not eliminated at
    ``alpha``) and ``eliminated`` (model name -> elimination p-value), using
    the semi-quadratic statistic T_SQ = max_{i<j} (d_bar_ij)^2 / var(d_bar_ij).
    """
    rng = np.random.default_rng(seed)
    t_len, m = loss_matrix.shape
    active = list(range(m))
    names = list(model_names)
    elimination_pvalues: dict[str, float] = {}
    boot_paths = _stationary_bootstrap_indices(t_len, n_boot, block_length, rng)

    while len(active) > 1:
        losses = loss_matrix[:, active]
        m_active = len(active)
        d_bar = np.array([
            [np.mean(losses[:, i] - losses[:, j]) for j in range(m_active)]
            for i in range(m_active)
        ])
        boot_losses = losses[boot_paths]  # shape (n_boot, T, m_active)
        boot_d_bar = boot_losses.mean(axis=1, keepdims=True) - boot_losses.mean(axis=1, keepdims=True).transpose(0, 2, 1)
        boot_mean = boot_d_bar.mean(axis=0)
        var_d = np.var(boot_d_bar - boot_mean[None, :, :] + d_bar[None, :, :], axis=0)
        var_d[var_d < 1e-12] = 1e-12
        t_stat = d_bar ** 2 / var_d
        np.fill_diagonal(t_stat, 0.0)
        t_sq = t_stat.max()
        boot_t_sq = np.zeros(n_boot)
        for b in range(n_boot):
            # Recenter each bootstrap replication on the bootstrap mean (not
            # on the observed d_bar) so that boot_t_sq approximates the
            # sampling distribution of the statistic under the null of equal
            # expected loss (Hansen, Lunde and Nason, 2011, Section 3).
            bd = boot_d_bar[b] - boot_mean
            bt = bd ** 2 / var_d
            np.fill_diagonal(bt, 0.0)
            boot_t_sq[b] = bt.max()
        p_value = float(np.mean(boot_t_sq >= t_sq))

        if p_value >= alpha or m_active <= 1:
            break
        # Eliminate the model with the largest average relative loss (Hansen
        # et al., 2011, elimination rule based on the model with the
        # largest standardized distance from the others).
        avg_rel = d_bar.sum(axis=1) / max(m_active - 1, 1)
        worst_local = int(np.argmax(avg_rel))
        worst_global = active[worst_local]
        elimination_pvalues[names[worst_global]] = p_value
        active.remove(worst_global)

    superior = [names[i] for i in active]
    return {"superior_set": superior, "eliminated_pvalues": elimination_pvalues}
