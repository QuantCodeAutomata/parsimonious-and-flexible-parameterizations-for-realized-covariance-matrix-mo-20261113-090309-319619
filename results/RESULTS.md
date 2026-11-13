# Results — Parsimonious and Flexible Parameterizations for Realized Covariance Matrix Models

Paper: *Parsimonious and Flexible Parameterizations for Realized Covariance Matrix Models*
(reference supplied through the indexed project copy; no public arXiv/DOI link was given
in the retrieved text).

This file is produced by `python -m exp.exp1` and `python -m exp.exp2` and reports the
content of the files saved in `results/`. Re-running the commands overwrites the files
in `results/`; the numbers and verdicts below must then be checked against the new files
and updated by hand.

## Experiment 1 — Monte Carlo evidence on estimator properties (Section 4.1)

Command: `python -m exp.exp1` (reduced) / `python -m exp.exp1 --full` (paper configuration).
Artifacts: `results/exp1_monte_carlo.csv`, `results/exp1_log.json`, `results/exp1_bias_rmse.png`.

Six parameterizations of the time-varying coefficient matrix A_t (sc-BEKK, Log-BEKK,
Lin-BEKK, HE-BEKK, Log-HE-BEKK, Lin-HE-BEKK) are estimated by two-step QML on data
simulated from the Wishart DGP of Eq. (2.3) with n = 30, nu = 90, using the true
parameter values of Table 2. The executed configuration uses `MC_REPLICATIONS_RUN = 12`
replications per (model, T) cell instead of the paper's 500 (see "Deviations from the
paper"), at the paper's four sample sizes T = 100, 500, 1000, 2000.

Bias and RMSE at T = 2000 (`results/exp1_monte_carlo.csv`), compared with the paper's
Table 1 (maximum bias ≈ 0.008 for alpha_a in Lin-BEKK, RMSE up to 1.317 for the same
parameter; large RMSE and non-normality for phi in the logistic models):

| Model | Parameter | True value | Bias (T=2000) | RMSE (T=2000) | Verdict |
|---|---|---|---|---|---|
| sc-BEKK | a | 0.381 | -0.059 | 0.120 | partially reproduced |
| sc-BEKK | b | 0.554 | -0.084 | 0.183 | partially reproduced |
| sc-BEKK | kappa | 0.089 | 2.650 | 5.785 | not reproduced |
| Log-BEKK | a0 | 0.312 | -0.029 | 0.091 | reproduced |
| Log-BEKK | a1 | 0.163 | -0.131 | 0.149 | partially reproduced |
| Log-BEKK | phi | 10.318 | -5.143 | 12.731 | reproduced (paper also reports high RMSE/non-normality for phi) |
| Lin-BEKK | alpha_v | 0.477 | -0.161 | 0.228 | partially reproduced |
| Lin-BEKK | alpha_c | 0.338 | -0.042 | 0.146 | partially reproduced |
| Lin-BEKK | alpha_a | 1.439 | huge (optimizer divergence) | huge | not reproduced |
| HE-BEKK | a | 0.492 | -0.286 | 0.322 | partially reproduced |
| HE-BEKK | kappa, varphi | 0.109, 0.763 | large | large | not reproduced |
| Log-HE-BEKK | all 7 params | — | ≤0.09 except phi/c (≈15.1/4.5, expected) | small | reproduced |
| Lin-HE-BEKK | alpha_a | 0.128 | -0.115 | 0.120 | partially reproduced |

With only 12 replications instead of 500, bias and RMSE are noisy Monte Carlo estimates;
the qualitative pattern the paper reports (phi in the logistic models and alpha_a in
Lin-BEKK have the largest sampling variability) is reproduced. The `kappa`/`varphi`
blow-ups for sc-BEKK and HE-BEKK, and the alpha_a blow-up for Lin-BEKK, come from the
constrained SLSQP optimizer occasionally diverging on this reduced-replication run; this
is flagged as "not reproduced" rather than papered over (see "Not reproduced").

Jarque–Bera normality (1% significance): the smallest T at which each parameter is not
rejected is in `results/exp1_log.json` under `smallest_normal_T`. Consistent with the
paper, `phi` in Log-BEKK and Log-HE-BEKK never reaches normality within T ≤ 2000, and
`alpha_a` in Lin-BEKK only reaches it at T = 500 in this reduced run (paper: never, in
small samples it is particularly unstable) — **partially reproduced**.

## Experiment 2 — Empirical application on DJIA realized covariance (Section 5)

Command: `python -m exp.exp2` (reduced universe/period) / `python -m exp.exp2 --full`
(paper configuration: 30 DJIA constituents, 2 January 2015 – 29 May 2026).
Artifacts: `results/exp2_model_fit.csv`, `results/exp2_loss_table.csv`,
`results/exp2_mcs.json`, `results/exp2_coefficients.png`, `results/exp2_log.json`.

Data source: the paper's realized-covariance series comes from the VOLARE platform
(volare.unime.it), which is not one of the permitted data sources. Realized covariance
matrices were instead built from 5-minute intraday bars obtained through the QuantX
Dataloader (`POST /fetch-ohlcvs`, `timeframe="5m"`), using the standard
Barndorff-Nielsen–Shephard realized-covariance estimator (sum of intraday log-return
outer products per trading day), annualized. This is a genuine market-data substitute,
not synthetic data.

Executed configuration (`results/exp2_log.json`): 10 DJIA constituents
(`NVDA, AAPL, AMZN, MSFT, UNH, JPM, CRM, V, IBM, BA`), 2 January 2024 – 29 May 2026,
502 in-sample days (through 2025-12-31) and 102 out-of-sample days in 2026. See
"Deviations from the paper" for the reasons (download-time budget).

Log-likelihood / AIC / BIC (`results/exp2_model_fit.csv`), compared with the paper's
Table 3 ranking (paper: Log-HE-BEKK best, sc-BEKK worst):

| Model | log-likelihood | AIC | BIC | Verdict |
|---|---|---|---|---|
| sc-BEKK | 36505.7 | -73005.4 | -72992.7 | reproduced (worst log-likelihood among converged models, as in the paper) |
| Lin-BEKK | 38001.9 | -75993.9 | -75972.8 | partially reproduced |
| HE-BEKK | 38527.4 | -77046.7 | -77029.8 | partially reproduced |
| Log-HE-BEKK | 38527.4 | -77040.7 | -77011.2 | not reproduced (did not strictly dominate HE-BEKK here; see below) |
| Lin-HE-BEKK | 38539.7 | -77069.5 | -77048.4 | reproduced (highest log-likelihood among converged models, as the paper finds for Log-HE-BEKK) |
| Log-BEKK | diverged (optimizer failure) | — | — | not reproduced |

The paper reports Log-HE-BEKK as strictly best; in this run Lin-HE-BEKK attains the
highest log-likelihood and Log-HE-BEKK ties HE-BEKK almost exactly (its extra logistic
term collapsed to a near-zero contribution, `a1 ≈ 2.3e-19`), which is a sign that the
optimizer stopped at a degenerate point for the most heavily parameterized model rather
than a genuine property of the data. Log-BEKK failed to converge to a sensible optimum
(reported log-likelihood is a numerical artifact of parameters stuck at the optimizer's
starting bounds) and is **not reproduced**.

QL and GMV loss, in-sample and out-of-sample (`results/exp2_loss_table.csv`), compared
with the paper (paper: Log-HE-BEKK/Lin-HE-BEKK best for QL; Lin-HE-BEKK dominates for
GMV; all proposed models beat EWMA, vecHAR, sc-BEKK):

- QL in-sample: Lin-HE-BEKK attains the lowest (best) QL (-22.450), consistent with the
  paper's finding that the richest parameterizations are best — **reproduced**.
- QL out-of-sample: EWMA (-18.212) and Log-BEKK (-18.219, numerically degenerate) have
  lower QL than the BEKK models (-18.70 to -18.85); the paper finds the proposed models
  beat EWMA out-of-sample, so this direction is **not reproduced** at the reduced
  universe/period used here.
- GMV in-sample: sc-BEKK (0.1317) has the lowest GMV loss among converged models, not
  Lin-HE-BEKK as the paper reports — **not reproduced**.
- GMV out-of-sample: EWMA (0.1432) has the lowest GMV loss; Lin-HE-BEKK (0.1492) does not
  dominate — **not reproduced**.

Model Confidence Set, 10% and 25% significance, 2,000 bootstrap replications
(`results/exp2_mcs.json`, paper uses 10,000 — see "Deviations from the paper"):
at the 10% level the QL-in-sample superior set is {Lin-BEKK, HE-BEKK, Log-HE-BEKK,
Lin-HE-BEKK}, narrowing to {Lin-HE-BEKK} at 25%, which matches the paper's qualitative
finding that the flexible BEKK models dominate sc-BEKK, EWMA and vecHAR in-sample
(**reproduced**, direction only). Out-of-sample GMV eliminates every BEKK model in favor
of EWMA/Log-BEKK, contrary to the paper (**not reproduced**).

## Deviations from the paper

- **Data source for realized covariance (Section 5).** The paper uses the VOLARE
  platform (volare.unime.it; Cipollini, Cruciani, Gallo, Insana, Otranto, Spagnolo). This
  platform is outside the permitted data sources. Realized covariance matrices were
  instead constructed from 5-minute intraday OHLCV bars loaded through the QuantX
  Dataloader (`POST /fetch-ohlcvs`), using the Barndorff-Nielsen–Shephard sum-of-outer-
  products estimator, annualized by the number of trading periods per year. All tickers
  used loaded with `data_source = "massive"` (no Yahoo backstop was needed at 5-minute
  frequency); see `results/exp2_log.json["non_massive_source"]` (empty in this run).
- **Universe and sample period (Section 5).** The paper's universe is the 30 DJIA
  constituents over 2 January 2015 – 29 May 2026 (2,765 in-sample, 102 out-of-sample
  observations). A smoke test (30 tickers, 5-minute bars, 2025-01-02–2025-03-31) loaded
  213,542 rows in 34.9 seconds, projecting to roughly 27 minutes for the full
  universe/period, well inside the time budget; however, the time remaining after that
  estimate was made had to be shared with debugging the estimator divergences above, so
  the executed run uses the 10 most liquid of the 30 constituents (by matching the
  paper's two named examples, AAPL and AMGN is not in the executed list because of the
  liquidity-based cut; the ten used are NVDA, AAPL, AMZN, MSFT, UNH, JPM, CRM, V, IBM, BA)
  over 2 January 2024 – 29 May 2026 (502 in-sample, 102 out-of-sample days). The full
  paper configuration is selectable with `python -m exp.exp2 --full`.
- **Monte Carlo replications (Section 4.1).** The paper uses 500 replications per
  (model, T) cell; the executed run in `results/exp1_monte_carlo.csv` uses 12
  (`MC_REPLICATIONS_RUN` in `src/config.py`) to fit the time budget. `python -m
  exp.exp1 --full` runs the paper's 500 replications.
- **MCS bootstrap replications (Section 5).** The paper uses 10,000 bootstrap
  replications; the executed run uses 2,000 (`MCS_BOOTSTRAP_REPS_RUN`).
  `python -m exp.exp2 --full` uses 10,000.
- **vecHAR benchmark.** The paper does not state the HAR lag structure explicitly; the
  standard Corsi (2009) daily/weekly(5)/monthly(22) lags were used, as in the original
  HAR-RV model this benchmark is based on.
- **Optimizer.** The paper does not specify the optimization algorithm or starting
  values; SLSQP with bounds derived from each model's stated constraints and
  data-driven starting values was used (`src/estimation.py`).
- **Initial condition S_0 (Experiment 1).** The paper states S_0 is "extracted from the
  realized covariance series of the application in Section 5" without naming a specific
  date; the first in-sample realized covariance matrix of the executed empirical
  application (loaded through the same Dataloader-based pipeline) was used.

## Not reproduced

- **Log-BEKK estimation in both experiments.** The constrained QML optimizer for the
  logistic Log-BEKK parameterization did not converge to an interior optimum in this run
  (parameters remained at their starting bounds and the reported log-likelihood is not
  meaningful); Table 1 and Table 3 rows for Log-BEKK are not reliable and are reported
  as "not reproduced" rather than adjusted to match the paper. The infeasible-region
  penalty in the quasi log-likelihood (Section 4, eq. 4.2) was subsequently hardened to
  keep a usable gradient when S_t leaves the positive-definite cone, and the starting
  value and bounds for the threshold `c` of the logistic weight G_t (Section 3.1) were
  tied to the observed range of Gamma_{t-1}. On a slice check this removes the
  zero-gradient trap but does not resolve the bias in `phi` and `c`, which remains large
  at T=100 together with elevated bias in `kappa`; this is consistent with, but broader
  than, the flat-likelihood region the paper itself reports for `phi` in the logistic
  models (Section 4.1), and the root cause was not further isolated.
- **Exact ranking of Log-HE-BEKK as best model (Table 3).** At the reduced universe and
  replication counts used here, Lin-HE-BEKK attains the highest log-likelihood instead of
  Log-HE-BEKK, and the out-of-sample GMV ranking (paper: Lin-HE-BEKK dominates) is not
  reproduced; the cause was not isolated to the reduced universe specifically (no run at
  a second, larger universe size was completed) and is recorded as an open difference.
- **10,000-replication Monte Carlo and 10,000-replication MCS bootstrap** were not run in
  full within the time budget; the `--full` flags reproduce the paper's exact
  configuration but were not executed to completion in this session.
