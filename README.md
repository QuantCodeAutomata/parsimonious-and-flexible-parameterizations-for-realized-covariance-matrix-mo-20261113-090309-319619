# Parsimonious and Flexible Parameterizations for Realized Covariance Matrix Models

Replication of the paper "Parsimonious and Flexible Parameterizations for
Realized Covariance Matrix Models", implementing:

- **Experiment 1** (Section 4.1, "Monte Carlo Evidence"): a Monte Carlo study
  of the finite-sample properties (bias, RMSE, normality) of the two-step QML
  estimator for six coefficient-matrix parameterizations of model (2.3):
  sc-BEKK, Log-BEKK, Lin-BEKK, HE-BEKK, Log-HE-BEKK and Lin-HE-BEKK.
- **Experiment 2** (Section 5, "Empirical Application"): estimation and
  in-sample/out-of-sample forecasting comparison of the six models plus the
  EWMA and vecHAR benchmarks on daily realized covariance matrices of DJIA
  constituents, with model selection by AIC/BIC and the Model Confidence Set
  (Hansen, Lunde and Nason, 2011).

## Repository layout

```
src/     model parameterizations, QML likelihood/estimation, metrics, loss
         functions, Model Confidence Set, experiment parameters (config.py)
exp/     experiment scripts (exp1: Monte Carlo, exp2: empirical application)
data/    data/generate_synthetic.py (Monte Carlo DGP) and data/data_loader.py
         (DJIA intraday bars and realized-covariance construction)
tests/   pytest unit and market-data tests
results/ saved tables, figures and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for exp2
```

## Tests

```bash
python -m pytest -m "not market_data"     # unit tests, no network access
python -m pytest                          # full suite, requires API keys for exp2
```

## Running the experiments

```bash
python -m exp.exp1                 # Monte Carlo study, reduced replications
python -m exp.exp1 --full          # paper configuration (500 replications)
python -m exp.exp2                 # empirical application, executed configuration
python -m exp.exp2 --full          # paper's full 30-stock, 2015-2026 configuration
```

All generated tables, figures and logs are written directly to `results/`.
See `results/RESULTS.md` for the comparison against the paper's reported
figures, deviations and items not reproduced.

## Reproduce everything

```bash
bash reproduce.sh
```
