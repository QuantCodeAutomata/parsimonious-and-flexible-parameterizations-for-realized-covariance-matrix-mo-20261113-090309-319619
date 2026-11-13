# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this implements

Replication of "Parsimonious and Flexible Parameterizations for Realized
Covariance Matrix Models":

- **Experiment 1** (Section 4.1): Monte Carlo study of the finite-sample
  properties (bias, RMSE, Jarque-Bera normality) of the two-step QML
  estimator for the six coefficient-matrix parameterizations of model (2.3) —
  sc-BEKK, Log-BEKK, Lin-BEKK, HE-BEKK, Log-HE-BEKK, Lin-HE-BEKK.
- **Experiment 2** (Section 5): estimation of the same six models plus EWMA
  and vecHAR benchmarks on daily realized covariance matrices of DJIA
  constituents, with in-sample/out-of-sample comparison via log-likelihood,
  AIC/BIC, Quasi-Likelihood and Global Minimum Variance loss, and the Model
  Confidence Set (Hansen, Lunde and Nason, 2011).

## Layout

```
src/     model parameterizations (models.py), QML likelihood (likelihood.py)
         and two-step estimation (estimation.py), metrics (metrics.py), loss
         functions and benchmarks (loss.py), Model Confidence Set (mcs.py),
         all experiment parameters in config.py (module-level variables only)
exp/     exp1.py (Monte Carlo), exp2.py (empirical application); call src/
         and data/ with their documented signatures only
data/    data_loader.py (QuantX Dataloader intraday bars and realized
         covariance construction for exp2) and generate_synthetic.py
         (Wishart DGP simulation for exp1); downloaded data is never
         committed and is cached under data/cache/ (gitignored)
tests/   pytest unit tests (synthetic fixtures) and market_data-marked tests
         (require DATALOADER_API_KEY/MASSIVE_API_KEY, skipped otherwise)
results/ tables, figures and logs written by exp1/exp2, plus RESULTS.md
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for exp2

python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, needs API keys for exp2

python -m exp.exp1            # Monte Carlo, reduced replication count
python -m exp.exp1 --full     # paper configuration (500 replications per T)
python -m exp.exp2            # empirical application, executed configuration
python -m exp.exp2 --full     # paper's full 30-stock, 2015-2026 configuration

bash reproduce.sh             # tests + both experiments with the commands above
```

## Conventions

- All experiment and model parameters live only in `src/config.py` as
  module-level `UPPER_CASE` variables, each with a comment on meaning, unit
  and paper source; no parameter classes or dicts, and no parameters hidden
  in `exp/` or `data/`.
- Market data is loaded only through `data/data_loader.py` (QuantX Dataloader
  REST API, `POST /fetch-ohlcvs`); synthetic Monte Carlo data is generated
  only through `data/generate_synthetic.py`. No other module fetches or
  generates data.
- `exp/exp1.py` and `exp/exp2.py` call the public functions of `src/` and
  `data/` with the argument names and return keys those modules define; do
  not add keyword arguments they do not accept.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` via `python-dotenv`; `.env.example` holds only empty
  placeholders and must never contain a real key.
- Running an experiment overwrites the corresponding files in `results/`;
  after a rerun, check the new files and update the numbers and verdicts in
  `results/RESULTS.md` by hand — they are not regenerated automatically.
- Unit tests use small synthetic fixtures; tests marked `market_data` use a
  small, cached real-data slice and are skipped automatically when the API
  keys are not set. Run `python -m pytest -m "not market_data"` for routine
  checks and the full suite (including `market_data` tests) after a change to
  the data path or before relying on results from real data.
