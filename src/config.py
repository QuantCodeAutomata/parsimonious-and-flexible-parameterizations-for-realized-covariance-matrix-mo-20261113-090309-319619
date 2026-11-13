"""Model and experiment parameters for the realized-covariance BEKK family.

All numeric parameters that define the method or the paper's reported figures
live here as module-level variables. Each has a comment with its meaning,
unit, and the paper section/equation it is taken from (Section 2 "The Model
Framework", Section 3 "Feasible Parameterizations", Section 4 "Estimator
Properties", Section 5 "Empirical Application").
"""

# ---------------------------------------------------------------------------
# Common model dimension (Section 4.1 Monte Carlo Evidence / Section 5)
# ---------------------------------------------------------------------------
N_ASSETS = 30          # dimension n of the covariance matrices (Section 4.1, Section 5)
NU = 3 * N_ASSETS      # Wishart degrees of freedom, nu = 3n = 90 (Section 4.1, footnote 5)

MODEL_NAMES = [
    "sc-BEKK",
    "Log-BEKK",
    "Lin-BEKK",
    "HE-BEKK",
    "Log-HE-BEKK",
    "Lin-HE-BEKK",
]

# Free parameters of A_t (and kappa, b) for each parameterization, in the
# order used by src.models.unpack_params (Section 3, equations 3.1-3.8).
PARAM_NAMES = {
    "sc-BEKK": ["kappa", "a", "b"],
    "Log-BEKK": ["kappa", "a0", "a1", "phi", "c", "b"],
    "Lin-BEKK": ["kappa", "alpha_v", "alpha_c", "alpha_a", "b"],
    "HE-BEKK": ["kappa", "a", "varphi", "b"],
    "Log-HE-BEKK": ["kappa", "a", "a1", "phi", "c", "varphi", "b"],
    "Lin-HE-BEKK": ["kappa", "a", "varphi", "alpha_a", "b"],
}

# True parameter values used as the Monte Carlo data-generating parameters,
# taken from Table 2 (Section 5, in-sample estimates of the empirical
# application on the DJIA realized covariance series).
TRUE_PARAMS = {
    "sc-BEKK": {"kappa": 0.089, "a": 0.381, "b": 0.554},
    "Log-BEKK": {"kappa": 0.144, "a0": 0.312, "a1": 0.163, "phi": 10.318, "c": 0.780, "b": 0.355},
    "Lin-BEKK": {"kappa": 0.111, "alpha_v": 0.477, "alpha_c": 0.338, "alpha_a": 1.439, "b": 0.381},
    "HE-BEKK": {"kappa": 0.109, "a": 0.492, "varphi": 0.763, "b": 0.362},
    "Log-HE-BEKK": {"kappa": 0.131, "a": 0.419, "a1": 0.069, "phi": 15.062, "c": 1.994, "varphi": 0.491, "b": 0.350},
    "Lin-HE-BEKK": {"kappa": 0.109, "a": 0.492, "varphi": 0.764, "alpha_a": 0.128, "b": 0.362},
}

# ---------------------------------------------------------------------------
# Experiment 1: Monte Carlo evidence (Section 4.1)
# ---------------------------------------------------------------------------
MC_SAMPLE_SIZES = [100, 500, 1000, 2000]   # T values used in Table 1 (Section 4.1)
MC_REPLICATIONS_PAPER = 500                # replications per (model, T) cell in the paper (Section 4.1)
MC_REPLICATIONS_RUN = 12                   # replications executed by default; see "Deviations from the paper"
MC_WORKERS = 4                              # parallel worker processes for the Monte Carlo study
MC_JB_SIGNIFICANCE = 0.01                  # Jarque-Bera significance level for the normality check (Section 4.1)
MC_SEED = 20261113                         # seed for the Monte Carlo random generator (not specified in the paper)
MC_BURNIN = 50                             # recursion burn-in periods before recording Gamma_t (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 2: Empirical application on DJIA realized covariance (Section 5)
# ---------------------------------------------------------------------------
# Paper's own sample period and universe (Section 5): daily annualized realized
# covariance matrices of the 30 DJIA constituents, Jan 2, 2015 to May 29, 2026,
# in-sample through end of 2025 (T=2765), out-of-sample 102 observations of 2026.
DJIA_START_DATE_PAPER = "2015-01-02"
DJIA_END_DATE_PAPER = "2026-05-29"
DJIA_IN_SAMPLE_END_PAPER = "2025-12-31"
DJIA_T_IN_SAMPLE_PAPER = 2765
DJIA_T_OUT_OF_SAMPLE_PAPER = 102
DJIA_UNIVERSE_SIZE_PAPER = 30

# Executed configuration: realized covariance matrices are not available from
# the Dataloader/Massive data path (the paper's source, the VOLARE platform,
# is not reachable from this environment) and must instead be built from
# intraday bars using the realized-covariance estimator of Barndorff-Nielsen
# and Shephard (2004). A timed download of all 30 constituents' 5-minute bars
# over three months (213,542 rows, 34.9 seconds) scales to an estimated 26.6
# minutes for the full Jan 2, 2015 - May 29, 2026 window, well inside the
# download-time budget, so the full 30-stock universe and the paper's full
# sample period are both used (see results/RESULTS.md for the measured
# timing and realized download log).
DJIA_TICKERS = [
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JPM", "JNJ", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT",
]  # 30 DJIA constituents (not specified in the paper beyond the AAPL/AMGN examples)
DJIA_UNIVERSE_SIZE_RUN = 30          # number of DJIA constituents used in the executed run
RC_BAR_TIMEFRAME = "5m"              # intraday bar size used to build realized covariance (standard RV estimator)
DJIA_IDENTITY_DATE = "2026-05-29"    # point-in-time date for ticker identity (last date of the sample period)
DJIA_START_DATE_RUN = DJIA_START_DATE_PAPER  # executed sample start equals the paper's sample start
DJIA_END_DATE_RUN = DJIA_END_DATE_PAPER      # executed sample end equals the paper's sample end
DJIA_IN_SAMPLE_END_RUN = DJIA_IN_SAMPLE_END_PAPER  # executed in-sample/out-of-sample split equals the paper's

EWMA_DECAY = 0.96          # EWMA decay parameter (Section 5: "decay parameter equal to 0.96")
VECHAR_DAILY_LAG = 1       # vecHAR daily component lag, in trading days (Corsi 2009 HAR structure)
VECHAR_WEEKLY_LAG = 5      # vecHAR weekly component window, in trading days
VECHAR_MONTHLY_LAG = 22    # vecHAR monthly component window, in trading days

MCS_ALPHA_LEVELS = [0.10, 0.25]   # Model Confidence Set significance levels (Section 5)
MCS_BOOTSTRAP_REPS_PAPER = 10_000  # bootstrap replications used by the paper (Section 5, footnote 9)
MCS_BOOTSTRAP_REPS_RUN = 2_000      # bootstrap replications executed by default; see "Deviations from the paper"
MCS_BLOCK_LENGTH = 10               # stationary-bootstrap block length (not specified in the paper)

# ---------------------------------------------------------------------------
# QML optimization settings (not specified in the paper)
# ---------------------------------------------------------------------------
OPTIMIZER_METHOD = "SLSQP"
OPTIMIZER_MAXITER = 60  # capped to bound worst-case runtime of the Monte Carlo study
OPTIMIZER_FTOL = 1e-8
