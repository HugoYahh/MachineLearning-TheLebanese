from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# --- Reproducibility ---
SEED = 42

# --- Universe: {ticker: sector}, 40 stocks ---
TICKER_SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "IBM": "Information Technology",
    "ORCL": "Information Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "WFC": "Financials",
    "JNJ": "Health Care",
    "UNH": "Health Care",
    "PFE": "Health Care",
    "MRK": "Health Care",
    "CAT": "Industrials",
    "DE": "Industrials",
    "HON": "Industrials",
    "UNP": "Industrials",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "FCX": "Materials",
    "SHW": "Materials",
    "APD": "Materials",
    "AMT": "Real Estate",
    "PSA": "Real Estate",
    "O": "Real Estate",
}
TICKERS = list(TICKER_SECTORS)
FRED_SERIES = ["VIXCLS", "T10Y2Y"]

# --- Dates ---
DOWNLOAD_START = "2009-01-01"  # Warm-up
DOWNLOAD_END = "2026-01-01"    # Exclusive in yfinance

TRAIN_START, TRAIN_END = "2010-01-01", "2018-12-31"
VAL_START, VAL_END = "2019-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2025-12-31"

# --- Target and split ---
HORIZON = 5                    # Trading sessions
EMBARGO = 5                    # Trading sessions
DRAWDOWN_THRESHOLD = None      # Set after comparing 5% and 7% on train only
N_FOLDS = None                 # Decide when setting up walk-forward validation

# --- Dataset schema ---
ID_COLS = ["date", "ticker", "sector"]
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_22d",
    "park_vol_5d", "park_vol_22d",
    "dist_high_252d", "volume_ratio_20d",
    "vix", "vix_chg_5d", "t10y2y",
]
TARGET_COL = "y"
B2_FEATURES = None             # Select after training-set EDA

# --- Evaluation ---
# Recall is reported at a precision of at least one true event per four alerts.
TARGET_PRECISION = 0.25

# A missed drawdown is assumed to cost five times as much as a false alarm.
# For both B1 and B2, select the threshold that minimizes this cost on validation.
COST_FN = 5
COST_FP = 1