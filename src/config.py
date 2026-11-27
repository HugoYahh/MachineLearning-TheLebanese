from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
SNAPSHOT_DATE = "2026-11-27"

SEED = 42   # Use this seed to make random operations reproducible.

# Universe: {ticker: sector}, 40 stocks 
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
DOWNLOAD_START = "2009-01-01"  
DOWNLOAD_END = "2026-01-01"    

TRAIN_START, TRAIN_END = "2010-01-01", "2018-12-31"
VAL_START, VAL_END = "2019-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2025-12-31"

# Target and split 
HORIZON = 5                    # Number of future trading sessions used to build the target.
EMBARGO = 5                    # Drop the last five trading sessions from train and validation so labels stay within each split.
DRAWDOWN_THRESHOLD = 0.05      # Choose the drop threshold (5% or 7%) using only the training set. Finally, we chose 5% because it is more common and allows for more training data.
N_FOLDS = 4                 # Number of walk-forward splits within the training set, to be decided.

# Dataset schema 
# Columns expected in the dataset. Each row represents one stock on one trading day.
ID_COLS = ["date", "ticker", "sector"]
FEATURE_COLS = ["ret_1d", "ret_5d", "ret_22d","park_vol_5d", "park_vol_22d","dist_high_252d", "volume_ratio_20d","vix", "vix_chg_5d", "t10y2y",]
TARGET_COL = "y"
B2_FEATURES = FEATURE_COLS.copy()            # Select after training-set EDA

# --- Evaluation ---
# Set the target precision to twice the prevalence in the training set.
TARGET_PRECISION_MULTIPLIER = 2.0

# A missed drawdown is assumed to cost five times as much as a false alarm.
# For both B1 and B2, select the threshold that minimizes this cost on validation.
COST_FN = 5
COST_FP = 1