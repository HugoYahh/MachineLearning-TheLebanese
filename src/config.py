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
    # "TICKER": "Sector",
}
TICKERS = list(TICKER_SECTORS)
FRED_SERIES = ["VIXCLS", "T10Y2Y"]

# --- Dates ---
DOWNLOAD_START = "2009-01-01"   # warm-up
DOWNLOAD_END = "2026-01-01"     # excluded by yfinance

TRAIN_START, TRAIN_END = "2010-01-01", "2018-12-31"
VAL_START, VAL_END = "2019-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2025-12-31"

# --- Target and split ---
HORIZON = 5                     # trading days
EMBARGO = 5                     # trading days
DRAWDOWN_THRESHOLD = None       # X: set after EDA on train only
N_FOLDS = None                  # walk-forward folds, to decide

# --- Dataset schema ---
ID_COLS = ["date", "ticker", "sector"]
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_22d",
    "park_vol_5d", "park_vol_22d",
    "dist_high_252d", "volume_ratio_20d",
    "vix", "vix_chg_5d", "t10y2y",
]
TARGET_COL = "y"
B2_FEATURES = None              # subset to decide

# --- Evaluation: set before seeing any result ---
TARGET_PRECISION = None         # for recall at fixed precision
COST_FN = None                  # cost of a missed drawdown
COST_FP = None                  # cost of a false alarm