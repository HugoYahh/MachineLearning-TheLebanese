import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TRAIN_START, ID_COLS, FEATURE_COLS


def rolling_by_ticker(table: pd.DataFrame, column: str, window: int, stat: str) -> pd.Series:
   # Helper Function
    # This function split rows using the ticker, then choose a column for example "volume"
    # Then it applies the "stat" calculation asked in the parameter, for the actual row and the window - 1 preceding
    
    return table.groupby("ticker")[column].transform(
        lambda values: values.rolling(window).agg(stat)
    )


def add_features(clean: pd.DataFrame) -> pd.DataFrame:
    
    # Rows must be in time order: shift and rolling rely on it
    table = clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = table.groupby("ticker")

    # 1. Returns over the last 1, 5 and 22 trading days
    for days in [1, 5, 22]:
        past_close = by_ticker["close"].shift(days)
        table[f"ret_{days}d"] = table["close"] / past_close - 1

    # 2. Parkinson volatility (daily scale, not annualized)
    table["park_var"] = np.log(table["high"] / table["low"]) ** 2 / (4 * np.log(2))
    for days in [5, 22]:
        mean_var = rolling_by_ticker(table, "park_var", days, "mean")
        table[f"park_vol_{days}d"] = np.sqrt(mean_var)

    # 3. Distance from the highest close of the last 252 days (0 or negative)
    high_252d = rolling_by_ticker(table, "close", 252, "max")
    table["dist_high_252d"] = table["close"] / high_252d - 1

    # 4. Volume compared with its 20-day average (1 = usual volume)
    mean_volume = rolling_by_ticker(table, "volume", 20, "mean")
    table["volume_ratio_20d"] = table["volume"] / mean_volume

    # 5. VIX change over 5 trading days, in VIX points
    # ("vix" and "t10y2y" are already in the table)
    table["vix_chg_5d"] = table["vix"] - by_ticker["vix"].shift(5)

    return table.drop(columns="park_var")


def remove_warmup(table: pd.DataFrame) -> pd.DataFrame:
    # Keep rows from TRAIN_START and check that no feature is missing.

    # The 2009 rows are only used to fill the rolling windows.
    
    kept = table[table["date"] >= TRAIN_START]

    # replace: an infinite value (division by zero) also counts as missing
    invalid = kept[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).isna().sum()
    if invalid.sum() > 0:
        raise ValueError(f"Invalid feature values:\n{invalid[invalid > 0]}")

    return kept.reset_index(drop=True)


def main() -> None:
    # Build the features and save them.
    clean = pd.read_csv(PROCESSED_DIR / "clean_prices.csv", parse_dates=["date"])

    table = add_features(clean)
    table = remove_warmup(table)

    # "close" is kept ONLY to build the target. It is not a feature.
    table = table[[*ID_COLS, "close", *FEATURE_COLS]]

    path = PROCESSED_DIR / "features.csv"
    table.to_csv(path, index=False)

    print(f"Saved {table.shape} -> {path}")
    print(f"Dates: {table['date'].min().date()} to {table['date'].max().date()}")
    print(table.groupby("ticker").size().agg(["min", "max"]))
    print(f"Missing feature values: {table[FEATURE_COLS].isna().sum().sum()}")


if __name__ == "__main__":
    main()