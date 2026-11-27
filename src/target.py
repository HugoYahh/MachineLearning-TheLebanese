import pandas as pd

from src.config import PROCESSED_DIR, HORIZON, TARGET_COL


def future_min_close(table: pd.DataFrame, horizon: int) -> pd.Series:
    # Return the lowest close in the 5 next business days
    close_by_ticker = table.groupby("ticker")["close"]

    # One column per future day: close at t+1, t+2, ..., t+horizon
    future_closes = []
    for step in range(1, horizon + 1):
        future_closes.append(close_by_ticker.shift(-step))  # negative = future

    future_table = pd.concat(future_closes, axis=1)

    # skipna=False: if one future day is missing, the result is NaN
    return future_table.min(axis=1, skipna=False)


def add_target(features: pd.DataFrame, threshold: float, horizon: int = HORIZON) -> pd.DataFrame:
    # Return the target value : If fall greater than X = 5% for example, y=1, else, y=0
    
    if threshold is None or not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1, e.g. 0.05")

    table = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    future_min = future_min_close(table, horizon)

    # Remove rows with an unknown future BEFORE comparing:
    # a comparison with NaN returns False and would create fake zeros
    known = future_min.notna()
    table = table[known].copy()
    future_min = future_min[known]

    limit = table["close"] * (1 - threshold)
    table[TARGET_COL] = (future_min <= limit).astype(int)

    return table.reset_index(drop=True)


if __name__ == "__main__":
    # Technical check only. 0.05 is NOT the final choice of X.
    # The share of positives is deliberately not printed here:
    # it must be studied on the training set only, after the split.
    data = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    labelled = add_target(data, threshold=0.05)

    print(f"Shape: {labelled.shape}")
    print(f"Rows removed: {len(data) - len(labelled)}")
    print(f"Last labelled date: {labelled['date'].max().date()}")
    print(f"Target values: {sorted(labelled[TARGET_COL].unique().tolist())}")