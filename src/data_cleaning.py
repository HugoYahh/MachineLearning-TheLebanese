import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR, SNAPSHOT_DATE, TICKER_SECTORS

PRICE_COLS = ["open", "high", "low", "close"]
TOLERANCE = 1e-6    # ignore tiny rounding errors in adjusted prices
BIG_MOVE = 0.25     # daily moves above 25% are listed for manual review, this is to avoid outliers


def load_snapshot(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}_{SNAPSHOT_DATE}.csv"
    table = pd.read_csv(path, parse_dates=["date"])
    return table


def check_prices(prices: pd.DataFrame) -> None:
    # 1. Same stock and same date appearing twice
    duplicates = prices.duplicated(["date", "ticker"]).sum()

    # 2. Empty cells
    missing_values = prices.isna().sum().sum()

    # 3. Missing dates: each stock should have every date of the dataset.
    # Limit: a date missing for ALL stocks cannot be detected this way.
    n_dates = prices["date"].nunique()
    n_tickers = prices["ticker"].nunique()
    n_unique_rows = len(prices.drop_duplicates(["date", "ticker"]))
    missing_rows = n_dates * n_tickers - n_unique_rows

    # 4. Prices must be positive, volume cannot be negative
    bad_prices = (prices[PRICE_COLS] <= 0).any(axis=1).sum()
    negative_volume = (prices["volume"] < 0).sum()
    zero_volume = (prices["volume"] == 0).sum()

    # 5. High must be the highest price of the day, low the lowest
    highest = prices[["open", "close", "low"]].max(axis=1)
    lowest = prices[["open", "close", "high"]].min(axis=1)
    high_too_low = prices["high"] < highest * (1 - TOLERANCE)
    low_too_high = prices["low"] > lowest * (1 + TOLERANCE)
    incoherent = prices[high_too_low | low_too_high]

    # 6. Very large daily moves, computed stock by stock
    ordered = prices.sort_values(["ticker", "date"])
    previous_close = ordered.groupby("ticker")["close"].shift(1)
    daily_return = ordered["close"] / previous_close - 1
    big_moves = ordered[daily_return.abs() > BIG_MOVE].copy()
    big_moves["return"] = daily_return[big_moves.index]

    print("=== Price quality report ===")
    print(f"Duplicated rows      : {duplicates}")
    print(f"Missing values       : {missing_values}")
    print(f"Missing stock-days   : {missing_rows}")
    print(f"Non-positive prices  : {bad_prices}")
    print(f"Negative volumes     : {negative_volume}")
    print(f"Zero volumes         : {zero_volume}")
    print(f"Incoherent high/low  : {len(incoherent)}")
    print(f"Moves above {BIG_MOVE:.0%}      : {len(big_moves)}")

    if len(incoherent) > 0:
        print("\nIncoherent rows (first 10):")
        print(incoherent.head(10).to_string(index=False))

    if len(big_moves) > 0:
        print("\nLarge moves to review:")
        print(big_moves[["date", "ticker", "close", "return"]].to_string(index=False))

    blocking = [duplicates, missing_values, missing_rows, bad_prices, negative_volume]
    if sum(blocking) > 0:
        raise ValueError("Blocking problem found, see the report above.")


def align_fred(prices: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
   # Goal here is to align fred data and yahoo finance data.
   # Missing values are forward filled with last known data
    fred = fred.rename(columns={"VIXCLS": "vix", "T10Y2Y": "t10y2y"})
    fred = fred.set_index("date").sort_index()

    trading_days = pd.DatetimeIndex(prices["date"].unique()).sort_values()

    print("\n=== FRED missing values on trading days (before fill) ===")
    print(fred.reindex(trading_days).isna().sum())

    # Keep ALL dates while filling, so that a value dated on a day
    # without stock trading is not lost. Then keep trading days only.
    all_days = fred.index.union(trading_days)
    aligned = fred.reindex(all_days).ffill().loc[trading_days]

    if aligned.isna().any().any():
        raise ValueError("Some trading days have no past FRED value.")

    return aligned.rename_axis("date").reset_index()


def main() -> None:
    prices = load_snapshot("prices")
    fred = load_snapshot("fred")

    check_prices(prices)
    macro = align_fred(prices, fred)

    # Add the FRED values to every stock (same value for all stocks on a date)
    clean = pd.merge(prices, macro, on="date", how="left", validate="many_to_one")

    # Add the sector of each stock
    clean["sector"] = clean["ticker"].map(TICKER_SECTORS)
    if clean["sector"].isna().any():
        raise ValueError("Some tickers have no sector in config.py.")

    clean = clean[["date", "ticker", "sector", *PRICE_COLS, "volume", "vix", "t10y2y"]]
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "clean_prices.csv"
    clean.to_csv(path, index=False)

    print(f"\nSaved {clean.shape} -> {path}")


if __name__ == "__main__":
    main()