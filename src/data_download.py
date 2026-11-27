import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path

from src.config import TICKERS, DOWNLOAD_START, DOWNLOAD_END, FRED_SERIES,RAW_DIR


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # We extract the price for each ticker, clean the row, then concatenate all rows
    all_tables = []

    for ticker in tickers:
        # 1. Download one stock
        table = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,         
            multi_level_index=False, 
            progress=False,
        )

        # 2. Remove rows where every value is missing
        table = table.dropna(how="all")

        # 3. Stop if nothing was downloaded
        if table.empty:
            raise ValueError(f"No data for {ticker}")

        # 4. Turn the date index into a normal column
        table = table.reset_index()

        # 5. Use lowercase column names ("Close" -> "close")
        table.columns = table.columns.str.lower()

        # 6. Remember which stock these rows belong to
        table["ticker"] = ticker

        # 7. Keep our columns, in a fixed order
        table = table[["date", "ticker", "open", "high", "low", "close", "volume"]]

        all_tables.append(table)

    # 8. Stack all the stocks on top of each other
    prices = pd.concat(all_tables)

    # 9. Sort and renumber the rows
    prices = prices.sort_values(["ticker", "date"])
    prices = prices.reset_index(drop=True)
    prices.columns.name = None  # remove the extra "Price" label

    return prices


def download_fred(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily series from FRED.

    Parameters
    ----------
    series_ids : list of FRED series names, e.g. ["VIXCLS", "T10Y2Y"].
    start : first date (included), "YYYY-MM-DD".
    end : last date (excluded), "YYYY-MM-DD".

    Returns
    -------
    One row per date, with columns: date, then one column per series.
    Sorted by date.
    """
    result = None

    for series in series_ids:
        current_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        table = pd.read_csv(current_url, na_values=".")
        table = table.rename(columns={"observation_date":"date"})
        table["date"] = pd.to_datetime(table["date"])
       
        if result is None :
            result=table
        else:
            result = pd.merge(result, table, on="date",how="outer")
    
    result = result[(result["date"]>=start) & (result["date"]<end)]
    result = result.sort_values("date")
    result = result.reset_index(drop=True) # Since we selected only some rows, we have to reset index
    return result


def save_snapshot(table: pd.DataFrame, name: str) -> Path:
    
    # 1. Create the folder if it does not exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Build the file path, with today's date in the name
    today = date.today().isoformat()
    path = RAW_DIR / f"{name}_{today}.csv"

    # 3. Never overwrite an existing snapshot
    if path.exists():
        raise FileExistsError(f"Snapshot already exists: {path}")

    # 4. Save without the index
    table.to_csv(path, index=False)

    return path



def main() -> None:
    """Download the raw data and save a dated snapshot."""
    # 1. Check the configuration
    if len(TICKERS) != 40:
        raise ValueError(f"Expected 40 tickers, got {len(TICKERS)}")

    # 2. Download everything before saving anything
    prices = download_prices(TICKERS, DOWNLOAD_START, DOWNLOAD_END)
    fred = download_fred(FRED_SERIES, DOWNLOAD_START, DOWNLOAD_END)

    # 3. Save the snapshot
    prices_path = save_snapshot(prices, "prices")
    fred_path = save_snapshot(fred, "fred")

    # 4. Print a summary to check the result
    print(prices.groupby("ticker")["date"].agg(["min", "max", "count"]).to_string())
    print(prices.isna().sum())
    print(fred.isna().sum())
    print(f"Prices: {prices.shape} -> {prices_path}")
    print(f"FRED: {fred.shape} -> {fred_path}")


if __name__ == "__main__":
    main()


