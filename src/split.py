import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, EMBARGO, N_FOLDS,
)
from src.target import add_target


def select_period(table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # Rows dated between start and end (both included)
    return table[table["date"].between(start, end)]


def remove_last_dates(table: pd.DataFrame, n_days: int) -> pd.DataFrame:
    # Remove the last days: their labels use prices from the next period
    
    dates = np.sort(table["date"].unique())

    if len(dates) <= n_days:
        raise ValueError("Not enough dates to apply the embargo.")

    removed_dates = dates[len(dates) - n_days:]
    return table[~table["date"].isin(removed_dates)]


def split_by_date(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Chronological train / validation / test split with embargo

    
    train = select_period(table, TRAIN_START, TRAIN_END)
    validation = select_period(table, VAL_START, VAL_END)
    test = select_period(table, TEST_START, TEST_END)

    # The labels of the last days look into the next period: remove them.
    # No embargo for the test set: there is no period after it.
    train = remove_last_dates(train, EMBARGO)
    validation = remove_last_dates(validation, EMBARGO)

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def walk_forward_folds(train: pd.DataFrame, n_folds: int = N_FOLDS, embargo: int = EMBARGO) -> list[tuple[np.ndarray, np.ndarray]]:
    # Repeat multiple time : Learn on past, validate on next year

    
    years = np.sort(train["date"].dt.year.unique())

    if n_folds < 1 or n_folds >= len(years):
        raise ValueError("n_folds must be between 1 and (number of years - 1).")

    folds = []
    for year in years[-n_folds:]:
        validation_mask = train["date"].dt.year == year
        first_validation_date = train.loc[validation_mask, "date"].min()

        past = train[train["date"] < first_validation_date]
        past = remove_last_dates(past, embargo)
        fit_mask = train["date"].isin(past["date"])

        folds.append((np.flatnonzero(fit_mask), np.flatnonzero(validation_mask)))

    return folds


def describe(name: str, table: pd.DataFrame) -> None:
    #Print the size and dates of a set (no target statistics)
    print(f"{name:<12} rows={len(table):>7}  "
          f"{table['date'].min().date()} -> {table['date'].max().date()}")


if __name__ == "__main__":
    # Technical check only: 0.05 is NOT the final choice of X.
    data = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    labelled = add_target(data, threshold=0.05)

    train, validation, test = split_by_date(labelled)
    describe("Train", train)
    describe("Validation", validation)
    describe("Test", test)

    for number, (fit_pos, val_pos) in enumerate(walk_forward_folds(train), start=1):
        print(f"\nFold {number}")
        describe("  fit", train.iloc[fit_pos])
        describe("  validation", train.iloc[val_pos])