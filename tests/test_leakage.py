"""Leakage tests on small simulated data (the real snapshot is not needed)."""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, TARGET_COL, HORIZON
from src.data_cleaning import align_fred
from src.features import add_features
from src.target import add_target
from src.split import split_by_date, walk_forward_folds

DATES = pd.bdate_range("2009-01-01", "2025-12-31")


@pytest.fixture
def clean():
    """Random clean table with two stocks, same columns as clean_prices.csv."""
    rng = np.random.default_rng(0)
    vix = rng.uniform(10, 40, len(DATES))
    tables = []
    for ticker in ["AAA", "BBB"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DATES))))
        tables.append(pd.DataFrame({
            "date": DATES, "ticker": ticker, "sector": "Test",
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.integers(100, 1000, len(DATES)),
            "vix": vix, "t10y2y": rng.uniform(-1, 2, len(DATES)),
        }))
    return pd.concat(tables, ignore_index=True)


def dates_between(earlier, later):
    """Number of trading dates strictly between two dates."""
    return DATES.get_loc(later) - DATES.get_loc(earlier) - 1


# ---------- Features ----------

def test_features_do_not_use_the_future(clean):
    cutoff = DATES[1000]
    full = add_features(clean)
    truncated = add_features(clean[clean["date"] <= cutoff])
    expected = full[full["date"] <= cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(truncated[FEATURE_COLS], expected[FEATURE_COLS])


def test_features_do_not_mix_stocks(clean):
    both = add_features(clean)
    alone = add_features(clean[clean["ticker"] == "BBB"])
    expected = both[both["ticker"] == "BBB"].reset_index(drop=True)

    pd.testing.assert_frame_equal(alone[FEATURE_COLS], expected[FEATURE_COLS])


def test_forbidden_columns_are_not_features():
    forbidden = {TARGET_COL, "close", "date", "ticker"}
    assert forbidden.isdisjoint(FEATURE_COLS)


def test_fred_alignment_does_not_use_the_future():
    days = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"date": days})
    fred = pd.DataFrame({
        "date": days,
        "VIXCLS": [10.0, np.nan, 30.0, 40.0],
        "T10Y2Y": [1.0, np.nan, 3.0, 4.0],
    })

    aligned = align_fred(prices, fred)

    # The gap takes the previous value (10), not the next one (30)
    assert aligned["vix"].tolist() == [10.0, 10.0, 30.0, 40.0]
    assert aligned["t10y2y"].tolist() == [1.0, 1.0, 3.0, 4.0]


# ---------- Labels ----------

def make_prices(ticker, closes):
    days = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"date": days, "ticker": ticker, "close": closes})


def test_label_looks_exactly_at_the_next_days():
    # The drop happens on day t+5 of the first row, and t+6 is ignored
    inside = make_prices("AAA", [100, 100, 100, 100, 100, 90, 100])
    outside = make_prices("BBB", [100, 100, 100, 100, 100, 100, 90])

    labelled = add_target(pd.concat([inside, outside]), threshold=0.05)
    first_rows = labelled.groupby("ticker")[TARGET_COL].first()

    assert first_rows["AAA"] == 1
    assert first_rows["BBB"] == 0


def test_unknown_future_is_removed_not_set_to_zero():
    prices = make_prices("AAA", [100.0] * 8)

    labelled = add_target(prices, threshold=0.05)

    assert len(labelled) == len(prices) - HORIZON
    assert labelled["date"].max() == prices["date"].iloc[-HORIZON - 1]


# ---------- Splits ----------

def test_split_has_an_embargo(clean):
    train, validation, test = split_by_date(add_target(clean, threshold=0.05))

    # At least HORIZON dates in between: labels cannot reach the next set
    assert dates_between(train["date"].max(), validation["date"].min()) >= HORIZON
    assert dates_between(validation["date"].max(), test["date"].min()) >= HORIZON


def test_folds_are_chronological_with_an_embargo(clean):
    train, _, _ = split_by_date(add_target(clean, threshold=0.05))

    folds = walk_forward_folds(train)

    previous_size = 0
    for fit_positions, validation_positions in folds:
        fit = train.iloc[fit_positions]
        validation = train.iloc[validation_positions]

        assert dates_between(fit["date"].max(), validation["date"].min()) >= HORIZON
        assert len(fit) > previous_size  # expanding window
        previous_size = len(fit)