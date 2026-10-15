"""Tests for fixed, chronological ML validation windows."""

import pandas as pd
import pytest

from src.ml_validation import (
    build_validation_actuals,
    validation_windows,
)


@pytest.fixture
def sample_sales():
    """Two eligible series and one series introduced too late."""
    dates = pd.date_range("2015-10-01", "2016-01-03")

    records = []

    for series_id in ("A", "B", "C"):
        for day, date in enumerate(dates):
            # C has fewer than 28 days of history at V1.
            # It becomes eligible later, but must stay excluded.
            if series_id == "C" and date < pd.Timestamp("2015-11-01"):
                continue

            records.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "quantity": day % 7,
                }
            )

    return pd.DataFrame(records)


def test_validation_windows_are_fixed():
    windows = validation_windows()

    assert len(windows) == 2

    assert windows[0].origin == pd.Timestamp("2015-11-08")
    assert windows[0].start == pd.Timestamp("2015-11-09")
    assert windows[0].end == pd.Timestamp("2015-12-06")

    assert windows[1].origin == pd.Timestamp("2015-12-06")
    assert windows[1].start == pd.Timestamp("2015-12-07")
    assert windows[1].end == pd.Timestamp("2016-01-03")

    for window in windows:
        assert (window.end - window.start).days + 1 == 28


def test_fixed_validation_cohort(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    # A and B have enough history at V1; C does not.
    assert set(actuals["series_id"]) == {"A", "B"}

    # Two eligible series, two windows, 28 days each.
    assert len(actuals) == 2 * 2 * 28

    cohorts = [set(group["series_id"]) for _, group in actuals.groupby("origin")]

    assert cohorts == [{"A", "B"}, {"A", "B"}]


def test_validation_dates_and_horizons(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    assert actuals["horizon"].between(1, 28).all()

    expected_dates = actuals["origin"] + pd.to_timedelta(actuals["horizon"], unit="D")

    pd.testing.assert_series_equal(
        actuals["date"],
        expected_dates,
        check_names=False,
    )

    assert not actuals.duplicated(["series_id", "origin", "horizon"]).any()


def test_actuals_match_original_sales(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    matched = actuals.merge(
        sample_sales,
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    assert matched["quantity"].notna().all()
    assert (matched["actual"] == matched["quantity"]).all()


def test_missing_future_target_raises_error(sample_sales):
    broken = sample_sales.loc[
        ~(
            (sample_sales["series_id"] == "A")
            & (sample_sales["date"] == pd.Timestamp("2015-12-25"))
        )
    ].copy()

    with pytest.raises(ValueError, match="Missing observations"):
        build_validation_actuals(broken)


def test_missing_v2_history_raises_error(sample_sales):
    broken = sample_sales.loc[
        ~(
            (sample_sales["series_id"] == "B")
            & (sample_sales["date"] == pd.Timestamp("2015-12-03"))
        )
    ].copy()

    with pytest.raises(ValueError, match="Missing observations"):
        build_validation_actuals(broken)


def test_no_eligible_series_raises_error(sample_sales):
    recent_only = sample_sales.loc[
        sample_sales["date"] >= pd.Timestamp("2015-11-01")
    ].copy()

    with pytest.raises(ValueError, match="Not enough calendar history"):
        build_validation_actuals(recent_only)
