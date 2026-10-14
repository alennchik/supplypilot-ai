"""Tests for converting M5 data into unified daily sales."""

import pandas as pd
import pytest

from src.m5_unified import UNIFIED_COLUMNS, build_unified_batch


@pytest.fixture
def m5_data():
    """Create two series with different price histories."""
    sales = pd.DataFrame(
        {
            "id": ["ITEM_A_CA_1_validation", "ITEM_B_CA_1_validation"],
            "item_id": ["ITEM_A", "ITEM_B"],
            "store_id": ["CA_1", "CA_1"],
            "d_1": [0, 1],
            "d_2": [0, 0],
            "d_3": [2, 3],
            "d_4": [0, 4],
        }
    )

    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3", "d_4", "d_5"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
            ],
            "wm_yr_wk": [1, 1, 2, 3, 3],
        }
    )

    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "CA_1", "CA_1"],
            "item_id": ["ITEM_A", "ITEM_B", "ITEM_B", "ITEM_B"],
            "wm_yr_wk": [2, 1, 2, 3],
            "sell_price": [5.0, 2.0, 3.0, 4.0],
        }
    )

    return sales, calendar, prices


def test_output_schema(m5_data):
    result = build_unified_batch(*m5_data)

    assert result.columns.tolist() == UNIFIED_COLUMNS

    # Accept valid datetime types regardless of time precision.
    assert pd.api.types.is_datetime64_any_dtype(result["date"])

    assert result["price_available"].dtype == bool


def test_first_observed_price_trimming(m5_data):
    result = build_unified_batch(*m5_data)

    item_a = result[result["series_id"] == "ITEM_A_CA_1_validation"]

    assert len(item_a) == 2
    assert item_a["date"].min() == pd.Timestamp("2020-01-03")
    assert item_a["quantity"].tolist() == [2, 0]


def test_missing_price_after_first_price_preserved(m5_data):
    result = build_unified_batch(*m5_data)

    item_a = result[result["series_id"] == "ITEM_A_CA_1_validation"]

    assert item_a["price_available"].tolist() == [True, False]
    assert item_a["price"].iloc[0] == 5.0
    assert pd.isna(item_a["price"].iloc[1])


def test_zero_sales_preserved(m5_data):
    result = build_unified_batch(*m5_data)

    item_b = result[result["series_id"] == "ITEM_B_CA_1_validation"]

    assert item_b["quantity"].tolist() == [1, 0, 3, 4]


def test_no_extra_calendar_days(m5_data):
    result = build_unified_batch(*m5_data)

    assert len(result) == 6
    assert result["date"].max() == pd.Timestamp("2020-01-04")
    assert not result.duplicated(["series_id", "date"]).any()


def test_missing_calendar_day_rejected(m5_data):
    sales, calendar, prices = m5_data

    calendar = calendar[calendar["d"] != "d_3"]

    with pytest.raises(
        ValueError,
        match="Missing calendar mapping",
    ):
        build_unified_batch(sales, calendar, prices)


def test_duplicate_price_keys_rejected(m5_data):
    sales, calendar, prices = m5_data

    prices = pd.concat(
        [prices, prices.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate weekly price keys",
    ):
        build_unified_batch(sales, calendar, prices)


def test_negative_quantity_rejected(m5_data):
    sales, calendar, prices = m5_data

    sales = sales.copy()
    sales.loc[0, "d_3"] = -1

    with pytest.raises(
        ValueError,
        match="Invalid daily sales quantities",
    ):
        build_unified_batch(sales, calendar, prices)


def test_series_without_observed_price_rejected(m5_data):
    sales, calendar, prices = m5_data

    prices = prices[prices["item_id"] != "ITEM_A"]

    with pytest.raises(
        ValueError,
        match="Some series have no observed price",
    ):
        build_unified_batch(sales, calendar, prices)


def test_missing_sales_day_column_rejected(m5_data):
    sales, calendar, prices = m5_data

    sales = sales.drop(columns=["d_2"])

    with pytest.raises(
        ValueError,
        match="Sales day columns must be consecutive",
    ):
        build_unified_batch(sales, calendar, prices)


@pytest.mark.parametrize(
    "replacement_date",
    ["2020-01-02", "2020-01-10"],
)
def test_invalid_calendar_sequence_rejected(m5_data, replacement_date):
    sales, calendar, prices = m5_data

    calendar = calendar.copy()
    calendar.loc[calendar["d"] == "d_3", "date"] = replacement_date

    with pytest.raises(
        ValueError,
        match="Calendar dates must be consecutive",
    ):
        build_unified_batch(sales, calendar, prices)
