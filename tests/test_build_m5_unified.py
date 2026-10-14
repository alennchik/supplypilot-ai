"""Tests for the end-to-end M5 unified build script."""

import pandas as pd
import pytest

from scripts import build_m5_unified as builder


@pytest.fixture
def build_files(tmp_path, monkeypatch):
    """Create small source datasets and redirect output to tmp_path."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()

    sales = pd.DataFrame(
        {
            "id": [
                "ITEM_A_CA_1_validation",
                "ITEM_B_CA_1_validation",
                "ITEM_C_CA_1_validation",
            ],
            "item_id": ["ITEM_A", "ITEM_B", "ITEM_C"],
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "d_1": [0, 1, 9],
            "d_2": [2, 0, 9],
            "d_3": [0, 3, 9],
        }
    )

    series = pd.DataFrame(
        {
            "series_id": [
                "ITEM_A_CA_1_validation",
                "ITEM_B_CA_1_validation",
            ],
            "item_id": ["ITEM_A", "ITEM_B"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
        }
    )

    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "wm_yr_wk": [1, 2, 2],
        }
    )

    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "item_id": ["ITEM_A", "ITEM_B", "ITEM_B"],
            "wm_yr_wk": [2, 1, 2],
            "sell_price": [5.0, 2.0, 3.0],
        }
    )

    source_path = processed / "source.parquet"
    series_path = processed / "series.parquet"
    calendar_path = raw / "calendar.csv"
    prices_path = raw / "prices.csv"
    output_path = processed / "sales.parquet"

    sales.to_parquet(source_path, index=False)
    series.to_parquet(series_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    prices.to_csv(prices_path, index=False)

    monkeypatch.setattr(builder, "SOURCE_PATH", source_path)
    monkeypatch.setattr(builder, "SERIES_PATH", series_path)
    monkeypatch.setattr(builder, "CALENDAR_PATH", calendar_path)
    monkeypatch.setattr(builder, "PRICES_PATH", prices_path)
    monkeypatch.setattr(builder, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(builder, "SOURCE_BATCH_SIZE", 1)
    monkeypatch.setattr(builder, "PRICE_CHUNK_SIZE", 1)

    return source_path, output_path


def test_build_creates_correct_unified_dataset(build_files):
    _, output_path = build_files

    builder.main()

    result = pd.read_parquet(output_path)

    assert len(result) == 5
    assert result["series_id"].nunique() == 2
    assert not result["series_id"].str.contains("ITEM_C").any()
    assert not result.duplicated(["series_id", "date"]).any()

    item_a = result[result["series_id"] == "ITEM_A_CA_1_validation"]

    assert item_a["quantity"].tolist() == [2, 0]
    assert item_a["price"].tolist() == [5.0, 5.0]


def test_failed_build_preserves_existing_output(build_files):
    source_path, output_path = build_files

    builder.main()
    original_bytes = output_path.read_bytes()

    # Make the second source series invalid.
    sales = pd.read_parquet(source_path)
    sales.loc[1, "d_3"] = -4
    sales.to_parquet(source_path, index=False)

    with pytest.raises(
        ValueError,
        match="Invalid daily sales quantities",
    ):
        builder.main()

    assert output_path.read_bytes() == original_bytes
    assert list(output_path.parent.glob(".sales_*.parquet")) == []
