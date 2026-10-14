"""Tests for M5 CSV-to-Parquet conversion."""

import pandas as pd
import pytest

from src.parquet_storage import convert_sales_to_parquet


@pytest.fixture
def sample_sales(tmp_path):
    """Create a small M5-style CSV for testing."""
    csv_path = tmp_path / "sales.csv"

    df = pd.DataFrame({
        "id": ["ITEM_A_STORE_1", "ITEM_B_STORE_1"],
        "item_id": ["ITEM_A", "ITEM_B"],
        "dept_id": ["FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS"],
        "store_id": ["STORE_1", "STORE_1"],
        "state_id": ["CA", "CA"],
        "d_1": [0, 3],
        "d_2": [5, 32767],
    })

    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_successful_conversion(sample_sales, tmp_path):
    csv_path, original = sample_sales
    parquet_path = tmp_path / "sales.parquet"

    rows, columns = convert_sales_to_parquet(
        csv_path, parquet_path
    )

    result = pd.read_parquet(parquet_path)

    assert parquet_path.exists()
    assert rows == 2
    assert columns == 8
    assert result.shape == original.shape
    assert result.columns.tolist() == original.columns.tolist()

    for column in original.columns:
        assert result[column].tolist() == original[column].tolist()


def test_sales_stored_as_int16(sample_sales, tmp_path):
    csv_path, _ = sample_sales
    parquet_path = tmp_path / "sales.parquet"

    convert_sales_to_parquet(csv_path, parquet_path)
    result = pd.read_parquet(parquet_path)

    assert str(result["d_1"].dtype) == "int16"
    assert str(result["d_2"].dtype) == "int16"


def test_negative_sales_rejected(sample_sales, tmp_path):
    csv_path, df = sample_sales
    df.loc[0, "d_1"] = -1
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Sales values must"):
        convert_sales_to_parquet(
            csv_path, tmp_path / "sales.parquet"
        )


def test_int16_overflow_rejected(sample_sales, tmp_path):
    csv_path, df = sample_sales
    df.loc[0, "d_1"] = 32768
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Sales values must"):
        convert_sales_to_parquet(
            csv_path, tmp_path / "sales.parquet"
        )


def test_missing_identifier_rejected(sample_sales, tmp_path):
    csv_path, df = sample_sales
    df = df.drop(columns=["store_id"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing identifier"):
        convert_sales_to_parquet(
            csv_path, tmp_path / "sales.parquet"
        )


def test_missing_day_columns_rejected(sample_sales, tmp_path):
    csv_path, df = sample_sales
    df = df.drop(columns=["d_1", "d_2"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No daily sales"):
        convert_sales_to_parquet(
            csv_path, tmp_path / "sales.parquet"
        )


def test_failed_conversion_preserves_existing_file(
    sample_sales, tmp_path
):
    csv_path, df = sample_sales
    parquet_path = tmp_path / "sales.parquet"

    # Create a valid existing output.
    convert_sales_to_parquet(csv_path, parquet_path)
    original_bytes = parquet_path.read_bytes()

    # Make the input invalid.
    df.loc[0, "d_1"] = -5
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        convert_sales_to_parquet(csv_path, parquet_path)

    assert parquet_path.read_bytes() == original_bytes
