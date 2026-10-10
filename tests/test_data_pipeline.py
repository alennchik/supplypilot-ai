import pandas as pd
import pytest

from src.data_pipeline import (
    load_sales_data,
    process_sales_in_chunks,
    reshape_sales_data,
)


def test_reshape_sales_data():
    # Create a small sample of sales data
    df = pd.DataFrame({
        "id": ["APPLE_CA_1", "BANANA_CA_1"],
        "item_id": ["APPLE", "BANANA"],
        "dept_id": ["FOOD", "FOOD"],
        "cat_id": ["FOODS", "FOODS"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
        "d_1": [5, 2],
        "d_2": [3, 4],
    })

    result = reshape_sales_data(df)

    assert len(result) == 4
    assert list(result["day"]) == ["d_1", "d_1", "d_2", "d_2"]
    assert list(result["sales"]) == [5, 2, 3, 4]

def test_load_sales_data(tmp_path):
    # Create a temporary CSV file
    csv_path = tmp_path / "sales.csv"

    expected = pd.DataFrame({
        "item_id": ["APPLE", "BANANA"],
        "d_1": [5, 2],
    })

    expected.to_csv(csv_path, index=False)

    result = load_sales_data(csv_path)

    pd.testing.assert_frame_equal(result, expected)

def test_missing_required_columns():
    df = pd.DataFrame({
        "item_id": ["APPLE"],
        "d_1": [5],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        reshape_sales_data(df)


def test_missing_daily_sales_columns():
    df = pd.DataFrame({
        "id": ["APPLE_CA_1"],
        "item_id": ["APPLE"],
        "dept_id": ["FOOD"],
        "cat_id": ["FOODS"],
        "store_id": ["CA_1"],
        "state_id": ["CA"],
    })

    with pytest.raises(ValueError, match="No daily sales columns found"):
        reshape_sales_data(df)

def test_process_sales_in_chunks(tmp_path):
    csv_path = tmp_path / "sales.csv"

    df = pd.DataFrame({
        "id": ["A_CA_1", "B_CA_1", "C_CA_1"],
        "item_id": ["A", "B", "C"],
        "dept_id": ["FOOD"] * 3,
        "cat_id": ["FOODS"] * 3,
        "store_id": ["CA_1"] * 3,
        "state_id": ["CA"] * 3,
        "d_1": [5, 2, 7],
        "d_2": [3, 4, 1],
    })

    df.to_csv(csv_path, index=False)

    chunks = list(process_sales_in_chunks(csv_path, chunk_size=2))

    assert len(chunks) == 2
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 2
    assert chunks[0]["sales"].tolist() == [5, 2, 3, 4]
    assert chunks[1]["sales"].tolist() == [7, 1]

def test_invalid_chunk_size(tmp_path):
    csv_path = tmp_path / "sales.csv"

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        next(process_sales_in_chunks(csv_path, chunk_size=0))