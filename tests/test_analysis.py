import pandas as pd
import pytest

from src.analysis import (
    analyze_sales_dataset,
    calculate_sales_statistics,
    get_daily_sales,
    get_products_with_most_zero_sales,
    get_top_products,
)


def test_calculate_sales_statistics():
    df = pd.DataFrame({"sales": [5, 0, 8, 3]})

    result = calculate_sales_statistics(df)

    assert result["total_sales"] == 16
    assert result["average_sales"] == 4.0
    assert result["max_sales"] == 8
    assert result["zero_sales_days"] == 1


def test_all_zero_sales():
    df = pd.DataFrame({"sales": [0, 0, 0]})

    result = calculate_sales_statistics(df)

    assert result["total_sales"] == 0
    assert result["average_sales"] == 0.0
    assert result["max_sales"] == 0
    assert result["zero_sales_days"] == 3


def test_single_day_sales():
    df = pd.DataFrame({"sales": [12]})

    result = calculate_sales_statistics(df)

    assert result["total_sales"] == 12
    assert result["average_sales"] == 12.0
    assert result["max_sales"] == 12
    assert result["zero_sales_days"] == 0


def test_analyze_sales_dataset(tmp_path):
    df = pd.DataFrame({
        "id": ["A", "B", "C"],
        "item_id": ["A", "B", "C"],
        "dept_id": ["FOODS"] * 3,
        "cat_id": ["FOODS"] * 3,
        "store_id": ["CA_1"] * 3,
        "state_id": ["CA"] * 3,
        "d_1": [5, 0, 7],
        "d_2": [3, 4, 1],
    })

    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    result = analyze_sales_dataset(str(csv_path), chunk_size=2)

    assert result["total_sales"] == 20
    assert result["total_rows"] == 6
    assert result["average_sales"] == 20 / 6
    assert result["max_sales"] == 7
    assert result["zero_sales_days"] == 1


def test_get_top_products(tmp_path):
    df = pd.DataFrame({
        "id": ["A_CA", "A_TX", "B_CA"],
        "item_id": ["A", "A", "B"],
        "dept_id": ["FOODS"] * 3,
        "cat_id": ["FOODS"] * 3,
        "store_id": ["CA_1", "TX_1", "CA_1"],
        "state_id": ["CA", "TX", "CA"],
        "d_1": [5, 3, 10],
        "d_2": [4, 2, 1],
    })

    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    result = get_top_products(str(csv_path), top_n=2, chunk_size=2)

    assert result.index.tolist() == ["A", "B"]
    assert result.tolist() == [14, 11]


def test_invalid_top_n(tmp_path):
    with pytest.raises(ValueError, match="top_n must be positive"):
        get_top_products(str(tmp_path / "sales.csv"), top_n=0)



def test_get_daily_sales(tmp_path):
    df = pd.DataFrame({
        "id": ["A", "B"],
        "item_id": ["A", "B"],
        "dept_id": ["FOODS"] * 2,
        "cat_id": ["FOODS"] * 2,
        "store_id": ["CA_1"] * 2,
        "state_id": ["CA"] * 2,
        "d_1": [5, 3],
        "d_2": [8, 2],
        "d_10": [4, 7],
    })

    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    result = get_daily_sales(str(csv_path), chunk_size=1)

    assert result.index.tolist() == ["d_1", "d_2", "d_10"]
    assert result.tolist() == [8, 10, 11]
    assert result.sum() == 29



def test_get_products_with_most_zero_sales(tmp_path):
    df = pd.DataFrame({
        "id": ["A_CA", "B_CA"],
        "item_id": ["A", "B"],
        "dept_id": ["FOODS"] * 2,
        "cat_id": ["FOODS"] * 2,
        "store_id": ["CA_1"] * 2,
        "state_id": ["CA"] * 2,
        "d_1": [0, 5],
        "d_2": [0, 0],
        "d_3": [4, 3],
    })

    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)

    result = get_products_with_most_zero_sales(
        str(csv_path),
        top_n=2,
        chunk_size=1,
    )

    assert result.index.tolist() == ["A", "B"]
    assert result["zero_days"].tolist() == [2, 1]
    assert result["total_days"].tolist() == [3, 3]
    assert result["zero_percentage"].iloc[0] == pytest.approx(200 / 3)


def test_invalid_zero_sales_top_n(tmp_path):
    with pytest.raises(ValueError, match="top_n must be positive"):
        get_products_with_most_zero_sales(
            str(tmp_path / "sales.csv"),
            top_n=0,
        )
