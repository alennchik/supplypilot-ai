
import pandas as pd
import pytest

from src.visualization import plot_daily_sales, plot_top_products


def test_plot_top_products(tmp_path):
    sales = pd.Series({
        "FOODS_1": 100,
        "FOODS_2": 250,
        "FOODS_3": 150,
    })

    output_path = tmp_path / "top_products.png"

    plot_top_products(sales, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_empty_products(tmp_path):
    sales = pd.Series(dtype="int64")
    output_path = tmp_path / "empty.png"

    with pytest.raises(ValueError, match="Product sales cannot be empty"):
        plot_top_products(sales, output_path)

    assert not output_path.exists()


def test_plot_daily_sales(tmp_path):
    daily_sales = pd.Series(
        [100, 150, 120, 180],
        index=["d_1", "d_2", "d_3", "d_4"],
    )

    output_path = tmp_path / "daily_sales.png"

    plot_daily_sales(daily_sales, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_empty_daily_sales(tmp_path):
    daily_sales = pd.Series(dtype="int64")
    output_path = tmp_path / "empty_daily.png"

    with pytest.raises(ValueError, match="Daily sales cannot be empty"):
        plot_daily_sales(daily_sales, output_path)

    assert not output_path.exists()

