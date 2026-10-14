"""Evaluate two one-step-ahead baselines on a small Walmart M5 sample.

Run from the repository root:
    python -m scripts.evaluate_baselines
"""

from pathlib import Path

from src.baselines import naive_forecast, seasonal_naive_forecast
from src.data_pipeline import process_sales_in_chunks
from src.forecasting import (
    add_forecasting_features,
    prepare_daily_sales,
    split_train_validation,
)
from src.metrics import mae, rmse, wape

DATA_PATH = Path("data/raw/sales_train_validation.csv")


def main() -> None:
    """Load 20 series, forecast the final 28 days, and print error metrics."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing M5 dataset: {DATA_PATH}")

    chunk = next(process_sales_in_chunks(DATA_PATH, chunk_size=20))

    daily_sales = chunk.groupby("day")["sales"].sum()

    prepared = prepare_daily_sales(daily_sales)
    featured = add_forecasting_features(prepared)

    train, validation = split_train_validation(
        featured,
        validation_days=28,
    )

    actual = validation["sales"]

    forecasts = {
        "Naive (yesterday)": naive_forecast(validation),
        "Seasonal naive (last week)": seasonal_naive_forecast(validation),
    }

    print(f"M5 sample: 20 product-store series, {len(prepared)} days")
    print(f"Train: {len(train)} days | Validation: {len(validation)} days")
    print("One-step-ahead evaluation with observed daily sales updates")

    print(f"{'Model':<30} {'MAE':>10} {'RMSE':>10} {'WAPE':>10}")

    for name, predicted in forecasts.items():
        print(
            f"{name:<30} "
            f"{mae(actual, predicted):>10.2f} "
            f"{rmse(actual, predicted):>10.2f} "
            f"{wape(actual, predicted):>9.2f}%"
        )


if __name__ == "__main__":
    main()
