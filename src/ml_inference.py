"""Build leakage-safe 28-day LightGBM forecasts from historical sales."""

import numpy as np
import pandas as pd

from src.backtesting import build_sales_matrix
from src.forecast_baselines import FORECAST_HORIZON
from src.ml_dataset import METADATA_COLUMNS, _calendar_date, _metadata_by_series
from src.ml_features import DEMAND_FEATURE_NAMES, MIN_HISTORY, demand_features
from src.ml_model import GlobalDemandModel, predict_global_model


def build_ml_inference_examples(
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
    origin,
) -> pd.DataFrame:
    """Create prediction rows using only sales available through origin.

    A series is eligible if its most recent 28 calendar days, including
    origin, have observed sales quantities.

    Future sales are never included in the historical demand features.
    Future target values are not needed or generated.
    """
    forecast_origin = _calendar_date(origin, "Forecast origin")

    matrix = build_sales_matrix(sales)
    by_series = _metadata_by_series(metadata, matrix.index)

    # Only demand through the requested forecast origin is accessible.
    history_dates = pd.date_range(
        end=forecast_origin,
        periods=56,
        freq="D",
    )
    history = matrix.reindex(columns=history_dates)

    # Match the exact eligibility rule used during ML training.
    eligible = history.iloc[:, -MIN_HISTORY:].notna().all(axis=1)
    chosen = matrix.index[eligible]

    if len(chosen) == 0:
        raise ValueError(
            f"No eligible series at forecast origin {forecast_origin.date()}"
        )

    # Reuse the same feature computation as the training dataset.
    historical = history.loc[chosen].to_numpy(dtype=np.float64)
    feature_dicts = [demand_features(row) for row in historical]

    features = pd.DataFrame(
        feature_dicts,
        columns=DEMAND_FEATURE_NAMES,
    )

    n_series = len(chosen)

    forecast_dates = pd.date_range(
        start=forecast_origin + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D",
    )

    horizons = np.arange(
        1,
        FORECAST_HORIZON + 1,
        dtype=np.int8,
    )

    examples = pd.DataFrame(
        {
            "series_id": np.repeat(
                chosen.to_numpy(),
                FORECAST_HORIZON,
            ),
            "origin": forecast_origin,
            "date": np.tile(
                forecast_dates.to_numpy(),
                n_series,
            ),
            "horizon": np.tile(
                horizons,
                n_series,
            ),
        }
    )

    for column in METADATA_COLUMNS:
        examples[column] = np.repeat(
            by_series.loc[chosen, column].to_numpy(),
            FORECAST_HORIZON,
        )

    for column in DEMAND_FEATURE_NAMES:
        examples[column] = np.repeat(
            features[column].to_numpy(dtype=np.float32),
            FORECAST_HORIZON,
        )

    if examples.duplicated(["series_id", "origin", "horizon"]).any():
        raise ValueError("Duplicate inference example keys")

    return examples


def predict_at_origin(
    model: GlobalDemandModel,
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
    origin,
) -> pd.DataFrame:
    """Generate 28 nonnegative forecasts per eligible product-store series."""
    examples = build_ml_inference_examples(
        sales=sales,
        metadata=metadata,
        origin=origin,
    )

    return predict_global_model(model, examples)
