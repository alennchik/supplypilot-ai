"""Simple one-step-ahead demand forecasting baselines."""

import numpy as np
import pandas as pd


def _predict_from_lag(
    validation: pd.DataFrame, column: str, name: str
) -> pd.Series:
    """Return a lag column as predictions without reading target sales."""
    if validation.empty:
        raise ValueError("Validation data cannot be empty")

    if column not in validation.columns:
        raise ValueError(f"Missing required feature: {column}")

    predictions = validation[column].copy()

    if not pd.api.types.is_numeric_dtype(predictions):
        raise TypeError("Lag values must be numeric")

    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Lag values must be finite")

    if (predictions < 0).any():
        raise ValueError("Lag values cannot be negative")

    return predictions.rename(name)


def naive_forecast(validation: pd.DataFrame) -> pd.Series:
    """Predict each day using the previous day's actual sales."""
    return _predict_from_lag(validation, "lag_1", "naive")


def seasonal_naive_forecast(validation: pd.DataFrame) -> pd.Series:
    """Predict each day using actual sales from seven days earlier."""
    return _predict_from_lag(validation, "lag_7", "seasonal_naive")
