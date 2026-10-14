"""Metrics for evaluating demand forecast accuracy."""

import numpy as np
import pandas as pd


def _validate_inputs(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Convert equally aligned one-dimensional sequences into numeric arrays."""
    if (
        isinstance(actual, pd.Series)
        and isinstance(predicted, pd.Series)
        and not actual.index.equals(predicted.index)
    ):
        raise ValueError("Actual and predicted indexes must match")

    try:
        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Actual and predicted values must be numeric") from exc

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional")

    if actual_values.size == 0 or predicted_values.size == 0:
        raise ValueError("Actual and predicted values cannot be empty")

    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same length")

    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual and predicted values must be finite")

    if (actual_values < 0).any():
        raise ValueError("Actual sales cannot be negative")

    return actual_values, predicted_values


def mae(actual, predicted) -> float:
    """Mean absolute error, in units sold."""
    actual_values, predicted_values = _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def rmse(actual, predicted) -> float:
    """Root mean squared error, in units sold."""
    actual_values, predicted_values = _validate_inputs(actual, predicted)
    return float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))


def wape(actual, predicted) -> float:
    """Weighted absolute percentage error as a percentage."""
    actual_values, predicted_values = _validate_inputs(actual, predicted)

    total_actual = float(np.abs(actual_values).sum())

    if total_actual == 0:
        raise ValueError("WAPE is undefined when total actual sales are zero")

    return float(
        np.abs(actual_values - predicted_values).sum() / total_actual * 100
    )
