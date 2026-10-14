"""Direct 28-day demand forecasting baselines."""

import numpy as np

FORECAST_HORIZON = 28
SEASON_LENGTH = 7
MIN_HISTORY = 28


def _validate_history(history) -> np.ndarray:
    """Validate historical sales available at forecast origin."""
    try:
        values = np.asarray(history, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Historical sales must be numeric") from exc

    if values.ndim != 1:
        raise ValueError("Historical sales must be one-dimensional")

    if len(values) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} historical days are required")

    if not np.isfinite(values).all():
        raise ValueError("Historical sales must be finite")

    if (values < 0).any():
        raise ValueError("Historical sales cannot be negative")

    return values


def naive_forecast(history) -> np.ndarray:
    """Repeat the last observed sale for the next 28 days."""
    values = _validate_history(history)

    return np.full(
        FORECAST_HORIZON,
        values[-1],
        dtype=float,
    )


def seasonal_naive_forecast(history) -> np.ndarray:
    """Repeat the last seven observed days over 28 days."""
    values = _validate_history(history)

    last_week = values[-SEASON_LENGTH:]

    return np.tile(
        last_week,
        FORECAST_HORIZON // SEASON_LENGTH,
    )


def moving_average_forecast(history) -> np.ndarray:
    """Forecast using mean sales from the last 28 days."""
    values = _validate_history(history)

    average = np.mean(values[-MIN_HISTORY:])

    return np.full(
        FORECAST_HORIZON,
        average,
        dtype=float,
    )
