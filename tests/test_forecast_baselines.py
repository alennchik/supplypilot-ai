"""Tests for direct 28-day forecasting baselines."""

import numpy as np
import pytest

from src.forecast_baselines import (
    FORECAST_HORIZON,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


@pytest.fixture
def history():
    """Provide 28 days of historical demand."""
    return np.arange(1, 29, dtype=float)


@pytest.mark.parametrize(
    "forecast_function",
    [
        naive_forecast,
        seasonal_naive_forecast,
        moving_average_forecast,
    ],
)
def test_forecast_length(history, forecast_function):
    predictions = forecast_function(history)

    assert len(predictions) == FORECAST_HORIZON
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_naive_forecast(history):
    predictions = naive_forecast(history)

    np.testing.assert_array_equal(
        predictions,
        np.full(28, 28.0),
    )


def test_seasonal_naive_forecast(history):
    predictions = seasonal_naive_forecast(history)

    expected = np.tile(
        np.arange(22, 29, dtype=float),
        4,
    )

    np.testing.assert_array_equal(predictions, expected)


def test_moving_average_forecast(history):
    predictions = moving_average_forecast(history)

    np.testing.assert_allclose(
        predictions,
        np.full(28, 14.5),
    )


@pytest.mark.parametrize(
    "forecast_function",
    [
        naive_forecast,
        seasonal_naive_forecast,
        moving_average_forecast,
    ],
)
def test_zero_demand_history(forecast_function):
    predictions = forecast_function(np.zeros(28))

    np.testing.assert_array_equal(
        predictions,
        np.zeros(28),
    )


@pytest.mark.parametrize(
    "forecast_function",
    [
        naive_forecast,
        seasonal_naive_forecast,
        moving_average_forecast,
    ],
)
def test_insufficient_history_rejected(forecast_function):
    with pytest.raises(ValueError, match="At least 28"):
        forecast_function([1, 2, 3])


@pytest.mark.parametrize(
    "forecast_function",
    [
        naive_forecast,
        seasonal_naive_forecast,
        moving_average_forecast,
    ],
)
def test_negative_sales_rejected(forecast_function):
    history = np.ones(28)
    history[-1] = -1

    with pytest.raises(ValueError, match="cannot be negative"):
        forecast_function(history)


@pytest.mark.parametrize(
    "forecast_function",
    [
        naive_forecast,
        seasonal_naive_forecast,
        moving_average_forecast,
    ],
)
def test_nonfinite_sales_rejected(forecast_function):
    history = np.ones(28)
    history[0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        forecast_function(history)


def test_moving_average_uses_only_last_28_days():
    history = np.concatenate([np.full(10, 1000.0), np.full(28, 2.0)])

    predictions = moving_average_forecast(history)

    np.testing.assert_array_equal(
        predictions,
        np.full(28, 2.0),
    )


def test_forecasting_does_not_modify_history(history):
    original = history.copy()

    naive_forecast(history)
    seasonal_naive_forecast(history)
    moving_average_forecast(history)

    np.testing.assert_array_equal(history, original)
