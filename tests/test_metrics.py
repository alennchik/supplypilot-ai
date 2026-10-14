import math

import pandas as pd
import pytest

from src.metrics import mae, rmse, wape


def test_mae():
    assert mae([10, 20, 30], [8, 22, 26]) == pytest.approx(8 / 3)


def test_rmse():
    assert rmse([10, 20, 30], [8, 22, 26]) == pytest.approx(math.sqrt(8))


def test_wape_is_a_percentage():
    assert wape([10, 20, 30], [8, 22, 26]) == pytest.approx(100 * 8 / 60)


def test_perfect_forecast_has_zero_error():
    for metric in (mae, rmse, wape):
        assert metric([10, 20], [10, 20]) == 0.0


def test_wape_handles_some_zero_sales_days():
    assert wape([0, 10], [2, 8]) == pytest.approx(40.0)


def test_wape_rejects_all_zero_actuals():
    with pytest.raises(ValueError, match="undefined"):
        wape([0, 0], [0, 1])


@pytest.mark.parametrize("metric", [mae, rmse, wape])
def test_empty_inputs_are_rejected(metric):
    with pytest.raises(ValueError, match="empty"):
        metric([], [])


@pytest.mark.parametrize("metric", [mae, rmse, wape])
def test_mismatched_lengths_are_rejected(metric):
    with pytest.raises(ValueError, match="same length"):
        metric([1, 2], [1])


@pytest.mark.parametrize("metric", [mae, rmse, wape])
def test_nan_values_are_rejected(metric):
    with pytest.raises(ValueError, match="finite"):
        metric([1, float("nan")], [1, 2])


@pytest.mark.parametrize("metric", [mae, rmse, wape])
def test_non_numeric_values_are_rejected(metric):
    with pytest.raises(ValueError, match="numeric"):
        metric([1, "text"], [1, 2])


def test_series_indexes_must_match():
    actual = pd.Series([10, 20], index=[1, 2])
    predicted = pd.Series([10, 20], index=[2, 3])

    with pytest.raises(ValueError, match="indexes must match"):
        mae(actual, predicted)


def test_multidimensional_arrays_are_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        rmse([[1, 2]], [[1, 2]])


def test_negative_actual_sales_are_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        mae([-1, 3], [0, 2])
