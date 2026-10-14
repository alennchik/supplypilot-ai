import pandas as pd
import pytest

from src.baselines import naive_forecast, seasonal_naive_forecast


@pytest.fixture
def validation():
    return pd.DataFrame(
        {
            "sales": [999, 888, 777],
            "lag_1": [10, 20, 30],
            "lag_7": [7, 14, 21],
        },
        index=[10, 11, 12],
    )


def test_naive_forecast_uses_previous_day(validation):
    result = naive_forecast(validation)

    assert result.tolist() == [10, 20, 30]
    assert result.name == "naive"
    assert result.index.equals(validation.index)


def test_seasonal_naive_uses_previous_week(validation):
    result = seasonal_naive_forecast(validation)

    assert result.tolist() == [7, 14, 21]
    assert result.name == "seasonal_naive"


def test_baselines_ignore_current_day_sales(validation):
    original_naive = naive_forecast(validation)
    original_seasonal = seasonal_naive_forecast(validation)

    changed = validation.copy()
    changed["sales"] = [999999, 999999, 999999]

    pd.testing.assert_series_equal(naive_forecast(changed), original_naive)
    pd.testing.assert_series_equal(
        seasonal_naive_forecast(changed), original_seasonal
    )


def test_baselines_do_not_modify_input(validation):
    before = validation.copy(deep=True)

    naive_forecast(validation)
    seasonal_naive_forecast(validation)

    pd.testing.assert_frame_equal(validation, before)


@pytest.mark.parametrize("method", [naive_forecast, seasonal_naive_forecast])
def test_empty_validation_is_rejected(method):
    with pytest.raises(ValueError, match="cannot be empty"):
        method(pd.DataFrame())


@pytest.mark.parametrize(
    ("method", "missing_column"),
    [
        (naive_forecast, "lag_1"),
        (seasonal_naive_forecast, "lag_7"),
    ],
)
def test_missing_lag_is_rejected(validation, method, missing_column):
    with pytest.raises(ValueError, match="Missing required feature"):
        method(validation.drop(columns=missing_column))


def test_missing_lag_value_is_rejected(validation):
    validation.loc[10, "lag_1"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        naive_forecast(validation)


def test_infinite_lag_value_is_rejected(validation):
    validation["lag_1"] = validation["lag_1"].astype(float)
    validation.loc[10, "lag_1"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        naive_forecast(validation)


def test_negative_lag_value_is_rejected(validation):
    validation.loc[10, "lag_1"] = -1

    with pytest.raises(ValueError, match="negative"):
        naive_forecast(validation)


def test_non_numeric_lag_value_is_rejected(validation):
    validation["lag_1"] = ["a", "b", "c"]

    with pytest.raises(TypeError, match="numeric"):
        naive_forecast(validation)
