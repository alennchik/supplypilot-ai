import numpy as np
import pandas as pd
import pytest

from src.ml_features import DEMAND_FEATURE_NAMES, demand_features, features_at_origin


def test_constant_demand() -> None:
    result = demand_features(np.full(56, 3))
    assert tuple(result) == DEMAND_FEATURE_NAMES
    assert result["sales_at_origin"] == 3
    assert result["sales_lag_28_at_origin"] == 3
    assert result["rolling_mean_56"] == 3
    assert result["rolling_std_28"] == 0
    assert result["rolling_sum_7"] == 21
    assert result["zero_fraction_56"] == 0


def test_lags_use_origin_not_target_day() -> None:
    result = demand_features(np.arange(1, 57))
    assert result["sales_at_origin"] == 56
    assert result["sales_lag_1_at_origin"] == 55
    assert result["sales_lag_7_at_origin"] == 49
    assert result["sales_lag_14_at_origin"] == 42
    assert result["sales_lag_28_at_origin"] == 28
    assert result["rolling_mean_7"] == 53
    assert result["rolling_sum_28"] == sum(range(29, 57))
    assert result["rolling_std_7"] == pytest.approx(np.std(np.arange(50, 57)))


def test_real_zeros_are_not_missing() -> None:
    history = np.zeros(56)
    result = demand_features(history)
    assert result["zero_fraction_28"] == 1
    assert result["zero_fraction_56"] == 1
    assert result["rolling_mean_28"] == 0
    assert result["sales_at_origin"] == 0


def test_incomplete_56_day_history_produces_missing_long_features() -> None:
    result = demand_features(np.ones(28))
    assert np.isnan(result["sales_lag_28_at_origin"])
    assert np.isnan(result["rolling_mean_56"])
    assert np.isnan(result["zero_fraction_56"])
    assert result["rolling_mean_28"] == 1


def test_missing_older_observation_disables_only_affected_windows() -> None:
    history = np.ones(56)
    history[4] = np.nan
    result = demand_features(history)
    assert np.isnan(result["rolling_mean_56"])
    assert result["rolling_mean_28"] == 1


@pytest.mark.parametrize(
    "history",
    [np.ones(27), np.concatenate([np.ones(27), [np.nan]])],
)
def test_insufficient_recent_history_rejected(history: np.ndarray) -> None:
    with pytest.raises(ValueError, match="28 consecutive"):
        demand_features(history)


@pytest.mark.parametrize("bad", [-1, 2.5, np.inf])
def test_invalid_observed_quantities_rejected(bad: float) -> None:
    history = np.ones(56)
    history[0] = bad
    with pytest.raises(ValueError, match="quantities|Quantities"):
        demand_features(history)


def test_matrix_input_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        demand_features(np.ones((28, 1)))


def test_future_sales_cannot_change_features() -> None:
    dates = pd.date_range("2016-01-01", periods=65)
    sales = pd.DataFrame({"date": dates, "quantity": np.arange(65) % 4})
    origin = dates[55]

    initial = features_at_origin(sales, origin)

    modified = sales.copy()
    modified.loc[modified["date"] > origin, "quantity"] = 1_000_000

    assert features_at_origin(modified, origin) == initial


def test_missing_calendar_day_is_not_treated_as_zero() -> None:
    dates = pd.date_range("2016-01-01", periods=40)
    sales = pd.DataFrame({"date": dates, "quantity": np.ones(40)})
    sales = sales.loc[sales["date"] != dates[-4]]

    with pytest.raises(ValueError, match="28 consecutive"):
        features_at_origin(sales, dates[-1])


def test_missing_observation_outside_last_28_returns_nan_long_window() -> None:
    dates = pd.date_range("2016-01-01", periods=56)
    sales = pd.DataFrame({"date": dates, "quantity": np.ones(56)})
    sales = sales.loc[sales["date"] != dates[0]]

    result = features_at_origin(sales, dates[-1])

    assert np.isnan(result["rolling_mean_56"])
    assert result["rolling_mean_28"] == 1


def test_duplicate_past_dates_rejected() -> None:
    dates = pd.date_range("2016-01-01", periods=28)
    sales = pd.DataFrame({"date": dates, "quantity": np.ones(28)})
    sales = pd.concat([sales, sales.iloc[[-1]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        features_at_origin(sales, dates[-1])


def test_null_recorded_quantity_rejected() -> None:
    dates = pd.date_range("2016-01-01", periods=28)
    sales = pd.DataFrame({"date": dates, "quantity": np.ones(28)})
    sales.loc[2, "quantity"] = np.nan

    with pytest.raises(ValueError, match="cannot be null"):
        features_at_origin(sales, dates[-1])
