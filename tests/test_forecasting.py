import pandas as pd
import pytest

from src.forecasting import (
    FEATURE_COLUMNS,
    add_forecasting_features,
    prepare_daily_sales,
    split_train_validation,
)


@pytest.fixture
def sales_50_days():
    return pd.Series(range(1, 51), index=[f"d_{day}" for day in range(1, 51)])


def test_prepare_daily_sales_sorts_days_numerically():
    sales = pd.Series([30, 10, 20], index=["d_3", "d_1", "d_2"])

    actual = prepare_daily_sales(sales)

    assert actual["day"].tolist() == ["d_1", "d_2", "d_3"]
    assert actual["day_number"].tolist() == [1, 2, 3]
    assert actual["sales"].tolist() == [10, 20, 30]


def test_prepare_daily_sales_rejects_empty_series():
    with pytest.raises(ValueError, match="cannot be empty"):
        prepare_daily_sales(pd.Series(dtype="int64"))


def test_prepare_daily_sales_rejects_invalid_day_labels():
    with pytest.raises(ValueError, match="Day labels"):
        prepare_daily_sales(pd.Series([10], index=["Monday"]))


def test_prepare_daily_sales_rejects_duplicate_days():
    with pytest.raises(ValueError, match="unique"):
        prepare_daily_sales(pd.Series([10, 20], index=["d_1", "d_1"]))


def test_prepare_daily_sales_rejects_missing_days():
    with pytest.raises(ValueError, match="consecutive"):
        prepare_daily_sales(pd.Series([10, 20], index=["d_1", "d_3"]))


def test_prepare_daily_sales_rejects_negative_sales():
    with pytest.raises(ValueError, match="non-negative"):
        prepare_daily_sales(pd.Series([10, -2], index=["d_1", "d_2"]))


def test_prepare_daily_sales_rejects_missing_sales():
    with pytest.raises(ValueError, match="no missing values"):
        prepare_daily_sales(pd.Series([10.0, float("nan")], index=["d_1", "d_2"]))


def test_lag_features_reference_past_days(sales_50_days):
    prepared = prepare_daily_sales(sales_50_days)
    featured = add_forecasting_features(prepared)
    day_29 = featured.loc[featured["day_number"] == 29].iloc[0]

    assert day_29["sales"] == 29
    assert day_29["lag_1"] == 28
    assert day_29["lag_7"] == 22
    assert day_29["lag_28"] == 1


def test_rolling_features_exclude_current_day(sales_50_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))
    day_29 = featured.loc[featured["day_number"] == 29].iloc[0]

    assert day_29["rolling_mean_7"] == 25.0
    assert day_29["rolling_mean_28"] == 14.5


def test_first_28_days_have_incomplete_features(sales_50_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))

    assert pd.isna(featured.loc[0, "lag_1"])
    assert pd.isna(featured.loc[27, "lag_28"])
    assert featured.loc[28, list(FEATURE_COLUMNS)].notna().all()


def test_changing_future_sales_does_not_change_earlier_features(sales_50_days):
    original = add_forecasting_features(prepare_daily_sales(sales_50_days))

    changed = sales_50_days.copy()
    changed.loc["d_40"] = 9999
    recomputed = add_forecasting_features(prepare_daily_sales(changed))

    pd.testing.assert_frame_equal(
        original.loc[:39, list(FEATURE_COLUMNS)],
        recomputed.loc[:39, list(FEATURE_COLUMNS)],
    )

    assert recomputed.loc[40, "lag_1"] == 9999


def test_feature_generation_does_not_mutate_input(sales_50_days):
    prepared = prepare_daily_sales(sales_50_days)
    before = prepared.copy(deep=True)

    add_forecasting_features(prepared)

    pd.testing.assert_frame_equal(prepared, before)


def test_split_is_chronological_and_no_target_rows_overlap(sales_50_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))
    train, validation = split_train_validation(featured, validation_days=7)

    assert train["day_number"].tolist() == list(range(29, 44))
    assert validation["day_number"].tolist() == list(range(44, 51))
    assert train["day_number"].max() < validation["day_number"].min()
    assert not train[list(FEATURE_COLUMNS)].isna().any().any()
    assert not validation[list(FEATURE_COLUMNS)].isna().any().any()


def test_validation_can_use_previous_observed_sales(sales_50_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))
    _, validation = split_train_validation(featured, validation_days=7)

    day_45 = validation.loc[validation["day_number"] == 45].iloc[0]

    assert day_45["lag_1"] == 44


def test_split_rejects_validation_longer_than_history(sales_50_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))

    with pytest.raises(ValueError, match="Not enough"):
        split_train_validation(featured, validation_days=22)


@pytest.mark.parametrize("validation_days", [0, -1])
def test_split_rejects_invalid_validation_values(sales_50_days, validation_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))

    with pytest.raises(ValueError, match="positive integer"):
        split_train_validation(featured, validation_days=validation_days)


@pytest.mark.parametrize("validation_days", [1.5, True])
def test_split_rejects_invalid_validation_types(sales_50_days, validation_days):
    featured = add_forecasting_features(prepare_daily_sales(sales_50_days))

    with pytest.raises(TypeError, match="positive integer"):
        split_train_validation(featured, validation_days=validation_days)
