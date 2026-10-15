"""Tests for demand classification and pooled forecasting metrics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import build_demand_profile, evaluate_forecasts

ORIGIN = pd.Timestamp("2016-01-03")


def make_sales(n_days=28):
    dates = pd.date_range(end=ORIGIN, periods=n_days)

    return pd.DataFrame(
        {
            "series_id": np.repeat(["A", "B", "C", "D"], n_days),
            "date": np.tile(dates, 4),
            "quantity": np.concatenate(
                [
                    np.ones(n_days),
                    np.r_[
                        np.zeros(n_days // 4),
                        np.ones(n_days - n_days // 4),
                    ],
                    np.r_[
                        np.zeros(n_days * 3 // 4),
                        np.ones(n_days - n_days * 3 // 4),
                    ],
                    np.r_[
                        np.zeros(n_days * 3 // 4 + 1),
                        np.ones(n_days - n_days * 3 // 4 - 1),
                    ],
                ]
            ),
        }
    )


def make_predictions():
    return pd.DataFrame(
        {
            "series_id": ["A", "B", "A", "B"],
            "origin": [ORIGIN] * 4,
            "date": [ORIGIN + pd.Timedelta(days=1)] * 4,
            "model": [
                "naive",
                "naive",
                "seasonal_naive",
                "seasonal_naive",
            ],
            "actual": [2, 0, 2, 0],
            "predicted": [1, 1, 2, 0],
        }
    )


def test_demand_threshold_boundaries():
    profile = build_demand_profile(
        make_sales(),
        ["A", "B", "C", "D"],
        ORIGIN,
    )

    groups = profile.set_index("series_id")["demand_group"].to_dict()

    assert groups == {
        "A": "frequent",
        "B": "mixed",
        "C": "mixed",
        "D": "sparse",
    }

    assert (profile["observed_days"] == 28).all()


def test_lookback_ignores_older_history():
    dates = pd.date_range(end=ORIGIN, periods=380)

    sales = pd.DataFrame(
        {
            "series_id": "A",
            "date": dates,
            "quantity": [0] * 15 + [1] * 365,
        }
    )

    profile = build_demand_profile(sales, ["A"], ORIGIN)

    assert profile.iloc[0]["observed_days"] == 365
    assert profile.iloc[0]["zero_fraction"] == 0


def test_future_changes_do_not_affect_profile():
    sales = make_sales()

    future = sales.loc[sales["date"] == ORIGIN].copy()
    future["date"] = ORIGIN + pd.Timedelta(days=1)
    future["quantity"] = 999

    extended = pd.concat([sales, future], ignore_index=True)

    pd.testing.assert_frame_equal(
        build_demand_profile(sales, ["A", "B"], ORIGIN),
        build_demand_profile(extended, ["A", "B"], ORIGIN),
    )


def test_incomplete_recent_history_is_error():
    sales = make_sales()

    sales = sales.loc[~((sales["series_id"] == "A") & (sales["date"] == ORIGIN))]

    with pytest.raises(ValueError, match="28 consecutive"):
        build_demand_profile(sales, ["A"], ORIGIN)


def test_duplicate_history_is_error():
    sales = make_sales()
    sales = pd.concat([sales, sales.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_demand_profile(sales, ["A"], ORIGIN)


def test_pooled_wape_and_breakdowns():
    profile = build_demand_profile(make_sales(), ["A", "B"], ORIGIN)

    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    results = evaluate_forecasts(make_predictions(), metadata, profile)

    overall = results["overall"].set_index("model")

    assert overall.loc["naive", "wape"] == pytest.approx(100)
    assert overall.loc["naive", "mae"] == pytest.approx(1)
    assert overall.loc["seasonal_naive", "wape"] == pytest.approx(0)

    assert overall.loc["naive", "series_count"] == 2
    assert len(results["by_window"]) == 2
    assert len(results["by_category"]) == 4
    assert len(results["by_sparsity"]) == 4

    zero_demand = results["by_category"].query(
        "model == 'naive' and cat_id == 'HOBBIES'"
    )

    assert np.isnan(zero_demand.iloc[0]["wape"])


def test_missing_category_is_error():
    profile = build_demand_profile(make_sales(), ["A", "B"], ORIGIN)

    metadata = pd.DataFrame(
        {
            "series_id": ["A"],
            "cat_id": ["FOODS"],
        }
    )

    with pytest.raises(ValueError, match="Missing category"):
        evaluate_forecasts(make_predictions(), metadata, profile)


def test_duplicate_forecast_keys_are_error():
    profile = build_demand_profile(make_sales(), ["A", "B"], ORIGIN)

    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    predictions = make_predictions()

    predictions = pd.concat(
        [predictions, predictions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate forecast"):
        evaluate_forecasts(predictions, metadata, profile)
