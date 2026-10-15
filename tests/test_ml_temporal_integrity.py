"""Integration tests for temporal boundaries in ML training data."""

import numpy as np
import pandas as pd
import pytest

from src.ml_dataset import (
    build_ml_training_dataset,
    select_training_origins,
)
from src.ml_features import DEMAND_FEATURE_NAMES


def make_data(days: int = 150):
    """Create two daily series with deterministic, nonnegative demand."""
    dates = pd.date_range("2015-01-01", periods=days, freq="D")

    sales = pd.DataFrame(
        {
            "series_id": np.repeat(["A", "B"], days),
            "date": np.tile(dates, 2),
            "quantity": np.concatenate(
                [
                    np.arange(days, dtype=np.int32) % 6,
                    np.zeros(days, dtype=np.int32),
                ]
            ),
        }
    )

    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "store_id": ["S1", "S2"],
            "dept_id": ["D1", "D2"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    return sales, metadata, dates


def test_m5_validation_and_final_origins_match_frozen_protocol():
    final_cutoff = pd.Timestamp("2016-01-03")  # M5 d_1801
    validation_cutoff = final_cutoff - pd.Timedelta(days=56)

    validation_origins = select_training_origins(validation_cutoff, 8)
    final_origins = select_training_origins(final_cutoff, 8)

    assert validation_origins[-1] == validation_cutoff - pd.Timedelta(days=28)
    assert final_origins[-1] == final_cutoff - pd.Timedelta(days=28)

    assert validation_origins[0] == validation_cutoff - pd.Timedelta(days=77)
    assert final_origins[0] == final_cutoff - pd.Timedelta(days=77)

    assert all(origin.dayofweek == 6 for origin in validation_origins)
    assert all(origin.dayofweek == 6 for origin in final_origins)


def test_every_target_matches_its_series_and_calendar_date():
    sales, metadata, dates = make_data()

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=3)

    assert (result["horizon"] >= 1).all()
    assert (result["horizon"] <= 28).all()

    expected_dates = result["origin"] + pd.to_timedelta(result["horizon"], unit="D")

    pd.testing.assert_series_equal(
        result["date"],
        expected_dates,
        check_names=False,
    )

    lookup = sales.set_index(["series_id", "date"])["quantity"]
    keys = pd.MultiIndex.from_frame(result[["series_id", "date"]])
    expected_targets = lookup.reindex(keys)

    assert expected_targets.notna().all()
    np.testing.assert_array_equal(
        result["target"].to_numpy(),
        expected_targets.to_numpy(),
    )

    assert (result["date"] <= dates[-1]).all()


def test_rows_after_cutoff_have_no_effect_on_training_dataset():
    sales, metadata, dates = make_data()
    cutoff = dates[-11]

    full_input = build_ml_training_dataset(sales, metadata, cutoff, n_origins=8)

    truncated_sales = sales.loc[sales["date"] <= cutoff].copy()

    truncated_input = build_ml_training_dataset(
        truncated_sales, metadata, cutoff, n_origins=8
    )

    pd.testing.assert_frame_equal(full_input, truncated_input)


def test_changing_future_demand_does_not_change_origin_features():
    sales, metadata, dates = make_data()
    cutoff = dates[-1]

    original = build_ml_training_dataset(sales, metadata, cutoff, n_origins=3)

    for origin in select_training_origins(cutoff, 3):
        changed_sales = sales.copy()
        changed_sales.loc[changed_sales["date"] > origin, "quantity"] = 999

        changed = build_ml_training_dataset(
            changed_sales, metadata, cutoff, n_origins=3
        )

        original_features = original.loc[
            original["origin"] == origin, DEMAND_FEATURE_NAMES
        ].reset_index(drop=True)

        changed_features = changed.loc[
            changed["origin"] == origin, DEMAND_FEATURE_NAMES
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            original_features,
            changed_features,
        )


def test_missing_target_in_earlier_training_window_is_rejected():
    sales, metadata, dates = make_data()

    origins = select_training_origins(dates[-1], n_origins=2)
    missing_target_date = origins[0] + pd.Timedelta(days=1)

    damaged = sales.loc[
        ~((sales["series_id"] == "A") & (sales["date"] == missing_target_date))
    ].copy()

    with pytest.raises(ValueError, match="Missing target"):
        build_ml_training_dataset(damaged, metadata, dates[-1], n_origins=2)


def test_history_eligibility_is_recomputed_at_each_origin():
    sales, metadata, dates = make_data()

    origins = select_training_origins(dates[-1], n_origins=2)
    second_origin = origins[-1]

    # B has exactly 28 recorded days at the second origin,
    # but does not have 28 days at the first origin.
    start_of_b = second_origin - pd.Timedelta(days=27)

    sales = sales.loc[
        (sales["series_id"] != "B") | (sales["date"] >= start_of_b)
    ].copy()

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=2)

    first_series = set(result.loc[result["origin"] == origins[0], "series_id"])
    second_series = set(result.loc[result["origin"] == origins[1], "series_id"])

    assert first_series == {"A"}
    assert second_series == {"A", "B"}
    assert len(result) == 3 * 28


def test_shuffled_input_rows_produce_identical_training_data():
    sales, metadata, dates = make_data()

    expected = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=3)

    shuffled_sales = sales.sample(frac=1, random_state=42)
    shuffled_metadata = metadata.sample(frac=1, random_state=17)

    actual = build_ml_training_dataset(
        shuffled_sales,
        shuffled_metadata,
        dates[-1],
        n_origins=3,
    )

    pd.testing.assert_frame_equal(expected, actual)
