"""Opt-in calendar integration retains Cycle 1 chronology and sales features."""

import numpy as np
import pandas as pd
import pytest

from src.calendar_features import CALENDAR_FEATURE_NAMES
from src.ml_dataset import build_ml_training_dataset, select_training_origins
from src.ml_inference import build_ml_inference_examples, predict_at_origin
from src.ml_model import train_global_model
from src.ml_validation import (
    TRAINING_CUTOFF,
    V1_ORIGIN,
    V2_ORIGIN,
    build_validation_feature_sets,
)


@pytest.fixture
def calendar_data():
    dates = pd.date_range("2015-04-01", "2016-01-03")
    sales = pd.DataFrame(
        {
            "series_id": np.repeat(["CA_item", "TX_item", "WI_item"], len(dates)),
            "date": np.tile(dates, 3),
            "quantity": np.tile(np.arange(len(dates)) % 9 + 1, 3),
            "price": np.tile(np.arange(len(dates)) + 1.0, 3),
            "price_available": True,
        }
    )
    metadata = pd.DataFrame(
        {
            "series_id": ["CA_item", "TX_item", "WI_item"],
            "store_id": ["CA_1", "TX_2", "WI_3"],
            "state_id": ["CA", "TX", "WI"],
            "dept_id": ["FOODS_1"] * 3,
            "cat_id": ["FOODS"] * 3,
        }
    )
    calendar = pd.DataFrame(
        {
            "date": dates,
            "event_name_1": pd.Series([None] * len(dates), dtype=object),
            "event_type_1": pd.Series([None] * len(dates), dtype=object),
            "event_name_2": pd.Series([None] * len(dates), dtype=object),
            "event_type_2": pd.Series([None] * len(dates), dtype=object),
            "snap_CA": np.arange(len(dates)) % 2,
            "snap_TX": 0,
            "snap_WI": 1,
        }
    )
    christmas = calendar["date"].eq(pd.Timestamp("2015-12-25"))
    calendar.loc[christmas, ["event_name_2", "event_type_2"]] = [
        "Christmas",
        "National",
    ]
    return sales, metadata, calendar


@pytest.mark.parametrize("n_origins", [8, 16])
def test_calendar_training_changes_only_seven_features_and_keeps_cutoff(
    calendar_data, n_origins
):
    sales, metadata, calendar = calendar_data
    original = build_ml_training_dataset(sales, metadata, TRAINING_CUTOFF, n_origins)
    result = build_ml_training_dataset(
        sales, metadata, TRAINING_CUTOFF, n_origins, calendar=calendar
    )
    pd.testing.assert_frame_equal(
        result.drop(columns=list(CALENDAR_FEATURE_NAMES)), original
    )
    assert len(result) == 3 * n_origins * 28
    assert result["date"].max() == TRAINING_CUTOFF
    assert tuple(sorted(result["origin"].unique())) == select_training_origins(
        TRAINING_CUTOFF, n_origins
    )
    assert result["origin"].dt.dayofweek.eq(6).all()
    assert not result.duplicated(["series_id", "origin", "horizon", "date"]).any()
    assert "state_id" not in result


def test_training_and_inference_schema_values_identical(calendar_data):
    sales, metadata, calendar = calendar_data
    training = build_ml_training_dataset(
        sales, metadata, V2_ORIGIN + pd.Timedelta(days=28), 1, calendar=calendar
    )
    inference = build_ml_inference_examples(
        sales, metadata, V2_ORIGIN, calendar=calendar
    )
    pd.testing.assert_frame_equal(training.drop(columns="target"), inference)
    christmas = inference["date"].eq(pd.Timestamp("2015-12-25"))
    assert inference.loc[christmas, "target_event_type_2"].eq("National").all()
    assert inference.loc[christmas, "target_is_christmas"].eq(1).all()
    # At origin (December 6), the event is absent. December 25 must still join.
    assert inference.loc[christmas, "target_is_event"].eq(1).all()


def test_calendar_model_predicts_through_origin_api(calendar_data):
    sales, metadata, calendar = calendar_data
    training = build_ml_training_dataset(
        sales, metadata, TRAINING_CUTOFF, calendar=calendar
    )
    model = train_global_model(
        training, calendar_features=True, objective="tweedie", n_estimators=5
    )
    predictions = predict_at_origin(
        model, sales, metadata, V2_ORIGIN, calendar=calendar
    )
    assert len(predictions) == 3 * 28
    assert np.isfinite(predictions["prediction"]).all()
    assert predictions["prediction"].ge(0).all()
    assert predictions["origin"].eq(V2_ORIGIN).all()


def test_only_scheduled_calendar_can_supply_post_origin_predictors(calendar_data):
    sales, metadata, calendar = calendar_data
    original = build_ml_inference_examples(
        sales, metadata, V2_ORIGIN, calendar=calendar
    )
    changed = sales.copy()
    future = changed["date"] > V2_ORIGIN
    changed.loc[future, "quantity"] = 999999
    changed.loc[future, "price"] = 888888.0
    changed.loc[future, "price_available"] = False
    pd.testing.assert_frame_equal(
        original,
        build_ml_inference_examples(changed, metadata, V2_ORIGIN, calendar=calendar),
    )
    pd.testing.assert_frame_equal(
        original,
        build_ml_inference_examples(
            sales.loc[~future], metadata, V2_ORIGIN, calendar=calendar
        ),
    )
    scheduled = calendar.copy()
    scheduled.loc[scheduled["date"] > V2_ORIGIN, "snap_CA"] = (
        1 - scheduled.loc[scheduled["date"] > V2_ORIGIN, "snap_CA"]
    )
    result = build_ml_inference_examples(sales, metadata, V2_ORIGIN, calendar=scheduled)
    pd.testing.assert_frame_equal(
        original.drop(columns="target_is_snap"), result.drop(columns="target_is_snap")
    )
    assert not original["target_is_snap"].equals(result["target_is_snap"])
    assert "price" not in result and "price_available" not in result


def test_fit_targets_and_features_ignore_sales_after_frozen_cutoff(calendar_data):
    sales, metadata, calendar = calendar_data
    original = build_ml_training_dataset(
        sales, metadata, TRAINING_CUTOFF, calendar=calendar
    )
    changed = sales.copy()
    changed.loc[changed["date"] > TRAINING_CUTOFF, ["quantity", "price"]] = 98765
    result = build_ml_training_dataset(
        changed, metadata, TRAINING_CUTOFF, calendar=calendar
    )
    pd.testing.assert_frame_equal(original, result)


def test_validation_opt_in_preserves_fixed_cohort_labels_and_dates(calendar_data):
    sales, metadata, calendar = calendar_data
    original_v1, original_v2 = build_validation_feature_sets(sales, metadata)
    v1, v2 = build_validation_feature_sets(sales, metadata, calendar=calendar)
    for original, result, origin in (
        (original_v1, v1, V1_ORIGIN),
        (original_v2, v2, V2_ORIGIN),
    ):
        pd.testing.assert_frame_equal(
            result.drop(columns=list(CALENDAR_FEATURE_NAMES)), original
        )
        assert len(result) == 3 * 28
        assert result["origin"].eq(origin).all()
        assert set(result["series_id"]) == {"CA_item", "TX_item", "WI_item"}
        assert set(result["horizon"]) == set(range(1, 29))
        assert not result.duplicated(["series_id", "origin", "horizon", "date"]).any()


@pytest.mark.parametrize("builder", ["training", "inference", "validation"])
def test_original_metadata_state_is_checked_before_discarding_it(
    calendar_data, builder
):
    sales, metadata, calendar = calendar_data
    metadata.loc[0, "state_id"] = "TX"
    with pytest.raises(ValueError, match="Inconsistent state_id"):
        if builder == "training":
            build_ml_training_dataset(
                sales, metadata, TRAINING_CUTOFF, calendar=calendar
            )
        elif builder == "inference":
            build_ml_inference_examples(sales, metadata, V2_ORIGIN, calendar=calendar)
        else:
            build_validation_feature_sets(sales, metadata, calendar=calendar)


def test_sunday_rule_only_applies_to_calendar_opt_in(calendar_data):
    sales, metadata, calendar = calendar_data
    monday = V2_ORIGIN + pd.Timedelta(days=1)
    assert len(build_ml_inference_examples(sales, metadata, monday)) == 3 * 28
    with pytest.raises(ValueError, match="Sunday"):
        build_ml_inference_examples(sales, metadata, monday, calendar=calendar)
    with pytest.raises(ValueError, match="Sunday"):
        build_ml_training_dataset(
            sales, metadata, TRAINING_CUTOFF + pd.Timedelta(days=1), calendar=calendar
        )
