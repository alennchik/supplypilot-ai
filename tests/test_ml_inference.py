"""Tests for leakage-safe 28-day LightGBM inference."""

import numpy as np
import pandas as pd
import pytest

from src.ml_dataset import build_ml_training_dataset
from src.ml_inference import (
    build_ml_inference_examples,
    predict_at_origin,
)
from src.ml_model import train_global_model


@pytest.fixture
def sample_data():
    """Three series: two eligible, one with insufficient history."""
    origin = pd.Timestamp("2015-09-05")
    dates = pd.date_range("2015-08-01", "2015-10-05")

    rows = []

    for series_number, series_id in enumerate(("A", "B", "C")):
        for day_number, date in enumerate(dates):
            # B has fewer than 28 observed days at the origin.
            if series_id == "B" and date < pd.Timestamp("2015-08-20"):
                continue

            rows.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "quantity": int((day_number + series_number) % 11),
                }
            )

    sales = pd.DataFrame(rows)

    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B", "C"],
            "store_id": ["S1", "S1", "S2"],
            "dept_id": ["D1", "D1", "D2"],
            "cat_id": ["FOODS", "FOODS", "HOBBIES"],
        }
    )

    return sales, metadata, origin


def test_generates_28_forecasts_per_eligible_series(sample_data):
    sales, metadata, origin = sample_data

    examples = build_ml_inference_examples(sales, metadata, origin)

    assert examples.groupby("series_id").size().to_dict() == {
        "A": 28,
        "C": 28,
    }
    assert "target" not in examples.columns
    assert examples["horizon"].between(1, 28).all()
    assert (
        examples["date"]
        == examples["origin"] + pd.to_timedelta(examples["horizon"], unit="D")
    ).all()
    assert examples["date"].min() == origin + pd.Timedelta(days=1)
    assert examples["date"].max() == origin + pd.Timedelta(days=28)


def test_inference_matches_training_features(sample_data):
    sales, metadata, origin = sample_data

    training = build_ml_training_dataset(
        sales,
        metadata,
        training_cutoff=origin + pd.Timedelta(days=28),
        n_origins=1,
    )

    inference = build_ml_inference_examples(sales, metadata, origin)

    # Training contains observed future targets; inference does not.
    # All identifiers, metadata and historical features must match.
    expected = training.drop(columns=["target"])

    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        inference.reset_index(drop=True),
        check_dtype=False,
    )


def test_future_sales_changes_cannot_affect_features(sample_data):
    sales, metadata, origin = sample_data

    original = build_ml_inference_examples(sales, metadata, origin)

    changed = sales.copy()
    changed.loc[changed["date"] > origin, "quantity"] += 100

    result = build_ml_inference_examples(changed, metadata, origin)

    pd.testing.assert_frame_equal(original, result)


def test_future_sales_not_required(sample_data):
    sales, metadata, origin = sample_data

    original = build_ml_inference_examples(sales, metadata, origin)

    historical_only = sales.loc[sales["date"] <= origin].copy()

    result = build_ml_inference_examples(
        historical_only,
        metadata,
        origin,
    )

    pd.testing.assert_frame_equal(original, result)


def test_no_eligible_series_raises_error(sample_data):
    sales, metadata, _ = sample_data

    with pytest.raises(ValueError, match="No eligible series"):
        build_ml_inference_examples(
            sales,
            metadata,
            origin="2015-07-30",
        )


def test_end_to_end_model_prediction(sample_data):
    sales, metadata, origin = sample_data

    training = build_ml_training_dataset(
        sales,
        metadata,
        training_cutoff=origin + pd.Timedelta(days=28),
        n_origins=1,
    )

    model = train_global_model(
        training,
        n_estimators=5,
        n_jobs=2,
    )

    predictions = predict_at_origin(
        model,
        sales,
        metadata,
        origin,
    )

    assert len(predictions) == 56
    assert set(predictions["series_id"]) == {"A", "C"}
    assert np.isfinite(predictions["prediction"]).all()
    assert (predictions["prediction"] >= 0).all()
    assert not predictions.duplicated(["series_id", "origin", "horizon"]).any()
