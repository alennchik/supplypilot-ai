"""Integration tests for the global LightGBM training pipeline."""

import json

import pandas as pd
import pytest

from scripts.train_ml_model import (
    train_from_parquet,
    validate_training_dataset,
)
from src.ml_features import DEMAND_FEATURE_NAMES
from src.ml_model import load_global_model, predict_global_model


def make_dataset():
    """Create two complete 28-day forecast paths."""
    origin = pd.Timestamp("2015-09-06")
    rows = []

    for series_number in range(2):
        for horizon in range(1, 29):
            row = {
                "series_id": f"item_{series_number}",
                "origin": origin,
                "date": origin + pd.Timedelta(days=horizon),
                "horizon": horizon,
                "store_id": f"S{series_number}",
                "dept_id": "D1",
                "cat_id": "FOODS",
                "target": float(series_number + horizon % 7),
            }

            for feature in DEMAND_FEATURE_NAMES:
                row[feature] = float(series_number + 1)

            rows.append(row)

    return pd.DataFrame(rows)


def test_valid_training_dataset():
    data = make_dataset()

    validate_training_dataset(data, "2015-10-04")


def test_reject_future_training_targets():
    data = make_dataset()

    with pytest.raises(ValueError, match="training cutoff"):
        validate_training_dataset(data, "2015-10-03")


def test_reject_incomplete_forecast_path():
    data = make_dataset().iloc[:-1].copy()

    with pytest.raises(ValueError, match="all 28 horizons"):
        validate_training_dataset(data, "2015-10-04")


def test_reject_duplicate_examples():
    data = make_dataset()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_training_dataset(data, "2015-10-04")


def test_reject_misaligned_target_date():
    data = make_dataset()
    data.loc[0, "date"] += pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="origin plus horizon"):
        validate_training_dataset(data, "2015-10-04")


def test_train_save_reload_pipeline(tmp_path):
    data = make_dataset()

    input_path = tmp_path / "training.parquet"
    model_path = tmp_path / "models" / "global.txt"
    manifest_path = tmp_path / "reports" / "manifest.json"

    data.to_parquet(input_path, index=False)

    summary = train_from_parquet(
        input_path=input_path,
        model_path=model_path,
        manifest_path=manifest_path,
        cutoff="2015-10-04",
        n_estimators=5,
        n_jobs=2,
    )

    assert summary["training_rows"] == 56
    assert summary["unique_series"] == 2
    assert summary["training_cutoff"] == "2015-10-04"

    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()
    assert manifest_path.exists()

    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["training_rows"] == 56

    restored_model = load_global_model(model_path)
    predictions = predict_global_model(restored_model, data.head(28))

    assert len(predictions) == 28
    assert predictions["prediction"].notna().all()
    assert (predictions["prediction"] >= 0).all()
