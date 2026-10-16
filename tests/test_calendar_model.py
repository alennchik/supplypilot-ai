"""Calendar model schema, categorical parity and serialization contracts."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.calendar_features import CALENDAR_FEATURE_NAMES
from src.ml_model import (
    CATEGORICAL_FEATURES,
    CYCLE2_CATEGORICAL_FEATURES,
    CYCLE2_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    _prepare_features,
    load_global_model,
    predict_global_model,
    save_global_model,
    train_global_model,
)


def calendar_examples():
    """Small multi-series direct examples; no real M5 observations are used."""
    rows = []
    for origin_number in range(3):
        origin = pd.Timestamp("2015-09-06") + pd.Timedelta(weeks=origin_number)
        for series_number in range(4):
            for horizon in range(1, 29):
                target_date = origin + pd.Timedelta(days=horizon)
                row = {name: float(series_number + 1) for name in NUMERIC_FEATURES}
                row.update(
                    series_id=f"item_{series_number}",
                    origin=origin,
                    date=target_date,
                    horizon=horizon,
                    store_id=f"CA_{series_number % 2}",
                    dept_id=f"DEPT_{series_number % 2}",
                    cat_id="FOODS",
                    target_month=target_date.month,
                    target_day_of_month=target_date.day,
                    target_is_event=int(horizon % 2 == 0),
                    target_event_type_1="Cultural" if horizon % 2 == 0 else "None",
                    target_event_type_2="Religious" if horizon % 4 == 0 else "None",
                    target_is_christmas=0,
                    target_is_snap=int(horizon % 3 == 0),
                    target=float(series_number + horizon % 7 + 1),
                )
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def calendar_model():
    data = calendar_examples()
    training = data.loc[data["origin"] < data["origin"].max()]
    validation = data.loc[data["origin"] == data["origin"].max()]
    return train_global_model(
        training,
        validation_data=validation,
        calendar_features=True,
        objective="tweedie",
        n_estimators=30,
        early_stopping_rounds=5,
    )


def test_schema_adds_exactly_seven_and_preserves_cycle1_order():
    data = calendar_examples()
    data["event_name_1"] = "Must never enter model"
    data["future_observed_price"] = 99999
    features, levels = _prepare_features(data, calendar_features=True)

    assert CYCLE2_FEATURE_COLUMNS == (*FEATURE_COLUMNS, *CALENDAR_FEATURE_NAMES)
    assert len(CYCLE2_FEATURE_COLUMNS) == len(FEATURE_COLUMNS) + 7
    assert CYCLE2_CATEGORICAL_FEATURES == (
        *CATEGORICAL_FEATURES,
        "target_event_type_1",
        "target_event_type_2",
    )
    assert features.columns.tolist() == list(CYCLE2_FEATURE_COLUMNS)
    assert list(levels) == list(CYCLE2_CATEGORICAL_FEATURES)
    assert not {"event_name_1", "future_observed_price", "target"}.intersection(
        features
    )
    assert "weekday" not in features


def test_calendar_schema_is_explicit_opt_in():
    data = calendar_examples()
    features, levels = _prepare_features(data)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert tuple(levels) == CATEGORICAL_FEATURES
    model = train_global_model(data, n_estimators=3)
    assert model.calendar_features is False
    assert model.feature_columns == FEATURE_COLUMNS


def test_literal_none_is_a_category_but_unseen_event_types_are_missing(calendar_model):
    examples = calendar_examples().head(28)
    original, _ = _prepare_features(
        examples, calendar_model.category_levels, calendar_features=True
    )
    for name in ("target_event_type_1", "target_event_type_2"):
        none_rows = examples[name] == "None"
        assert "None" in original[name].cat.categories
        assert not original.loc[none_rows, name].isna().any()
        assert (original.loc[none_rows, name] == "None").all()

    examples = examples.copy()
    examples["target_event_type_1"] = "UnseenEventType"
    prepared, learned = _prepare_features(
        examples, calendar_model.category_levels, calendar_features=True
    )
    assert prepared["target_event_type_1"].isna().all()
    assert learned == calendar_model.category_levels
    predictions = predict_global_model(calendar_model, examples)
    assert np.isfinite(predictions["prediction"]).all()


def test_none_level_is_reserved_when_every_training_date_has_events():
    training = calendar_examples()
    training["target_event_type_1"] = "Cultural"
    training["target_event_type_2"] = "Religious"
    training["target_is_event"] = 1
    model = train_global_model(training, calendar_features=True, n_estimators=3)
    inference = training.head(28).copy()
    inference["target_event_type_1"] = "None"
    inference["target_event_type_2"] = "None"
    inference["target_is_event"] = 0

    features, _ = _prepare_features(
        inference, model.category_levels, calendar_features=True
    )
    for name in ("target_event_type_1", "target_event_type_2"):
        assert "None" in model.category_levels[name]
        assert (features[name] == "None").all()
        assert not features[name].isna().any()
    assert np.isfinite(predict_global_model(model, inference)["prediction"]).all()


def test_training_inference_and_reload_preserve_exact_schema(calendar_model, tmp_path):
    examples = calendar_examples().sample(frac=1, random_state=11).head(56)
    expected = predict_global_model(calendar_model, examples)
    path = tmp_path / "calendar.txt"
    save_global_model(calendar_model, path)
    loaded = load_global_model(path)
    actual = predict_global_model(loaded, examples)

    assert loaded.calendar_features is True
    assert loaded.feature_columns == CYCLE2_FEATURE_COLUMNS
    assert loaded.booster.feature_name() == list(CYCLE2_FEATURE_COLUMNS)
    assert loaded.category_levels == calendar_model.category_levels
    assert loaded.objective == "tweedie"
    assert loaded.booster.params["tweedie_variance_power"] == 1.5
    assert loaded.best_iteration == calendar_model.best_iteration
    assert loaded.validation_mae_history == calendar_model.validation_mae_history
    pd.testing.assert_frame_equal(expected, actual)

    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["categorical_features"] == list(CYCLE2_CATEGORICAL_FEATURES)
    assert metadata["unseen_category_policy"] == "missing"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("target_month", 0),
        ("target_month", 13),
        ("target_month", 2.5),
        ("target_day_of_month", 0),
        ("target_day_of_month", 32),
        ("target_is_event", 2),
        ("target_is_snap", -1),
        ("target_is_christmas", 0.5),
        ("target_is_snap", np.nan),
        ("target_month", np.inf),
    ],
)
def test_calendar_numeric_validation(name, value):
    data = calendar_examples().head(28)
    data[name] = data[name].astype(float)
    data.loc[data.index[0], name] = value
    with pytest.raises(ValueError, match=name):
        _prepare_features(data, calendar_features=True)


@pytest.mark.parametrize("name", ["target_event_type_1", "target_event_type_2"])
@pytest.mark.parametrize("value", [None, "", " "])
def test_event_missing_values_must_use_literal_none(name, value):
    data = calendar_examples().head(28)
    data.loc[data.index[0], name] = value
    with pytest.raises(ValueError, match="Invalid categorical"):
        _prepare_features(data, calendar_features=True)


def test_calendar_model_requires_all_seven_fields_at_inference(calendar_model):
    data = calendar_examples().drop(columns=list(CALENDAR_FEATURE_NAMES))
    with pytest.raises(ValueError, match="Missing model features"):
        predict_global_model(calendar_model, data)


@pytest.mark.parametrize("invalid", [None, 0, "yes"])
def test_calendar_opt_in_rejects_non_boolean(invalid):
    with pytest.raises(TypeError, match="calendar_features must be a boolean"):
        train_global_model(calendar_examples(), calendar_features=invalid)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("feature_columns", list(reversed(CYCLE2_FEATURE_COLUMNS)), "feature schema"),
        ("categorical_features", ["target_event_type_1"], "categorical schema"),
        ("unseen_category_policy", "new_level", "unseen category policy"),
        ("category_levels", {}, "categorical schema"),
    ],
)
def test_reload_rejects_schema_tampering(calendar_model, tmp_path, key, value, message):
    path = tmp_path / "calendar.txt"
    save_global_model(calendar_model, path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata[key] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_global_model(path)


def test_reload_verifies_booster_feature_names(calendar_model, tmp_path):
    path = tmp_path / "calendar.txt"
    save_global_model(calendar_model, path)
    cycle1 = train_global_model(calendar_examples(), n_estimators=3)
    cycle1.booster.save_model(str(path))
    with pytest.raises(ValueError, match="booster feature names"):
        load_global_model(path)


def test_save_verifies_category_encoding_order(calendar_model, tmp_path):
    levels = {
        name: list(values) for name, values in calendar_model.category_levels.items()
    }
    levels["target_event_type_1"].reverse()
    altered = replace(calendar_model, category_levels=levels)
    with pytest.raises(ValueError, match="booster category levels"):
        save_global_model(altered, tmp_path / "mismatched.txt")


def test_save_rejects_duplicate_category_levels(calendar_model, tmp_path):
    levels = {
        name: list(values) for name, values in calendar_model.category_levels.items()
    }
    levels["target_event_type_1"] = ["None", "None"]
    with pytest.raises(ValueError, match="category levels"):
        save_global_model(
            replace(calendar_model, category_levels=levels), tmp_path / "bad.txt"
        )


def test_save_requires_the_explicit_none_level(calendar_model, tmp_path):
    levels = {
        name: list(values) for name, values in calendar_model.category_levels.items()
    }
    levels["target_event_type_1"].remove("None")
    with pytest.raises(ValueError, match="must include literal None"):
        save_global_model(
            replace(calendar_model, category_levels=levels), tmp_path / "bad.txt"
        )
