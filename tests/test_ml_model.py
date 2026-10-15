"""Synthetic tests for global LightGBM training and inference."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    GlobalDemandModel,
    _prepare_features,
    clipped_mae,
    load_global_model,
    predict_global_model,
    save_global_model,
    train_global_model,
)


def make_training_data() -> pd.DataFrame:
    """Create examples for six series, three origins and 28 horizons."""
    rows = []
    first_origin = pd.Timestamp("2015-09-06")

    for origin_number in range(3):
        origin = first_origin + pd.Timedelta(days=7 * origin_number)

        for series_number in range(6):
            mean = float(1 + series_number + origin_number)

            for horizon in range(1, 29):
                rows.append(
                    {
                        "series_id": f"item_{series_number}",
                        "origin": origin,
                        "date": origin + pd.Timedelta(days=horizon),
                        "horizon": horizon,
                        "store_id": f"S{series_number % 2}",
                        "dept_id": f"D{series_number % 3}",
                        "cat_id": ("FOODS" if series_number % 2 else "HOBBIES"),
                        "sales_at_origin": mean,
                        "sales_lag_1_at_origin": mean + 1,
                        "sales_lag_7_at_origin": mean + 2,
                        "sales_lag_14_at_origin": mean,
                        "sales_lag_28_at_origin": mean,
                        "rolling_mean_7": mean,
                        "rolling_mean_28": mean + 0.5,
                        "rolling_mean_56": mean + 1,
                        "rolling_std_7": 0.5,
                        "rolling_std_28": 1.0,
                        "zero_fraction_28": 0.1,
                        "zero_fraction_56": 0.2,
                        "rolling_sum_7": 7 * mean,
                        "rolling_sum_28": 28 * (mean + 0.5),
                        "target": (2 * mean + (horizon % 7) + series_number % 2),
                    }
                )

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_model():
    return train_global_model(
        make_training_data(),
        n_estimators=30,
        n_jobs=2,
    )


def test_exact_predictor_whitelist_excludes_future_and_identifiers():
    data = make_training_data()
    data["future_sales"] = 999

    features, levels = _prepare_features(data)

    assert features.columns.tolist() == list(FEATURE_COLUMNS)

    forbidden = {
        "target",
        "series_id",
        "origin",
        "date",
        "future_sales",
    }
    assert not forbidden.intersection(features.columns)

    assert all(
        str(features[column].dtype) == "category" for column in CATEGORICAL_FEATURES
    )
    assert levels["store_id"] == ["S0", "S1"]


def test_fit_and_predict_preserve_identity_and_28_horizons(
    trained_model,
):
    data = make_training_data()
    examples = data.loc[data["origin"] == data["origin"].max()]

    output = predict_global_model(trained_model, examples)

    assert output.columns.tolist() == [
        *OUTPUT_COLUMNS,
        "prediction",
    ]
    assert len(output) == 6 * 28
    assert output["horizon"].unique().tolist() == list(range(1, 29))

    pd.testing.assert_frame_equal(
        output[list(OUTPUT_COLUMNS)],
        examples[list(OUTPUT_COLUMNS)].reset_index(drop=True),
    )

    assert np.isfinite(output["prediction"]).all()
    assert (output["prediction"] >= 0).all()


def test_target_and_unselected_columns_do_not_affect_predictions(
    trained_model,
):
    examples = make_training_data().head(56).copy()

    original = predict_global_model(trained_model, examples)

    modified = examples.copy()
    modified["target"] = 1234567
    modified["future_sales"] = -999

    result = predict_global_model(trained_model, modified)

    pd.testing.assert_frame_equal(original, result)


def test_shuffled_input_preserves_row_alignment(trained_model):
    examples = make_training_data().head(112)
    expected = predict_global_model(trained_model, examples)

    shuffled = examples.sample(frac=1, random_state=5)
    actual = predict_global_model(trained_model, shuffled)

    assert actual["series_id"].tolist() == shuffled["series_id"].tolist()

    np.testing.assert_allclose(
        actual["prediction"],
        expected.loc[shuffled.index, "prediction"],
        atol=1e-12,
    )


def test_unseen_category_becomes_missing_and_prediction_stays_finite(
    trained_model,
):
    examples = make_training_data().head(28).copy()
    examples["store_id"] = "NEW_STORE"

    features, _ = _prepare_features(
        examples,
        trained_model.category_levels,
    )

    assert features["store_id"].isna().all()

    result = predict_global_model(trained_model, examples)

    assert np.isfinite(result["prediction"]).all()


def test_model_save_reload_preserves_predictions_and_vocabulary(
    trained_model, tmp_path
):
    path = tmp_path / "subfolder" / "model.txt"
    examples = make_training_data().head(56)

    before = predict_global_model(trained_model, examples)

    save_global_model(trained_model, path)
    restored = load_global_model(path)

    after = predict_global_model(restored, examples)

    assert path.exists()
    assert path.with_suffix(".json").exists()
    assert restored.category_levels == trained_model.category_levels

    np.testing.assert_allclose(
        before["prediction"],
        after["prediction"],
        rtol=0,
        atol=1e-10,
    )


def test_saved_schema_mismatch_rejected(trained_model, tmp_path):
    path = tmp_path / "model.txt"
    save_global_model(trained_model, path)

    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_columns"] = ["wrong"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="feature schema"):
        load_global_model(path)


@pytest.mark.parametrize("bad_target", [-1, np.nan, np.inf])
def test_reject_invalid_training_targets(bad_target):
    examples = make_training_data()
    examples.loc[0, "target"] = bad_target

    with pytest.raises(ValueError, match="finite and nonnegative"):
        train_global_model(examples, n_estimators=5)


@pytest.mark.parametrize("bad_horizon", [0, 29, 2.5, np.nan])
def test_reject_invalid_horizons(bad_horizon):
    examples = make_training_data()
    examples["horizon"] = examples["horizon"].astype(float)
    examples.loc[0, "horizon"] = bad_horizon

    with pytest.raises(ValueError, match="horizon"):
        _prepare_features(examples)


def test_missing_feature_and_invalid_category_rejected():
    examples = make_training_data()

    with pytest.raises(ValueError, match="Missing model features"):
        _prepare_features(examples.drop(columns=["rolling_mean_7"]))

    examples.loc[0, "store_id"] = ""

    with pytest.raises(ValueError, match="Invalid categorical"):
        _prepare_features(examples)


def test_negative_raw_predictions_are_clipped_to_zero():
    class FakeBooster:
        def predict(self, frame, *, validate_features):
            assert validate_features is True
            return [-2.5, 3.25]

    examples = make_training_data().head(2)
    _, levels = _prepare_features(examples)

    fake_model = GlobalDemandModel(FakeBooster(), levels)
    result = predict_global_model(fake_model, examples)

    assert result["prediction"].tolist() == [0.0, 3.25]


@pytest.mark.parametrize("objective", ["regression", "poisson", "tweedie"])
def test_early_stopping_for_all_objectives(objective):
    """All objectives must select trees using the same V1 MAE metric."""
    data = make_training_data()
    origins = sorted(data["origin"].unique())

    training = data.loc[data["origin"].isin(origins[:2])].copy()
    validation = data.loc[data["origin"] == origins[2]].copy()

    model = train_global_model(
        training,
        validation_data=validation,
        objective=objective,
        n_estimators=60,
        early_stopping_rounds=10,
        n_jobs=2,
    )

    assert model.best_iteration is not None
    assert 1 <= model.best_iteration <= 60

    assert model.validation_mae_history is not None
    assert len(model.validation_mae_history) >= model.best_iteration
    assert len(model.validation_mae_history) <= 60

    assert np.isfinite(model.validation_mae_history).all()
    assert np.all(np.array(model.validation_mae_history) >= 0)

    predictions = predict_global_model(model, validation)

    assert len(predictions) == len(validation)
    assert np.isfinite(predictions["prediction"]).all()
    assert (predictions["prediction"] >= 0).all()


def test_default_training_remains_backward_compatible():
    """Existing training calls must still work without validation."""
    model = train_global_model(
        make_training_data(),
        n_estimators=20,
    )

    assert model.best_iteration == 20
    assert model.validation_mae_history is None


def test_invalid_objective_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LightGBM objective"):
        train_global_model(
            make_training_data(),
            objective="invalid",
        )


def test_validation_requires_target_column():
    data = make_training_data()

    with pytest.raises(ValueError, match="Validation data must include target"):
        train_global_model(
            data,
            validation_data=data.drop(columns=["target"]),
            n_estimators=10,
        )


def test_invalid_early_stopping_patience_is_rejected():
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        train_global_model(
            make_training_data(),
            early_stopping_rounds=0,
        )


def test_early_stopped_model_save_reload(tmp_path):
    """Reloading must preserve the V1-selected model configuration."""
    data = make_training_data()
    origins = sorted(data["origin"].unique())

    training = data.loc[data["origin"].isin(origins[:2])].copy()
    validation = data.loc[data["origin"] == origins[2]].copy()

    model = train_global_model(
        training,
        validation_data=validation,
        objective="poisson",
        n_estimators=60,
        early_stopping_rounds=10,
        n_jobs=2,
    )

    predictions_before = predict_global_model(model, validation)

    path = tmp_path / "early_stopped.txt"
    save_global_model(model, path)
    loaded = load_global_model(path)

    predictions_after = predict_global_model(loaded, validation)

    assert loaded.objective == "poisson"
    assert loaded.best_iteration == model.best_iteration
    assert loaded.validation_mae_history == model.validation_mae_history

    assert loaded.booster.num_trees() == model.best_iteration

    np.testing.assert_allclose(
        predictions_before["prediction"],
        predictions_after["prediction"],
        rtol=0,
        atol=1e-10,
    )


def test_legacy_model_metadata_still_loads(tmp_path):
    """Models saved before PR 3 remain readable."""
    model = train_global_model(
        make_training_data(),
        n_estimators=20,
    )

    path = tmp_path / "legacy_model.txt"
    save_global_model(model, path)

    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())

    metadata.pop("objective")
    metadata.pop("best_iteration")
    metadata.pop("validation_mae_history")

    metadata_path.write_text(json.dumps(metadata))

    loaded = load_global_model(path)

    assert loaded.objective == "regression"
    assert loaded.best_iteration is None
    assert loaded.validation_mae_history is None

    predictions = predict_global_model(
        loaded,
        make_training_data().head(28),
    )

    assert np.isfinite(predictions["prediction"]).all()


def test_clipped_mae_matches_inference_behavior():
    """Early stopping must evaluate nonnegative predictions."""
    actual = np.array([0.0, 1.0, 3.0])
    predicted = np.array([-2.0, 2.0, 5.0])

    name, value, higher_is_better = clipped_mae(actual, predicted)

    # After clipping: [0, 2, 5]
    # Absolute errors: [0, 1, 2]
    # MAE = (0 + 1 + 2) / 3 = 1
    assert name == "clipped_mae"
    assert value == pytest.approx(1.0)
    assert higher_is_better is False

    # Without clipping, the error would be larger.
    raw_mae = np.mean(np.abs(actual - predicted))
    assert raw_mae > value
