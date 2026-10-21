"""Synthetic-only protocol, chronology, fitting and pooled scoring tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import ml_final_evaluation as final
from src.backtesting import run_backtesting
from src.ml_dataset import build_ml_training_dataset
from src.ml_model import CYCLE2_FEATURE_COLUMNS


def make_synthetic_inputs():
    dates = pd.date_range("2015-08-01", "2016-04-24")
    sales = pd.DataFrame(
        [
            {
                "series_id": name,
                "date": date,
                "quantity": (day * (i + 3) + day // 5) % 17,
            }
            for i, name in enumerate(["CA_food", "TX_hobby", "WI_house"])
            for day, date in enumerate(dates)
        ]
    )
    metadata = pd.DataFrame(
        {
            "series_id": ["CA_food", "TX_hobby", "WI_house"],
            "store_id": ["CA_1", "TX_1", "WI_1"],
            "state_id": ["CA", "TX", "WI"],
            "dept_id": ["FOODS_1", "HOBBIES_1", "HOUSEHOLD_1"],
            "cat_id": ["FOODS", "HOBBIES", "HOUSEHOLD"],
        }
    )
    calendar = pd.DataFrame(
        {
            "date": dates,
            "event_name_1": "",
            "event_name_2": "",
            "event_type_1": "",
            "event_type_2": "",
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": (dates.day <= 12).astype(int),
            "snap_WI": (dates.day <= 15).astype(int),
        }
    )
    calendar.loc[calendar.date > final.CUTOFF, ["event_name_1", "event_type_1"]] = [
        "Future event",
        "Unseen type",
    ]
    baseline = run_backtesting(sales, "2016-04-24")
    return sales, metadata, calendar, baseline


@pytest.fixture
def synthetic_inputs():
    return make_synthetic_inputs()


@pytest.fixture
def aligned(synthetic_inputs):
    _, metadata, _, baseline = synthetic_inputs
    ml = baseline.loc[baseline.model == "naive"].copy()
    ml["model"] = final.MODEL
    forecasts = pd.concat([baseline, ml], ignore_index=True)
    forecasts["horizon"] = (forecasts.date - forecasts.origin).dt.days
    lookups = metadata[["series_id", "cat_id"]].copy()
    lookups["demand_group"] = ["frequent", "mixed", "sparse"]
    return forecasts, lookups


def test_frozen_protocol():
    p = final.protocol()
    assert p["parameters"]["n_estimators"] == 1743
    assert not p["early_stopping"]
    assert p["daily_rows_per_method"] == 335552
    assert p["total_predictions"] == 1342208
    assert len(p["feature_columns"]) == 25
    assert p["training_origins"] == [
        "2015-10-18",
        "2015-10-25",
        "2015-11-01",
        "2015-11-08",
        "2015-11-15",
        "2015-11-22",
        "2015-11-29",
        "2015-12-06",
    ]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "actual",
        "date",
        "origin",
        "negative",
        "nan",
        "infinite",
        "cohort",
        "method",
        "horizon",
    ],
)
def test_alignment_rejects_damage(aligned, damage):
    frame, lookups = aligned
    row = frame.index[frame.model == final.MODEL][0]
    if damage == "missing":
        frame = frame.drop(index=row)
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.loc[[row]]])
    elif damage == "actual":
        frame.loc[row, "actual"] += 1
    elif damage == "date":
        frame.loc[row, "date"] += pd.Timedelta(days=1)
    elif damage == "origin":
        frame.loc[row, "origin"] += pd.Timedelta(days=1)
    elif damage in ("negative", "nan", "infinite"):
        frame["predicted"] = frame.predicted.astype(float)
        frame.loc[row, "predicted"] = {
            "negative": -1,
            "nan": np.nan,
            "infinite": np.inf,
        }[damage]
    elif damage == "cohort":
        frame.loc[row, "series_id"] = "other"
    elif damage == "method":
        frame.loc[row, "model"] = "other"
    else:
        frame.loc[row, "horizon"] = 0
    with pytest.raises(ValueError):
        final.align_methods(frame, lookups.series_id)


def test_shuffled_alignment_is_identical(aligned):
    frame, lookups = aligned
    expected = final.align_methods(frame, lookups.series_id)
    pd.testing.assert_frame_equal(
        expected,
        final.align_methods(frame.sample(frac=1, random_state=42), lookups.series_id),
    )


def test_pooled_metrics_cancellation_and_zero():
    frame = pd.DataFrame(
        {
            "series_id": ["a"] * 28,
            "origin": final.CUTOFF,
            "horizon_week": np.repeat([1, 2, 3, 4], 7),
            "actual": [1.0] * 28,
            "predicted": [2.0, 0.0] * 14,
        }
    )
    metrics = final.pooled_metrics(frame)
    assert metrics["daily_wape"] == 100
    assert metrics["weekly_wape"] == pytest.approx(100 / 7)
    assert metrics["day28_wape"] == 0
    assert metrics["mae"] == 1
    assert metrics["rmse"] == 1
    frame["actual"] = 0.0
    for key in ("daily_wape", "weekly_wape", "day28_wape"):
        assert np.isnan(final.pooled_metrics(frame)[key])


def test_pooled_ratio_not_mean_of_series_percentages():
    frame = pd.DataFrame(
        {
            "series_id": ["a", "b"],
            "origin": final.CUTOFF,
            "horizon_week": 1,
            "actual": [1.0, 99.0],
            "predicted": [0.0, 99.0],
        }
    )
    assert final.pooled_metrics(frame)["daily_wape"] == 1


def test_all_diagnostics_and_counts(aligned):
    frame, lookups = aligned
    reports = final.reveal_metrics(frame, lookups)
    assert set(reports) == {
        "overall",
        "by_window",
        "by_category",
        "by_demand_group",
        "by_horizon_week",
    }
    assert reports["overall"].observations.tolist() == [336] * 4
    assert len(reports["by_window"]) == 16
    assert len(reports["by_horizon_week"]) == 16


def fake_model(trees=1743):
    return SimpleNamespace(
        booster=SimpleNamespace(
            num_trees=lambda: trees, current_iteration=lambda: trees
        ),
        best_iteration=1743,
        validation_mae_history=None,
        feature_columns=CYCLE2_FEATURE_COLUMNS,
        objective="tweedie",
        category_levels={},
    )


def test_blind_preflight_fit_once_origin_safety(monkeypatch, synthetic_inputs):
    sales, metadata, calendar, baseline = synthetic_inputs
    fit_calls, prediction_origins = [], []

    def fit(training, **kwargs):
        fit_calls.append(kwargs)
        assert training.date.max() == final.CUTOFF
        assert set(training.origin) == set(final.TRAINING_ORIGINS)
        assert set(training.target_event_type_1) == {"None"}
        assert "validation_data" not in kwargs
        return fake_model()

    original = final.build_ml_inference_examples

    def examples(history, meta, origin, **kwargs):
        assert history.date.max() == origin
        return original(history, meta, origin, **kwargs)

    def predict(model, examples):
        assert "actual" not in examples and "target" not in examples
        prediction_origins.append(examples.origin.iloc[0])
        out = examples[final.KEYS].copy()
        out["prediction"] = 1.0
        return out

    def forbidden(*a, **k):
        pytest.fail("Blind preflight attempted metric calculation")

    monkeypatch.setattr(final, "train_global_model", fit)
    monkeypatch.setattr(final, "build_ml_inference_examples", examples)
    monkeypatch.setattr(final, "predict_global_model", predict)
    monkeypatch.setattr(final, "pooled_metrics", forbidden)
    monkeypatch.setattr(final, "reveal_metrics", forbidden)
    _, forecasts, lookups, checks = final.structural_preflight(
        sales, metadata, calendar, baseline, expected_series=3
    )
    assert len(fit_calls) == 1
    assert fit_calls[0] == {
        "n_estimators": 1743,
        "n_jobs": 2,
        "objective": "tweedie",
        "calendar_features": True,
    }
    assert prediction_origins == list(final.ORIGINS)
    assert len(forecasts) == 1344 and len(lookups) == 3
    assert checks["accuracy_calculated"] is False
    assert not any("wape" in key or "mae" in key for key in checks)


@pytest.mark.parametrize("trees", [1, 1742, 1744])
def test_realized_tree_count_required(monkeypatch, synthetic_inputs, trees):
    sales, metadata, calendar, _ = synthetic_inputs
    training = build_ml_training_dataset(
        sales, metadata, final.CUTOFF, 8, calendar=calendar
    )
    monkeypatch.setattr(final, "train_global_model", lambda *a, **k: fake_model(trees))
    with pytest.raises(ValueError, match="1743"):
        final.fit_frozen_model(training)


def test_real_synthetic_model_reproducible(synthetic_inputs):
    sales, metadata, calendar, _ = synthetic_inputs
    training = build_ml_training_dataset(
        sales, metadata, final.CUTOFF, 8, calendar=calendar
    )
    first = final.fit_frozen_model(training)
    second = final.fit_frozen_model(training)
    assert first.booster.num_trees() == 1743
    assert first.category_levels == second.category_levels
    assert "Unseen type" not in first.category_levels["target_event_type_1"]
    np.testing.assert_array_equal(
        final.predict_global_model(first, training).prediction,
        final.predict_global_model(second, training).prediction,
    )


@pytest.mark.parametrize("method", final.BASELINES)
@pytest.mark.parametrize("replacement", [12345.0, "next_float"])
def test_preflight_rejects_substituted_baseline_predictions(
    synthetic_inputs, monkeypatch, capsys, method, replacement
):
    sales, metadata, calendar, baselines = synthetic_inputs
    baselines = baselines.copy()
    baselines["predicted"] = baselines.predicted.astype(float)
    row = baselines.index[baselines.model == method][0]
    baselines.loc[row, "predicted"] = (
        np.nextafter(baselines.loc[row, "predicted"], np.inf)
        if replacement == "next_float"
        else replacement
    )

    def forbidden(*a, **k):
        pytest.fail("Invalid baseline reached fitting or metric calculation")

    monkeypatch.setattr(final, "fit_frozen_model", forbidden)
    monkeypatch.setattr(final, "pooled_metrics", forbidden)
    monkeypatch.setattr(final, "reveal_metrics", forbidden)
    with pytest.raises(ValueError, match="unchanged Phase 1 definitions"):
        final.structural_preflight(
            sales, metadata, calendar, baselines, expected_series=3
        )
    assert capsys.readouterr().out == ""


def test_baseline_provenance_ignores_supplied_order(synthetic_inputs):
    sales, _, _, baselines = synthetic_inputs
    cohort = pd.Index(sorted(sales.series_id.unique()))
    expected = final.verified_phase1_baselines(sales, baselines, cohort)
    actual = final.verified_phase1_baselines(
        sales, baselines.sample(frac=1, random_state=42), cohort
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_original_all_baseline_substitution_is_rejected(synthetic_inputs, monkeypatch):
    sales, metadata, calendar, baselines = synthetic_inputs
    substituted = baselines.copy()
    substituted["predicted"] = 12345.0
    monkeypatch.setattr(
        final,
        "fit_frozen_model",
        lambda *a: pytest.fail("Substituted forecasts reached fitting"),
    )
    with pytest.raises(ValueError, match="unchanged Phase 1 definitions"):
        final.structural_preflight(
            sales, metadata, calendar, substituted, expected_series=3
        )


def test_post_cutoff_sample_member_preserves_training_and_fixed_cohort(
    synthetic_inputs, monkeypatch
):
    sales, metadata, calendar, baselines = synthetic_inputs
    expected_training = build_ml_training_dataset(
        sales.loc[sales.date <= final.CUTOFF],
        metadata,
        final.CUTOFF,
        8,
        calendar=calendar,
    )
    late = sales.loc[
        (sales.series_id == "CA_food") & (sales.date > final.CUTOFF)
    ].copy()
    late["series_id"] = "late_sample_member"
    late_metadata = metadata.loc[metadata.series_id == "CA_food"].copy()
    late_metadata["series_id"] = "late_sample_member"
    seen_origins = []
    model = fake_model()

    def fit(training, **kwargs):
        pd.testing.assert_frame_equal(training, expected_training)
        return model

    original_builder = final.build_ml_inference_examples

    def examples(history, origin_metadata, origin, **kwargs):
        assert set(origin_metadata.series_id) == set(history.series_id)
        assert ("late_sample_member" in set(origin_metadata.series_id)) == (
            origin > final.CUTOFF
        )
        return original_builder(history, origin_metadata, origin, **kwargs)

    def predict(fitted, examples):
        assert fitted is model
        assert set(examples.series_id) == set(metadata.series_id)
        seen_origins.append(examples.origin.iloc[0])
        result = examples[final.KEYS].copy()
        result["prediction"] = 1.0
        return result

    monkeypatch.setattr(final, "train_global_model", fit)
    monkeypatch.setattr(final, "build_ml_inference_examples", examples)
    monkeypatch.setattr(final, "predict_global_model", predict)
    _, forecasts, lookups, checks = final.structural_preflight(
        pd.concat([sales, late], ignore_index=True),
        pd.concat([metadata, late_metadata], ignore_index=True),
        calendar,
        baselines,
        expected_series=3,
    )
    assert seen_origins == list(final.ORIGINS)
    assert set(forecasts.series_id) == set(lookups.series_id) == set(metadata.series_id)
    assert len(forecasts) == 3 * 4 * 28 * 4
    assert checks["training_series"] == 3
    assert "exact_phase1_baseline_reproduction" in checks["checks"]


def test_preflight_does_not_drop_eligible_series_with_missing_future_sales(
    synthetic_inputs, monkeypatch
):
    sales, metadata, calendar, baselines = synthetic_inputs
    damaged = sales.drop(
        sales.index[(sales.series_id == "CA_food") & (sales.date == "2016-02-10")]
    )
    monkeypatch.setattr(
        final,
        "fit_frozen_model",
        lambda *a: pytest.fail("Missing targets reached fitting"),
    )
    with pytest.raises(ValueError, match="Missing observations"):
        final.structural_preflight(
            damaged, metadata, calendar, baselines, expected_series=3
        )


def test_week_report_does_not_label_seven_days_as_28_day_wape():
    paths = []
    for origin in final.ORIGINS:
        for method in final.METHODS:
            paths.append(
                pd.DataFrame(
                    {
                        "series_id": "s",
                        "origin": origin,
                        "date": pd.date_range(
                            origin + pd.Timedelta(days=1), periods=28
                        ),
                        "horizon": range(1, 29),
                        "model": method,
                        "actual": 1.0,
                        "predicted": [2.0] * 7 + [0.0] * 7 + [1.0] * 14,
                    }
                )
            )
    reports = final.reveal_metrics(
        pd.concat(paths, ignore_index=True),
        pd.DataFrame(
            {
                "series_id": ["s"],
                "cat_id": ["FOODS"],
                "demand_group": ["frequent"],
            }
        ),
    )
    overall = reports["overall"]
    assert overall.day28_wape.eq(0).all()
    assert overall.weekly_wape.eq(50).all()
    assert overall.daily_wape.eq(50).all()
    week = reports["by_horizon_week"]
    assert "day28_wape" not in week.columns
    assert week.loc[week.horizon_week.isin([1, 2]), "weekly_wape"].eq(100).all()
    assert week.loc[week.horizon_week.isin([3, 4]), "weekly_wape"].eq(0).all()
    for name in ("by_window", "by_category", "by_demand_group"):
        assert reports[name].day28_wape.eq(0).all()
