"""Synthetic tests for Phase 3C LightGBM forecast paths (need LightGBM)."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_ml_final_evaluation import frame_hash
from src import inventory_ml_forecasts as forecasts
from src import ml_final_evaluation as final
from src.inventory_inputs import (
    ISSUE28_ORIGINS,
    LIGHTGBM_MODEL,
    MA28_MODEL,
    PREDICTION_TOLERANCE,
    REVIEW_ORIGINS,
    check_paths_match,
    cohort_sha256,
    ma28_paths,
    paths_from_predictions,
)
from src.ml_dataset import build_ml_training_dataset
from src.ml_model import save_global_model, train_global_model
from tests.test_final_evaluation import make_synthetic_inputs


def calendar_through(end):
    """The synthetic Issue #28 calendar, extended to cover later target dates."""
    dates = pd.date_range("2015-08-01", end)
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
    return calendar


@pytest.fixture(scope="module")
def issue28_run():
    """One synthetic Issue #28 preflight: frozen 1,743-tree fit and predictions."""
    sales, metadata, calendar, baselines = make_synthetic_inputs()
    model, aligned, lookups, _ = final.structural_preflight(
        sales, metadata, calendar, baselines, expected_series=3
    )
    cohort = pd.Index(lookups.series_id).sort_values()
    return sales, metadata, calendar, model, aligned, cohort


def test_saved_frozen_model_reproduces_issue28_predictions(issue28_run, tmp_path):
    sales, metadata, calendar, model, aligned, cohort = issue28_run
    save_global_model(model, tmp_path / "model.txt")
    loaded = forecasts.load_frozen_model(tmp_path)

    paths = forecasts.lightgbm_paths(
        loaded, sales, metadata, calendar, cohort, ISSUE28_ORIGINS
    )
    reference = paths_from_predictions(
        aligned.loc[aligned.model == LIGHTGBM_MODEL],
        cohort,
        ISSUE28_ORIGINS,
        value_column="predicted",
    )

    largest = check_paths_match(
        paths, reference, tolerance=PREDICTION_TOLERANCE, label="LightGBM"
    )
    assert largest <= PREDICTION_TOLERANCE


def test_ma28_paths_match_issue28_baseline_predictions(issue28_run):
    sales, _, _, _, aligned, cohort = issue28_run
    reference = paths_from_predictions(
        aligned.loc[aligned.model == MA28_MODEL],
        cohort,
        ISSUE28_ORIGINS,
        value_column="predicted",
    )

    np.testing.assert_array_equal(ma28_paths(sales, cohort, ISSUE28_ORIGINS), reference)


def test_weekly_paths_use_only_sales_through_each_origin(issue28_run, monkeypatch):
    sales, metadata, calendar, model, _, cohort = issue28_run
    # Origins after 2016-03-27 forecast past the synthetic calendar's last day.
    calendar = calendar_through("2016-05-15")
    original = forecasts.build_ml_inference_examples
    seen = []

    def examples(history, history_metadata, origin, **kwargs):
        assert history.date.max() == origin
        seen.append(origin)
        return original(history, history_metadata, origin, **kwargs)

    monkeypatch.setattr(forecasts, "build_ml_inference_examples", examples)
    paths = forecasts.lightgbm_paths(model, sales, metadata, calendar, cohort)

    assert seen == list(REVIEW_ORIGINS)
    assert paths.shape == (3, 16, 28)
    assert np.isfinite(paths).all() and (paths >= 0).all()


def test_a_model_other_than_the_frozen_one_is_rejected(tmp_path):
    sales, metadata, calendar, _ = make_synthetic_inputs()
    training = build_ml_training_dataset(
        sales, metadata, final.CUTOFF, 8, calendar=calendar
    )
    small = train_global_model(
        training, n_estimators=5, objective="tweedie", calendar_features=True
    )
    save_global_model(small, tmp_path / "model.txt")

    with pytest.raises(ValueError, match="frozen"):
        forecasts.load_frozen_model(tmp_path)


def test_cohort_hash_equals_the_issue28_frame_hash():
    lookups = pd.DataFrame(
        {"series_id": ["TX_hobby", "CA_food", "WI_house"], "cat_id": ["H", "F", "W"]}
    )

    assert cohort_sha256(lookups.series_id) == frame_hash(lookups, ["series_id"])
