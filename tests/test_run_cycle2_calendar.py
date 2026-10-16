"""Synthetic CI coverage for the isolated, preregistered Cycle 2 runner.

These fixtures are deliberately tiny and are never full-M5 experiments.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_cycle2_calendar as runner
from src.calendar_features import CALENDAR_FEATURE_NAMES
from src.cycle2_metrics import aggregation_metrics
from src.ml_dataset import select_training_origins
from src.ml_model import CYCLE2_CATEGORICAL_FEATURES, CYCLE2_FEATURE_COLUMNS


@pytest.fixture
def inputs():
    """Two states with enough daily history for all 16 weekly origins."""
    dates = pd.date_range("2015-04-01", "2016-01-03")
    sales = pd.DataFrame(
        [
            {
                "series_id": series_id,
                "date": date,
                "quantity": float(2 + (day + number * 3) % 9),
                "price": float(1000 + day),
                "price_available": True,
            }
            for number, series_id in enumerate(("series_CA", "series_TX"))
            for day, date in enumerate(dates)
        ]
    )
    metadata = pd.DataFrame(
        {
            "series_id": ["series_CA", "series_TX"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
            "dept_id": ["FOODS_1", "HOBBIES_1"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )
    calendar = pd.DataFrame(
        {
            "date": dates,
            "event_name_1": "",
            "event_type_1": "",
            "event_name_2": "",
            "event_type_2": "",
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": ((dates.day >= 3) & (dates.day <= 12)).astype(int),
            "snap_WI": (dates.day % 2 == 0).astype(int),
        }
    )
    calendar.loc[calendar["date"].dt.day.eq(13), ["event_name_1", "event_type_1"]] = [
        "Scheduled event",
        "Cultural",
    ]
    calendar.loc[calendar["date"].dt.day.eq(17), ["event_name_2", "event_type_2"]] = [
        "Second event",
        "National",
    ]
    calendar.loc[
        calendar["date"].eq("2015-12-25"), ["event_name_1", "event_type_1"]
    ] = [
        "Christmas",
        "Religious",
    ]
    return sales, metadata, calendar


def scored_at_wape(v2, wape):
    result = v2[[*runner.KEYS, "target"]].copy()
    result["prediction"] = result["target"] * (1 - wape / 100)
    return result


def candidate_result(candidate, training, v1, v2, wape=70.0):
    scored = scored_at_wape(v2, wape)
    metrics = aggregation_metrics(scored)
    return {
        "candidate": candidate.name,
        "objective": "tweedie",
        "n_origins": candidate.n_origins,
        "best_iteration": 2,
        "v2_wape": metrics["daily_wape"],
        "v2_mae": metrics["mae"],
        "v2_rmse": metrics["rmse"],
        "v1_best_mae": 0.5,
        "v1_mae_history": [0.75, 0.5, 0.6],
        "fit_seconds": 0.01,
        "training_rows": len(training),
        "training_series": int(training["series_id"].nunique()),
        "training_origins": [
            str(d.date()) for d in sorted(training["origin"].unique())
        ],
        "category_levels": {column: ["None"] for column in CYCLE2_CATEGORICAL_FEATURES},
        "near_tree_cap": False,
    }, scored


def install_frozen_reference(
    monkeypatch, calls=None, *, bad_reference=False, bad_ma28=False
):
    """Replace only costly historical-model recovery with explicit test scores."""

    def frozen_run(candidate, training, v1, v2):
        if calls is not None:
            calls.append(candidate.name)
        assert candidate == runner.cycle1.Candidate("tweedie", 8)
        assert not set(CALENDAR_FEATURE_NAMES).intersection(training.columns)
        assert not set(CALENDAR_FEATURE_NAMES).intersection(v1.columns)
        assert not set(CALENDAR_FEATURE_NAMES).intersection(v2.columns)
        return {
            "candidate": runner.FROZEN_CYCLE1_NAME,
            "best_iteration": runner.FROZEN_CYCLE1_ITERATION,
            "v2_wape": runner.FROZEN_CYCLE1_WAPE + (0.1 if bad_reference else 0),
            "v1_best_mae": 0.5,
            "v1_mae_history": [0.5] * runner.FROZEN_CYCLE1_ITERATION,
        }, scored_at_wape(v2, runner.FROZEN_CYCLE1_WAPE)

    def ma28(sales, reference):
        if calls is not None:
            calls.append("MA28")
        return (
            runner.FROZEN_MA28_WAPE + (0.1 if bad_ma28 else 0),
            scored_at_wape(reference, runner.FROZEN_MA28_WAPE),
        )

    monkeypatch.setattr(runner.cycle1, "run_candidate", frozen_run)
    monkeypatch.setattr(runner.cycle1, "score_ma28_on_v2", ma28)


@pytest.fixture
def small_experiment(inputs, monkeypatch):
    """Exercise real feature/cohort construction, substituting only model fits."""
    sales, metadata, calendar = inputs
    monkeypatch.setattr(runner, "EXPECTED_SERIES", 2)
    install_frozen_reference(monkeypatch)
    monkeypatch.setattr(runner, "run_candidate", candidate_result)
    summary, forecasts, details = runner.run_experiment(sales, metadata, calendar)
    return summary, forecasts, metadata, details


def test_exact_two_candidates_and_frozen_protocol():
    assert [(c.n_origins, c.name) for c in runner.candidate_grid()] == [
        (8, "cycle2_tweedie_8_origins_calendar"),
        (16, "cycle2_tweedie_16_origins_calendar"),
    ]
    protocol = runner.protocol()
    assert protocol["training_cutoff"] == protocol["v1_origin"] == "2015-11-08"
    assert protocol["v1_targets"] == ["2015-11-09", "2015-12-06"]
    assert protocol["v2_origin"] == "2015-12-06"
    assert protocol["v2_targets"] == ["2015-12-07", "2016-01-03"]
    assert protocol["evaluation_series"] == 2992
    assert protocol["v2_rows"] == 83776
    assert protocol["model_parameters"] == {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "early_stopping_metric": "V1 clipped MAE",
        "random_state": 42,
        "n_jobs": 2,
        "deterministic": True,
        "force_col_wise": True,
    }
    assert protocol["added_features"] == list(CALENDAR_FEATURE_NAMES)
    assert len(protocol["added_features"]) == 7
    assert protocol["feature_columns"] == list(CYCLE2_FEATURE_COLUMNS)
    assert protocol["categorical_features"] == list(CYCLE2_CATEGORICAL_FEATURES)
    assert protocol["adoption_threshold_exclusive"] == runner.FROZEN_CYCLE1_WAPE - 1
    assert "known at origin" in protocol["calendar_assumption"]
    assert "Sunday" in protocol["weekday"]
    assert "selection optimism" in " ".join(protocol["limitations"])
    assert "No final holdout scoring" in " ".join(protocol["limitations"])


@pytest.mark.parametrize(
    "owner,name,value",
    [
        (runner, "TRAINING_CUTOFF", pd.Timestamp("2015-11-09")),
        (runner, "V1_ORIGIN", pd.Timestamp("2015-11-09")),
        (runner, "V2_ORIGIN", pd.Timestamp("2015-12-07")),
        (runner, "VALIDATION_END", pd.Timestamp("2016-01-04")),
        (runner.cycle1, "MAX_ESTIMATORS", 5000),
        (runner.cycle1, "EARLY_STOPPING_ROUNDS", 50),
        (runner.cycle1, "N_JOBS", 4),
    ],
)
def test_protocol_guard_rejects_changed_dates_or_frozen_parameters(
    monkeypatch, owner, name, value
):
    runner.validate_protocol()
    monkeypatch.setattr(owner, name, value)
    with pytest.raises(ValueError, match="chronology or frozen Cycle 1 parameters"):
        runner.validate_protocol()


def test_dry_run_never_reads_data_trains_or_writes(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run accessed data, fitted, or wrote reports")

    monkeypatch.setattr(runner.pd, "read_parquet", forbidden)
    monkeypatch.setattr(runner.pd, "read_csv", forbidden)
    monkeypatch.setattr(runner, "run_experiment", forbidden)
    monkeypatch.setattr(runner, "save_reports", forbidden)
    output = tmp_path / "cycle2_calendar"
    assert (
        runner.main(
            [
                "--sales",
                str(tmp_path / "absent.parquet"),
                "--calendar",
                str(tmp_path / "absent.csv"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert not output.exists()
    text = capsys.readouterr().out
    assert "DRY RUN: no data read, no fitting, no reports written" in text
    assert "cycle2_tweedie_16_origins_calendar" in text
    assert "Pass --execute" in text


def test_execute_missing_input_fails_before_training(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        runner, "run_experiment", lambda *a: pytest.fail("Must fail before fitting")
    )
    with pytest.raises(SystemExit) as error:
        runner.main(["--execute", "--sales", str(tmp_path / "missing.parquet")])
    assert error.value.code == 2
    assert "Missing sales input" in capsys.readouterr().err


def test_execute_cli_reads_only_permitted_sales_and_writes_separate_reports(
    inputs,
    small_experiment,
    monkeypatch,
    tmp_path,
    capsys,
):
    """Use actual Parquet filtering, including a forbidden later sales sentinel."""
    sales, metadata, calendar = inputs
    summary, forecasts, _, details = small_experiment
    later = sales.iloc[[0]].copy()
    later["date"] = pd.Timestamp("2016-01-04")
    later["quantity"] = 1e12
    sales_path, metadata_path, calendar_path = (
        tmp_path / "sales.parquet",
        tmp_path / "series.parquet",
        tmp_path / "calendar.csv",
    )
    pd.concat([sales, later], ignore_index=True).to_parquet(sales_path, index=False)
    metadata.to_parquet(metadata_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    calls = []

    def check_sample(actual, manifest):
        pd.testing.assert_frame_equal(actual, metadata)
        assert manifest.name == "sample_series.csv"
        calls.append("frozen_sample")

    def run(actual_sales, actual_metadata, actual_calendar):
        assert set(actual_sales) == {"series_id", "date", "quantity"}
        assert len(actual_sales) == len(sales)
        assert actual_sales["date"].max() == pd.Timestamp("2016-01-03")
        assert actual_sales["quantity"].max() < 1e12
        pd.testing.assert_frame_equal(actual_metadata, metadata)
        assert actual_calendar["event_type_1"].eq("").any()
        assert not actual_calendar["event_type_1"].isna().any()
        calls.append("experiment")
        return summary, forecasts, copy.deepcopy(details)

    monkeypatch.setattr(runner, "validate_sample_metadata", check_sample)
    monkeypatch.setattr(runner, "run_experiment", run)
    output = tmp_path / "cycle2_calendar" / "cli"
    arguments = [
        "--execute",
        "--sales",
        str(sales_path),
        "--metadata",
        str(metadata_path),
        "--calendar",
        str(calendar_path),
        "--output-dir",
        str(output),
    ]
    assert runner.main(arguments) == 0
    assert calls == ["frozen_sample", "experiment"]
    assert "Saved 7 reports" in capsys.readouterr().out
    manifest = json.loads((output / "protocol_selection.json").read_text())
    assert set(manifest["execution"]["inputs"]) == {
        "sales",
        "metadata",
        "calendar",
        "sample_manifest",
    }
    assert manifest["execution"]["inputs"]["sales"]["sha256"] == runner.file_sha256(
        sales_path
    )
    # Existing reports reject a rerun before reading data or fitting anything.
    calls.clear()
    with pytest.raises(ValueError, match="already exist"):
        runner.main(arguments)
    assert calls == []


@pytest.mark.parametrize("n_origins", [8, 16])
def test_prepare_training_uses_exact_origins_cutoff_and_calendar(inputs, n_origins):
    sales, metadata, calendar = inputs
    training = runner.prepare_training_dataset(
        sales, metadata, calendar, runner.Candidate(n_origins)
    )
    expected = select_training_origins(runner.TRAINING_CUTOFF, n_origins)
    assert set(training["origin"]) == set(expected)
    assert all(origin.dayofweek == 6 for origin in expected)
    assert len(training) == 2 * n_origins * 28
    assert training["date"].max() == runner.TRAINING_CUTOFF
    assert training["date"].le(runner.TRAINING_CUTOFF).all()
    assert set(CALENDAR_FEATURE_NAMES).issubset(training.columns)
    assert not {
        "price",
        "price_available",
        "event_name_1",
        "event_name_2",
    }.intersection(training.columns)
    assert training["target_month"].eq(training["date"].dt.month).all()
    assert training["target_day_of_month"].eq(training["date"].dt.day).all()
    assert not training.duplicated(runner.KEYS).any()


@pytest.mark.parametrize("n_origins", [0, 4, 32])
def test_prepare_and_run_reject_non_preregistered_candidate(n_origins):
    candidate = runner.Candidate(n_origins)
    with pytest.raises(ValueError, match="preregistered"):
        runner.prepare_training_dataset(None, None, None, candidate)
    with pytest.raises(ValueError, match="preregistered"):
        runner.run_candidate(candidate, None, None, None)


def test_prepare_rejects_builder_origin_drift(inputs, monkeypatch):
    sales, metadata, calendar = inputs
    training = runner.prepare_training_dataset(
        sales, metadata, calendar, runner.Candidate(8)
    )
    training = training.loc[training["origin"].ne(training["origin"].min())].copy()
    monkeypatch.setattr(runner, "build_ml_training_dataset", lambda *a, **k: training)
    with pytest.raises(ValueError, match="Training origins"):
        runner.prepare_training_dataset(sales, metadata, calendar, runner.Candidate(8))


def test_run_candidate_uses_v1_only_fixed_fit_arguments_and_target_free_v2(
    inputs, monkeypatch
):
    sales, metadata, calendar = inputs
    candidate = runner.Candidate(8)
    training = runner.prepare_training_dataset(sales, metadata, calendar, candidate)
    v1, v2 = runner.build_validation_feature_sets(sales, metadata, calendar=calendar)
    calls = []

    def train(examples, **kwargs):
        calls.append("fit")
        assert examples is training
        assert kwargs == {
            "objective": "tweedie",
            "n_estimators": 2000,
            "n_jobs": 2,
            "validation_data": v1,
            "early_stopping_rounds": 100,
            "calendar_features": True,
        }
        assert kwargs["validation_data"] is not v2
        assert examples["date"].le(runner.TRAINING_CUTOFF).all()
        return SimpleNamespace(
            best_iteration=2,
            validation_mae_history=[2.0, 1.0, 1.5],
            category_levels={"target_event_type_1": ["None", "Cultural"]},
        )

    def predict(model, examples):
        calls.append("predict")
        assert "target" not in examples
        pd.testing.assert_frame_equal(examples, v2.drop(columns="target"))
        output = examples[runner.KEYS].copy()
        output["prediction"] = v2["target"].to_numpy() / 2
        return output.sample(frac=1, random_state=42)

    monkeypatch.setattr(runner, "train_global_model", train)
    monkeypatch.setattr(runner, "predict_global_model", predict)
    result, scored = runner.run_candidate(candidate, training, v1, v2)
    assert calls == ["fit", "predict"]
    assert result["candidate"] == candidate.name
    assert result["v2_wape"] == pytest.approx(50)
    assert result["best_iteration"] == 2
    assert result["v1_best_mae"] == 1
    assert result["training_rows"] == 448
    assert result["training_series"] == 2
    assert result["fit_seconds"] >= 0
    assert result["near_tree_cap"] is False
    np.testing.assert_allclose(scored["prediction"], scored["target"] / 2)


@pytest.mark.parametrize(
    "corruption", ["future_target", "wrong_origins", "swapped_windows"]
)
def test_run_candidate_rejects_temporal_protocol_drift(inputs, monkeypatch, corruption):
    sales, metadata, calendar = inputs
    training = runner.prepare_training_dataset(
        sales, metadata, calendar, runner.Candidate(8)
    )
    v1, v2 = runner.build_validation_feature_sets(sales, metadata, calendar=calendar)
    if corruption == "future_target":
        training.loc[0, "date"] = runner.TRAINING_CUTOFF + pd.Timedelta(days=1)
    elif corruption == "wrong_origins":
        training = training.loc[training["origin"].ne(training["origin"].min())]
    else:
        v1, v2 = v2, v1
    monkeypatch.setattr(
        runner,
        "train_global_model",
        lambda *a, **k: pytest.fail("Invalid protocol fitted"),
    )
    with pytest.raises(ValueError):
        runner.run_candidate(runner.Candidate(8), training, v1, v2)


@pytest.mark.parametrize(
    "history,iteration", [([], 1), ([np.nan], 1), ([-1.0], 1), ([1.0], 2)]
)
def test_run_candidate_requires_valid_v1_curve(inputs, monkeypatch, history, iteration):
    sales, metadata, calendar = inputs
    training = runner.prepare_training_dataset(
        sales, metadata, calendar, runner.Candidate(8)
    )
    v1, v2 = runner.build_validation_feature_sets(sales, metadata, calendar=calendar)
    monkeypatch.setattr(
        runner,
        "train_global_model",
        lambda *a, **k: SimpleNamespace(
            best_iteration=iteration,
            validation_mae_history=history,
            category_levels={},
        ),
    )
    monkeypatch.setattr(
        runner, "predict_global_model", lambda *a: pytest.fail("Invalid history scored")
    )
    with pytest.raises(ValueError, match="V1 early-stopping history"):
        runner.run_candidate(runner.Candidate(8), training, v1, v2)


def test_real_tweedie_calendar_candidate_end_to_end(inputs, monkeypatch):
    """Real LightGBM uses the full frozen cap; small synthetic data stops early."""
    sales, metadata, calendar = inputs
    candidate = runner.Candidate(8)
    training = runner.prepare_training_dataset(sales, metadata, calendar, candidate)
    v1, v2 = runner.build_validation_feature_sets(sales, metadata, calendar=calendar)
    actual_train = runner.train_global_model
    captured = []

    def capture(*args, **kwargs):
        model = actual_train(*args, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(runner, "train_global_model", capture)
    result, scored = runner.run_candidate(candidate, training, v1, v2)
    model = captured[0]
    for name, expected in runner.protocol()["model_parameters"].items():
        if name not in {
            "n_estimators",
            "early_stopping_rounds",
            "early_stopping_metric",
        }:
            actual_name = "num_threads" if name == "n_jobs" else name
            assert model.booster.params[actual_name] == expected
    assert model.feature_columns == CYCLE2_FEATURE_COLUMNS
    assert set(model.category_levels) == set(CYCLE2_CATEGORICAL_FEATURES)
    assert "None" in model.category_levels["target_event_type_1"]
    assert 1 <= result["best_iteration"] <= len(result["v1_mae_history"]) <= 2000
    assert len(scored) == 56
    assert scored["prediction"].ge(0).all()
    assert np.isfinite(scored["prediction"]).all()
    assert result["v2_wape"] == aggregation_metrics(scored)["daily_wape"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate", "tweedie_16_origins"),
        ("best_iteration", 1943),
        ("v2_wape", runner.FROZEN_CYCLE1_WAPE + 1e-7),
        ("v2_wape", np.nan),
    ],
)
def test_frozen_reference_rejects_drift(field, value):
    reference = {
        "candidate": runner.FROZEN_CYCLE1_NAME,
        "best_iteration": 1944,
        "v2_wape": runner.FROZEN_CYCLE1_WAPE,
    }
    runner.verify_frozen_reference(reference)
    reference[field] = value
    with pytest.raises(ValueError, match="frozen WAPE/iteration"):
        runner.verify_frozen_reference(reference)


def test_run_experiment_exact_sequence_and_no_holdout_sales_or_prices(
    inputs, monkeypatch
):
    sales, metadata, calendar = inputs
    holdout = pd.DataFrame(
        [
            {
                "series_id": "series_CA",
                "date": pd.Timestamp("2016-01-04"),
                "quantity": 1e12,
                "price": 1e12,
                "price_available": True,
            }
        ]
    )
    sales = pd.concat([sales, holdout], ignore_index=True)
    monkeypatch.setattr(runner, "EXPECTED_SERIES", 2)
    calls = []
    install_frozen_reference(monkeypatch, calls)
    real_build = runner.build_validation_feature_sets

    def build(safe_sales, metadata, **kwargs):
        assert set(safe_sales) == {"series_id", "date", "quantity"}
        assert safe_sales["date"].max() == runner.VALIDATION_END
        assert not safe_sales["quantity"].eq(1e12).any()
        calls.append("calendar_features" if "calendar" in kwargs else "cycle1_features")
        return real_build(safe_sales, metadata, **kwargs)

    def run(candidate, training, v1, v2):
        calls.append(candidate.name)
        assert set(CALENDAR_FEATURE_NAMES).issubset(training.columns)
        return candidate_result(candidate, training, v1, v2)

    monkeypatch.setattr(runner, "build_validation_feature_sets", build)
    monkeypatch.setattr(runner, "run_candidate", run)
    summary, forecasts, details = runner.run_experiment(sales, metadata, calendar)
    assert calls == [
        "calendar_features",
        "cycle1_features",
        "tweedie_8_origins",
        "cycle2_tweedie_8_origins_calendar",
        "cycle2_tweedie_16_origins_calendar",
        "MA28",
    ]
    assert summary["candidate"].tolist() == [c.name for c in runner.candidate_grid()]
    assert set(forecasts) == {runner.FROZEN_CYCLE1_NAME, "MA28", *summary["candidate"]}
    reference = (
        forecasts[runner.FROZEN_CYCLE1_NAME]
        .sort_values(runner.KEYS)
        .reset_index(drop=True)
    )
    for frame in forecasts.values():
        frame = frame.sort_values(runner.KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            frame[[*runner.KEYS, "target"]],
            reference[[*runner.KEYS, "target"]],
            check_dtype=False,
        )
    assert details["evaluation_series"] == 2
    assert details["v1_rows"] == details["v2_rows"] == 56
    assert (
        details["cohort_sha256"] == hashlib.sha256(b"series_CA\nseries_TX").hexdigest()
    )
    assert len(details["v2_observations_sha256"]) == 64
    assert details["experiment_wall_seconds"] >= 0
    assert details["process_cumulative_peak_rss_mib"] > 0
    assert details["feature_schema"] == list(CYCLE2_FEATURE_COLUMNS)


def test_production_runner_rejects_small_or_changed_cohort(inputs, monkeypatch):
    monkeypatch.setattr(
        runner.cycle1, "run_candidate", lambda *a: pytest.fail("Wrong cohort fitted")
    )
    with pytest.raises(ValueError, match="2992-series / 83776-row cohort"):
        runner.run_experiment(*inputs)


@pytest.mark.parametrize("bad_reference,bad_ma28", [(True, False), (False, True)])
def test_experiment_aborts_on_frozen_baseline_mismatch(
    inputs, monkeypatch, bad_reference, bad_ma28
):
    monkeypatch.setattr(runner, "EXPECTED_SERIES", 2)
    install_frozen_reference(
        monkeypatch, bad_reference=bad_reference, bad_ma28=bad_ma28
    )
    monkeypatch.setattr(runner, "run_candidate", candidate_result)
    with pytest.raises(ValueError, match="Cycle 1 reproduction|Matched MA28"):
        runner.run_experiment(*inputs)


def test_reports_include_four_methods_curves_schema_diagnostics_and_preserve_history(
    small_experiment, tmp_path
):
    summary, forecasts, metadata, details = small_experiment
    output = tmp_path / "reports" / "cycle2_calendar" / "synthetic_run"
    historical = tmp_path / "reports" / "ml_validation_selection.json"
    historical.parent.mkdir()
    historical.write_bytes(b"frozen Cycle 1 sentinel\n")
    paths = runner.save_reports(summary, forecasts, metadata, details, output)
    assert {path.name for path in paths} == {
        "candidate_scores.csv",
        "method_metrics.csv",
        "v1_curves.csv",
        "horizon_diagnostics.csv",
        "category_diagnostics.csv",
        "christmas_sensitivity.csv",
        "protocol_selection.json",
    }
    assert all(path.parent == output and path.stat().st_size > 0 for path in paths)
    assert historical.read_bytes() == b"frozen Cycle 1 sentinel\n"
    methods = set(forecasts)
    metrics = pd.read_csv(output / "method_metrics.csv")
    assert set(metrics["method"]) == methods
    assert set(metrics).issuperset(
        {
            "daily_wape",
            "weekly_wape",
            "period_28_wape",
            "mae",
            "rmse",
            "week_1_wape",
            "week_2_wape",
            "week_3_wape",
            "week_4_wape",
        }
    )
    for filename in (
        "horizon_diagnostics.csv",
        "category_diagnostics.csv",
        "christmas_sensitivity.csv",
    ):
        assert set(pd.read_csv(output / filename)["method"]) == methods
    curves = pd.read_csv(output / "v1_curves.csv")
    assert set(curves["candidate"]) == methods - {"MA28"}
    assert curves.groupby("candidate")["is_best_iteration"].sum().eq(1).all()
    assert len(curves) == 1944 + 6
    manifest = json.loads((output / "protocol_selection.json").read_text())
    assert manifest["selected_candidate"]["candidate"] == runner.Candidate(8).name
    assert set(manifest["datasets"]) == set(summary["candidate"])
    for name in summary["candidate"]:
        assert "category_levels" in manifest["datasets"][name]
        assert "training_origins" in manifest["datasets"][name]
    assert manifest["execution"]["v2_rows"] == 56
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(ValueError, match="already exist"):
        runner.save_reports(summary, forecasts, metadata, details, output)
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize(
    "corruption",
    ["wrong_namespace", "missing_method", "mismatched_actuals", "selection_wape"],
)
def test_report_validation_fails_before_writing(small_experiment, tmp_path, corruption):
    summary, forecasts, metadata, details = small_experiment
    summary = summary.copy()
    forecasts = {name: frame.copy() for name, frame in forecasts.items()}
    output = tmp_path / "cycle2_calendar" / "invalid"
    if corruption == "wrong_namespace":
        output = tmp_path / "reports"
    elif corruption == "missing_method":
        forecasts.pop("MA28")
    elif corruption == "mismatched_actuals":
        forecasts["MA28"].loc[0, "target"] += 1
    else:
        summary.loc[0, "v2_wape"] += 1
    with pytest.raises(ValueError):
        runner.save_reports(summary, forecasts, metadata, details, output)
    assert not output.exists()


def test_learning_curves_reject_inconsistent_best_mae(small_experiment):
    summary, _, _, details = small_experiment
    summary = summary.copy()
    summary.loc[0, "v1_best_mae"] = 999
    with pytest.raises(ValueError, match="Invalid V1 learning curve"):
        runner.build_learning_curves(summary, details)


def test_sample_metadata_checks_original_membership_not_just_count(tmp_path):
    manifest = tmp_path / "sample_series.csv"
    frozen = pd.DataFrame({"series_id": [f"frozen_{number}" for number in range(3000)]})
    frozen.to_csv(manifest, index=False)
    runner.validate_sample_metadata(frozen.sample(frac=1, random_state=42), manifest)
    changed = frozen.copy()
    changed.loc[0, "series_id"] = "different_series_same_count"
    with pytest.raises(ValueError, match="frozen 3000-series"):
        runner.validate_sample_metadata(changed, manifest)
    with pytest.raises(ValueError, match="frozen 3000-series"):
        runner.validate_sample_metadata(frozen.iloc[:-1], manifest)
    duplicate = pd.concat([frozen.iloc[:-1], frozen.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="frozen 3000-series"):
        runner.validate_sample_metadata(duplicate, manifest)
    duplicate.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="frozen 3000-series"):
        runner.validate_sample_metadata(frozen, manifest)


@pytest.mark.parametrize("field", [None, "best_iteration", "v2_wape"])
def test_independent_reproduction_checks_scores_and_iterations(
    small_experiment, tmp_path, field
):
    summary, _, _, _ = small_experiment
    previous = summary.copy()
    if field is not None:
        previous.loc[0, field] += 1
    previous.to_csv(tmp_path / "candidate_scores.csv", index=False)
    if field is None:
        runner.verify_reproduction(summary, tmp_path)
    else:
        with pytest.raises(ValueError, match="Reproduction"):
            runner.verify_reproduction(summary, tmp_path)


def test_independent_reproduction_checks_diagnostics_curves_and_provenance(
    small_experiment, tmp_path
):
    summary, forecasts, metadata, details = small_experiment
    input_path = tmp_path / "synthetic.csv"
    input_path.write_text("synthetic data\n", encoding="utf-8")
    details = {**details, **runner.execution_metadata({"synthetic": input_path})}
    output = tmp_path / "cycle2_calendar" / "original"
    runner.save_reports(summary, forecasts, metadata, details, output)
    runner.verify_reproduction(summary, output, forecasts, details)

    def reject_changed_file(filename, change, expected):
        path = output / filename
        original = path.read_bytes()
        change(path)
        with pytest.raises(ValueError, match=expected):
            runner.verify_reproduction(summary, output, forecasts, details)
        path.write_bytes(original)

    def alter_metrics(path):
        metrics = pd.read_csv(path)
        metrics.loc[0, "weekly_wape"] += 1
        metrics.to_csv(path, index=False)

    reject_changed_file("method_metrics.csv", alter_metrics, "four-method diagnostics")

    def alter_curve(path):
        curves = pd.read_csv(path)
        curves.loc[0, "v1_clipped_mae"] += 1
        curves.to_csv(path, index=False)

    reject_changed_file("v1_curves.csv", alter_curve, "V1 learning curves")
    for key in (
        "protocol",
        "cohort_sha256",
        "v2_observations_sha256",
        "source_files_sha256",
        "dependencies",
        "inputs",
    ):

        def alter_manifest(path, key=key):
            manifest = json.loads(path.read_text())
            if key == "protocol":
                manifest["protocol"]["training_cutoff"] = "2015-11-09"
            elif key == "inputs":
                manifest["execution"][key]["synthetic"]["sha256"] = "changed"
            else:
                manifest["execution"][key] = "changed"
            path.write_text(json.dumps(manifest), encoding="utf-8")

        reject_changed_file("protocol_selection.json", alter_manifest, "Reproduction")


def test_execution_metadata_records_real_source_dependencies_and_input_hashes(tmp_path):
    sample = tmp_path / "calendar.csv"
    sample.write_bytes(b"synthetic input, no M5 results\n")
    metadata = runner.execution_metadata({"calendar": sample})
    assert len(metadata["source_commit"]) == 40
    assert isinstance(metadata["source_dirty"], bool)
    assert metadata["source_files_sha256"][
        "scripts/run_cycle2_calendar.py"
    ] == runner.file_sha256(
        Path(runner.__file__),
    )
    assert set(metadata["dependencies"]) == {
        "lightgbm",
        "pandas",
        "numpy",
        "pyarrow",
        "scikit-learn",
        "scipy",
    }
    assert all(metadata["dependencies"].values())
    assert metadata["python"]
    assert metadata["inputs"]["calendar"] == {
        "path": str(sample),
        "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
    }
