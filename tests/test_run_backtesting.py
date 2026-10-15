"""Integration tests for the reproducible backtesting script."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_backtesting import main, run_experiment


@pytest.fixture
def sample_files(tmp_path):
    """Create small synthetic Parquet inputs with 140 daily dates."""
    dates = pd.date_range(
        end="2016-04-24",
        periods=140,
        freq="D",
    )

    sales = pd.DataFrame(
        {
            "series_id": np.repeat(["A", "B"], 140),
            "date": np.tile(dates, 2),
            "quantity": np.concatenate(
                [
                    np.arange(140) % 7,
                    np.full(140, 2, dtype=int),
                ]
            ),
        }
    )

    series = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    sales_path = tmp_path / "sales.parquet"
    series_path = tmp_path / "series.parquet"

    sales.to_parquet(sales_path, index=False)
    series.to_parquet(series_path, index=False)

    return sales_path, series_path


def test_experiment_generates_all_outputs(tmp_path, sample_files):
    sales_path, series_path = sample_files

    predictions_path = tmp_path / "processed/predictions.parquet"
    reports_dir = tmp_path / "reports"

    manifest = run_experiment(
        sales_path=sales_path,
        series_path=series_path,
        predictions_path=predictions_path,
        reports_dir=reports_dir,
    )

    predictions = pd.read_parquet(predictions_path)
    assert len(predictions) == 2 * 4 * 28 * 3

    assert manifest["eligible_series"] == 2
    assert manifest["excluded_series"] == 0
    assert manifest["prediction_rows"] == len(predictions)
    assert len(manifest["windows"]) == 4

    expected_reports = [
        "m5_baselines_overall.csv",
        "m5_baselines_by_window.csv",
        "m5_baselines_by_category.csv",
        "m5_baselines_by_sparsity.csv",
        "m5_demand_group_summary.csv",
        "m5_backtesting_manifest.json",
    ]

    for filename in expected_reports:
        assert (reports_dir / filename).is_file()

    overall = pd.read_csv(reports_dir / "m5_baselines_overall.csv")

    assert set(overall["model"]) == {
        "naive",
        "seasonal_naive",
        "moving_average_28",
    }
    assert overall["observations"].eq(2 * 4 * 28).all()
    assert overall["wape"].notna().all()

    stored_manifest = json.loads(
        (reports_dir / "m5_backtesting_manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == manifest


def test_repeated_runs_are_reproducible(tmp_path, sample_files):
    sales_path, series_path = sample_files

    kwargs = {
        "sales_path": sales_path,
        "series_path": series_path,
        "predictions_path": tmp_path / "predictions.parquet",
        "reports_dir": tmp_path / "reports",
    }

    first_manifest = run_experiment(**kwargs)
    first_predictions = pd.read_parquet(kwargs["predictions_path"])
    first_results = pd.read_csv(kwargs["reports_dir"] / "m5_baselines_overall.csv")

    second_manifest = run_experiment(**kwargs)
    second_predictions = pd.read_parquet(kwargs["predictions_path"])
    second_results = pd.read_csv(kwargs["reports_dir"] / "m5_baselines_overall.csv")

    assert first_manifest == second_manifest
    pd.testing.assert_frame_equal(
        first_predictions,
        second_predictions,
    )
    pd.testing.assert_frame_equal(first_results, second_results)


def test_missing_metadata_does_not_publish_outputs(tmp_path, sample_files):
    sales_path, series_path = sample_files

    # Remove one series from metadata to create an invalid join.
    pd.DataFrame({"series_id": ["A"], "cat_id": ["FOODS"]}).to_parquet(
        series_path, index=False
    )

    predictions_path = tmp_path / "predictions.parquet"
    reports_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="Missing category"):
        run_experiment(
            sales_path=sales_path,
            series_path=series_path,
            predictions_path=predictions_path,
            reports_dir=reports_dir,
        )

    assert not predictions_path.exists()
    assert not reports_dir.exists()


def test_cli_accepts_custom_paths(tmp_path, sample_files):
    sales_path, series_path = sample_files

    predictions_path = tmp_path / "predictions.parquet"
    reports_dir = tmp_path / "reports"

    main(
        [
            "--sales",
            str(sales_path),
            "--series",
            str(series_path),
            "--predictions",
            str(predictions_path),
            "--reports-dir",
            str(reports_dir),
            "--end-date",
            "2016-04-24",
        ]
    )

    assert predictions_path.is_file()
    assert (reports_dir / "m5_baselines_overall.csv").is_file()
