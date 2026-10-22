"""Synthetic end-to-end tests for the Phase 3D runner (need LightGBM and PyArrow).

A synthetic Issue #28 run is built in a temporary folder: preflight fit,
saved artifacts, a preflight.json with their hashes, and a published record
anchoring it. No real M5 input is accessed.
"""

import json
import shutil
from argparse import Namespace

import pandas as pd
import pytest

from scripts import run_inventory_simulation as runner
from scripts.run_cycle2_calendar import CALENDAR_INPUT_COLUMNS
from scripts.run_ml_final_evaluation import write_json
from src import ml_final_evaluation as final
from src.inventory_inputs import (
    ISSUE28_ARTIFACTS,
    PREDICTION_TOLERANCE,
    cohort_sha256,
    file_sha256,
)
from src.ml_model import save_global_model
from tests.test_final_evaluation import make_synthetic_inputs
from tests.test_inventory_ml_forecasts import calendar_through


@pytest.fixture(scope="module")
def synthetic_issue28(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3d")
    data = root / "data"
    data.mkdir()
    sales, metadata, _, baselines = make_synthetic_inputs()

    prices = {"CA_food": 2.5, "TX_hobby": 9.0, "WI_house": 4.25}
    sales.assign(
        price=sales.series_id.map(prices), price_available=True
    ).to_parquet(data / "sales.parquet", index=False)
    metadata.assign(item_id=metadata.series_id).to_parquet(
        data / "series.parquet", index=False
    )
    calendar_through("2016-05-15").assign(
        date=lambda frame: frame.date.dt.strftime("%Y-%m-%d")
    ).to_csv(data / "calendar.csv", index=False)
    baselines.to_parquet(data / "phase1.parquet", index=False)

    # Fit and predict exactly as Issue #28 did, from the files as written.
    calendar = pd.read_csv(
        data / "calendar.csv", usecols=CALENDAR_INPUT_COLUMNS, keep_default_na=False
    )
    model, aligned, lookups, _ = final.structural_preflight(
        sales, metadata, calendar, baselines, expected_series=3
    )

    artifacts = root / "artifacts"
    artifacts.mkdir()
    save_global_model(model, artifacts / "model.txt")
    aligned.to_parquet(artifacts / "predictions.parquet", index=False)
    lookups.to_parquet(artifacts / "lookups.parquet", index=False)
    inputs = {
        "sales": data / "sales.parquet",
        "metadata": data / "series.parquet",
        "calendar": data / "calendar.csv",
        "baselines": data / "phase1.parquet",
    }
    write_json(
        artifacts / "preflight.json",
        {
            "artifact_sha256": {
                name: file_sha256(artifacts / name) for name in ISSUE28_ARTIFACTS
            },
            "cohort_sha256": cohort_sha256(lookups.series_id),
            "execution": {
                "inputs": {
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in inputs.items()
                }
            },
        },
    )
    published = root / "published.json"
    published.write_text(
        json.dumps(
            {
                "preflight_sha256": file_sha256(artifacts / "preflight.json"),
                "prediction_sha256": file_sha256(artifacts / "predictions.parquet"),
            }
        )
    )
    return root, data, artifacts, published


def arguments(data, artifacts, output_dir):
    return Namespace(
        sales=data / "sales.parquet",
        metadata=data / "series.parquet",
        calendar=data / "calendar.csv",
        artifacts_dir=artifacts,
        phase1_predictions=data / "phase1.parquet",
        output_dir=output_dir,
    )


def test_checks_only_passes_every_check_and_writes_nothing(synthetic_issue28):
    root, data, artifacts, published = synthetic_issue28
    output = root / "reports" / "checks"

    checks = runner.run_checks_only(
        arguments(data, artifacts, output),
        published_manifest=published,
        expected_series=3,
    )

    assert checks["lightgbm_max_abs_difference_vs_issue28"] <= PREDICTION_TOLERANCE
    assert checks["ma28_max_abs_difference_vs_phase1"] == 0.0
    assert checks["cohort_series"] == 3
    assert checks["inventory_metrics_computed"] is False
    assert len(checks["review_origins"]) == 16
    assert not output.exists()


def test_execute_publishes_complete_reports_exactly_once(synthetic_issue28):
    root, data, artifacts, published = synthetic_issue28
    report_root = root / "reports"
    output = report_root / "primary"
    args = arguments(data, artifacts, output)

    destination, summary = runner.run_execute(
        args,
        published_manifest=published,
        report_root=report_root,
        n_resamples=50,
        expected_series=3,
    )

    assert destination == output.resolve()
    for name in (*runner.REPORT_TABLES, "summary.json", "manifest.json"):
        assert (output / name).is_file(), name

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"]["protocol_version"] == "1.0"
    assert manifest["checks"]["inventory_metrics_computed"] is True
    assert manifest["execution"]["runtime_seconds"] > 0
    assert all(
        not record["path"].startswith("/")
        for record in manifest["execution"]["inputs"].values()
    )

    saved = json.loads((output / "summary.json").read_text())
    assert saved["protocol_version"] == "1.0"
    assert set(saved) >= {"primary", "sensitivity", "starting_state_dependent"}
    assert summary["primary"]["n_resamples"] == 50

    curves = pd.read_csv(output / "curves.csv")
    assert len(curves) == 3 * 3 * 2 * 31

    with pytest.raises(ValueError, match="never overwritten"):
        runner.run_execute(
            args,
            published_manifest=published,
            report_root=report_root,
            n_resamples=50,
            expected_series=3,
        )


def test_a_changed_input_stops_the_run_before_any_report(synthetic_issue28, tmp_path):
    _, data, artifacts, published = synthetic_issue28
    edited = tmp_path / "data"
    shutil.copytree(data, edited)
    sales = pd.read_parquet(edited / "sales.parquet")
    sales.loc[sales.index[-1], "quantity"] += 1
    sales.to_parquet(edited / "sales.parquet", index=False)
    output = tmp_path / "reports" / "primary"

    with pytest.raises(ValueError, match="Input differs from the Issue #28 run: sales"):
        runner.run_execute(
            arguments(edited, artifacts, output),
            published_manifest=published,
            report_root=tmp_path / "reports",
            n_resamples=10,
            expected_series=3,
        )

    assert not output.exists()


def test_an_unfrozen_protocol_is_refused(tmp_path):
    document = tmp_path / "protocol.md"
    document.write_text("**Status: DRAFT v0.3.**")

    with pytest.raises(ValueError, match="not marked FROZEN"):
        runner.check_protocol_is_frozen(document)

    assert len(runner.check_protocol_is_frozen()) == 64


def test_the_command_line_requires_a_mode():
    with pytest.raises(SystemExit):
        runner.main([])
