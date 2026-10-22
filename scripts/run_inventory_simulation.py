"""Phase 3D: run the frozen inventory experiment (protocol v1.0) on real data.

    python -m scripts.run_inventory_simulation --checks-only
    python -m scripts.run_inventory_simulation --execute

--checks-only verifies the provenance chain (published Issue #28 record ->
original preflight.json -> model, cohort, data), builds the weekly forecast
paths, and runs every Section 12 check. It computes no inventory metric and
writes nothing.

--execute runs the same checks, then every pre-registered scenario, and
publishes the reports once under reports/phase3_inventory/. An existing
report directory is never overwritten.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_cycle2_calendar import CALENDAR_INPUT_COLUMNS, execution_metadata
from scripts.run_ml_final_evaluation import (
    publish_directory,
    validate_destination,
    write_json,
)
from scripts.train_ml_model import peak_rss_mib
from src.evaluation import build_demand_profile
from src.inventory_experiment import protocol_parameters, run_experiment
from src.inventory_inputs import (
    EXPECTED_SERIES,
    FIRST_REVIEW,
    ISSUE28_ORIGINS,
    LAST_SIMULATED_DAY,
    LIGHTGBM_MODEL,
    MA28_MODEL,
    PREDICTION_TOLERANCE,
    PUBLISHED_REVEAL_MANIFEST,
    REVIEW_ORIGINS,
    check_paths_match,
    cohort_sha256,
    demand_matrix,
    file_sha256,
    frozen_cohort,
    ma28_paths,
    paths_from_predictions,
    value_proxies,
    verify_input_files,
    verify_issue28_artifacts,
)
from src.inventory_ml_forecasts import lightgbm_paths, load_frozen_model

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOCUMENT = ROOT / "docs/phase3_inventory_protocol.md"
FROZEN_STATUS = "**Status: FROZEN v1.0"
REPORT_ROOT = ROOT / "reports/phase3_inventory"
REPORT_TABLES = (
    "curves.csv",
    "reference.csv",
    "endpoints.csv",
    "group_curves.csv",
    "group_endpoints.csv",
)


def check_protocol_is_frozen(document=PROTOCOL_DOCUMENT) -> str:
    """Refuse to run unless the protocol document is the frozen v1.0."""
    text = Path(document).read_text(encoding="utf-8")

    if FROZEN_STATUS not in text:
        raise ValueError("The Phase 3 protocol is not marked FROZEN v1.0")

    return file_sha256(document)


def _issue28_indices():
    return [list(REVIEW_ORIGINS).index(origin) for origin in ISSUE28_ORIGINS]


def _input_paths(args) -> dict:
    """Inputs keyed by the names Issue #28 recorded their hashes under."""
    return {
        "sales": Path(args.sales),
        "metadata": Path(args.metadata),
        "calendar": Path(args.calendar),
        "baselines": Path(args.phase1_predictions),
    }


def prepare_inputs(args, *, published_manifest, expected_series=EXPECTED_SERIES):
    """Load and check every input. Computes no inventory metric.

    expected_series and published_manifest are synthetic-test seams; the
    command line always uses 2,996 series and the committed reveal manifest.
    """
    artifacts = Path(args.artifacts_dir)
    manifest = verify_issue28_artifacts(artifacts, published_manifest)
    input_paths = _input_paths(args)
    verify_input_files(manifest, input_paths)

    lookups = pd.read_parquet(artifacts / "lookups.parquet")
    cohort = frozen_cohort(lookups, manifest, expected=expected_series)
    model = load_frozen_model(artifacts)

    sales_all = pd.read_parquet(args.sales)
    sales_all = sales_all.loc[sales_all["date"] <= LAST_SIMULATED_DAY]
    # Issue #28 read exactly these columns; LightGBM sees the same frame.
    sales = sales_all[["series_id", "date", "quantity"]]
    metadata = pd.read_parquet(args.metadata)
    calendar = pd.read_csv(
        args.calendar, usecols=CALENDAR_INPUT_COLUMNS, keep_default_na=False
    )

    paths = {
        "lightgbm": lightgbm_paths(model, sales, metadata, calendar, cohort),
        "ma28": ma28_paths(sales, cohort),
    }
    indices = _issue28_indices()

    predictions = pd.read_parquet(artifacts / "predictions.parquet")
    lightgbm_difference = check_paths_match(
        paths["lightgbm"][:, indices],
        paths_from_predictions(
            predictions.loc[predictions["model"] == LIGHTGBM_MODEL],
            cohort,
            ISSUE28_ORIGINS,
            value_column="predicted",
        ),
        tolerance=PREDICTION_TOLERANCE,
        label="LightGBM vs saved Issue #28 predictions",
    )

    phase1 = pd.read_parquet(args.phase1_predictions)
    ma28_difference = check_paths_match(
        paths["ma28"][:, indices],
        paths_from_predictions(
            phase1.loc[phase1["model"] == MA28_MODEL],
            cohort,
            ISSUE28_ORIGINS,
            value_column="predicted",
        ),
        tolerance=0.0,
        label="MA28 vs Phase 1 predictions",
    )

    demand = demand_matrix(sales, cohort)
    values = value_proxies(sales_all, cohort)

    by_series = metadata.set_index("series_id")

    if not cohort.isin(by_series.index).all():
        raise ValueError("Series metadata missing for cohort series")

    profile = build_demand_profile(
        sales.loc[sales["date"] <= FIRST_REVIEW], cohort, FIRST_REVIEW
    ).set_index("series_id")

    inputs = {
        "demand": demand,
        "paths": paths,
        "clusters": by_series.loc[cohort, "item_id"].to_numpy(),
        "groups": {
            "demand_group": profile.loc[cohort, "demand_group"].to_numpy(),
            "cat_id": by_series.loc[cohort, "cat_id"].to_numpy(),
        },
        "unit_values": values,
    }
    checks = {
        "provenance": "published Issue #28 record -> preflight.json -> artifacts",
        "published_preflight_sha256": file_sha256(artifacts / "preflight.json"),
        "inputs_verified": sorted(input_paths),
        "cohort_series": len(cohort),
        "cohort_sha256": cohort_sha256(cohort),
        "review_origins": [str(origin.date()) for origin in REVIEW_ORIGINS],
        "lightgbm_max_abs_difference_vs_issue28": lightgbm_difference,
        "ma28_max_abs_difference_vs_phase1": ma28_difference,
        "demand_days": int(demand.shape[1]),
        "value_proxy_series": len(values),
        "inventory_metrics_computed": False,
    }
    return inputs, checks


def _json_safe(value):
    """Convert numpy values and NaN to JSON-compatible equivalents."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _portable_inputs(inputs: dict) -> dict:
    """Keep input hashes, but never absolute local paths, in published reports."""
    portable = {}

    for name, record in inputs.items():
        path = Path(record["path"])

        if not path.is_absolute():
            shown = str(path)
        elif path.is_relative_to(ROOT):
            shown = str(path.relative_to(ROOT))
        else:
            shown = path.name

        portable[name] = {"path": shown, "sha256": record["sha256"]}

    return portable


def run_checks_only(args, *, published_manifest=PUBLISHED_REVEAL_MANIFEST, **seams):
    check_protocol_is_frozen()
    _, checks = prepare_inputs(args, published_manifest=published_manifest, **seams)
    return checks


def run_execute(
    args,
    *,
    published_manifest=PUBLISHED_REVEAL_MANIFEST,
    report_root=REPORT_ROOT,
    n_resamples=None,
    **seams,
):
    """Run the frozen experiment once and publish its reports."""
    protocol_sha256 = check_protocol_is_frozen()
    destination = validate_destination(args.output_dir, report_root)
    started = time.perf_counter()
    input_paths = _input_paths(args)
    # Hash sources and inputs before reading, to detect changes during the run.
    execution = execution_metadata(input_paths)

    inputs, checks = prepare_inputs(
        args, published_manifest=published_manifest, **seams
    )

    options = {} if n_resamples is None else {"n_resamples": n_resamples}
    results = run_experiment(
        inputs["demand"],
        inputs["paths"],
        inputs["clusters"],
        inputs["groups"],
        inputs["unit_values"],
        **options,
    )

    if execution_metadata(input_paths)["inputs"] != execution["inputs"]:
        raise ValueError("Inputs changed during the experiment")

    execution["inputs"] = _portable_inputs(execution["inputs"])
    execution.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_rss_mib": peak_rss_mib(),
            "rss_note": "Whole-process cumulative high-water mark",
        }
    )
    manifest = {
        "issue": 36,
        "protocol": protocol_parameters(),
        "protocol_document_sha256": protocol_sha256,
        "checks": {**checks, "inventory_metrics_computed": True},
        "execution": execution,
        "files": [*REPORT_TABLES, "summary.json"],
    }

    def writer(folder):
        for name in REPORT_TABLES:
            results[name.removesuffix(".csv")].to_csv(folder / name, index=False)
        write_json(folder / "summary.json", _json_safe(results["summary"]))
        write_json(folder / "manifest.json", _json_safe(manifest))

    publish_directory(destination, writer)
    return destination, results["summary"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checks-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--sales", type=Path, default=Path("data/processed/sales.parquet"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/series.parquet")
    )
    parser.add_argument("--calendar", type=Path, default=Path("data/raw/calendar.csv"))
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/processed/ml_final_evaluation/primary"),
    )
    parser.add_argument(
        "--phase1-predictions",
        type=Path,
        default=Path("data/processed/m5_backtest_predictions.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT / "primary")
    args = parser.parse_args(argv)

    if args.checks_only:
        checks = run_checks_only(args)
        print(json.dumps(_json_safe(checks), indent=2))
        print("All Section 12 checks passed. No inventory metric was computed.")
        return 0

    destination, summary = run_execute(args)
    primary = summary["primary"]
    print(f"Reports published to {destination}")
    print(
        "Primary verdict (L = 7, 95% fill): "
        f"{primary['verdict']}; point estimate {primary['point_estimate']}"
    )
    print(f"Depends on starting state: {summary['starting_state_dependent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
