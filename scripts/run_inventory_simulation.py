"""Phase 3D: run the frozen inventory experiment (protocol v1.0) once on real data.

    .venv/bin/python -m scripts.run_inventory_simulation --checks-only
    .venv/bin/python -m scripts.run_inventory_simulation --execute
    .venv/bin/python -m scripts.run_inventory_simulation --publish-only

--checks-only verifies the provenance chain (published Issue #28 record ->
original preflight.json -> model, cohort, data), builds the weekly forecast
paths from verified snapshots, and runs every Section 12 check. It computes
no inventory metric and writes nothing.

--execute refuses an unfrozen protocol and runs the same checks. Only then
does it register the one real run, exclusively and for every output location
(the registration is kept if anything later fails). It then runs every
pre-registered scenario, confirms that sources, protocol and inputs did not
change, seals the results privately under data/processed/phase3_inventory/
with checksums, and publishes them to reports/phase3_inventory/primary/.
Nothing is ever overwritten.

--publish-only publishes already sealed results if publication failed after
sealing. It never recomputes anything.
"""

import argparse
import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_cycle2_calendar import CALENDAR_INPUT_COLUMNS, execution_metadata
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
    verified_snapshots,
)
from src.inventory_ml_forecasts import lightgbm_paths, load_frozen_model
from src.inventory_publication import (
    PUBLIC_REPORT,
    RESERVATION,
    RESULT_STORE,
    json_bytes,
    publish_sealed_results,
    read_sealed_results,
    reserve_primary_run,
    seal_results,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOCUMENT = ROOT / "docs/phase3_inventory_protocol.md"
FROZEN_STATUS = "**Status: FROZEN v1.0"
REPORT_ROOT = ROOT / "reports/phase3_inventory"
PRIVATE_ROOT = ROOT / "data/processed/phase3_inventory"
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


def prepare_inputs(args, folder, *, published_manifest, expected_series=EXPECTED_SERIES):
    """Verify, snapshot and load every input, then run every check.

    Every file is read from a verified private snapshot, never from its
    original path. Computes no inventory metric. expected_series and
    published_manifest are synthetic-test seams; the command line always
    uses 2,996 series and the committed reveal manifest.
    """
    manifest, snapshots, provenance = verified_snapshots(
        args.artifacts_dir, _input_paths(args), folder, published_manifest
    )
    lookups = pd.read_parquet(snapshots["lookups.parquet"])
    cohort = frozen_cohort(lookups, manifest, expected=expected_series)
    model = load_frozen_model(folder)

    sales_all = pd.read_parquet(snapshots["sales"])
    sales_all = sales_all.loc[sales_all["date"] <= LAST_SIMULATED_DAY]
    # Issue #28 read exactly these columns; LightGBM sees the same frame.
    sales = sales_all[["series_id", "date", "quantity"]]
    metadata = pd.read_parquet(snapshots["metadata"])
    calendar = pd.read_csv(
        snapshots["calendar"], usecols=CALENDAR_INPUT_COLUMNS, keep_default_na=False
    )

    paths = {
        "lightgbm": lightgbm_paths(model, sales, metadata, calendar, cohort),
        "ma28": ma28_paths(sales, cohort),
    }
    indices = _issue28_indices()

    predictions = pd.read_parquet(snapshots["predictions.parquet"])
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

    phase1 = pd.read_parquet(snapshots["baselines"])
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
        "provenance": "published Issue #28 record -> preflight.json -> artifacts and inputs",
        "files_read_from_verified_snapshots": True,
        "inputs_verified": sorted(_input_paths(args)),
        "cohort_series": len(cohort),
        "cohort_sha256": cohort_sha256(cohort),
        "review_origins": [str(origin.date()) for origin in REVIEW_ORIGINS],
        "lightgbm_max_abs_difference_vs_issue28": lightgbm_difference,
        "ma28_max_abs_difference_vs_phase1": ma28_difference,
        "demand_days": int(demand.shape[1]),
        "value_proxy_series": len(values),
        "inventory_metrics_computed": False,
    }
    return inputs, checks, provenance


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

    with tempfile.TemporaryDirectory(prefix="phase3-snapshots-") as folder:
        _, checks, _ = prepare_inputs(
            args, folder, published_manifest=published_manifest, **seams
        )

    return checks


def run_execute(
    args,
    *,
    published_manifest=PUBLISHED_REVEAL_MANIFEST,
    report_root=REPORT_ROOT,
    private_root=PRIVATE_ROOT,
    n_resamples=None,
    **seams,
):
    """Run the frozen experiment once, seal its results, then publish them."""
    report_root, private_root = Path(report_root), Path(private_root)
    protocol_sha256 = check_protocol_is_frozen()

    if (private_root / RESERVATION).exists():
        raise ValueError(
            "A primary Phase 3 run is already registered; use --publish-only if "
            "its sealed results still need publishing"
        )

    for existing in (report_root / PUBLIC_REPORT, private_root / RESULT_STORE):
        if existing.exists():
            raise ValueError(f"{existing} already exists and is never overwritten")

    started = time.perf_counter()
    input_paths = _input_paths(args)
    # Hash sources and inputs before reading, to detect changes during the run.
    before = execution_metadata(input_paths)

    with tempfile.TemporaryDirectory(prefix="phase3-snapshots-") as folder:
        inputs, checks, provenance = prepare_inputs(
            args, folder, published_manifest=published_manifest, **seams
        )
        # Every check passed. Register the one real run before any simulation.
        reservation = reserve_primary_run(
            private_root,
            _json_safe(
                {
                    "issue": 36,
                    "protocol_document_sha256": protocol_sha256,
                    "source_commit": before["source_commit"],
                    "source_dirty": before["source_dirty"],
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "provenance": provenance,
                    "cohort_sha256": checks["cohort_sha256"],
                }
            ),
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

    after = execution_metadata(input_paths)

    if after["source_files_sha256"] != before["source_files_sha256"]:
        raise ValueError("Source files changed during the experiment")

    if after["inputs"] != before["inputs"]:
        raise ValueError("Inputs changed during the experiment")

    if check_protocol_is_frozen() != protocol_sha256:
        raise ValueError("The protocol document changed during the experiment")

    execution = {
        **before,
        "inputs": _portable_inputs(before["inputs"]),
        "runtime_seconds": time.perf_counter() - started,
        "peak_process_rss_mib": peak_rss_mib(),
        "rss_note": "Whole-process cumulative high-water mark",
    }
    manifest = {
        "issue": 36,
        "protocol": protocol_parameters(),
        "protocol_document_sha256": protocol_sha256,
        "provenance": provenance,
        "reservation_sha256": file_sha256(reservation),
        "checks": {**checks, "inventory_metrics_computed": True},
        "execution": execution,
        "files": [*REPORT_TABLES, "summary.json", "manifest.json"],
    }
    contents = {
        name: results[name.removesuffix(".csv")].to_csv(index=False).encode()
        for name in REPORT_TABLES
    }
    contents["summary.json"] = json_bytes(_json_safe(results["summary"]))
    contents["manifest.json"] = json_bytes(_json_safe(manifest))

    seal_results(private_root, contents)
    destination = publish_sealed_results(private_root, report_root)
    return destination, results["summary"]


def run_publish_only(*, report_root=REPORT_ROOT, private_root=PRIVATE_ROOT):
    """Publish sealed results after a failed publication; never recompute."""
    destination = publish_sealed_results(private_root, report_root)
    summary = json.loads(read_sealed_results(private_root)["summary.json"])
    return destination, summary


def _print_outcome(destination, summary) -> None:
    conclusion = summary["conclusion"]
    print(f"Conclusion: {conclusion['statement']}")
    print(
        "Verdicts: primary "
        f"{conclusion['primary_verdict']}, empty start "
        f"{conclusion['empty_start_verdict']}, round-half-up "
        f"{conclusion['half_up_verdict']}"
    )
    print(f"Reports published to {destination}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checks-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
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
    args = parser.parse_args(argv)

    if args.checks_only:
        checks = run_checks_only(args)
        print(json.dumps(_json_safe(checks), indent=2))
        print("All Section 12 checks passed. No inventory metric was computed.")
        return 0

    if args.publish_only:
        _print_outcome(*run_publish_only())
        return 0

    _print_outcome(*run_execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
