"""Opt-in, preregistered Calendar-Aware LightGBM Cycle 2 (Issue #26).

Scheduled M5 calendar information is assumed known at each origin. No
post-origin sales or observed prices are predictors. Later holdout sales
are excluded before constructing training or validation examples.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_ml_validation as cycle1
from scripts.train_ml_model import validate_training_dataset
from src.calendar_features import CALENDAR_FEATURE_NAMES
from src.cycle2_metrics import (
    FROZEN_CYCLE1_ITERATION,
    FROZEN_CYCLE1_NAME,
    FROZEN_CYCLE1_WAPE,
    aggregation_metrics,
    align_forecasts,
    select_cycle2,
)
from src.ml_dataset import build_ml_training_dataset, select_training_origins
from src.ml_model import (
    CYCLE2_CATEGORICAL_FEATURES,
    CYCLE2_FEATURE_COLUMNS,
    predict_global_model,
    train_global_model,
)
from src.ml_validation import (
    TRAINING_CUTOFF,
    V1_ORIGIN,
    V2_ORIGIN,
    build_validation_feature_sets,
)

EXPECTED_SERIES = 2992
VALIDATION_END = pd.Timestamp("2016-01-03")
FROZEN_MA28_WAPE = 83.41645395641076
DEFAULT_OUTPUT = Path("reports/cycle2_calendar")
KEYS = ["series_id", "origin", "horizon", "date"]
CALENDAR_INPUT_COLUMNS = [
    "date",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
REPORT_FILENAMES = (
    "candidate_scores.csv",
    "method_metrics.csv",
    "v1_curves.csv",
    "horizon_diagnostics.csv",
    "category_diagnostics.csv",
    "christmas_sensitivity.csv",
    "protocol_selection.json",
)


@dataclass(frozen=True)
class Candidate:
    n_origins: int

    @property
    def name(self) -> str:
        return f"cycle2_tweedie_{self.n_origins}_origins_calendar"


def candidate_grid() -> tuple[Candidate, ...]:
    """Exactly two sequential experiments; never expand based on scores."""
    return (Candidate(8), Candidate(16))


def validate_protocol() -> None:
    if (
        TRAINING_CUTOFF != pd.Timestamp("2015-11-08")
        or V1_ORIGIN != TRAINING_CUTOFF
        or V2_ORIGIN != pd.Timestamp("2015-12-06")
        or VALIDATION_END != V2_ORIGIN + pd.Timedelta(days=28)
        or cycle1.MAX_ESTIMATORS != 2000
        or cycle1.EARLY_STOPPING_ROUNDS != 100
        or cycle1.N_JOBS != 2
    ):
        raise ValueError("Cycle 2 chronology or frozen Cycle 1 parameters changed")


def protocol() -> dict:
    """Machine-readable frozen protocol, also available without any M5 data."""
    return {
        "issue": "https://github.com/alennchik/supplypilot-ai/issues/26",
        "training_cutoff": "2015-11-08",
        "v1_origin": "2015-11-08",
        "v1_targets": ["2015-11-09", "2015-12-06"],
        "v2_origin": "2015-12-06",
        "v2_targets": ["2015-12-07", "2016-01-03"],
        "evaluation_series": EXPECTED_SERIES,
        "v2_rows": EXPECTED_SERIES * 28,
        "candidates": [candidate.name for candidate in candidate_grid()],
        "model_parameters": {
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
        },
        "added_features": list(CALENDAR_FEATURE_NAMES),
        "feature_columns": list(CYCLE2_FEATURE_COLUMNS),
        "categorical_features": list(CYCLE2_CATEGORICAL_FEATURES),
        "calendar_assumption": "Dates, scheduled events and state SNAP are known at origin",
        "unseen_category_policy": "Encode unseen valid strings as missing; literal None is a level",
        "weekday": "All origins are Sunday; numeric horizon determines weekday",
        "selection_metric": "full V2 daily pooled WAPE, in percentage points",
        "adoption_threshold_exclusive": FROZEN_CYCLE1_WAPE - 1.0,
        "complexity_tolerance_percentage_points": 1.0,
        "frozen_cycle1": {
            "candidate": FROZEN_CYCLE1_NAME,
            "best_iteration": FROZEN_CYCLE1_ITERATION,
            "v2_wape": FROZEN_CYCLE1_WAPE,
            "ma28_v2_wape": FROZEN_MA28_WAPE,
            "reproduction_absolute_tolerance": 1e-9,
        },
        "metric_definitions": {
            "daily_wape": "100 * sum(abs(actual - prediction)) / sum(actual)",
            "weekly_wape": "Sum within series/origin/horizon-week, then pooled WAPE",
            "period_28_wape": "Sum within series/origin over 28 days, then pooled WAPE",
            "week_buckets": [[1, 7], [8, 14], [15, 21], [22, 28]],
            "zero_pooled_actual": "Fail explicitly; never silently omit a method or week",
        },
        "limitations": [
            "One-point threshold is a complexity rule, not statistical significance",
            "V2 is reused for development; metrics are subject to selection optimism",
            "Later Phase 1 baseline windows were already inspected",
            "No final holdout scoring; that requires a separate once-only frozen evaluation",
            "Candidates near the fixed 2000-tree cap are reported, never extended",
        ],
    }


def prepare_training_dataset(sales, metadata, calendar, candidate) -> pd.DataFrame:
    validate_protocol()
    if candidate not in candidate_grid():
        raise ValueError("Candidate is outside the preregistered Cycle 2 grid")
    training = build_ml_training_dataset(
        sales,
        metadata,
        TRAINING_CUTOFF,
        candidate.n_origins,
        calendar=calendar,
    )
    validate_training_dataset(training, cutoff="2015-11-08")
    expected = select_training_origins(TRAINING_CUTOFF, candidate.n_origins)
    if set(training["origin"]) != set(expected):
        raise ValueError("Training origins do not match the frozen protocol")
    if not all(origin.dayofweek == 6 for origin in expected):
        raise ValueError("Cycle 2 origins must be Sunday")
    return training


def run_candidate(candidate, training, v1, v2) -> tuple[dict, pd.DataFrame]:
    """V1 alone selects trees; V2 targets are removed before prediction."""
    validate_protocol()
    if candidate not in candidate_grid():
        raise ValueError("Candidate is outside the preregistered Cycle 2 grid")
    validate_training_dataset(training, cutoff="2015-11-08")
    if set(training["origin"]) != set(
        select_training_origins(TRAINING_CUTOFF, candidate.n_origins)
    ):
        raise ValueError("Incorrect training origins for candidate")
    cycle1.validate_validation_sets(v1, v2)
    started = time.perf_counter()
    model = train_global_model(
        training,
        objective="tweedie",
        n_estimators=2000,
        n_jobs=2,
        validation_data=v1,
        early_stopping_rounds=100,
        calendar_features=True,
    )
    elapsed = time.perf_counter() - started
    history = model.validation_mae_history
    iteration = model.best_iteration
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(iteration, int)
        or not 1 <= iteration <= len(history) <= 2000
        or not np.isfinite(history).all()
        or (np.asarray(history) < 0).any()
    ):
        raise ValueError("Incomplete V1 early-stopping history")
    predicted = predict_global_model(model, v2.drop(columns="target"))
    scored = v2[[*KEYS, "target"]].merge(
        predicted,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scored.pop("_merge").eq("both").all():
        raise ValueError("V2 prediction and target keys do not match")
    metrics = aggregation_metrics(scored)
    result = {
        "candidate": candidate.name,
        "objective": "tweedie",
        "n_origins": candidate.n_origins,
        "best_iteration": iteration,
        "v2_wape": metrics["daily_wape"],
        "v2_mae": metrics["mae"],
        "v2_rmse": metrics["rmse"],
        "v1_best_mae": history[iteration - 1],
        "v1_mae_history": history,
        "fit_seconds": elapsed,
        "training_rows": len(training),
        "training_series": int(training["series_id"].nunique()),
        "training_origins": [
            str(d.date()) for d in sorted(training["origin"].unique())
        ],
        "near_tree_cap": iteration >= 1900,
        "category_levels": model.category_levels,
    }
    return result, scored


def verify_frozen_reference(result: dict) -> None:
    """Reject baseline drift rather than silently redefining Cycle 1."""
    if (
        result["candidate"] != FROZEN_CYCLE1_NAME
        or result["best_iteration"] != FROZEN_CYCLE1_ITERATION
        or not np.isclose(result["v2_wape"], FROZEN_CYCLE1_WAPE, rtol=0, atol=1e-9)
    ):
        raise ValueError(
            "Cycle 1 reproduction differs from frozen WAPE/iteration; no adoption decision"
        )


def run_experiment(sales, metadata, calendar) -> tuple[pd.DataFrame, dict, dict]:
    """Run the full M5 protocol; small synthetic tests exercise its components."""
    validate_protocol()
    started = time.perf_counter()
    dates = pd.to_datetime(sales["date"], errors="raise")
    sales = sales.loc[dates <= VALIDATION_END, ["series_id", "date", "quantity"]].copy()
    metadata = metadata.loc[metadata["series_id"].isin(sales["series_id"])].copy()
    v1, v2 = build_validation_feature_sets(sales, metadata, calendar=calendar)
    cycle1.validate_validation_sets(v1, v2)
    if v1["series_id"].nunique() != EXPECTED_SERIES or len(v2) != EXPECTED_SERIES * 28:
        raise ValueError("Cycle 2 requires the original 2992-series / 83776-row cohort")

    # Recover predictions using only the original, unchanged Cycle 1 setup.
    old_v1, old_v2 = build_validation_feature_sets(sales, metadata)
    if not v2[[*KEYS, "target"]].equals(old_v2[[*KEYS, "target"]]):
        raise ValueError("Calendar features changed the original evaluation population")
    reference_candidate = cycle1.Candidate("tweedie", 8)
    reference_training = cycle1.prepare_training_dataset(
        sales, metadata, reference_candidate
    )
    reference_result, reference = cycle1.run_candidate(
        reference_candidate,
        reference_training,
        old_v1,
        old_v2,
    )
    verify_frozen_reference(reference_result)
    del reference_training, old_v1, old_v2
    gc.collect()
    forecasts = {FROZEN_CYCLE1_NAME: reference}
    results = []
    for candidate in candidate_grid():
        training = prepare_training_dataset(sales, metadata, calendar, candidate)
        result, scored = run_candidate(candidate, training, v1, v2)
        forecasts[candidate.name] = align_forecasts(reference, scored)
        results.append(result)
        del training
        gc.collect()
    ma28_wape, ma28 = cycle1.score_ma28_on_v2(sales, reference)
    if not np.isclose(ma28_wape, FROZEN_MA28_WAPE, rtol=0, atol=1e-9):
        raise ValueError("Matched MA28 differs from the frozen reference")
    forecasts["MA28"] = align_forecasts(reference, ma28)
    summary = pd.DataFrame(results)
    select_cycle2(summary)
    cohort = sorted(v2["series_id"].unique())
    details = {
        "reference_reproduction": {
            key: value
            for key, value in reference_result.items()
            if key != "v1_mae_history"
        },
        "reference_v1_mae_history": reference_result["v1_mae_history"],
        "evaluation_series": len(cohort),
        "v1_rows": len(v1),
        "v2_rows": len(v2),
        "cohort_sha256": hashlib.sha256("\n".join(cohort).encode()).hexdigest(),
        "v2_observations_sha256": hashlib.sha256(
            v2[[*KEYS, "target"]].sort_values(KEYS).to_csv(index=False).encode()
        ).hexdigest(),
        "experiment_wall_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "process_cumulative_peak_rss_mib": cycle1.peak_process_rss_mib(),
        "rss_note": "Whole-process high-water mark, not per-candidate memory",
        "feature_schema": list(CYCLE2_FEATURE_COLUMNS),
        "categorical_schema": list(CYCLE2_CATEGORICAL_FEATURES),
    }
    return summary, forecasts, details


def build_learning_curves(summary: pd.DataFrame, details: dict) -> pd.DataFrame:
    select_cycle2(summary)
    rows = []
    entries = [
        (row.candidate, row.best_iteration, row.v1_best_mae, row.v1_mae_history)
        for row in summary.itertuples(index=False)
    ]
    reference = details["reference_reproduction"]
    entries.append(
        (
            FROZEN_CYCLE1_NAME,
            reference["best_iteration"],
            reference["v1_best_mae"],
            details["reference_v1_mae_history"],
        )
    )
    for name, best, best_mae, history in entries:
        values = np.asarray(history, dtype=float)
        if (
            values.ndim != 1
            or not 1 <= best <= len(values) <= 2000
            or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values[best - 1], best_mae, rtol=0, atol=1e-12)
        ):
            raise ValueError(f"Invalid V1 learning curve: {name}")
        rows.extend(
            {
                "candidate": name,
                "iteration": i,
                "v1_clipped_mae": float(value),
                "is_best_iteration": i == best,
            }
            for i, value in enumerate(values, 1)
        )
    return pd.DataFrame(rows)


def verify_reproduction(
    summary: pd.DataFrame,
    previous_dir: Path,
    forecasts: dict | None = None,
    details: dict | None = None,
) -> None:
    """Independent rerun check; never copy old scores into new results."""
    previous = pd.read_csv(Path(previous_dir) / "candidate_scores.csv")
    select_cycle2(previous)
    current = summary.set_index("candidate").sort_index()
    previous = previous.set_index("candidate").sort_index()
    if not current["best_iteration"].equals(previous["best_iteration"]):
        raise ValueError("Reproduction best iterations differ")
    if not np.allclose(current["v2_wape"], previous["v2_wape"], rtol=0, atol=1e-9):
        raise ValueError("Reproduction daily WAPEs differ")
    if forecasts is not None:
        previous_metrics = (
            pd.read_csv(Path(previous_dir) / "method_metrics.csv")
            .set_index("method")
            .sort_index()
        )
        current_metrics = (
            pd.DataFrame(
                [
                    {"method": name, **aggregation_metrics(scored)}
                    for name, scored in forecasts.items()
                ]
            )
            .set_index("method")
            .sort_index()
        )
        if (
            not previous_metrics.index.equals(current_metrics.index)
            or set(previous_metrics.columns) != set(current_metrics.columns)
            or not np.allclose(
                current_metrics,
                previous_metrics[current_metrics.columns],
                rtol=0,
                atol=1e-9,
            )
        ):
            raise ValueError("Reproduction four-method diagnostics differ")
    if details is not None:
        previous_manifest = json.loads(
            (Path(previous_dir) / "protocol_selection.json").read_text(encoding="utf-8")
        )
        if previous_manifest["protocol"] != protocol():
            raise ValueError("Reproduction protocol differs")
        for key in (
            "cohort_sha256",
            "v2_observations_sha256",
            "source_files_sha256",
            "dependencies",
        ):
            if previous_manifest["execution"][key] != details[key]:
                raise ValueError(f"Reproduction metadata differs: {key}")
        previous_inputs = {
            name: value["sha256"]
            for name, value in previous_manifest["execution"]["inputs"].items()
        }
        current_inputs = {
            name: value["sha256"] for name, value in details["inputs"].items()
        }
        if previous_inputs != current_inputs:
            raise ValueError("Reproduction input hashes differ")
        current_curves = (
            build_learning_curves(summary, details)
            .sort_values(["candidate", "iteration"])
            .reset_index(drop=True)
        )
        previous_curves = (
            pd.read_csv(Path(previous_dir) / "v1_curves.csv")
            .sort_values(["candidate", "iteration"])
            .reset_index(drop=True)
        )
        columns = ["candidate", "iteration", "is_best_iteration"]
        if not current_curves[columns].equals(
            previous_curves[columns]
        ) or not np.allclose(
            current_curves["v1_clipped_mae"],
            previous_curves["v1_clipped_mae"],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("Reproduction V1 learning curves differ")


def save_reports(
    summary, forecasts, metadata, details, output_dir=DEFAULT_OUTPUT
) -> list[Path]:
    """Validate every report first; use a separate, never-overwritten run namespace."""
    output_dir = Path(output_dir)
    validate_output_dir(output_dir)
    winner = select_cycle2(summary)
    expected = {
        FROZEN_CYCLE1_NAME,
        "MA28",
        *(candidate.name for candidate in candidate_grid()),
    }
    if set(forecasts) != expected:
        raise ValueError("Reports require all four aligned methods")
    reference = forecasts[FROZEN_CYCLE1_NAME]
    metrics, horizons, categories, holidays = [], [], [], []
    for method, scored in forecasts.items():
        scored = align_forecasts(reference, scored)
        metrics.append({"method": method, **aggregation_metrics(scored)})
        horizon = cycle1.build_v2_horizon_diagnostics(scored)
        category, holiday = cycle1.build_v2_breakdown_diagnostics(scored, metadata)
        for report, destination in (
            (horizon, horizons),
            (category, categories),
            (holiday, holidays),
        ):
            report.insert(0, "method", method)
            destination.append(report)
    metric_frame = pd.DataFrame(metrics)
    for row in summary.itertuples(index=False):
        score = metric_frame.set_index("method").loc[row.candidate, "daily_wape"]
        if score != row.v2_wape:
            raise ValueError(
                "Candidate selection WAPE differs from aligned daily predictions"
            )
    curves = build_learning_curves(summary, details)
    datasets = {
        row.candidate: {
            "training_rows": row.training_rows,
            "training_series": row.training_series,
            "training_origins": row.training_origins,
            "category_levels": row.category_levels,
        }
        for row in summary.itertuples(index=False)
    }
    manifest = {
        "protocol": protocol(),
        "selected_candidate": winner,
        "execution": {
            k: v for k, v in details.items() if k != "reference_v1_mae_history"
        },
        "datasets": datasets,
        "near_tree_cap_candidates": summary.loc[
            summary["near_tree_cap"], "candidate"
        ].tolist(),
    }
    reports = {
        "candidate_scores.csv": summary.drop(
            columns=["v1_mae_history", "category_levels", "training_origins"]
        ).to_csv(index=False),
        "method_metrics.csv": metric_frame.to_csv(index=False),
        "v1_curves.csv": curves.to_csv(index=False),
        "horizon_diagnostics.csv": pd.concat(horizons).to_csv(index=False),
        "category_diagnostics.csv": pd.concat(categories).to_csv(index=False),
        "christmas_sensitivity.csv": pd.concat(holidays).to_csv(index=False),
        "protocol_selection.json": json.dumps(manifest, indent=2, allow_nan=False)
        + "\n",
    }
    paths = [output_dir / filename for filename in reports]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError(
            "Cycle 2 run reports already exist; choose a fresh subdirectory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in reports.items():
        with (output_dir / filename).open("x", encoding="utf-8") as stream:
            stream.write(contents)
    return paths


def validate_output_dir(output_dir: Path) -> None:
    if "cycle2_calendar" not in output_dir.parts:
        raise ValueError("Reports must be under a cycle2_calendar namespace")
    if any(
        (output_dir / name).exists() or (output_dir / name).is_symlink()
        for name in REPORT_FILENAMES
    ):
        raise ValueError(
            "Cycle 2 run reports already exist; choose a fresh subdirectory"
        )


def validate_sample_metadata(metadata: pd.DataFrame, manifest: Path) -> None:
    """The full-data entrypoint must retain the frozen deterministic M5 sample."""
    sample = pd.read_csv(manifest, usecols=["series_id"])
    if (
        len(sample) != 3000
        or sample["series_id"].duplicated().any()
        or metadata["series_id"].duplicated().any()
        or set(metadata["series_id"]) != set(sample["series_id"])
    ):
        raise ValueError("Metadata must match the frozen 3000-series M5 sample")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_metadata(paths: dict[str, Path]) -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return {
        "source_commit": commit,
        "source_dirty": dirty,
        "source_files_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for folder in ("src", "scripts")
            for path in sorted((root / folder).glob("*.py"))
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "lightgbm",
                "pandas",
                "numpy",
                "pyarrow",
                "scikit-learn",
                "scipy",
            )
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales", type=Path, default=Path("data/processed/sales.parquet")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/series.parquet")
    )
    parser.add_argument("--calendar", type=Path, default=Path("data/raw/calendar.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run two calendar candidates and recover frozen Cycle 1 predictions",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Compare a fresh independent run with earlier Cycle 2 reports",
    )
    args = parser.parse_args(argv)
    validate_protocol()
    if not args.execute:
        print(json.dumps(protocol(), indent=2))
        print(
            "DRY RUN: no data read, no fitting, no reports written. Pass --execute for full M5."
        )
        return 0
    paths = {"sales": args.sales, "metadata": args.metadata, "calendar": args.calendar}
    for name, path in paths.items():
        if not path.is_file():
            parser.error(f"Missing {name} input: {path}")
    validate_output_dir(args.output_dir)
    # Filter at the storage boundary; prices and later holdout sales are not read.
    sales = pd.read_parquet(
        args.sales,
        columns=["series_id", "date", "quantity"],
        filters=[("date", "<=", VALIDATION_END)],
    )
    metadata = pd.read_parquet(args.metadata)
    sample_manifest = Path(__file__).resolve().parents[1] / "reports/sample_series.csv"
    validate_sample_metadata(metadata, sample_manifest)
    paths["sample_manifest"] = sample_manifest
    calendar = pd.read_csv(
        args.calendar, usecols=CALENDAR_INPUT_COLUMNS, keep_default_na=False
    )
    summary, forecasts, details = run_experiment(sales, metadata, calendar)
    details.update(execution_metadata(paths))
    if args.verify_against:
        verify_reproduction(summary, args.verify_against, forecasts, details)
        details["independent_reproduction_verified_against"] = str(args.verify_against)
    reports = save_reports(summary, forecasts, metadata, details, args.output_dir)
    print(
        summary.drop(
            columns=["v1_mae_history", "category_levels", "training_origins"]
        ).to_string(index=False)
    )
    print(json.dumps(select_cycle2(summary), indent=2))
    print(f"Saved {len(reports)} reports under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
