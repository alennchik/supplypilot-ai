"""Train and persist the initial global LightGBM demand forecasting model."""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.ml_model import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    load_global_model,
    predict_global_model,
    save_global_model,
    train_global_model,
)

DEFAULT_INPUT = Path("data/processed/ml_training.parquet")
DEFAULT_MODEL = Path("models/global_lightgbm.txt")
DEFAULT_MANIFEST = Path("reports/ml_model_training_manifest.json")
DEFAULT_CUTOFF = "2015-11-08"


def validate_training_dataset(data: pd.DataFrame, cutoff: str) -> None:
    """Check complete, correctly aligned 28-day training targets."""
    required = {*FEATURE_COLUMNS, *OUTPUT_COLUMNS, "target"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Training dataset cannot be empty")

    cutoff_date = pd.Timestamp(cutoff)

    if (
        pd.isna(cutoff_date)
        or cutoff_date.tzinfo is not None
        or cutoff_date != cutoff_date.normalize()
    ):
        raise ValueError("Cutoff must be a timezone-naive calendar date")

    origins = pd.to_datetime(data["origin"], errors="raise")
    dates = pd.to_datetime(data["date"], errors="raise")
    horizons = pd.to_numeric(data["horizon"], errors="raise")

    if origins.isna().any() or dates.isna().any() or horizons.isna().any():
        raise ValueError("Missing training dates or forecast horizons")

    if (horizons < 1).any() or (horizons > 28).any() or not (horizons % 1 == 0).all():
        raise ValueError("Forecast horizons must be integers from 1 to 28")

    expected_dates = origins + pd.to_timedelta(horizons, unit="D")

    if not (dates == expected_dates).all():
        raise ValueError("Target dates must equal forecast origin plus horizon")

    if (dates > cutoff_date).any():
        raise ValueError("Training targets exceed the training cutoff")

    if data[["series_id", "origin", "horizon"]].isna().any().any():
        raise ValueError("Missing training identifiers")

    if data.duplicated(["series_id", "origin", "horizon"]).any():
        raise ValueError("Duplicate training example keys")

    per_path = data.groupby(["series_id", "origin"], sort=False)["horizon"].size()

    if not per_path.eq(28).all():
        raise ValueError("Every series/origin must have all 28 horizons")


def peak_rss_mib() -> float:
    """Return peak process memory in MiB on macOS or Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024

    return round(rss / divisor, 2)


def train_from_parquet(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    cutoff: str = DEFAULT_CUTOFF,
    n_estimators: int = 100,
    n_jobs: int = 2,
) -> dict:
    """Train a fixed initial model and verify predictions after reloading."""
    if n_estimators < 1 or n_jobs < 1:
        raise ValueError("n_estimators and n_jobs must be positive")

    input_path = Path(input_path)
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    category_path = model_path.with_suffix(".json")

    paths = (input_path, model_path, category_path, manifest_path)

    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Training input and artifact paths must be distinct")

    started = time.perf_counter()

    training = pd.read_parquet(input_path)
    validate_training_dataset(training, cutoff)

    fit_started = time.perf_counter()

    model = train_global_model(
        training,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )

    fit_seconds = round(time.perf_counter() - fit_started, 2)

    save_global_model(model, model_path)
    loaded = load_global_model(model_path)

    # Check one complete 28-day prediction path after model reload.
    first = training.iloc[0]

    smoke = training.loc[
        (training["series_id"] == first["series_id"])
        & (training["origin"] == first["origin"])
    ].sort_values("horizon")

    before = predict_global_model(model, smoke)
    after = predict_global_model(loaded, smoke)

    if len(after) != 28 or after["horizon"].tolist() != list(range(1, 29)):
        raise ValueError("Saved-model smoke test did not produce all 28 horizons")

    if not np.allclose(
        before["prediction"],
        after["prediction"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Reloaded model predictions do not match original predictions")

    if not np.isfinite(after["prediction"]).all() or (after["prediction"] < 0).any():
        raise ValueError("Invalid saved-model smoke test predictions")

    summary = {
        "training_cutoff": pd.Timestamp(cutoff).date().isoformat(),
        "training_rows": len(training),
        "unique_series": int(training["series_id"].nunique()),
        "training_origins": sorted(
            training["origin"].dt.strftime("%Y-%m-%d").unique().tolist()
        ),
        "forecast_horizon": 28,
        "feature_columns": list(FEATURE_COLUMNS),
        "category_levels": loaded.category_levels,
        "model_configuration": {
            "objective": "regression",
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": n_jobs,
            "deterministic": True,
            "force_col_wise": True,
        },
        "model_path": str(model_path),
        "model_size_mib": round(model_path.stat().st_size / (1024**2), 2),
        "fit_seconds": fit_seconds,
        "total_seconds": round(time.perf_counter() - started, 2),
        "peak_process_rss_mib": peak_rss_mib(),
        "smoke_test": ("28 direct forecasts, finite and nonnegative; reload matches"),
        "environment": {
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)

    args = parser.parse_args()

    summary = train_from_parquet(
        input_path=args.input,
        model_path=args.model,
        manifest_path=args.manifest,
        cutoff=args.cutoff,
        n_estimators=args.estimators,
        n_jobs=args.threads,
    )

    print(f"Training rows: {summary['training_rows']:,}")
    print(f"Series: {summary['unique_series']:,}")
    print(f"Training time: {summary['fit_seconds']:.2f}s")
    print(f"Peak process RSS: {summary['peak_process_rss_mib']:.2f} MiB")
    print(f"Model saved: {args.model}")
    print(f"Manifest saved: {args.manifest}")
    print("Saved-model prediction smoke test: PASS")


if __name__ == "__main__":
    main()
