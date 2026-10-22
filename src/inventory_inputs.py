"""Phase 3C inputs that do not need LightGBM.

Review calendar, demand matrix, MA28 forecast paths, value proxies, artifact
and input hash checks, and comparison of forecast paths with published
predictions (docs/phase3_inventory_protocol.md, Sections 2, 3, 10 and 12).
Nothing here computes an inventory metric.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting import build_sales_matrix
from src.forecast_baselines import (
    FORECAST_HORIZON,
    MIN_HISTORY,
    moving_average_forecast,
)
from src.inventory_metrics import value_proxy_from_prices

ROOT = Path(__file__).resolve().parents[1]
# Committed record of the published Issue #28 results: the anchor of the chain
# published result -> original preflight.json -> model, cohort, data hashes.
PUBLISHED_REVEAL_MANIFEST = (
    ROOT / "reports/ml_final_evaluation/primary/reveal/manifest.json"
)

FIRST_REVIEW = pd.Timestamp("2016-01-03")
LAST_SIMULATED_DAY = pd.Timestamp("2016-04-24")
REVIEW_PERIOD_DAYS = 7
REVIEW_ORIGINS = tuple(
    pd.date_range(
        FIRST_REVIEW,
        LAST_SIMULATED_DAY - pd.Timedelta(days=1),
        freq=f"{REVIEW_PERIOD_DAYS}D",
    )
)
ISSUE28_ORIGINS = tuple(
    pd.to_datetime(["2016-01-03", "2016-01-31", "2016-02-28", "2016-03-27"])
)
LIGHTGBM_MODEL = "cycle2_tweedie_8_origins_calendar"
MA28_MODEL = "moving_average_28"
EXPECTED_SERIES = 2996
ISSUE28_ARTIFACTS = ("model.txt", "model.json", "predictions.parquet", "lookups.parquet")
PREDICTION_TOLERANCE = 1e-9
VALUE_WINDOW_DAYS = 28


def simulated_days() -> pd.DatetimeIndex:
    """Demand days 2016-01-04 through 2016-04-24 (112 days)."""
    return pd.date_range(FIRST_REVIEW + pd.Timedelta(days=1), LAST_SIMULATED_DAY)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_sha256(series_ids) -> str:
    """Same canonical hash as Issue #28: sorted one-column CSV of series_id."""
    frame = pd.DataFrame({"series_id": pd.Series(series_ids, dtype=object)})
    canonical = frame.sort_values(["series_id"]).reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


def verify_issue28_artifacts(
    artifacts_dir,
    published_manifest=PUBLISHED_REVEAL_MANIFEST,
) -> dict:
    """Check the saved Issue #28 artifacts back to the published results.

    The local preflight.json must be byte-identical to the one whose hash the
    committed reveal manifest records, and predictions.parquet must match the
    published prediction hash. Only then are the other artifact hashes in that
    preflight trusted. A different set of files with its own consistent
    preflight.json is rejected.
    """
    folder = Path(artifacts_dir)
    manifest_path = folder / "preflight.json"
    published_path = Path(published_manifest)

    if not published_path.is_file():
        raise ValueError(f"Missing published Issue #28 manifest: {published_path}")

    published = json.loads(published_path.read_text(encoding="utf-8"))

    for key in ("preflight_sha256", "prediction_sha256"):
        if not isinstance(published.get(key), str) or len(published[key]) != 64:
            raise ValueError(f"Published Issue #28 manifest lacks {key}")

    if not manifest_path.is_file():
        raise ValueError(f"Missing Issue #28 preflight manifest: {manifest_path}")

    if file_sha256(manifest_path) != published["preflight_sha256"]:
        raise ValueError(
            "Local preflight.json is not the one behind the published Issue #28 results"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("artifact_sha256", {})

    if recorded.get("predictions.parquet") != published["prediction_sha256"]:
        raise ValueError("Preflight prediction hash differs from the published one")

    for name in ISSUE28_ARTIFACTS:
        path = folder / name

        if name not in recorded:
            raise ValueError(f"Preflight manifest has no hash for {name}")

        if not path.is_file():
            raise ValueError(f"Missing Issue #28 artifact: {name}")

        if file_sha256(path) != recorded[name]:
            raise ValueError(f"Issue #28 artifact changed: {name}")

    return manifest


def verify_input_files(manifest: dict, paths: dict) -> None:
    """Check data files against the input hashes recorded for Issue #28."""
    records = manifest.get("execution", {}).get("inputs", {})

    for name, path in paths.items():
        if name not in records:
            raise ValueError(f"No recorded Issue #28 hash for input: {name}")

        if file_sha256(path) != records[name]["sha256"]:
            raise ValueError(f"Input differs from the Issue #28 run: {name}")


def frozen_cohort(lookups: pd.DataFrame, manifest: dict, *, expected=EXPECTED_SERIES):
    """Return the sorted frozen cohort after checking its count and hash."""
    if "series_id" not in lookups.columns:
        raise ValueError("Lookups must contain series_id")

    ids = pd.Index(lookups["series_id"])

    if ids.empty or ids.has_duplicates or ids.isna().any():
        raise ValueError("Cohort must be nonempty, unique, and non-null")

    if len(ids) != expected:
        raise ValueError(f"Cohort has {len(ids)} series, expected {expected}")

    if cohort_sha256(ids) != manifest.get("cohort_sha256"):
        raise ValueError("Cohort hash differs from the Issue #28 preflight")

    return ids.sort_values()


def _cohort_rows(matrix: pd.DataFrame, cohort, dates) -> pd.DataFrame:
    cohort = pd.Index(cohort)
    missing_series = cohort.difference(matrix.index)

    if len(missing_series):
        raise ValueError(f"Sales missing for {len(missing_series)} cohort series")

    if not pd.DatetimeIndex(dates).isin(matrix.columns).all():
        raise ValueError("Sales do not cover every required date")

    return matrix.loc[cohort, dates]


def demand_matrix(sales: pd.DataFrame, cohort) -> np.ndarray:
    """Observed sales of each cohort series on the 112 simulated days."""
    rows = _cohort_rows(build_sales_matrix(sales), cohort, simulated_days())

    if rows.isna().any().any():
        raise ValueError("Missing sales for a cohort series on a simulated day")

    return rows.to_numpy(dtype=np.float64).astype(np.int64)


def ma28_paths(sales: pd.DataFrame, cohort, origins=REVIEW_ORIGINS) -> np.ndarray:
    """MA28 daily forecasts, shape (series, origins, 28), from sales through each origin.

    Uses the unchanged Phase 1 function on the same float32-to-float64 path as
    run_backtesting, so values at the Issue #28 origins match Phase 1 exactly.
    """
    matrix = build_sales_matrix(sales)
    cohort = pd.Index(cohort)
    paths = np.zeros((len(cohort), len(origins), FORECAST_HORIZON))

    for k, origin in enumerate(origins):
        history_dates = pd.date_range(end=pd.Timestamp(origin), periods=MIN_HISTORY)
        history = _cohort_rows(matrix, cohort, history_dates).to_numpy(dtype=np.float64)

        if np.isnan(history).any():
            raise ValueError(f"Incomplete 28-day history at origin {origin.date()}")

        paths[:, k, :] = np.vstack([moving_average_forecast(row) for row in history])

    return paths


def paths_from_predictions(
    frame: pd.DataFrame,
    cohort,
    origins,
    *,
    value_column: str,
) -> np.ndarray:
    """Reshape long predictions to (series, origins, 28); every key exactly once."""
    required = {"series_id", "origin", value_column}

    if not required.issubset(frame.columns):
        raise ValueError(f"Predictions need columns {sorted(required)}")

    data = frame.copy()
    data["origin"] = pd.to_datetime(data["origin"])

    if "horizon" not in data.columns:
        if "date" not in data.columns:
            raise ValueError("Predictions need horizon or date")
        data["horizon"] = (pd.to_datetime(data["date"]) - data["origin"]).dt.days

    cohort = pd.Index(cohort)
    origins = pd.DatetimeIndex(origins)
    data = data.loc[data["series_id"].isin(cohort) & data["origin"].isin(origins)]

    if data.duplicated(["series_id", "origin", "horizon"]).any():
        raise ValueError("Duplicate prediction keys")

    if not data["horizon"].isin(range(1, FORECAST_HORIZON + 1)).all():
        raise ValueError("Prediction horizons must be 1..28")

    expected = len(cohort) * len(origins) * FORECAST_HORIZON

    if len(data) != expected:
        raise ValueError(f"Expected {expected} prediction rows, found {len(data)}")

    values = data[value_column].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")

    series_index = cohort.get_indexer(data["series_id"])
    origin_index = origins.get_indexer(data["origin"])
    paths = np.full((len(cohort), len(origins), FORECAST_HORIZON), np.nan)
    paths[series_index, origin_index, data["horizon"].to_numpy(dtype=int) - 1] = values

    return paths


def check_paths_match(paths, reference, *, tolerance: float, label: str) -> float:
    """Raise unless two forecast-path arrays agree within tolerance."""
    a = np.asarray(paths, dtype=float)
    b = np.asarray(reference, dtype=float)

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")

    if a.shape != b.shape:
        raise ValueError(f"{label}: shapes differ {a.shape} vs {b.shape}")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{label}: forecast paths must be finite")

    largest = float(np.abs(a - b).max())

    if largest > tolerance:
        raise ValueError(
            f"{label}: forecasts differ from the reference by up to {largest:.3g}"
        )

    return largest


def value_proxies(sales: pd.DataFrame, cohort, as_of=FIRST_REVIEW) -> np.ndarray:
    """Median available sell price over the 28 days through as_of.

    Stops if a cohort series has no available price in that window.
    """
    if not {"series_id", "date", "price"}.issubset(sales.columns):
        raise ValueError("Sales need series_id, date, and price")

    end = pd.Timestamp(as_of)
    dates = pd.date_range(end=end, periods=VALUE_WINDOW_DAYS)
    data = sales.loc[sales["series_id"].isin(cohort) & sales["date"].isin(dates)].copy()

    if "price_available" in data.columns:
        data.loc[~data["price_available"].astype(bool), "price"] = np.nan

    cohort = pd.Index(cohort)
    prices = (
        data.pivot(index="series_id", columns="date", values="price")
        .reindex(index=cohort, columns=dates)
        .to_numpy(dtype=float)
    )

    return value_proxy_from_prices(prices, series_ids=cohort)
