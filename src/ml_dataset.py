"""Build supervised, origin-relative training examples for direct 28-day ML forecasts."""

import numpy as np
import pandas as pd

from src.backtesting import build_sales_matrix
from src.calendar_features import attach_calendar_features
from src.forecast_baselines import FORECAST_HORIZON
from src.ml_features import DEMAND_FEATURE_NAMES, MIN_HISTORY, demand_features

METADATA_COLUMNS = ("store_id", "dept_id", "cat_id")
IDENTIFIER_COLUMNS = ("series_id", "origin", "date", "horizon")
TARGET_COLUMN = "target"


def _calendar_date(value, name: str) -> pd.Timestamp:
    """Require a valid, timezone-naive calendar date."""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid calendar date") from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f"{name} must be a timezone-naive calendar date")
    return result


def select_training_origins(
    training_cutoff, n_origins: int = 8
) -> tuple[pd.Timestamp, ...]:
    """Choose the last N weekly origins with full 28-day labels by cutoff.

    Returns origins in chronological order. Calendar alignment follows the
    cutoff; the defaults therefore match the original Sunday M5 protocol.
    """
    cutoff = _calendar_date(training_cutoff, "Training cutoff")
    if isinstance(n_origins, bool) or not isinstance(n_origins, int) or n_origins < 1:
        raise ValueError("n_origins must be a positive integer")
    latest = cutoff - pd.Timedelta(days=FORECAST_HORIZON)
    return tuple(
        latest - pd.Timedelta(days=7 * i) for i in range(n_origins - 1, -1, -1)
    )


def _metadata_by_series(metadata: pd.DataFrame, expected_ids: pd.Index) -> pd.DataFrame:
    required = {"series_id", *METADATA_COLUMNS}
    absent = required - set(metadata.columns)
    if absent:
        raise ValueError(f"Missing required metadata columns: {sorted(absent)}")

    records = metadata.loc[:, ["series_id", *METADATA_COLUMNS]].copy()
    if records.empty or records.isna().any().any():
        raise ValueError("Metadata cannot be empty or contain null identifiers")

    for column in records.columns:
        if (
            not records[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Metadata column {column} must contain nonempty strings")

    if records["series_id"].duplicated().any():
        raise ValueError("Duplicate metadata series IDs")

    result = records.set_index("series_id").sort_index()
    if not result.index.equals(expected_ids):
        raise ValueError("Sales and metadata must contain the same series IDs")

    return result


def build_ml_training_dataset(
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
    training_cutoff,
    n_origins: int = 8,
    *,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate daily 1..28-horizon targets and origin-safe historical features.

    Only series with 28 continuous observed history days at each origin are
    eligible at that origin. Missing future targets for any eligible series
    cause an error rather than silently changing the evaluation population.
    Passing calendar opts into the seven Cycle 2 target-date features; its
    scheduled events and SNAP indicators are assumed known at each origin.
    """
    origins = select_training_origins(training_cutoff, n_origins)
    cutoff = _calendar_date(training_cutoff, "Training cutoff")

    matrix = build_sales_matrix(sales)
    by_series = _metadata_by_series(metadata, matrix.index)

    if cutoff > matrix.columns.max():
        raise ValueError("Training cutoff extends beyond available sales dates")

    frames: list[pd.DataFrame] = []

    for origin in origins:
        target_dates = pd.date_range(
            origin + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON,
            freq="D",
        )

        if target_dates[-1] > cutoff:
            raise ValueError("Training target date exceeds training cutoff")

        if not target_dates.isin(matrix.columns).all():
            raise ValueError("Required training target calendar dates are missing")

        # Unavailable older history is NaN, never zero.
        history_dates = pd.date_range(end=origin, periods=56, freq="D")
        history = matrix.reindex(columns=history_dates)

        eligible = history.iloc[:, -MIN_HISTORY:].notna().all(axis=1)
        chosen = matrix.index[eligible]

        if len(chosen) == 0:
            raise ValueError(f"No eligible series at training origin {origin.date()}")

        targets = matrix.loc[chosen, target_dates]

        if targets.isna().any().any():
            raise ValueError(
                f"Missing target values at training origin {origin.date()}"
            )

        historical = history.loc[chosen].to_numpy(dtype=np.float64)
        feature_dicts = [demand_features(row) for row in historical]

        features = pd.DataFrame(
            feature_dicts,
            columns=DEMAND_FEATURE_NAMES,
        )

        n_series = len(chosen)
        count = n_series * FORECAST_HORIZON

        result = pd.DataFrame(
            {
                "series_id": np.repeat(chosen.to_numpy(), FORECAST_HORIZON),
                "origin": pd.Timestamp(origin),
                "date": np.tile(target_dates.to_numpy(), n_series),
                "horizon": np.tile(
                    np.arange(1, FORECAST_HORIZON + 1, dtype=np.int8),
                    n_series,
                ),
            }
        )

        for column in METADATA_COLUMNS:
            result[column] = np.repeat(
                by_series.loc[chosen, column].to_numpy(),
                FORECAST_HORIZON,
            )

        for column in DEMAND_FEATURE_NAMES:
            result[column] = np.repeat(
                features[column].to_numpy(dtype=np.float32),
                FORECAST_HORIZON,
            )

        result[TARGET_COLUMN] = targets.to_numpy(dtype=np.float32).reshape(count)

        frames.append(result)

    output = pd.concat(frames, ignore_index=True)

    if output.duplicated(["series_id", "origin", "horizon"]).any():
        raise ValueError("Duplicate supervised training keys")

    if (output["date"] > cutoff).any():
        raise ValueError("Training target date exceeds training cutoff")

    if calendar is not None:
        output = attach_calendar_features(output, calendar, metadata)

    return output
