"""Aggregate forecasting metrics and classify demand using past observations."""

import numpy as np
import pandas as pd

from src.metrics import mae, rmse, wape

LOOKBACK_DAYS = 365
MIN_HISTORY = 28


def build_demand_profile(
    sales: pd.DataFrame,
    eligible_series,
    first_origin,
) -> pd.DataFrame:
    """Classify eligible series using at most 365 pre-test days."""
    required = {"series_id", "date", "quantity"}

    if not required.issubset(sales.columns):
        missing = sorted(required - set(sales.columns))
        raise ValueError(f"Missing sales columns: {missing}")

    series_ids = pd.Index(eligible_series)

    if series_ids.empty or series_ids.has_duplicates or series_ids.isna().any():
        raise ValueError("Eligible series must be nonempty, unique, and non-null")

    origin = pd.Timestamp(first_origin)

    if pd.isna(origin) or origin.tzinfo is not None or origin != origin.normalize():
        raise ValueError("First origin must be a valid timezone-naive date")

    start = origin - pd.Timedelta(days=LOOKBACK_DAYS - 1)

    # Future observations are excluded before calculating demand groups.
    data = sales.loc[
        sales["series_id"].isin(series_ids) & sales["date"].between(start, origin),
        ["series_id", "date", "quantity"],
    ].copy()

    if data.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate historical series/date pairs")

    if data.empty:
        raise ValueError("No historical observations for eligible series")

    quantities = pd.to_numeric(data["quantity"], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("Historical quantities must be finite and nonnegative")

    data["quantity"] = quantities

    required_dates = pd.date_range(
        end=origin,
        periods=MIN_HISTORY,
        freq="D",
    )

    recent = data.loc[data["date"].isin(required_dates)]

    complete = (
        recent.groupby("series_id")["date"].nunique().reindex(series_ids, fill_value=0)
    )

    if not complete.eq(MIN_HISTORY).all():
        raise ValueError("Eligible series lacks 28 consecutive days at first origin")

    counts = data.groupby("series_id")["quantity"].agg(
        observed_days="size",
        zero_days=lambda values: int(values.eq(0).sum()),
    )

    counts = counts.reindex(series_ids)

    counts["zero_fraction"] = counts["zero_days"] / counts["observed_days"]

    counts["demand_group"] = np.where(
        counts["zero_fraction"] < 0.25,
        "frequent",
        np.where(
            counts["zero_fraction"] > 0.75,
            "sparse",
            "mixed",
        ),
    )

    return (
        counts.rename_axis("series_id")
        .reset_index()
        .sort_values("series_id")
        .reset_index(drop=True)
    )


def evaluate_forecasts(
    predictions: pd.DataFrame,
    series_metadata: pd.DataFrame,
    demand_profile: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate pooled metrics by model, window, category, and sparsity."""
    required = {
        "series_id",
        "origin",
        "date",
        "model",
        "actual",
        "predicted",
    }

    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("Predictions must be nonempty with all required columns")

    if predictions.duplicated(["series_id", "origin", "date", "model"]).any():
        raise ValueError("Duplicate forecast keys")

    for lookup, fields in (
        (series_metadata, ["series_id", "cat_id"]),
        (demand_profile, ["series_id", "demand_group"]),
    ):
        if not set(fields).issubset(lookup.columns):
            raise ValueError(f"Missing lookup columns: {fields}")

        if lookup["series_id"].duplicated().any() or lookup[fields].isna().any().any():
            raise ValueError(
                "Lookup keys and classifications must be unique and non-null"
            )

    categories = series_metadata.set_index("series_id")["cat_id"]
    groups = demand_profile.set_index("series_id")["demand_group"]

    enriched = predictions.copy()

    enriched["cat_id"] = enriched["series_id"].map(categories)
    enriched["demand_group"] = enriched["series_id"].map(groups)

    if enriched[["cat_id", "demand_group"]].isna().any().any():
        raise ValueError("Missing category or demand group for predicted series")

    def aggregate(keys: list[str]) -> pd.DataFrame:
        records = []

        for values, frame in enriched.groupby(keys, sort=True):
            key_values = values if isinstance(values, tuple) else (values,)

            actual = frame["actual"].to_numpy(dtype=np.float64)
            predicted = frame["predicted"].to_numpy(dtype=np.float64)

            mean_error = mae(actual, predicted)
            root_error = rmse(actual, predicted)
            total = float(actual.sum())

            records.append(
                {
                    **dict(zip(keys, key_values, strict=True)),
                    "series_count": int(frame["series_id"].nunique()),
                    "observations": len(frame),
                    "total_actual": total,
                    "mae": mean_error,
                    "rmse": root_error,
                    "wape": (wape(actual, predicted) if total > 0 else np.nan),
                }
            )

        return pd.DataFrame.from_records(records)

    return {
        "overall": aggregate(["model"]),
        "by_window": aggregate(["model", "origin"]),
        "by_category": aggregate(["model", "cat_id"]),
        "by_sparsity": aggregate(["model", "demand_group"]),
    }
