"""Demand features calculated strictly from sales known at a forecast origin."""

import numpy as np
import pandas as pd

MIN_HISTORY = 28

DEMAND_FEATURE_NAMES = (
    "sales_at_origin",
    "sales_lag_1_at_origin",
    "sales_lag_7_at_origin",
    "sales_lag_14_at_origin",
    "sales_lag_28_at_origin",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_7",
    "rolling_std_28",
    "zero_fraction_28",
    "zero_fraction_56",
    "rolling_sum_7",
    "rolling_sum_28",
)


def _validated_history(history: np.ndarray) -> np.ndarray:
    """Validate ordered daily demand; NaN represents an absent calendar day."""
    try:
        values = np.asarray(history, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("History must contain numeric quantities") from exc

    if values.ndim != 1:
        raise ValueError("History must be a one-dimensional array")
    if len(values) < MIN_HISTORY or not np.isfinite(values[-MIN_HISTORY:]).all():
        raise ValueError("At least 28 consecutive observed days are required")

    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("Observed quantities must be finite")
    if (observed < 0).any() or (observed != np.floor(observed)).any():
        raise ValueError("Observed quantities must be nonnegative whole numbers")
    if (observed > 2**24).any():
        raise ValueError("Quantities exceed the exact float32 integer range")
    return values


def demand_features(history: np.ndarray) -> dict[str, float]:
    """Calculate features as of the final position of a daily demand array.

    Values are chronological, with history[-1] being the forecast origin.
    Never pass observations after the origin. Missing calendar days may be
    represented by NaN; statistics requiring incomplete windows are NaN.
    """
    values = _validated_history(history)

    def lag(days: int) -> float:
        return float(values[-1 - days]) if len(values) > days else float("nan")

    def window(days: int) -> np.ndarray | None:
        if len(values) < days:
            return None
        recent = values[-days:]
        return recent if np.isfinite(recent).all() else None

    w7, w28, w56 = (window(n) for n in (7, 28, 56))

    def statistic(recent: np.ndarray | None, kind: str) -> float:
        if recent is None:
            return float("nan")
        if kind == "mean":
            return float(recent.mean())
        if kind == "std":
            return float(recent.std(ddof=0))
        if kind == "zero_fraction":
            return float(np.mean(recent == 0))
        if kind == "sum":
            return float(recent.sum())
        raise ValueError(f"Unknown statistic: {kind}")

    return {
        "sales_at_origin": lag(0),
        "sales_lag_1_at_origin": lag(1),
        "sales_lag_7_at_origin": lag(7),
        "sales_lag_14_at_origin": lag(14),
        "sales_lag_28_at_origin": lag(28),
        "rolling_mean_7": statistic(w7, "mean"),
        "rolling_mean_28": statistic(w28, "mean"),
        "rolling_mean_56": statistic(w56, "mean"),
        "rolling_std_7": statistic(w7, "std"),
        "rolling_std_28": statistic(w28, "std"),
        "zero_fraction_28": statistic(w28, "zero_fraction"),
        "zero_fraction_56": statistic(w56, "zero_fraction"),
        "rolling_sum_7": statistic(w7, "sum"),
        "rolling_sum_28": statistic(w28, "sum"),
    }


def features_at_origin(sales: pd.DataFrame, origin) -> dict[str, float]:
    """Read one series' daily sales and ignore every observation after origin.

    Designed as a clear correctness reference. The training-dataset builder
    will use batched NumPy histories for efficiency on large M5 datasets.
    """
    if not {"date", "quantity"}.issubset(sales.columns):
        raise ValueError("Sales must contain date and quantity columns")

    cutoff = pd.Timestamp(origin)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("Origin must be a timezone-naive calendar date")

    try:
        dates = pd.to_datetime(sales["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid sales dates") from exc

    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError("Sales dates must be timezone-naive calendar dates")

    past = sales.loc[dates <= cutoff, ["quantity"]].copy()
    past.index = pd.DatetimeIndex(dates[dates <= cutoff])
    if past.empty:
        raise ValueError("No sales at or before the forecast origin")
    if past.index.has_duplicates:
        raise ValueError("Duplicate dates in historical sales")
    if (
        "series_id" in sales.columns
        and sales.loc[dates <= cutoff, "series_id"].nunique() != 1
    ):
        raise ValueError("Expected historical sales for exactly one series")

    try:
        quantities = pd.to_numeric(past["quantity"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid historical quantities") from exc
    if quantities.isna().any():
        raise ValueError("Historical quantities cannot be null")

    calendar = pd.date_range(start=past.index.min(), end=cutoff, freq="D")
    history = quantities.reindex(calendar).to_numpy(dtype=np.float64)
    return demand_features(history)
