"""Strict V2 alignment, aggregation and preregistered Cycle 2 selection.

Only daily pooled V2 WAPE enters the adoption decision. Weekly and 28-day
aggregation first combines days of the same series, never different series.
The fixed 2,992-series production cohort is checked by the experiment runner;
these helpers also accept complete, small synthetic cohorts for unit tests.
"""

import numpy as np
import pandas as pd

from src.forecast_baselines import FORECAST_HORIZON
from src.ml_validation import V2_ORIGIN

FROZEN_CYCLE1_WAPE = 80.86700130410776
FROZEN_CYCLE1_ITERATION = 1944
FROZEN_CYCLE1_NAME = "tweedie_8_origins"
ADOPTION_IMPROVEMENT = 1.0
COMPLEXITY_TOLERANCE = 1.0
CYCLE2_CANDIDATES = {
    "cycle2_tweedie_8_origins_calendar": 8,
    "cycle2_tweedie_16_origins_calendar": 16,
}
FORECAST_KEYS = ["series_id", "origin", "horizon", "date"]


def _numeric(values: pd.Series, label: str) -> np.ndarray:
    """Validate numeric finite values without accepting boolean flags."""
    if (
        pd.api.types.is_bool_dtype(values.dtype)
        or values.map(
            lambda value: isinstance(
                value, (bool, np.bool_, complex, np.complexfloating)
            )
        ).any()
    ):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not np.isfinite(numbers).all():
        raise ValueError(f"{label} must be finite")
    return numbers


def _dates(values: pd.Series, label: str) -> pd.Series:
    """Accept valid timezone-naive midnight dates, including ISO strings."""
    try:
        dates = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain valid dates") from exc
    if not pd.api.types.is_datetime64_dtype(dates.dtype) or dates.dt.tz is not None:
        raise ValueError(f"{label} dates must be timezone-naive")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{label} must contain nonmissing midnight dates")
    return dates


def _validated_scored(scored: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical copy while leaving the caller's frame unchanged."""
    required = [*FORECAST_KEYS, "target", "prediction"]
    if not isinstance(scored, pd.DataFrame) or scored.empty:
        raise ValueError("Scored V2 observations must be a nonempty DataFrame")
    if scored.columns.duplicated().any():
        raise ValueError("Scored V2 observations contain duplicate columns")
    if not set(required).issubset(scored.columns):
        raise ValueError("Missing scored V2 observation columns")
    if scored[required].isna().any().any():
        raise ValueError("Scored V2 observations contain missing values")
    if (
        not scored["series_id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("series_id must contain nonempty strings")

    frame = scored.copy()
    frame["origin"] = _dates(frame["origin"], "origin")
    frame["date"] = _dates(frame["date"], "date")
    if not frame["origin"].eq(V2_ORIGIN).all():
        raise ValueError("Scored observations must use the fixed V2 origin")

    horizons = _numeric(frame["horizon"], "horizon")
    if (
        (horizons != np.floor(horizons)).any()
        or (horizons < 1).any()
        or (horizons > FORECAST_HORIZON).any()
    ):
        raise ValueError("horizon must contain integers from 1 through 28")
    frame["horizon"] = horizons.astype(np.int64)
    expected_dates = V2_ORIGIN + pd.to_timedelta(frame["horizon"], unit="D")
    if not frame["date"].eq(expected_dates).all():
        raise ValueError("V2 dates must equal origin plus horizon")
    if frame.duplicated(FORECAST_KEYS).any():
        raise ValueError("Duplicate scored V2 forecast keys")
    counts = frame.groupby("series_id", sort=False)["horizon"].nunique()
    if not counts.eq(FORECAST_HORIZON).all():
        raise ValueError("Every V2 series must have all 28 forecast horizons")

    for column in ("target", "prediction"):
        numbers = _numeric(frame[column], column)
        if (numbers < 0).any():
            raise ValueError(f"{column} must be nonnegative")
        frame[column] = numbers
    return frame.sort_values(FORECAST_KEYS, kind="stable").reset_index(drop=True)


def validate_scored(scored: pd.DataFrame) -> None:
    """Reject invalid, duplicated or incomplete fixed-window V2 observations."""
    _validated_scored(scored)


def align_forecasts(
    reference: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Return sorted forecasts only when keys and actuals match exactly.

    Both inputs must contain their actual target values and nonnegative daily
    predictions. Comparing complete scored frames prevents a method from
    silently changing actuals or evaluating a different population.
    """
    expected = _validated_scored(reference)
    aligned = _validated_scored(forecasts)
    if not expected[FORECAST_KEYS].equals(aligned[FORECAST_KEYS]):
        raise ValueError("Forecasts must match the identical V2 key set")
    if not np.array_equal(expected["target"].to_numpy(), aligned["target"].to_numpy()):
        raise ValueError("Forecasts must contain identical V2 target values")
    return aligned


def _pooled_wape(frame: pd.DataFrame, metric: str) -> float:
    actual = frame["target"].to_numpy(dtype=np.float64)
    predicted = frame["prediction"].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = actual.sum()
        numerator = np.abs(actual - predicted).sum()
    if denominator == 0:
        raise ValueError(f"{metric} is undefined when pooled actual sales are zero")
    if not np.isfinite(denominator) or not np.isfinite(numerator):
        raise ValueError(f"{metric} pooled sums must be finite")
    result = float(100 * (numerator / denominator))
    if not np.isfinite(result):
        raise ValueError(f"{metric} must be finite")
    return result


def aggregation_metrics(scored: pd.DataFrame) -> dict[str, float]:
    """Compute daily, four-week, per-week and 28-day pooled errors.

    The absolute error is taken after summing days within each series-week or
    series-period, then pooled across those groups. Zero actual denominators
    fail explicitly, including an individual week with no demand anywhere.
    """
    frame = _validated_scored(scored)
    daily_wape = _pooled_wape(frame, "daily_wape")
    frame["week"] = (frame["horizon"] - 1) // 7 + 1
    weekly = frame.groupby(["series_id", "origin", "week"], sort=True, as_index=False)[
        ["target", "prediction"]
    ].sum()
    period = frame.groupby(["series_id", "origin"], sort=True, as_index=False)[
        ["target", "prediction"]
    ].sum()
    result = {
        "daily_wape": daily_wape,
        "weekly_wape": _pooled_wape(weekly, "weekly_wape"),
        "period_28_wape": _pooled_wape(period, "period_28_wape"),
    }
    for week in range(1, 5):
        metric = f"week_{week}_wape"
        result[metric] = _pooled_wape(weekly.loc[weekly["week"].eq(week)], metric)
    errors = frame["target"].to_numpy() - frame["prediction"].to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        result["mae"] = float(np.mean(np.abs(errors)))
        result["rmse"] = float(np.sqrt(np.mean(errors**2)))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Aggregated error metrics must be finite")
    return result


def select_cycle2(summary: pd.DataFrame) -> dict:
    """Apply the strict >1 point adoption rule and <=1 point complexity tie.

    V1 determines each supplied best iteration. Only V2 *daily* WAPE affects
    this decision; other diagnostic columns are deliberately ignored.
    """
    required = {"candidate", "n_origins", "v2_wape", "best_iteration"}
    if (
        not isinstance(summary, pd.DataFrame)
        or summary.columns.duplicated().any()
        or not required.issubset(summary.columns)
    ):
        raise ValueError("Missing or duplicate Cycle 2 selection columns")
    if (
        len(summary) != 2
        or summary["candidate"].duplicated().any()
        or set(summary["candidate"]) != set(CYCLE2_CANDIDATES)
    ):
        raise ValueError("Selection requires the exact two Cycle 2 candidates")

    records = summary.copy()
    for column in ("n_origins", "v2_wape", "best_iteration"):
        records[column] = _numeric(records[column], column)
    if (records["v2_wape"] < 0).any():
        raise ValueError("Candidate WAPE values must be nonnegative")
    iterations = records["best_iteration"]
    if (iterations != np.floor(iterations)).any() or not iterations.between(
        1, 2000
    ).all():
        raise ValueError("best_iteration must be an integer from 1 through 2000")
    expected_origins = records["candidate"].map(CYCLE2_CANDIDATES)
    if not records["n_origins"].eq(expected_origins).all():
        raise ValueError("Candidate n_origins does not match its name")
    if "objective" in records and not records["objective"].eq("tweedie").all():
        raise ValueError("Cycle 2 candidates must use the Tweedie objective")

    threshold = FROZEN_CYCLE1_WAPE - ADOPTION_IMPROVEMENT
    qualifying = records.loc[records["v2_wape"] < threshold]
    if qualifying.empty:
        winner = {
            "candidate": FROZEN_CYCLE1_NAME,
            "n_origins": 8,
            "best_iteration": FROZEN_CYCLE1_ITERATION,
            "v2_wape": FROZEN_CYCLE1_WAPE,
        }
        reason = "Neither calendar candidate strictly improves by more than 1 point"
    else:
        minimum = qualifying["v2_wape"].min()
        preferred = (
            qualifying.loc[qualifying["v2_wape"] <= minimum + COMPLEXITY_TOLERANCE]
            .sort_values("n_origins", kind="stable")
            .iloc[0]
        )
        winner = {
            "candidate": str(preferred["candidate"]),
            "n_origins": int(preferred["n_origins"]),
            "best_iteration": int(preferred["best_iteration"]),
            "v2_wape": float(preferred["v2_wape"]),
        }
        reason = "Qualifying daily V2 WAPE with the 1 point, fewer origins preference"
    return {
        **winner,
        "objective": "tweedie",
        "adopted_cycle2": not qualifying.empty,
        "selection_metric": "daily pooled V2 WAPE",
        "frozen_cycle1_v2_wape": FROZEN_CYCLE1_WAPE,
        "strict_adoption_threshold": threshold,
        "required_improvement_percentage_points": ADOPTION_IMPROVEMENT,
        "complexity_tolerance_percentage_points": COMPLEXITY_TOLERANCE,
        "best_observed_cycle2_v2_wape": float(records["v2_wape"].min()),
        "reason": reason,
    }
