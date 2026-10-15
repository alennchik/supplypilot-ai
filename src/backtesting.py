"""Leakage-free rolling-origin evaluation for 28-day demand forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast_baselines import (
    FORECAST_HORIZON,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)

N_WINDOWS = 4
MIN_HISTORY = 28
MAX_EXACT_FLOAT32_INTEGER = 2**24

MODELS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "moving_average_28": moving_average_forecast,
}


@dataclass(frozen=True)
class ForecastWindow:
    """Origin and inclusive forecast-date boundaries."""

    origin: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


def make_forecast_windows(evaluation_end_date) -> tuple[ForecastWindow, ...]:
    """Return four non-overlapping, chronological 28-day windows."""
    end = pd.Timestamp(evaluation_end_date)

    if pd.isna(end) or end.tzinfo is not None or end != end.normalize():
        raise ValueError("Evaluation end date must be a valid timezone-naive date")

    windows = []

    for i in range(N_WINDOWS):
        origin = end - pd.Timedelta(days=FORECAST_HORIZON * (N_WINDOWS - i))

        windows.append(
            ForecastWindow(
                origin=origin,
                start=origin + pd.Timedelta(days=1),
                end=origin + pd.Timedelta(days=FORECAST_HORIZON),
            )
        )

    return tuple(windows)


def build_sales_matrix(sales: pd.DataFrame) -> pd.DataFrame:
    """Pivot sales to series-by-day float32 values; missing is not zero."""
    required = {"series_id", "date", "quantity"}
    missing = required - set(sales.columns)

    if missing:
        raise ValueError(f"Missing required sales columns: {sorted(missing)}")

    if sales.empty:
        raise ValueError("Sales data cannot be empty")

    data = sales.loc[:, ["series_id", "date", "quantity"]].copy()

    if (
        data["series_id"].isna().any()
        or not data["series_id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Series IDs must be nonempty strings")

    try:
        dates = pd.to_datetime(data["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be valid timestamps") from exc

    if (
        dates.isna().any()
        or dates.dt.tz is not None
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError("Dates must be non-null timezone-naive calendar dates")

    data["date"] = dates

    try:
        quantity = pd.to_numeric(data["quantity"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Quantities must be numeric") from exc

    if not np.isfinite(quantity).all():
        raise ValueError("Quantities must be finite and non-null")

    if (quantity < 0).any() or (quantity != np.floor(quantity)).any():
        raise ValueError("Quantities must be nonnegative whole numbers")

    if (quantity > MAX_EXACT_FLOAT32_INTEGER).any():
        raise ValueError("Quantity exceeds float32 exact integer range")

    if data.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate (series_id, date) observations")

    calendar = pd.date_range(
        data["date"].min(),
        data["date"].max(),
        freq="D",
    )

    matrix = data.pivot(
        index="series_id",
        columns="date",
        values="quantity",
    )

    return matrix.sort_index().reindex(columns=calendar).astype(np.float32)


def select_eligible_series(
    matrix: pd.DataFrame,
    first_origin,
) -> pd.Index:
    """Choose series with 28 observed days at the first forecast origin."""
    origin = pd.Timestamp(first_origin)

    history_dates = pd.date_range(
        end=origin,
        periods=MIN_HISTORY,
        freq="D",
    )

    if not history_dates.isin(matrix.columns).all():
        raise ValueError("Not enough calendar history before the first origin")

    complete = matrix.loc[:, history_dates].notna().all(axis=1)
    selected = matrix.index[complete]

    if len(selected) == 0:
        raise ValueError("No series has the required 28-day history")

    return selected


def run_backtesting(
    sales: pd.DataFrame,
    evaluation_end_date,
) -> pd.DataFrame:
    """Generate forecasts using only sales available at each origin."""
    matrix = build_sales_matrix(sales)
    windows = make_forecast_windows(evaluation_end_date)

    selected = select_eligible_series(
        matrix,
        windows[0].origin,
    )

    # Freeze the cohort at the first origin.
    # Never silently remove series because of missing future observations.
    required_dates = pd.date_range(
        windows[0].origin - pd.Timedelta(days=MIN_HISTORY - 1),
        windows[-1].end,
        freq="D",
    )

    if not required_dates.isin(matrix.columns).all():
        raise ValueError("Missing calendar dates in the backtesting period")

    cohort = matrix.loc[selected, required_dates]

    if cohort.isna().any().any():
        raise ValueError(
            "Missing observations for eligible series in backtesting period"
        )

    names = selected.to_numpy()
    n_series = len(names)
    frames = []

    for window in windows:
        # Models receive only the 28 historical days known at this origin.
        history_dates = pd.date_range(
            end=window.origin,
            periods=MIN_HISTORY,
        )
        target_dates = pd.date_range(
            window.start,
            window.end,
            freq="D",
        )

        history = cohort.loc[:, history_dates].to_numpy(dtype=np.float64)
        actual = cohort.loc[:, target_dates].to_numpy(dtype=np.float64)

        for model_name, forecast in MODELS.items():
            predicted = np.vstack([forecast(row) for row in history])

            if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid forecast output from {model_name}")

            if (predicted < 0).any():
                raise ValueError(f"Negative predictions from {model_name}")

            frames.append(
                pd.DataFrame(
                    {
                        "series_id": np.repeat(names, FORECAST_HORIZON),
                        "origin": window.origin,
                        "date": np.tile(target_dates.to_numpy(), n_series),
                        "model": model_name,
                        "actual": actual.ravel(),
                        "predicted": predicted.ravel(),
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)
