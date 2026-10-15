"""Fixed chronological validation windows and evaluation observations."""

import numpy as np
import pandas as pd

from src.backtesting import (
    ForecastWindow,
    build_sales_matrix,
    select_eligible_series,
)
from src.forecast_baselines import FORECAST_HORIZON

TRAINING_CUTOFF = pd.Timestamp("2015-11-08")

V1_ORIGIN = TRAINING_CUTOFF
V2_ORIGIN = pd.Timestamp("2015-12-06")


def validation_windows() -> tuple[ForecastWindow, ...]:
    """Return the two preregistered 28-day validation windows."""
    windows = []

    for origin in (V1_ORIGIN, V2_ORIGIN):
        windows.append(
            ForecastWindow(
                origin=origin,
                start=origin + pd.Timedelta(days=1),
                end=origin + pd.Timedelta(days=FORECAST_HORIZON),
            )
        )

    return tuple(windows)


def build_validation_actuals(sales: pd.DataFrame) -> pd.DataFrame:
    """Return actual sales for one fixed cohort across V1 and V2.

    The cohort is selected using only the 28 historical days available
    at the V1 origin. Both validation windows must have complete
    observations for that cohort.

    This function prepares evaluation labels, not model features.
    """
    matrix = build_sales_matrix(sales)
    windows = validation_windows()

    selected = select_eligible_series(matrix, windows[0].origin)

    required_dates = pd.date_range(
        start=windows[0].origin - pd.Timedelta(days=27),
        end=windows[-1].end,
        freq="D",
    )

    if not required_dates.isin(matrix.columns).all():
        raise ValueError("Missing required validation calendar dates")

    cohort = matrix.loc[selected, required_dates]

    if cohort.isna().any().any():
        raise ValueError("Missing observations for the fixed validation cohort")

    names = selected.to_numpy()
    n_series = len(names)
    frames = []

    for window in windows:
        target_dates = pd.date_range(
            start=window.start,
            end=window.end,
            freq="D",
        )

        actual = cohort.loc[:, target_dates].to_numpy(dtype=np.float64)

        frame = pd.DataFrame(
            {
                "series_id": np.repeat(
                    names,
                    FORECAST_HORIZON,
                ),
                "origin": window.origin,
                "date": np.tile(
                    target_dates.to_numpy(),
                    n_series,
                ),
                "horizon": np.tile(
                    np.arange(
                        1,
                        FORECAST_HORIZON + 1,
                        dtype=np.int8,
                    ),
                    n_series,
                ),
                "actual": actual.ravel(),
            }
        )

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    if result.duplicated(["series_id", "origin", "horizon"]).any():
        raise ValueError("Duplicate validation observation keys")

    return result
