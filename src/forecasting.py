"""Prepare historical daily sales for one-step-ahead demand forecasting."""

import pandas as pd

FEATURE_COLUMNS = (
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
)


def prepare_daily_sales(daily_sales: pd.Series) -> pd.DataFrame:
    """Convert an M5 series indexed by d_1, d_2, ... into ordered rows."""
    if daily_sales.empty:
        raise ValueError("Daily sales cannot be empty")

    frame = daily_sales.rename("sales").rename_axis("day").reset_index()

    if not frame["day"].astype(str).str.fullmatch(r"d_[1-9]\d*").all():
        raise ValueError("Day labels must use the format d_1, d_2, ...")
    if frame["day"].duplicated().any():
        raise ValueError("Day labels must be unique")
    if not pd.api.types.is_numeric_dtype(frame["sales"]):
        raise ValueError("Sales must be numeric")
    if frame["sales"].isna().any() or (frame["sales"] < 0).any():
        raise ValueError("Sales must be non-negative and contain no missing values")

    frame["day_number"] = frame["day"].str[2:].astype(int)
    frame = frame.sort_values("day_number").reset_index(drop=True)

    if not frame["day_number"].diff().dropna().eq(1).all():
        raise ValueError("Daily sales must contain consecutive days")

    return frame[["day", "day_number", "sales"]]


def add_forecasting_features(daily_frame: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features using only observations from earlier days."""
    required = {"day", "day_number", "sales"}
    if not required.issubset(daily_frame.columns):
        raise ValueError("Input must contain day, day_number, and sales")
    if daily_frame.empty:
        raise ValueError("Input data cannot be empty")

    frame = daily_frame.sort_values("day_number").reset_index(drop=True).copy()

    for lag in (1, 7, 28):
        frame[f"lag_{lag}"] = frame["sales"].shift(lag)

    previous_sales = frame["sales"].shift(1)
    for window in (7, 28):
        frame[f"rolling_mean_{window}"] = previous_sales.rolling(
            window=window, min_periods=window
        ).mean()

    return frame


def split_train_validation(
    featured_frame: pd.DataFrame, validation_days: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last validation_days labeled rows for validation."""
    if isinstance(validation_days, bool) or not isinstance(validation_days, int):
        raise TypeError("validation_days must be a positive integer")
    if validation_days <= 0:
        raise ValueError("validation_days must be a positive integer")

    required = {"day", "day_number", "sales", *FEATURE_COLUMNS}
    if not required.issubset(featured_frame.columns):
        raise ValueError("Input must contain day, day_number, sales, and features")

    ready = featured_frame.dropna(subset=FEATURE_COLUMNS).copy()
    ready = ready.sort_values("day_number").reset_index(drop=True)
    if len(ready) <= validation_days:
        raise ValueError("Not enough feature-ready days for this validation split")

    train = ready.iloc[:-validation_days].copy()
    validation = ready.iloc[-validation_days:].copy()
    return train, validation
