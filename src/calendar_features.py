"""Seven preregistered, future-known M5 target-date calendar predictors.

M5 scheduled events and SNAP flags are explicitly assumed known at the
forecast origin. This module reads no sales or prices and never uses an
event's name as a predictor. Calendar features are opt-in for Cycle 2.
"""

import numpy as np
import pandas as pd

CALENDAR_FEATURE_NAMES = (
    "target_month",
    "target_day_of_month",
    "target_is_event",
    "target_event_type_1",
    "target_event_type_2",
    "target_is_christmas",
    "target_is_snap",
)
CALENDAR_CATEGORICAL_FEATURES = ("target_event_type_1", "target_event_type_2")
CALENDAR_NUMERIC_FEATURES = tuple(
    name for name in CALENDAR_FEATURE_NAMES if name not in CALENDAR_CATEGORICAL_FEATURES
)
STATES = ("CA", "TX", "WI")
CALENDAR_COLUMNS = (
    "date",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
)


def _dates(values: pd.Series, name: str) -> pd.Series:
    """Reject invalid dates, timestamps with a time, and timezone-aware dates."""
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        if (
            values.dt.tz is not None
            or values.isna().any()
            or values.ne(values.dt.normalize()).any()
        ):
            raise ValueError(f"{name} must contain timezone-naive calendar dates")
        try:
            return values.astype("datetime64[ns]")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain valid calendar dates") from exc
    result = []
    for value in values:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain valid calendar dates") from exc
        if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
            raise ValueError(f"{name} must contain timezone-naive calendar dates")
        result.append(parsed)
    try:
        return pd.Series(result, index=values.index, dtype="datetime64[ns]")
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain valid calendar dates") from exc


def _event_values(values: pd.Series, column: str) -> pd.Series:
    """Normalize empty M5 event slots while preserving real string levels."""
    # Calendar-only parquet extracts may encode these columns categorically;
    # normalize without requiring an empty level in the source vocabulary.
    values = values.astype(object)
    if not values.map(
        lambda value: (
            pd.api.types.is_scalar(value) and (pd.isna(value) or isinstance(value, str))
        )
    ).all():
        raise ValueError(f"Calendar {column} must contain strings or missing values")
    return values.fillna("").str.strip()


def validate_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Validate a calendar-only table without dropping dates or event slots."""
    if not isinstance(calendar, pd.DataFrame) or calendar.empty:
        raise ValueError("Calendar cannot be empty")
    if calendar.columns.duplicated().any():
        raise ValueError("Duplicate calendar columns")
    missing = set(CALENDAR_COLUMNS) - set(calendar.columns)
    if missing:
        raise ValueError(f"Missing required calendar columns: {sorted(missing)}")
    result = calendar.loc[:, list(CALENDAR_COLUMNS)].copy()
    result["date"] = _dates(result["date"], "Calendar date")
    if result["date"].duplicated().any():
        raise ValueError("Duplicate calendar dates")

    for slot in (1, 2):
        names = _event_values(result[f"event_name_{slot}"], f"event_name_{slot}")
        types = _event_values(result[f"event_type_{slot}"], f"event_type_{slot}")
        if names.ne("").ne(types.ne("")).any():
            raise ValueError(f"Inconsistent event name/type in calendar slot {slot}")
        result[f"event_name_{slot}"] = names
        # The explicit level is the string "None", never a missing value.
        result[f"event_type_{slot}"] = types.mask(types.eq(""), "None")

    for state in STATES:
        column = f"snap_{state}"
        try:
            values = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Calendar {column} must be binary 0/1") from exc
        if not values.isin([0, 1]).all():
            raise ValueError(f"Calendar {column} must be binary 0/1")
        result[column] = values.astype(np.int8)
    return result


def _states(examples: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.Series:
    """Resolve each series' state and require a consistent M5 store mapping."""
    if (
        not examples["series_id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Calendar series IDs must be nonempty strings")
    if metadata is None:
        records = examples.loc[:, ["series_id", "store_id"]].copy()
        if "state_id" in examples:
            records["state_id"] = examples["state_id"]
        records = records.drop_duplicates()
    else:
        if metadata.columns.duplicated().any():
            raise ValueError("Duplicate calendar metadata columns")
        required = {"series_id", "store_id"}
        if not required.issubset(metadata.columns):
            raise ValueError("Calendar metadata requires series_id and store_id")
        columns = ["series_id", "store_id"]
        if "state_id" in metadata:
            columns.append("state_id")
        records = metadata.loc[:, columns].copy()

    if (
        not records["series_id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Calendar series IDs must be nonempty strings")
    if records["series_id"].duplicated().any():
        raise ValueError("Calendar metadata requires unique nonmissing series IDs")
    stores = records["store_id"]
    if not stores.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Unsupported store-to-state mapping")
    derived = stores.str.extract(r"^(CA|TX|WI)_.+$", expand=False)
    if derived.isna().any():
        raise ValueError("Unsupported store-to-state mapping")

    if "state_id" in records:
        if not records["state_id"].isin(STATES).all():
            raise ValueError("Unsupported or missing state_id")
        if records["state_id"].ne(derived).any():
            raise ValueError("Inconsistent state_id and store_id mapping")
        states = records["state_id"]
    else:
        states = derived

    records = records.assign(_state=states).set_index("series_id")
    mapped = examples["series_id"].map(records["_state"])
    expected_stores = examples["series_id"].map(records["store_id"])
    if mapped.isna().any():
        raise ValueError("Missing state coverage for forecast series")
    if (
        examples["store_id"].isna().any()
        or expected_stores.ne(examples["store_id"]).any()
    ):
        raise ValueError("Inconsistent metadata and example store_id mapping")
    if "state_id" in examples and (
        examples["state_id"].isna().any() or examples["state_id"].ne(mapped).any()
    ):
        raise ValueError("Inconsistent example state_id mapping")
    return mapped


def build_calendar_features(
    examples: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return exactly seven predictors aligned to ``origin + horizon``.

    Every origin must be Sunday, as fixed by Cycle 2. The numerical horizon
    then already determines weekday. Only the explicit calendar's scheduled
    fields may describe dates after origin; future sales/prices are unused.
    Optional original metadata preserves state_id for validation, without
    introducing it as an eighth predictor.
    """
    required = {"series_id", "store_id", "origin", "date", "horizon"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Missing calendar example columns: {sorted(missing)}")
    if examples.empty or examples.columns.duplicated().any():
        raise ValueError("Calendar examples must be nonempty with unique columns")
    origins = _dates(examples["origin"], "Forecast origin")
    dates = _dates(examples["date"], "Target date")
    if not origins.dt.dayofweek.eq(6).all():
        raise ValueError("Calendar Cycle 2 requires Sunday forecast origins")
    try:
        horizons = pd.to_numeric(examples["horizon"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Calendar horizon must be an integer from 1 to 28") from exc
    if (
        not horizons.isin(range(1, 29)).all()
        or examples["horizon"]
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .any()
    ):
        raise ValueError("Calendar horizon must be an integer from 1 to 28")
    if dates.ne(origins + pd.to_timedelta(horizons, unit="D")).any():
        raise ValueError("Target date must equal origin + horizon")
    keys = examples.loc[:, ["series_id", "origin", "horizon"]].copy()
    keys["origin"] = origins
    if keys.duplicated().any():
        raise ValueError("Duplicate calendar example keys")

    validated = validate_calendar(calendar)
    states = _states(examples, metadata)
    # Each distinct target date has exactly one calendar row. Reindexing
    # preserves repeated series/date rows, their order, and their population.
    indexed = validated.set_index("date")
    if not dates.isin(indexed.index).all():
        raise ValueError("Missing calendar coverage for required target dates")
    aligned = indexed.loc[dates.to_numpy()].reset_index(drop=True)
    snap = aligned.loc[:, [f"snap_{state}" for state in STATES]].to_numpy()
    state_positions = states.map(
        {state: i for i, state in enumerate(STATES)}
    ).to_numpy()
    result = pd.DataFrame(
        {
            "target_month": dates.dt.month.to_numpy(dtype=np.int8),
            "target_day_of_month": dates.dt.day.to_numpy(dtype=np.int8),
            "target_is_event": (
                aligned["event_name_1"].ne("") | aligned["event_name_2"].ne("")
            ).to_numpy(dtype=np.int8),
            "target_event_type_1": aligned["event_type_1"].to_numpy(),
            "target_event_type_2": aligned["event_type_2"].to_numpy(),
            "target_is_christmas": (
                dates.dt.month.eq(12) & dates.dt.day.eq(25)
            ).to_numpy(dtype=np.int8),
            "target_is_snap": snap[np.arange(len(examples)), state_positions],
        },
        index=examples.index,
    )
    return result.loc[:, list(CALENDAR_FEATURE_NAMES)]


def attach_calendar_features(
    examples: pd.DataFrame,
    calendar: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append opt-in Cycle 2 columns while preserving the existing features."""
    if set(CALENDAR_FEATURE_NAMES) & set(examples.columns):
        raise ValueError("Calendar feature columns already exist")
    features = build_calendar_features(examples, calendar, metadata)
    result = examples.copy()
    for column in CALENDAR_FEATURE_NAMES:
        result[column] = features[column]
    return result
