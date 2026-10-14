"""Transform sampled M5 series into unified daily sales data."""

import re

import pandas as pd

UNIFIED_COLUMNS = [
    "series_id",
    "date",
    "quantity",
    "price",
    "price_available",
]


def build_unified_batch(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a small batch of M5 series to daily observations."""
    required_sales = {"id", "item_id", "store_id"}

    if not required_sales.issubset(sales.columns):
        raise ValueError("Missing required sales columns")

    if sales["id"].isna().any() or sales["id"].duplicated().any():
        raise ValueError("Invalid or duplicate series IDs")

    # Identify and sort M5 daily sales columns.
    day_columns = sorted(
        (column for column in sales if re.fullmatch(r"d_\d+", column)),
        key=lambda column: int(column.split("_")[1]),
    )

    if not day_columns:
        raise ValueError("No M5 daily sales columns found")

    # Ensure no daily sales columns are missing between the first and last day.
    day_numbers = [int(column[2:]) for column in day_columns]

    if day_numbers != list(range(day_numbers[0], day_numbers[-1] + 1)):
        raise ValueError("Sales day columns must be consecutive")

    # Validate required calendar columns.
    required_calendar = {"d", "date", "wm_yr_wk"}

    if not required_calendar.issubset(calendar.columns):
        raise ValueError("Missing required calendar columns")

    # Validate required weekly price columns.
    required_prices = {
        "store_id",
        "item_id",
        "wm_yr_wk",
        "sell_price",
    }

    if not required_prices.issubset(prices.columns):
        raise ValueError("Missing required price columns")

    # Prevent duplicate calendar and price join keys.
    if calendar["d"].duplicated().any():
        raise ValueError("Duplicate calendar day identifiers")

    price_keys = ["store_id", "item_id", "wm_yr_wk"]

    if prices.duplicated(price_keys).any():
        raise ValueError("Duplicate weekly price keys")

    # Keep only dates corresponding to available sales columns.
    calendar_subset = calendar.loc[
        calendar["d"].isin(day_columns),
        ["d", "date", "wm_yr_wk"],
    ].copy()

    if len(calendar_subset) != len(day_columns):
        raise ValueError("Missing calendar mapping for sales days")

    # Convert calendar strings into proper datetime values.
    calendar_subset["date"] = pd.to_datetime(
        calendar_subset["date"],
        errors="raise",
    )

    if calendar_subset[["date", "wm_yr_wk"]].isna().any().any():
        raise ValueError("Missing calendar dates or weeks")

    # Verify that dates advance by exactly one day.
    ordered_calendar = calendar_subset.assign(
        _day_number=calendar_subset["d"].str[2:].astype(int)
    ).sort_values("_day_number")

    date_differences = ordered_calendar["date"].diff().iloc[1:]

    if not date_differences.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Calendar dates must be consecutive")

    # Convert wide sales data into daily observations.
    daily = sales.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=day_columns,
        var_name="d",
        value_name="quantity",
    )

    # Reject missing, negative, or non-integer sales quantities.
    if (
        daily["quantity"].isna().any()
        or not pd.api.types.is_integer_dtype(daily["quantity"])
        or (daily["quantity"] < 0).any()
    ):
        raise ValueError("Invalid daily sales quantities")

    # Join daily sales to calendar dates and weeks.
    daily = daily.merge(
        calendar_subset,
        on="d",
        how="left",
        validate="many_to_one",
    )

    # Join weekly prices using store, item, and week.
    daily = daily.merge(
        prices[price_keys + ["sell_price"]],
        on=price_keys,
        how="left",
        validate="many_to_one",
    )

    # Preserve missing prices explicitly.
    daily["price_available"] = daily["sell_price"].notna()

    # Sort each series chronologically.
    daily = daily.sort_values(["id", "date"])

    # Keep every day from the first observed price onward.
    # Later missing prices and zero-sales days remain in the dataset.
    observed_price = daily["price_available"].groupby(daily["id"]).cummax()

    daily = daily.loc[observed_price].copy()

    # Every input series must have an observed price in the sales period.
    if daily["id"].nunique() != len(sales):
        raise ValueError("Some series have no observed price in the sales period")

    # Convert source-specific column names into our unified schema.
    daily = daily.rename(
        columns={
            "id": "series_id",
            "sell_price": "price",
        }
    )

    return daily[UNIFIED_COLUMNS].reset_index(drop=True)
