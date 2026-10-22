"""Order-up-to targets for the Phase 3 inventory simulation.

Every method uses the same rule: forecast demand over the protection interval
plus a safety stock of z * sqrt(forecast). See docs/phase3_inventory_protocol.md.
"""

import numpy as np

from src.inventory_simulation import MAX_UNITS, review_times, validate_day_count


def _validate_demand_matrix(demand) -> np.ndarray:
    try:
        values = np.asarray(demand, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Demand must be numeric") from exc

    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Demand must be a nonempty series-by-day matrix")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand must be finite and nonnegative")

    if (values != np.floor(values)).any() or (values > MAX_UNITS).any():
        raise ValueError(
            f"Demand must be whole units no larger than {MAX_UNITS}"
        )

    return values


def initial_stock_covering_lead_time(demand, lead_time_days: int) -> np.ndarray:
    """Starting on-hand equal to each series' demand over the first L days.

    The first order cannot arrive before day L + 1, so this stock serves
    exactly the days before it. It is the same for every method. An empty
    start would instead make the first order include demand that is already
    lost before the delivery arrives; for rare-demand series that surplus can
    stay on the shelf for the whole evaluation window.

    This is an idealized condition: it uses actual future demand, and because
    it is part of the inventory position it changes the first orders. It is
    never a forecasting input.
    """
    values = _validate_demand_matrix(demand)
    lead_time = validate_day_count(lead_time_days, "Lead time", minimum=0)

    return values[:, :lead_time].sum(axis=1).astype(np.int64)


def protection_interval_days(lead_time_days: int, review_period_days: int) -> int:
    """Days one order must cover: until the next order arrives (L + R)."""
    lead_time = validate_day_count(lead_time_days, "Lead time", minimum=0)
    period = validate_day_count(review_period_days, "Review period", minimum=1)

    return lead_time + period


def protection_forecast(daily_forecasts, protection_days: int) -> np.ndarray:
    """Sum horizons 1..P of each series-by-review forecast path.

    daily_forecasts has shape (series, reviews, horizons), with horizon 1 in
    position 0. Only the first P horizons are used.
    """
    try:
        values = np.asarray(daily_forecasts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Forecasts must be numeric") from exc

    if values.ndim != 3 or 0 in values.shape:
        raise ValueError("Forecasts must be a nonempty (series, reviews, horizons) array")

    protection_days = validate_day_count(
        protection_days, "Protection interval", minimum=1
    )

    if values.shape[2] < protection_days:
        raise ValueError("Forecast horizon is shorter than the protection interval")

    window = values[:, :, :protection_days]

    if not np.isfinite(window).all():
        raise ValueError("Forecasts must be finite")

    if (window < 0).any():
        raise ValueError("Forecasts cannot be negative")

    return window.sum(axis=2)


def poisson_scaled_order_up_to(forecast_sum, z: float) -> np.ndarray:
    """Order-up-to level S = F + z * sqrt(F).

    For count demand, the standard deviation of a Poisson total equals the
    square root of its mean. z is a lever for tracing the service/inventory
    trade-off, not a guaranteed service level.
    """
    try:
        values = np.asarray(forecast_sum, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Protection forecasts must be numeric") from exc

    if not np.isfinite(values).all():
        raise ValueError("Protection forecasts must be finite")

    if (values < 0).any():
        raise ValueError("Protection forecasts cannot be negative")

    try:
        factor = float(z)
    except (TypeError, ValueError) as exc:
        raise ValueError("Safety factor z must be numeric") from exc

    if isinstance(z, bool) or not np.isfinite(factor) or factor < 0:
        raise ValueError("Safety factor z must be a finite nonnegative number")

    return values + factor * np.sqrt(values)


def perfect_foresight_protection_demand(
    demand,
    *,
    lead_time_days: int,
    review_period_days: int = 7,
) -> np.ndarray:
    """Actual demand over each review's protection interval.

    Perfect-foresight policy reference: it uses future demand by construction
    and is not a feasible forecasting method. When the simulation starts with
    initial_stock_covering_lead_time(), this policy ends every day holding
    exactly the demand that must be served before the next scheduled delivery.
    Any policy with the same review days and lead time that loses no sales
    must hold at least that much, so under those conditions it is the
    minimum stock for a 100% fill rate. With an empty start it is not.

    Demand after the simulated span is unknown and counts as zero. Sales in
    the span never change. With weekly reviews, a 112-day span and L = 7, no
    delivery inside the span is affected. With shorter lead times, the
    reference holds less stock in the last few days, because it does not
    stock for demand after the span.
    """
    values = _validate_demand_matrix(demand)
    protection = protection_interval_days(lead_time_days, review_period_days)
    times = review_times(values.shape[1], review_period_days)

    # cumulative[:, d] = demand on days 1..d (cumulative[:, 0] = 0).
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)],
        axis=1,
    )
    last_day = values.shape[1]
    ends = np.minimum(times + protection, last_day)

    return cumulative[:, ends] - cumulative[:, times]
