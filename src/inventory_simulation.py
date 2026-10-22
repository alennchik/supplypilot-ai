"""Periodic-review, order-up-to inventory simulation with lost sales.

Daily event order for every series (see docs/phase3_inventory_protocol.md):

1. Orders due today arrive before demand.
2. Demand is served from stock; unmet demand is lost (no backorders).
3. End-of-day on-hand is recorded.
4. On a review day, inventory position = on-hand + outstanding orders, and the
   order raises it to at least the order-up-to level. An order placed at the
   end of day t arrives at the start of day t + lead_time + 1.

Day numbers are 1-based simulated days. Review time 0 is the end of the
origin day, before simulated day 1. Reviews happen at times 0, R, 2R, ...
"""

from dataclasses import dataclass

import numpy as np

ORDER_TOLERANCE = 1e-9

# Far above any M5 value, and small enough that pooled sums over thousands of
# series and days stay exact in int64 and float64.
MAX_UNITS = 10**9


@dataclass(frozen=True)
class SimulationResult:
    """Daily and per-review outcomes, one row per series."""

    on_hand: np.ndarray
    sales: np.ndarray
    lost: np.ndarray
    receipts: np.ndarray
    orders: np.ndarray
    inventory_position_before_order: np.ndarray
    review_times: np.ndarray
    arrival_days: np.ndarray


def validate_day_count(value, name: str, *, minimum: int) -> int:
    """Return value as int if it is a whole number of days >= minimum."""
    is_integer = isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not is_integer or value < minimum:
        raise ValueError(f"{name} must be a whole number of days, at least {minimum}")

    return int(value)


def review_times(n_days: int, review_period_days: int) -> np.ndarray:
    """Return review times 0, R, 2R, ... that fall before the last simulated day."""
    days = validate_day_count(n_days, "Simulated span", minimum=1)
    period = validate_day_count(review_period_days, "Review period", minimum=1)

    return np.arange(0, days, period, dtype=np.int64)


def _validate_demand(demand) -> np.ndarray:
    try:
        values = np.asarray(demand, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Demand must be numeric") from exc

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Demand must be a nonempty series-by-day matrix")

    if not np.isfinite(values).all():
        raise ValueError("Demand must be finite")

    if (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Demand must be nonnegative whole units")

    if (values > MAX_UNITS).any():
        raise ValueError(f"Demand exceeds the supported range of {MAX_UNITS} units")

    return values.astype(np.int64)


def _validate_targets(order_up_to, shape: tuple[int, int]) -> np.ndarray:
    try:
        values = np.asarray(order_up_to, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Order-up-to levels must be numeric") from exc

    if values.shape != shape:
        raise ValueError(
            f"Order-up-to levels must have shape {shape} (series, reviews)"
        )

    if not np.isfinite(values).all():
        raise ValueError("Order-up-to levels must be finite")

    if (values < 0).any():
        raise ValueError("Order-up-to levels cannot be negative")

    if (values > MAX_UNITS).any():
        raise ValueError(
            f"Order-up-to levels exceed the supported range of {MAX_UNITS} units"
        )

    return values


def _validate_initial(initial_on_hand, n_series: int) -> np.ndarray:
    if initial_on_hand is None:
        return np.zeros(n_series, dtype=np.int64)

    try:
        values = np.asarray(initial_on_hand, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Initial on-hand must be numeric") from exc

    if values.shape != (n_series,):
        raise ValueError("Initial on-hand must have one value per series")

    if not np.isfinite(values).all():
        raise ValueError("Initial on-hand must be finite")

    if (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Initial on-hand must be nonnegative whole units")

    if (values > MAX_UNITS).any():
        raise ValueError(
            f"Initial on-hand exceeds the supported range of {MAX_UNITS} units"
        )

    return values.astype(np.int64)


def order_quantity(order_up_to, inventory_position) -> np.ndarray:
    """Units needed to raise inventory position to at least the target.

    The tolerance stops floating-point noise from adding a unit:
    a target of 3.0000000001 with position 0 orders 3, not 4.
    """
    target = np.asarray(order_up_to, dtype=float)
    position = np.asarray(inventory_position, dtype=float)

    if not np.isfinite(target).all() or not np.isfinite(position).all():
        raise ValueError("Order-up-to levels and inventory positions must be finite")

    needed = np.maximum(np.ceil(target - position - ORDER_TOLERANCE), 0)

    if (needed > MAX_UNITS).any():
        raise ValueError(
            f"Order quantity exceeds the supported range of {MAX_UNITS} units"
        )

    return needed.astype(np.int64)


def simulate_periodic_review(
    demand,
    order_up_to,
    *,
    lead_time_days: int,
    review_period_days: int = 7,
    initial_on_hand=None,
) -> SimulationResult:
    """Simulate every series independently under the same ordering rule.

    demand: series-by-day matrix of whole units for simulated days 1..N.
    order_up_to: series-by-review matrix of targets, one column per review
        time returned by review_times(N, review_period_days).
    """
    units = _validate_demand(demand)
    n_series, n_days = units.shape

    lead_time = validate_day_count(lead_time_days, "Lead time", minimum=0)
    period = validate_day_count(review_period_days, "Review period", minimum=1)

    times = review_times(n_days, period)
    targets = _validate_targets(order_up_to, (n_series, len(times)))
    stock = _validate_initial(initial_on_hand, n_series).copy()

    arrival_days = times + lead_time + 1

    on_hand = np.zeros((n_series, n_days), dtype=np.int64)
    sales = np.zeros((n_series, n_days), dtype=np.int64)
    lost = np.zeros((n_series, n_days), dtype=np.int64)
    receipts = np.zeros((n_series, n_days), dtype=np.int64)
    orders = np.zeros((n_series, len(times)), dtype=np.int64)
    position_before = np.zeros((n_series, len(times)), dtype=np.int64)

    # Index 0 is unused so pipeline[:, day] holds units due on 1-based day.
    pipeline = np.zeros((n_series, n_days + lead_time + 2), dtype=np.int64)
    outstanding = np.zeros(n_series, dtype=np.int64)

    def place_order(review: int, time: int) -> None:
        position = stock + outstanding
        quantity = order_quantity(targets[:, review], position)

        position_before[:, review] = position
        orders[:, review] = quantity

        pipeline[:, time + lead_time + 1] += quantity
        outstanding[:] += quantity

    # Review time 0 is the end of the origin day, before day 1.
    place_order(0, 0)
    next_review = 1

    for day in range(1, n_days + 1):
        column = day - 1

        arriving = pipeline[:, day]
        stock += arriving
        outstanding -= arriving
        receipts[:, column] = arriving

        served = np.minimum(stock, units[:, column])
        sales[:, column] = served
        lost[:, column] = units[:, column] - served
        stock -= served
        on_hand[:, column] = stock

        if next_review < len(times) and times[next_review] == day:
            place_order(next_review, day)
            next_review += 1

    return SimulationResult(
        on_hand=on_hand,
        sales=sales,
        lost=lost,
        receipts=receipts,
        orders=orders,
        inventory_position_before_order=position_before,
        review_times=times,
        arrival_days=arrival_days,
    )
