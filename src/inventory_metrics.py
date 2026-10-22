"""Pooled service and inventory metrics for the Phase 3 simulation.

Warm-up days are simulated but excluded from every metric. Demand is recovered
as sales + lost, so it always matches the simulated outcome.
"""

import numpy as np

from src.inventory_simulation import SimulationResult, validate_day_count


def _evaluated_columns(result: SimulationResult, warmup_days: int) -> slice:
    n_days = result.on_hand.shape[1]
    warmup = validate_day_count(warmup_days, "Warm-up", minimum=0)

    if warmup >= n_days:
        raise ValueError("Warm-up leaves no evaluated days")

    return slice(warmup, n_days)


def per_series_totals(
    result: SimulationResult,
    *,
    warmup_days: int,
) -> dict[str, np.ndarray]:
    """Per-series sums over evaluated days, for pooling and paired bootstrap."""
    days = _evaluated_columns(result, warmup_days)

    sales = result.sales[:, days]
    lost = result.lost[:, days]
    demand = sales + lost
    receipts = result.receipts[:, days]

    return {
        "demand_units": demand.sum(axis=1),
        "sales_units": sales.sum(axis=1),
        "lost_units": lost.sum(axis=1),
        "on_hand_unit_days": result.on_hand[:, days].sum(axis=1),
        "stockout_days": (lost > 0).sum(axis=1),
        "demand_days": (demand > 0).sum(axis=1),
        "deliveries": (receipts > 0).sum(axis=1),
        "units_received": receipts.sum(axis=1),
    }


def _validate_unit_values(unit_values, n_series: int) -> np.ndarray:
    try:
        values = np.asarray(unit_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Unit values must be numeric") from exc

    if values.shape != (n_series,):
        raise ValueError("Unit values must have one value per series")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Unit values must be finite and nonnegative")

    return values


def summarize_inventory(
    result: SimulationResult,
    *,
    warmup_days: int,
    unit_values=None,
) -> dict:
    """Pool metrics over all series and evaluated days.

    unit_values is an optional per-series value proxy (for example a sell
    price). It weights stock and lost sales; it is not a cost.
    """
    days = _evaluated_columns(result, warmup_days)
    n_series = result.on_hand.shape[0]
    evaluated_days = days.stop - days.start
    totals = per_series_totals(result, warmup_days=warmup_days)

    demand = int(totals["demand_units"].sum())

    if demand == 0:
        raise ValueError("Fill rate is undefined when evaluated demand is zero")

    sales = int(totals["sales_units"].sum())
    demand_days = int(totals["demand_days"].sum())
    stockout_days = int(totals["stockout_days"].sum())

    summary = {
        "series_count": n_series,
        "evaluated_days": evaluated_days,
        "demand_units": demand,
        "sales_units": sales,
        "lost_units": int(totals["lost_units"].sum()),
        "fill_rate": sales / demand,
        "stockout_series_days": stockout_days,
        "demand_series_days": demand_days,
        "stockout_rate": stockout_days / demand_days,
        "average_on_hand_units": float(totals["on_hand_unit_days"].sum())
        / evaluated_days,
        "deliveries": int(totals["deliveries"].sum()),
        "units_received": int(totals["units_received"].sum()),
    }

    if unit_values is not None:
        values = _validate_unit_values(unit_values, n_series)
        summary["average_on_hand_value"] = (
            float((totals["on_hand_unit_days"] * values).sum()) / evaluated_days
        )
        summary["lost_sales_value"] = float((totals["lost_units"] * values).sum())

    return summary


def inventory_at_fill_rate(fill_rates, inventories, target: float) -> float:
    """Interpolate the inventory needed to reach a target fill rate.

    Points must be ordered by safety factor z. Uses the first pair of
    adjacent points whose fill rates bracket the target. Returns NaN when
    the target is not reached within the grid, or is already exceeded at the
    first point (no lower bracket).
    """
    try:
        fills = np.asarray(fill_rates, dtype=float)
        stock = np.asarray(inventories, dtype=float)
        goal = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fill rates, inventories, and target must be numeric") from exc

    if fills.ndim != 1 or fills.size == 0 or fills.shape != stock.shape:
        raise ValueError("Fill rates and inventories must be equal-length 1-D arrays")

    if not np.isfinite(fills).all() or not np.isfinite(stock).all():
        raise ValueError("Fill rates and inventories must be finite")

    if ((fills < 0) | (fills > 1)).any():
        raise ValueError("Fill rates must be between 0 and 1")

    if not np.isfinite(goal) or not 0 < goal <= 1:
        raise ValueError("Target fill rate must be in (0, 1]")

    for i, fill in enumerate(fills):
        if fill == goal:
            return float(stock[i])

        if fill > goal:
            if i == 0:
                return float("nan")

            weight = (goal - fills[i - 1]) / (fill - fills[i - 1])
            return float(stock[i - 1] + weight * (stock[i] - stock[i - 1]))

    return float("nan")
