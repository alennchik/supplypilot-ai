"""Pooled service and inventory metrics for the Phase 3 simulation.

Warm-up days are simulated but excluded from every metric. Demand is recovered
as sales + lost, so it always matches the simulated outcome. Fill rate is the
share of demanded units that were sold, not a share of customers.
"""

from dataclasses import dataclass

import numpy as np

from src.inventory_simulation import SimulationResult, validate_day_count

EXACT = "exact"
INTERPOLATED = "interpolated"
ABOVE_TARGET_AT_MIN_Z = "above_target_at_min_z"
NOT_REACHED = "not_reached"
ESTIMABLE = (EXACT, INTERPOLATED)


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


@dataclass(frozen=True)
class FillRateEndpoint:
    """Inventory at a target fill rate, read off a z-ordered curve.

    status is one of:
    - "exact": a simulated point hits the target exactly;
    - "interpolated": linear interpolation between the two simulated points
      that first bracket the target (lower_index, upper_index);
    - "above_target_at_min_z": the smallest z already exceeds the target;
    - "not_reached": no simulated point reaches the target.
    first_reaching_index is the first simulated point with fill >= target,
    a policy that was actually run. inventory is NaN unless estimable.
    """

    status: str
    inventory: float
    lower_index: int | None
    upper_index: int | None
    first_reaching_index: int | None


def interpolated_inventory_at_fill_rate(
    fill_rates, inventories, target: float
) -> FillRateEndpoint:
    """Estimate the inventory at a target fill rate from simulated points.

    Points must be ordered by safety factor z. The interpolated value is an
    estimate between two simulated policies, not a simulated policy itself.
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

    if (stock < 0).any():
        raise ValueError("Inventories cannot be negative")

    if not np.isfinite(goal) or not 0 < goal <= 1:
        raise ValueError("Target fill rate must be in (0, 1]")

    reaching = np.flatnonzero(fills >= goal)

    if reaching.size == 0:
        return FillRateEndpoint(NOT_REACHED, float("nan"), None, None, None)

    first = int(reaching[0])

    if fills[first] == goal:
        return FillRateEndpoint(EXACT, float(stock[first]), first, first, first)

    if first == 0:
        return FillRateEndpoint(ABOVE_TARGET_AT_MIN_Z, float("nan"), None, 0, 0)

    lower = first - 1
    weight = (goal - fills[lower]) / (fills[first] - fills[lower])
    estimate = stock[lower] + weight * (stock[first] - stock[lower])

    return FillRateEndpoint(INTERPOLATED, float(estimate), lower, first, first)


def classify_relative_difference(ci_low: float, ci_high: float, band: float = 0.02) -> str:
    """Classify an interval for delta = (inventory_A - inventory_B) / inventory_B.

    Lower inventory at the same fill rate is better. Equivalence needs the
    whole interval inside (-band, band); a small point estimate is not enough.
    """
    values = np.array([ci_low, ci_high, band], dtype=float)

    if not np.isfinite(values).all() or ci_low > ci_high or band <= 0:
        raise ValueError("Interval bounds must be finite and ordered; band positive")

    if -band < ci_low and ci_high < band:
        return "practically_equivalent"

    if ci_high < 0:
        return "a_needs_less_inventory"

    if ci_low > 0:
        return "b_needs_less_inventory"

    return "inconclusive"


def _curve_arrays(values, name: str, n_series: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != n_series:
        raise ValueError(f"{name} must have shape (z values, series)")

    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")

    return array


def _pooled_endpoint(sales, on_hand, demand_total, weights, days, target):
    fills = (sales @ weights) / demand_total
    stock = (on_hand @ weights) / days
    return interpolated_inventory_at_fill_rate(fills, stock, target)


def paired_cluster_bootstrap(
    *,
    demand_units,
    sales_a,
    on_hand_a,
    sales_b,
    on_hand_b,
    evaluated_days: int,
    clusters,
    target: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
    band: float = 0.02,
    max_invalid_share: float = 0.01,
) -> dict:
    """Paired percentile bootstrap of the relative inventory difference.

    Inputs are per-series totals over evaluated days: demand_units has shape
    (series,); sales_* and on_hand_* (unit-days) have shape (z values, series)
    in z order. Whole clusters (for example all stores of one item) are
    resampled with replacement, and the same resample is used for both
    methods. Inside each resample, pooled fill rates and inventories are
    recomputed for every z and the endpoint is re-interpolated.

    A resample is invalid when its demand is zero or either endpoint is not
    estimable. The interval uses valid resamples only and is flagged
    unreliable when invalid resamples exceed max_invalid_share.
    """
    demand = np.asarray(demand_units, dtype=float)

    if demand.ndim != 1 or demand.size == 0:
        raise ValueError("Demand must be a nonempty per-series vector")

    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Demand must be finite and nonnegative")

    n_series = demand.size
    curves = {
        name: _curve_arrays(values, name, n_series)
        for name, values in [
            ("sales_a", sales_a),
            ("on_hand_a", on_hand_a),
            ("sales_b", sales_b),
            ("on_hand_b", on_hand_b),
        ]
    }

    if len({array.shape for array in curves.values()}) != 1:
        raise ValueError("Both methods must use the same z grid")

    if ((curves["sales_a"] > demand) | (curves["sales_b"] > demand)).any():
        raise ValueError("Sales cannot exceed demand")

    days = validate_day_count(evaluated_days, "Evaluated days", minimum=1)

    is_count = isinstance(n_resamples, (int, np.integer)) and not isinstance(
        n_resamples, bool
    )

    if not is_count or n_resamples < 1:
        raise ValueError("Resample count must be a positive integer")

    resamples = int(n_resamples)

    labels = np.asarray(clusters)

    if labels.shape != (n_series,):
        raise ValueError("Clusters must have one label per series")

    if not 0 < confidence < 1 or not 0 <= max_invalid_share < 1:
        raise ValueError("Confidence must be in (0, 1); invalid share in [0, 1)")

    def delta_for(weights):
        demand_total = float(demand @ weights)

        if demand_total == 0:
            return None, None, None

        a = _pooled_endpoint(
            curves["sales_a"], curves["on_hand_a"], demand_total, weights, days, target
        )
        b = _pooled_endpoint(
            curves["sales_b"], curves["on_hand_b"], demand_total, weights, days, target
        )

        if a.status not in ESTIMABLE or b.status not in ESTIMABLE or b.inventory == 0:
            return None, a, b

        return (a.inventory - b.inventory) / b.inventory, a, b

    point, endpoint_a, endpoint_b = delta_for(np.ones(n_series))

    _, codes = np.unique(labels, return_inverse=True)
    n_clusters = int(codes.max()) + 1
    rng = np.random.default_rng(seed)

    deltas = []

    for _ in range(resamples):
        draws = rng.integers(0, n_clusters, size=n_clusters)
        weights = np.bincount(draws, minlength=n_clusters)[codes].astype(float)
        delta, _, _ = delta_for(weights)

        if delta is not None:
            deltas.append(delta)

    invalid = resamples - len(deltas)
    invalid_share = invalid / resamples
    reliable = invalid_share <= max_invalid_share

    if deltas:
        tail = (1 - confidence) / 2 * 100
        ci_low, ci_high = (float(v) for v in np.percentile(deltas, [tail, 100 - tail]))
    else:
        ci_low = ci_high = float("nan")

    if point is None:
        verdict = "not_estimable"
    elif not reliable or not deltas:
        verdict = "unreliable"
    else:
        verdict = classify_relative_difference(ci_low, ci_high, band)

    return {
        "target_fill_rate": float(target),
        "point_estimate": float("nan") if point is None else float(point),
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "interval": "percentile",
        "n_resamples": resamples,
        "n_clusters": n_clusters,
        "invalid_resamples": invalid,
        "invalid_share": invalid_share,
        "reliable": reliable,
        "verdict": verdict,
        "resample_deltas": np.asarray(deltas, dtype=float),
    }


def illustrative_costs(
    totals: dict[str, np.ndarray],
    unit_values,
    *,
    holding_rate_per_year: float = 0.25,
    penalty_fractions=(0.25, 0.5, 1.0),
) -> dict:
    """Illustrative costs in the currency of unit_values (sell-price proxy).

    holding = sum_i value_i * on_hand_unit_days_i * holding_rate / 365
    penalty(p) = sum_i value_i * lost_units_i * p
    These are not Walmart costs: M5 has sell prices, not unit costs.
    """
    on_hand = np.asarray(totals["on_hand_unit_days"], dtype=float)
    lost = np.asarray(totals["lost_units"], dtype=float)
    values = _validate_unit_values(unit_values, on_hand.shape[0])

    if not np.isfinite(holding_rate_per_year) or holding_rate_per_year < 0:
        raise ValueError("Holding rate must be finite and nonnegative")

    fractions = np.asarray(penalty_fractions, dtype=float)

    if fractions.ndim != 1 or not np.isfinite(fractions).all() or (fractions < 0).any():
        raise ValueError("Penalty fractions must be finite and nonnegative")

    holding = float((values * on_hand).sum()) * holding_rate_per_year / 365
    lost_value = float((values * lost).sum())

    return {
        "holding_cost": holding,
        "penalty_cost": {float(p): lost_value * float(p) for p in fractions},
        "total_cost": {float(p): holding + lost_value * float(p) for p in fractions},
    }


def value_proxy_from_prices(prices) -> np.ndarray:
    """Median available price per series; missing prices are NaN.

    A series with no available price stops the run: it is never dropped
    silently or given a zero value.
    """
    try:
        values = np.asarray(prices, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prices must be numeric or NaN") from exc

    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Prices must be a nonempty series-by-day matrix")

    if np.isinf(values).any() or (values[~np.isnan(values)] <= 0).any():
        raise ValueError("Available prices must be finite and positive")

    missing = np.isnan(values).all(axis=1)

    if missing.any():
        rows = np.flatnonzero(missing).tolist()
        raise ValueError(f"No available price for series at rows {rows}")

    return np.nanmedian(values, axis=1)
