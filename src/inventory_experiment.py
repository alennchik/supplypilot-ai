"""Phase 3D: the frozen inventory experiment on prepared arrays.

Runs every pre-registered scenario of docs/phase3_inventory_protocol.md
(v1.0) and returns report tables. Inputs are already-checked arrays in cohort
order; loading files and the Section 12 checks live in the runner script.
Every parameter below is fixed by the frozen protocol.
"""

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.inventory_metrics import (
    ESTIMABLE,
    illustrative_costs,
    interpolated_inventory_at_fill_rate,
    paired_cluster_bootstrap,
    per_series_totals,
    summarize_inventory,
)
from src.inventory_policy import (
    initial_stock_covering_lead_time,
    perfect_foresight_protection_demand,
    poisson_scaled_order_up_to,
    protection_forecast,
    protection_interval_days,
)
from src.inventory_simulation import review_times, simulate_periodic_review

PROTOCOL_VERSION = "1.0"
REVIEW_PERIOD_DAYS = 7
SIMULATED_DAYS = 112
LEAD_TIMES = (7, 3, 1)
PRIMARY_LEAD_TIME = 7
Z_GRID = tuple(round(0.1 * step, 1) for step in range(31))
WARMUP_DAYS = 28
TARGETS = (0.90, 0.95, 0.98)
PRIMARY_TARGET = 0.95
METHODS = ("lightgbm", "ma28")
REFERENCE = "perfect_foresight"
# (name, starting stock, rounding rule): primary first, then robustness runs.
VARIANTS = (
    ("primary", "lead_time_demand", "ceil"),
    ("empty_start", "empty", "ceil"),
    ("half_up", "lead_time_demand", "half_up"),
)
BOOTSTRAP = {
    "n_resamples": 2000,
    "seed": 42,
    "confidence": 0.95,
    "band": 0.02,
    "max_invalid_share": 0.01,
}
HOLDING_RATE_PER_YEAR = 0.25
PENALTY_FRACTIONS = (0.25, 0.5, 1.0)


def protocol_parameters() -> dict:
    """The frozen parameters, recorded in every report manifest."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "review_period_days": REVIEW_PERIOD_DAYS,
        "simulated_days": SIMULATED_DAYS,
        "lead_times": list(LEAD_TIMES),
        "primary_lead_time": PRIMARY_LEAD_TIME,
        "z_grid": list(Z_GRID),
        "warmup_days": WARMUP_DAYS,
        "targets": list(TARGETS),
        "primary_target": PRIMARY_TARGET,
        "methods": list(METHODS),
        "variants": [list(variant) for variant in VARIANTS],
        "bootstrap": dict(BOOTSTRAP),
        "holding_rate_per_year": HOLDING_RATE_PER_YEAR,
        "penalty_fractions": list(PENALTY_FRACTIONS),
    }


def _validate_inputs(demand, paths, clusters, groups, unit_values) -> np.ndarray:
    units = np.asarray(demand)

    if units.ndim != 2 or units.shape[1] != SIMULATED_DAYS or units.shape[0] == 0:
        raise ValueError(f"Demand must have shape (series, {SIMULATED_DAYS})")

    n_series = units.shape[0]
    n_reviews = len(review_times(SIMULATED_DAYS, REVIEW_PERIOD_DAYS))

    if set(paths) != set(METHODS):
        raise ValueError(f"Forecast paths needed for exactly {METHODS}")

    for method in METHODS:
        if np.asarray(paths[method]).shape != (n_series, n_reviews, 28):
            raise ValueError(f"{method} paths must have shape (series, {n_reviews}, 28)")

    if len(clusters) != n_series or len(unit_values) != n_series:
        raise ValueError("Clusters and unit values need one entry per series")

    for name, labels in groups.items():
        if len(labels) != n_series:
            raise ValueError(f"Group {name} needs one label per series")

    return units


def _starting_stock(demand, kind: str, lead_time: int) -> np.ndarray:
    if kind == "empty":
        return np.zeros(demand.shape[0], dtype=np.int64)

    return initial_stock_covering_lead_time(demand, lead_time)


def _endpoint_record(fills, inventories, target) -> dict:
    endpoint = interpolated_inventory_at_fill_rate(fills, inventories, target)
    record = asdict(endpoint)

    for key, index in [
        ("lower", endpoint.lower_index),
        ("upper", endpoint.upper_index),
        ("first_reaching", endpoint.first_reaching_index),
    ]:
        record[f"{key}_z"] = np.nan if index is None else Z_GRID[index]
        record[f"{key}_fill_rate"] = np.nan if index is None else float(fills[index])
        record[f"{key}_inventory"] = (
            np.nan if index is None else float(inventories[index])
        )

    return record


def run_experiment(
    demand,
    paths: dict,
    clusters,
    groups: dict,
    unit_values,
    *,
    n_resamples: int = BOOTSTRAP["n_resamples"],
) -> dict:
    """Run every frozen scenario and return report tables.

    paths maps "lightgbm" and "ma28" to daily forecasts with shape
    (series, 16 review origins, 28 horizons). groups maps a breakdown name
    (for example "demand_group", "cat_id") to one label per series.
    """
    units = _validate_inputs(demand, paths, clusters, groups, unit_values)
    values = np.asarray(unit_values, dtype=float)

    curve_rows = []
    reference_rows = []
    totals_store = {}

    for variant, start_kind, rounding in VARIANTS:
        for lead_time in LEAD_TIMES:
            protection = protection_interval_days(lead_time, REVIEW_PERIOD_DAYS)
            start = _starting_stock(units, start_kind, lead_time)

            for method in METHODS:
                forecast_sums = protection_forecast(paths[method], protection)
                store = {"sales": [], "on_hand": [], "lost": []}

                for z in Z_GRID:
                    result = simulate_periodic_review(
                        units,
                        poisson_scaled_order_up_to(forecast_sums, z),
                        lead_time_days=lead_time,
                        review_period_days=REVIEW_PERIOD_DAYS,
                        initial_on_hand=start,
                        rounding=rounding,
                    )
                    totals = per_series_totals(result, warmup_days=WARMUP_DAYS)
                    summary = summarize_inventory(
                        result, warmup_days=WARMUP_DAYS, unit_values=values
                    )
                    costs = illustrative_costs(
                        totals,
                        values,
                        holding_rate_per_year=HOLDING_RATE_PER_YEAR,
                        penalty_fractions=PENALTY_FRACTIONS,
                    )
                    curve_rows.append(
                        {
                            "variant": variant,
                            "lead_time": lead_time,
                            "method": method,
                            "z": z,
                            **summary,
                            "holding_cost": costs["holding_cost"],
                            **{
                                f"total_cost_penalty_{p}": cost
                                for p, cost in costs["total_cost"].items()
                            },
                        }
                    )
                    store["sales"].append(totals["sales_units"])
                    store["on_hand"].append(totals["on_hand_unit_days"])
                    store["lost"].append(totals["lost_units"])
                    store["demand"] = totals["demand_units"]

                totals_store[(variant, lead_time, method)] = {
                    key: np.vstack(value) if key != "demand" else value
                    for key, value in store.items()
                }

            reference = simulate_periodic_review(
                units,
                perfect_foresight_protection_demand(
                    units,
                    lead_time_days=lead_time,
                    review_period_days=REVIEW_PERIOD_DAYS,
                ),
                lead_time_days=lead_time,
                review_period_days=REVIEW_PERIOD_DAYS,
                initial_on_hand=start,
                rounding=rounding,
            )
            reference_rows.append(
                {
                    "variant": variant,
                    "lead_time": lead_time,
                    "method": REFERENCE,
                    **summarize_inventory(
                        reference, warmup_days=WARMUP_DAYS, unit_values=values
                    ),
                }
            )

    curves = pd.DataFrame(curve_rows)
    endpoints = _endpoints(curves)
    bootstrap = {
        variant: _primary_bootstrap(totals_store, variant, clusters, n_resamples)
        for variant, _, _ in VARIANTS
    }
    group_curves, group_endpoints = _group_breakdowns(totals_store, groups)

    primary_verdict = bootstrap["primary"]["verdict"]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "primary": bootstrap["primary"],
        "sensitivity": {
            "empty_start": bootstrap["empty_start"],
            "half_up": bootstrap["half_up"],
        },
        "starting_state_dependent": bootstrap["empty_start"]["verdict"]
        != primary_verdict,
        "rounding_check_agrees": bootstrap["half_up"]["verdict"] == primary_verdict,
    }

    return {
        "curves": curves,
        "reference": pd.DataFrame(reference_rows),
        "endpoints": endpoints,
        "group_curves": group_curves,
        "group_endpoints": group_endpoints,
        "summary": summary,
    }


def _endpoints(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (variant, lead_time), scenario in curves.groupby(
        ["variant", "lead_time"], sort=False
    ):
        by_method = {
            method: scenario.loc[scenario.method == method].sort_values("z")
            for method in METHODS
        }

        for target in TARGETS:
            records = {
                method: _endpoint_record(
                    frame.fill_rate.to_numpy(),
                    frame.average_on_hand_units.to_numpy(),
                    target,
                )
                for method, frame in by_method.items()
            }
            a, b = records["lightgbm"], records["ma28"]
            estimable = (
                a["status"] in ESTIMABLE and b["status"] in ESTIMABLE and b["inventory"] > 0
            )
            delta = (a["inventory"] - b["inventory"]) / b["inventory"] if estimable else np.nan

            for method, record in records.items():
                rows.append(
                    {
                        "variant": variant,
                        "lead_time": lead_time,
                        "target_fill_rate": target,
                        "method": method,
                        **record,
                        "relative_difference_lightgbm_vs_ma28": delta,
                    }
                )

    return pd.DataFrame(rows)


def _bootstrap_record(result: dict) -> dict:
    record = {
        key: value
        for key, value in result.items()
        if key not in ("endpoint_a", "endpoint_b", "resample_deltas")
    }

    for label, key in [("lightgbm", "endpoint_a"), ("ma28", "endpoint_b")]:
        endpoint = result[key]
        record[f"endpoint_{label}"] = None if endpoint is None else asdict(endpoint)

    return record


def _primary_bootstrap(totals_store, variant, clusters, n_resamples) -> dict:
    a = totals_store[(variant, PRIMARY_LEAD_TIME, "lightgbm")]
    b = totals_store[(variant, PRIMARY_LEAD_TIME, "ma28")]

    result = paired_cluster_bootstrap(
        demand_units=a["demand"],
        sales_a=a["sales"],
        on_hand_a=a["on_hand"],
        sales_b=b["sales"],
        on_hand_b=b["on_hand"],
        evaluated_days=SIMULATED_DAYS - WARMUP_DAYS,
        clusters=np.asarray(clusters),
        target=PRIMARY_TARGET,
        n_resamples=n_resamples,
        seed=BOOTSTRAP["seed"],
        confidence=BOOTSTRAP["confidence"],
        band=BOOTSTRAP["band"],
        max_invalid_share=BOOTSTRAP["max_invalid_share"],
    )
    record = _bootstrap_record(result)
    record.update({"variant": variant, "lead_time": PRIMARY_LEAD_TIME})
    return record


def _group_breakdowns(totals_store, groups) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows, endpoint_rows = [], []
    days = SIMULATED_DAYS - WARMUP_DAYS

    for group_type, labels in groups.items():
        labels = np.asarray(labels)

        for group in sorted(pd.unique(labels)):
            members = labels == group
            records = {}

            for method in METHODS:
                totals = totals_store[("primary", PRIMARY_LEAD_TIME, method)]
                demand_total = float(totals["demand"][members].sum())

                if demand_total == 0:
                    continue

                fills = totals["sales"][:, members].sum(axis=1) / demand_total
                stock = totals["on_hand"][:, members].sum(axis=1) / days

                for z, fill, inventory in zip(Z_GRID, fills, stock, strict=True):
                    curve_rows.append(
                        {
                            "group_type": group_type,
                            "group": group,
                            "series_count": int(members.sum()),
                            "method": method,
                            "z": z,
                            "fill_rate": float(fill),
                            "average_on_hand_units": float(inventory),
                        }
                    )

                records[method] = _endpoint_record(fills, stock, PRIMARY_TARGET)

            if len(records) != len(METHODS):
                continue

            a, b = records["lightgbm"], records["ma28"]
            estimable = (
                a["status"] in ESTIMABLE and b["status"] in ESTIMABLE and b["inventory"] > 0
            )
            delta = (a["inventory"] - b["inventory"]) / b["inventory"] if estimable else np.nan

            for method, record in records.items():
                endpoint_rows.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "series_count": int(members.sum()),
                        "target_fill_rate": PRIMARY_TARGET,
                        "method": method,
                        **record,
                        "relative_difference_lightgbm_vs_ma28": delta,
                    }
                )

    return pd.DataFrame(curve_rows), pd.DataFrame(endpoint_rows)
