"""Inventory what-if over the published Phase 3 trade-off curves.

The engine reads the sealed reports in reports/phase3_inventory/primary/ and
checks every file it uses against checksums.json first. Nothing is simulated
again:

- stock for a target fill rate uses the same interpolation and statuses as the
  preregistered endpoints;
- fill rate for a stock budget interpolates the same curves the other way;
- costs are recomputed for any holding rate and lost-sale penalty from the
  published average stock value and lost-sales value.

Only the protocol's own cost settings are preregistered: holding at 25% a year
and a penalty of 0.25, 0.5 or 1.0 times the sell price. Any other setting is
exploratory, and every result says which kind it is. Interpolated values are
estimates between two simulated policies, not policies that were simulated.
"""

import hashlib
import io
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory_metrics import ESTIMABLE, interpolated_inventory_at_fill_rate

REPORT_DIR = (
    Path(__file__).resolve().parents[1] / "reports" / "phase3_inventory" / "primary"
)
FILES = ("curves.csv", "reference.csv", "manifest.json")

METHODS = ("lightgbm", "ma28")
METHOD_LABELS = {"lightgbm": "LightGBM", "ma28": "MA28"}
VARIANT_LABELS = {
    "primary": "Primary start",
    "empty_start": "Empty start",
    "half_up": "Round-half-up ordering",
}
CURVE_COLUMNS = (
    "variant",
    "lead_time",
    "method",
    "z",
    "series_count",
    "evaluated_days",
    "demand_units",
    "fill_rate",
    "average_on_hand_units",
    "average_on_hand_value",
    "lost_sales_value",
)

# Accepted input ranges. Wider values are refused rather than extrapolated.
HOLDING_RATE_RANGE = (0.0, 1.0)
PENALTY_RANGE = (0.0, 5.0)
STOCK_BUDGET_MAX = 1e7
DEFAULT_PENALTY = 0.5


@dataclass(frozen=True)
class Reports:
    """The published Phase 3 files the what-if engine reads."""

    curves: pd.DataFrame
    reference: pd.DataFrame
    manifest: dict

    @property
    def protocol(self) -> dict:
        return self.manifest["protocol"]


def read_verified(report_dir, names) -> dict[str, bytes]:
    """The bytes of each named report file, after checking it against checksums.json.

    Callers parse the same bytes that were hashed, so a file replaced after the
    check cannot be read by mistake.
    """
    folder = Path(report_dir)
    checksums = json.loads((folder / "checksums.json").read_text())
    contents = {}

    for name in names:
        data = (folder / name).read_bytes()
        if checksums.get(name) != hashlib.sha256(data).hexdigest():
            raise ValueError(
                f"{name} does not match checksums.json; the published reports changed"
            )
        contents[name] = data

    return contents


def load_reports(report_dir=REPORT_DIR) -> Reports:
    """Read the published reports after checking them against checksums.json."""
    contents = read_verified(report_dir, FILES)

    curves = pd.read_csv(io.BytesIO(contents["curves.csv"]))
    reference = pd.read_csv(io.BytesIO(contents["reference.csv"]))
    manifest = json.loads(contents["manifest.json"])

    missing = [column for column in CURVE_COLUMNS if column not in curves.columns]
    if missing:
        raise ValueError(f"curves.csv lacks columns: {', '.join(missing)}")

    return Reports(curves=curves, reference=reference, manifest=manifest)


def _number(name: str, value, low: float, high: float, *, open_low=False) -> float:
    is_number = isinstance(value, (int, float, np.number)) and not isinstance(
        value, bool
    )
    number = float(value) if is_number else math.nan
    too_low = number <= low if open_low else number < low

    if not is_number or not math.isfinite(number) or too_low or number > high:
        bound = "above" if open_low else "at least"
        requirement = (
            f"{bound} {low:g} and at most {high:g}" if is_number else "a number"
        )
        raise ValueError(f"{name} must be {requirement}")

    return number


def _evaluation_window(manifest: dict) -> tuple[str, str]:
    """First and last evaluation dates, from the run's origins and protocol."""
    protocol = manifest["protocol"]
    first_origin = date.fromisoformat(manifest["checks"]["review_origins"][0])
    start = first_origin + timedelta(days=1 + protocol["warmup_days"])
    end = first_origin + timedelta(days=protocol["simulated_days"])
    return start.isoformat(), end.isoformat()


def scenarios(reports: Reports) -> dict:
    """What the published run covers, its protocol settings and provenance."""
    protocol = reports.protocol
    first = reports.curves.iloc[0]
    start, end = _evaluation_window(reports.manifest)

    return {
        "methods": [
            {"id": method, "label": METHOD_LABELS[method]} for method in METHODS
        ],
        "variants": [
            {"id": variant[0], "label": VARIANT_LABELS.get(variant[0], variant[0])}
            for variant in protocol["variants"]
        ],
        "lead_times": [int(value) for value in protocol["lead_times"]],
        "z_grid": [float(value) for value in protocol["z_grid"]],
        "defaults": {
            "variant": "primary",
            "lead_time": int(protocol["primary_lead_time"]),
            "target_fill_rate": float(protocol["primary_target"]),
            "holding_rate": float(protocol["holding_rate_per_year"]),
            "penalty_fraction": DEFAULT_PENALTY,
        },
        "preregistered": {
            "targets": [float(value) for value in protocol["targets"]],
            "holding_rate": float(protocol["holding_rate_per_year"]),
            "penalty_fractions": [
                float(value) for value in protocol["penalty_fractions"]
            ],
        },
        "limits": {
            "holding_rate": list(HOLDING_RATE_RANGE),
            "penalty_fraction": list(PENALTY_RANGE),
            "stock_budget_max": STOCK_BUDGET_MAX,
        },
        "provenance": {
            "protocol_version": protocol["protocol_version"],
            "issue": reports.manifest["issue"],
            "source_commit": reports.manifest["execution"]["source_commit"],
            "series": int(first["series_count"]),
            "evaluated_days": int(first["evaluated_days"]),
            "demand_units": int(first["demand_units"]),
            "evaluation_start": start,
            "evaluation_end": end,
        },
    }


def curve(reports: Reports, variant: str, lead_time: int, method: str) -> pd.DataFrame:
    """One method's published trade-off curve, ordered by z."""
    curves = reports.curves
    rows = curves[
        (curves["variant"] == variant)
        & (curves["lead_time"] == lead_time)
        & (curves["method"] == method)
    ].sort_values("z")

    if rows.empty:
        raise ValueError(
            f"No published curve for {method}, variant {variant!r}, lead time {lead_time}"
        )

    return rows.reset_index(drop=True)


def _optional_z(z: np.ndarray, index: int | None) -> float | None:
    return None if index is None else float(z[index])


def stock_for_fill_rate(frame: pd.DataFrame, target: float) -> dict:
    """Average stock needed for a target fill rate, read off one curve."""
    endpoint = interpolated_inventory_at_fill_rate(
        frame["fill_rate"], frame["average_on_hand_units"], target
    )
    z = frame["z"].to_numpy(dtype=float)

    return {
        "status": endpoint.status,
        "units": float(endpoint.inventory) if endpoint.status in ESTIMABLE else None,
        "lower_z": _optional_z(z, endpoint.lower_index),
        "upper_z": _optional_z(z, endpoint.upper_index),
        "first_reaching_z": _optional_z(z, endpoint.first_reaching_index),
        "max_fill_rate": float(frame["fill_rate"].max()),
    }


def fill_rate_for_stock(frame: pd.DataFrame, units: float) -> dict:
    """Fill rate reached with a given average stock, read off one curve.

    Only stock levels between the smallest and largest simulated policy are
    answered; outside that range the status says which side it fell on.
    """
    stock = frame["average_on_hand_units"].to_numpy(dtype=float)
    fills = frame["fill_rate"].to_numpy(dtype=float)
    z = frame["z"].to_numpy(dtype=float)

    if (np.diff(stock) <= 0).any() or (np.diff(fills) < 0).any():
        raise ValueError("A curve must gain stock and fill rate as z rises")

    answer = {
        "status": "",
        "fill_rate": None,
        "lower_z": None,
        "upper_z": None,
        "min_units": float(stock[0]),
        "max_units": float(stock[-1]),
    }

    if units < stock[0]:
        return {**answer, "status": "below_grid", "upper_z": float(z[0])}

    if units > stock[-1]:
        return {**answer, "status": "above_grid", "lower_z": float(z[-1])}

    upper = int(np.searchsorted(stock, units, side="left"))

    if stock[upper] == units:
        return {
            **answer,
            "status": "exact",
            "fill_rate": float(fills[upper]),
            "lower_z": float(z[upper]),
            "upper_z": float(z[upper]),
        }

    lower = upper - 1
    weight = (units - stock[lower]) / (stock[upper] - stock[lower])

    return {
        **answer,
        "status": "interpolated",
        "fill_rate": float(fills[lower] + weight * (fills[upper] - fills[lower])),
        "lower_z": float(z[lower]),
        "upper_z": float(z[upper]),
    }


def cost_table(
    frame: pd.DataFrame, *, holding_rate: float, penalty_fraction: float
) -> pd.DataFrame:
    """Holding, lost-sale and total cost of every simulated policy on one curve.

    holding = average stock value x evaluated days x yearly rate / 365
    penalty = lost-sales value x penalty fraction
    At the protocol's settings this reproduces the published cost columns.
    """
    days = frame["evaluated_days"].to_numpy(dtype=float)
    holding = (
        frame["average_on_hand_value"].to_numpy(dtype=float) * days * holding_rate / 365
    )
    penalty = frame["lost_sales_value"].to_numpy(dtype=float) * penalty_fraction

    return pd.DataFrame(
        {
            "z": frame["z"].to_numpy(dtype=float),
            "holding_cost": holding,
            "penalty_cost": penalty,
            "total_cost": holding + penalty,
        }
    )


def cheapest(frame: pd.DataFrame, costs: pd.DataFrame) -> dict:
    """The simulated policy with the lowest total cost.

    grid_edge says when that policy is the first or last z on the grid: the
    true minimum may then lie outside the simulated range.
    """
    totals = costs["total_cost"].to_numpy(dtype=float)
    best = int(totals.argmin())

    if best == 0:
        edge = "lowest_z"
    elif best == len(totals) - 1:
        edge = "highest_z"
    else:
        edge = None

    return {
        "z": float(costs["z"].iloc[best]),
        "total_cost": float(totals[best]),
        "holding_cost": float(costs["holding_cost"].iloc[best]),
        "penalty_cost": float(costs["penalty_cost"].iloc[best]),
        "fill_rate": float(frame["fill_rate"].iloc[best]),
        "average_on_hand_units": float(frame["average_on_hand_units"].iloc[best]),
        "grid_edge": edge,
    }


def _policies(frame: pd.DataFrame, costs: pd.DataFrame) -> list[dict]:
    return [
        {
            "z": float(row.z),
            "fill_rate": float(row.fill_rate),
            "average_on_hand_units": float(row.average_on_hand_units),
            "holding_cost": float(cost.holding_cost),
            "penalty_cost": float(cost.penalty_cost),
            "total_cost": float(cost.total_cost),
        }
        for row, cost in zip(
            frame.itertuples(index=False), costs.itertuples(index=False), strict=True
        )
    ]


def _relative(value: float, base: float) -> float | None:
    return None if base == 0 else (value - base) / base


def _settings(
    protocol: dict, *, variant, lead_time, target, holding_rate, penalty
) -> dict:
    costs_preregistered = math.isclose(
        holding_rate, protocol["holding_rate_per_year"], abs_tol=1e-9
    ) and any(
        math.isclose(penalty, value, abs_tol=1e-9)
        for value in protocol["penalty_fractions"]
    )
    target_preregistered = any(
        math.isclose(target, value, abs_tol=1e-9) for value in protocol["targets"]
    )
    primary = variant == "primary" and lead_time == protocol["primary_lead_time"]

    notes = []
    if costs_preregistered:
        notes.append(
            "Cost settings are one of the protocol's preregistered cost grids."
        )
    else:
        notes.append(
            "Cost settings are exploratory: the protocol fixed holding at "
            f"{protocol['holding_rate_per_year']:.0%} a year and penalties at "
            f"{', '.join(f'{value:g}' for value in protocol['penalty_fractions'])} "
            "times the sell price."
        )
    if not target_preregistered:
        notes.append(
            "This target fill rate is an exploratory reading of the published curves; "
            "the protocol reported 90%, 95% and 98%."
        )
    if not primary:
        notes.append(
            "The primary comparison is the primary start at a "
            f"{protocol['primary_lead_time']}-day lead time; this scenario is a "
            "secondary or robustness analysis."
        )

    return {
        "costs_preregistered": costs_preregistered,
        "target_preregistered": target_preregistered,
        "primary_scenario": primary,
        "notes": notes,
    }


def what_if(
    reports: Reports,
    *,
    variant: str = "primary",
    lead_time: int = 7,
    target_fill_rate: float = 0.95,
    holding_rate: float = 0.25,
    penalty_fraction: float = DEFAULT_PENALTY,
    stock_budget: float | None = None,
) -> dict:
    """Answer one what-if question for both methods from the published curves.

    Raises ValueError for an unknown scenario or a value outside the accepted
    ranges. The result contains only JSON-safe values.
    """
    protocol = reports.protocol
    variants = [value[0] for value in protocol["variants"]]

    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}; choose one of {variants}")

    if isinstance(lead_time, bool) or lead_time not in protocol["lead_times"]:
        raise ValueError(
            f"Lead time must be one of {list(protocol['lead_times'])} days"
        )

    lead_time = int(lead_time)
    target = _number("Target fill rate", target_fill_rate, 0.0, 1.0, open_low=True)
    holding = _number("Holding rate", holding_rate, *HOLDING_RATE_RANGE)
    penalty = _number("Penalty fraction", penalty_fraction, *PENALTY_RANGE)
    budget = (
        None
        if stock_budget is None
        else _number("Stock budget", stock_budget, 0.0, STOCK_BUDGET_MAX, open_low=True)
    )

    methods = {}
    for method in METHODS:
        frame = curve(reports, variant, lead_time, method)
        costs = cost_table(frame, holding_rate=holding, penalty_fraction=penalty)
        methods[method] = {
            "label": METHOD_LABELS[method],
            "policies": _policies(frame, costs),
            "stock_for_target": stock_for_fill_rate(frame, target),
            "fill_for_budget": None
            if budget is None
            else fill_rate_for_stock(frame, budget),
            "cheapest": cheapest(frame, costs),
        }

    stock = {method: methods[method]["stock_for_target"] for method in METHODS}
    comparable = all(stock[method]["status"] in ESTIMABLE for method in METHODS)
    cheapest_cost = {
        method: methods[method]["cheapest"]["total_cost"] for method in METHODS
    }

    reference = reports.reference[
        (reports.reference["variant"] == variant)
        & (reports.reference["lead_time"] == lead_time)
    ]
    if len(reference) != 1:
        raise ValueError(
            f"No perfect-foresight reference for {variant!r}, {lead_time} days"
        )
    oracle = reference.iloc[0]

    return {
        "scenario": {
            "variant": variant,
            "lead_time": lead_time,
            "target_fill_rate": target,
            "holding_rate": holding,
            "penalty_fraction": penalty,
            "stock_budget": budget,
        },
        "methods": methods,
        "comparison": {
            "stock_difference_lightgbm_vs_ma28": (
                _relative(stock["lightgbm"]["units"], stock["ma28"]["units"])
                if comparable
                else None
            ),
            "stock_comparable": comparable,
            "cheapest_cost_difference_lightgbm_vs_ma28": _relative(
                cheapest_cost["lightgbm"], cheapest_cost["ma28"]
            ),
        },
        "reference": {
            "label": "Perfect foresight",
            "fill_rate": float(oracle["fill_rate"]),
            "average_on_hand_units": float(oracle["average_on_hand_units"]),
        },
        "settings": _settings(
            protocol,
            variant=variant,
            lead_time=lead_time,
            target=target,
            holding_rate=holding,
            penalty=penalty,
        ),
    }
