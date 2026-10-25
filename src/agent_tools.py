"""Tools the planning agent may call: deterministic reads of the what-if engine.

Every tool answers from the published Phase 3 reports through
src/inventory_whatif.py; nothing is simulated or estimated by the model.
Results are compact and already rounded the way an answer should quote them:
whole units and dollars, one decimal for percentages and z. Comparisons the
user may ask for (stock and cost differences, fill-rate gaps) are computed
here, so the model never has to do arithmetic.
"""

import math

from src.inventory_whatif import Reports, scenarios, what_if

LEAD_TIME_ARGUMENT = {
    "type": "integer",
    "enum": [1, 3, 7],
    "description": "Supplier lead time in days.",
}
VARIANT_ARGUMENT = {
    "type": "string",
    "enum": ["primary", "empty_start", "half_up"],
    "description": (
        "Starting condition. Use 'primary' unless the user asks about the "
        "robustness checks: 'empty_start' (shelves start empty) or 'half_up' "
        "(orders rounded half up instead of up)."
    ),
}

TOOL_SPECS = [
    {
        "name": "list_scenarios",
        "description": (
            "What the published simulation covers: lead times, starting conditions, "
            "the tested safety-factor range, the protocol's targets and costs, and "
            "the data window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "stock_for_fill_rate",
        "description": (
            "Average units on hand that LightGBM and MA28 each need to reach a target "
            "fill rate, read off the published trade-off curves, with the difference "
            "between them and the perfect-foresight reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_time_days": LEAD_TIME_ARGUMENT,
                "target_fill_rate_percent": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 100,
                    "description": "Target share of demanded units served, in percent.",
                },
                "variant": VARIANT_ARGUMENT,
            },
            "required": ["lead_time_days", "target_fill_rate_percent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fill_rate_for_stock",
        "description": (
            "Fill rate that LightGBM and MA28 each reach with a given average stock "
            "(units on hand, all series combined), and the gap between them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_time_days": LEAD_TIME_ARGUMENT,
                "average_units_on_hand": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 10_000_000,
                    "description": "Average units on hand across all series.",
                },
                "variant": VARIANT_ARGUMENT,
            },
            "required": ["lead_time_days", "average_units_on_hand"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_costs",
        "description": (
            "Cheapest simulated policy of each forecaster for a holding cost and a "
            "lost-sale penalty, with its cost breakdown, and the cost difference. "
            "The protocol's own settings are 25% a year holding and a penalty of "
            "0.25, 0.5 or 1.0 times the sell price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lead_time_days": LEAD_TIME_ARGUMENT,
                "holding_rate_percent_per_year": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Yearly holding cost as a percent of the sell price.",
                },
                "lost_sale_penalty_times_price": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "Cost of a lost sale as a multiple of the sell price.",
                },
                "variant": VARIANT_ARGUMENT,
            },
            "required": ["lead_time_days"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = tuple(spec["name"] for spec in TOOL_SPECS)

STATUS_MEANING = {
    "exact": "reached exactly at a simulated policy",
    "interpolated": "reached; interpolated between two simulated policies",
    "above_target_at_min_z": "already above the target at z = 0, so no stock estimate",
    "not_reached": "not reached anywhere on the tested z range",
    "below_grid": "below the stock of the smallest tested policy (z = 0), so not simulated",
    "above_grid": "above the stock of the largest tested policy, so not simulated",
}


class ToolError(ValueError):
    """A call the tools cannot answer; the message goes back to the model."""


def _percent(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 1)


def _units(value: float | None) -> int | None:
    return None if value is None else round(value)


def _z(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _number(arguments: dict, name: str, *, default=None, required=False) -> float:
    value = arguments.get(name, default)
    if value is None:
        if required:
            raise ToolError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ToolError(f"{name} must be finite")
    return float(value)


def _scenario(reports: Reports, arguments: dict) -> tuple[str, int]:
    lead_time = arguments.get("lead_time_days")
    lead_times = list(reports.protocol["lead_times"])
    if isinstance(lead_time, bool) or lead_time not in lead_times:
        raise ToolError(f"lead_time_days must be one of {sorted(lead_times)}")
    variant = arguments.get("variant") or "primary"
    variants = [value[0] for value in reports.protocol["variants"]]
    if variant not in variants:
        raise ToolError(f"variant must be one of {variants}")
    return variant, int(lead_time)


def _notes(settings: dict, *, costs=False, target=False) -> list[str]:
    notes = []
    if target and not settings["target_preregistered"]:
        notes.append(
            "Exploratory target: the protocol reported 90%, 95% and 98% fill rates."
        )
    if costs and not settings["costs_preregistered"]:
        notes.append(
            "Exploratory costs: the protocol fixed holding at 25% a year and a "
            "penalty of 0.25, 0.5 or 1.0 times the sell price."
        )
    if not settings["primary_scenario"]:
        notes.append(
            "Secondary scenario: the primary comparison is the primary start at a "
            "7-day lead time."
        )
    return notes


def _between(result: dict) -> list[float] | None:
    if result["status"] in ("exact", "interpolated"):
        return [_z(result["lower_z"]), _z(result["upper_z"])]
    return None


def _ask(reports: Reports, **question) -> dict:
    try:
        return what_if(reports, **question)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def list_scenarios(reports: Reports, arguments: dict) -> dict:
    overview = scenarios(reports)
    provenance = overview["provenance"]
    preregistered = overview["preregistered"]
    z_grid = overview["z_grid"]
    return {
        "lead_times_days": sorted(overview["lead_times"]),
        "primary_lead_time_days": overview["defaults"]["lead_time"],
        "starting_conditions": {
            item["id"]: item["label"] for item in overview["variants"]
        },
        "safety_factor_z_range": [min(z_grid), max(z_grid)],
        "protocol_targets_percent": [
            _percent(value) for value in preregistered["targets"]
        ],
        "protocol_holding_rate_percent_per_year": _percent(
            preregistered["holding_rate"]
        ),
        "protocol_penalties_times_price": preregistered["penalty_fractions"],
        "series": provenance["series"],
        "evaluation_start": provenance["evaluation_start"],
        "evaluation_end": provenance["evaluation_end"],
        "evaluation_days": provenance["evaluated_days"],
        "demanded_units": provenance["demand_units"],
    }


def stock_for_fill_rate(reports: Reports, arguments: dict) -> dict:
    variant, lead_time = _scenario(reports, arguments)
    target = _number(arguments, "target_fill_rate_percent", required=True)
    if not 0 < target <= 100:
        raise ToolError("target_fill_rate_percent must be above 0 and at most 100")

    answer = _ask(
        reports, variant=variant, lead_time=lead_time, target_fill_rate=target / 100
    )
    methods = {}
    for method, label in (("lightgbm", "LightGBM"), ("ma28", "MA28")):
        result = answer["methods"][method]["stock_for_target"]
        methods[method] = {
            "label": label,
            "status": result["status"],
            "status_meaning": STATUS_MEANING[result["status"]],
            "average_units_on_hand": _units(result["units"]),
            "between_z": _between(result),
            "best_fill_rate_percent_on_tested_range": _percent(result["max_fill_rate"]),
        }
    difference = answer["comparison"]["stock_difference_lightgbm_vs_ma28"]

    return {
        "lead_time_days": lead_time,
        "variant": variant,
        "target_fill_rate_percent": round(target, 1),
        "lightgbm": methods["lightgbm"],
        "ma28": methods["ma28"],
        "lightgbm_stock_difference_vs_ma28_percent": _percent(difference),
        "stock_difference_note": (
            "Positive means LightGBM needs more stock than MA28."
            if difference is not None
            else "Not comparable: at least one method does not reach the target."
        ),
        "perfect_foresight_units_for_100_percent": _units(
            answer["reference"]["average_on_hand_units"]
        ),
        "notes": _notes(answer["settings"], target=True),
    }


def fill_rate_for_stock(reports: Reports, arguments: dict) -> dict:
    variant, lead_time = _scenario(reports, arguments)
    units = _number(arguments, "average_units_on_hand", required=True)
    if not 0 < units <= 10_000_000:
        raise ToolError("average_units_on_hand must be above 0 and at most 10,000,000")

    answer = _ask(reports, variant=variant, lead_time=lead_time, stock_budget=units)
    methods = {}
    for method, label in (("lightgbm", "LightGBM"), ("ma28", "MA28")):
        result = answer["methods"][method]["fill_for_budget"]
        methods[method] = {
            "label": label,
            "status": result["status"],
            "status_meaning": STATUS_MEANING[result["status"]],
            "fill_rate_percent": _percent(result["fill_rate"]),
            "between_z": _between(result),
            "tested_stock_range_units": [
                _units(result["min_units"]),
                _units(result["max_units"]),
            ],
        }
    fills = [
        answer["methods"][method]["fill_for_budget"]["fill_rate"]
        for method in ("lightgbm", "ma28")
    ]
    gap = None if None in fills else round((fills[1] - fills[0]) * 100, 1)

    return {
        "lead_time_days": lead_time,
        "variant": variant,
        "average_units_on_hand": _units(units),
        "lightgbm": methods["lightgbm"],
        "ma28": methods["ma28"],
        "ma28_minus_lightgbm_fill_rate_points": gap,
        "notes": _notes(answer["settings"]),
    }


def compare_costs(reports: Reports, arguments: dict) -> dict:
    variant, lead_time = _scenario(reports, arguments)
    protocol = reports.protocol
    holding = _number(
        arguments,
        "holding_rate_percent_per_year",
        default=protocol["holding_rate_per_year"] * 100,
    )
    penalty = _number(arguments, "lost_sale_penalty_times_price", default=0.5)

    answer = _ask(
        reports,
        variant=variant,
        lead_time=lead_time,
        holding_rate=holding / 100,
        penalty_fraction=penalty,
    )
    methods = {}
    for method, label in (("lightgbm", "LightGBM"), ("ma28", "MA28")):
        best = answer["methods"][method]["cheapest"]
        methods[method] = {
            "label": label,
            "cheapest_z": _z(best["z"]),
            "total_cost_dollars": round(best["total_cost"]),
            "holding_cost_dollars": round(best["holding_cost"]),
            "lost_sale_cost_dollars": round(best["penalty_cost"]),
            "fill_rate_percent": _percent(best["fill_rate"]),
            "average_units_on_hand": _units(best["average_on_hand_units"]),
            "at_edge_of_tested_range": best["grid_edge"] is not None,
        }

    return {
        "lead_time_days": lead_time,
        "variant": variant,
        "holding_rate_percent_per_year": round(holding, 1),
        "lost_sale_penalty_times_price": round(penalty, 2),
        "cost_period": "84 evaluation days, illustrative dollars at sell price",
        "lightgbm": methods["lightgbm"],
        "ma28": methods["ma28"],
        "lightgbm_cheapest_cost_vs_ma28_percent": _percent(
            answer["comparison"]["cheapest_cost_difference_lightgbm_vs_ma28"]
        ),
        "edge_note": (
            "A cheapest policy at the edge of the tested z range means the true "
            "minimum may lie outside it."
        ),
        "notes": _notes(answer["settings"], costs=True),
    }


HANDLERS = {
    "list_scenarios": list_scenarios,
    "stock_for_fill_rate": stock_for_fill_rate,
    "fill_rate_for_stock": fill_rate_for_stock,
    "compare_costs": compare_costs,
}


def run_tool(reports: Reports, name: str, arguments) -> dict:
    """Run one tool call; raises ToolError with a message meant for the model."""
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolError(
            f"Unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}"
        )
    if not isinstance(arguments, dict):
        raise ToolError("Tool arguments must be a JSON object")

    spec = next(spec for spec in TOOL_SPECS if spec["name"] == name)
    allowed = set(spec["input_schema"]["properties"])
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolError(f"Unknown arguments for {name}: {', '.join(unknown)}")

    return handler(reports, arguments)
