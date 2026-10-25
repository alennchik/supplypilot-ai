import json

import pytest

from src.agent_tools import HANDLERS, TOOL_NAMES, TOOL_SPECS, ToolError, run_tool
from src.inventory_whatif import load_reports


@pytest.fixture(scope="module")
def reports():
    return load_reports()


def test_every_spec_has_a_handler_and_matches_the_protocol(reports):
    assert set(TOOL_NAMES) == set(HANDLERS)
    for spec in TOOL_SPECS:
        properties = spec["input_schema"]["properties"]
        if "lead_time_days" in properties:
            assert sorted(properties["lead_time_days"]["enum"]) == sorted(
                reports.protocol["lead_times"]
            )
        if "variant" in properties:
            assert properties["variant"]["enum"] == [
                variant[0] for variant in reports.protocol["variants"]
            ]


def test_stock_for_ninety_percent_at_seven_days(reports):
    result = run_tool(
        reports,
        "stock_for_fill_rate",
        {"lead_time_days": 7, "target_fill_rate_percent": 90},
    )

    assert result["lightgbm"]["average_units_on_hand"] == 39157
    assert result["ma28"]["average_units_on_hand"] == 33044
    assert result["lightgbm"]["between_z"] == [2.1, 2.2]
    assert result["lightgbm_stock_difference_vs_ma28_percent"] == 18.5
    assert result["perfect_foresight_units_for_100_percent"] == 14285
    assert result["notes"] == []
    json.dumps(result, allow_nan=False)


def test_a_target_lightgbm_never_reaches(reports):
    result = run_tool(
        reports,
        "stock_for_fill_rate",
        {"lead_time_days": 7, "target_fill_rate_percent": 95},
    )

    assert result["lightgbm"]["status"] == "not_reached"
    assert result["lightgbm"]["average_units_on_hand"] is None
    assert result["lightgbm"]["best_fill_rate_percent_on_tested_range"] == 92.2
    assert result["ma28"]["average_units_on_hand"] == 51465
    assert result["lightgbm_stock_difference_vs_ma28_percent"] is None
    assert "Not comparable" in result["stock_difference_note"]


def test_exploratory_target_and_secondary_scenario_are_noted(reports):
    result = run_tool(
        reports,
        "stock_for_fill_rate",
        {"lead_time_days": 3, "target_fill_rate_percent": 93, "variant": "empty_start"},
    )

    assert result["variant"] == "empty_start"
    assert any(note.startswith("Exploratory target") for note in result["notes"])
    assert any(note.startswith("Secondary scenario") for note in result["notes"])


def test_fill_rate_for_a_stock_budget(reports):
    result = run_tool(
        reports,
        "fill_rate_for_stock",
        {"lead_time_days": 7, "average_units_on_hand": 35000},
    )

    assert result["lightgbm"]["fill_rate_percent"] == 88.4
    assert result["ma28"]["fill_rate_percent"] == 90.9
    assert result["ma28_minus_lightgbm_fill_rate_points"] == 2.5
    assert result["ma28"]["between_z"] == [1.3, 1.4]


def test_a_budget_below_the_tested_range(reports):
    result = run_tool(
        reports,
        "fill_rate_for_stock",
        {"lead_time_days": 7, "average_units_on_hand": 12000},
    )

    for method in ("lightgbm", "ma28"):
        assert result[method]["status"] == "below_grid"
        assert result[method]["fill_rate_percent"] is None
    assert result["lightgbm"]["tested_stock_range_units"] == [21111, 47331]
    assert result["ma28_minus_lightgbm_fill_rate_points"] is None


def test_costs_at_the_protocol_defaults(reports):
    result = run_tool(reports, "compare_costs", {"lead_time_days": 7})

    assert result["holding_rate_percent_per_year"] == 25.0
    assert result["lost_sale_penalty_times_price"] == 0.5
    assert result["lightgbm"]["total_cost_dollars"] == 44877
    assert result["ma28"]["total_cost_dollars"] == 33559
    assert result["lightgbm_cheapest_cost_vs_ma28_percent"] == 33.7
    assert result["ma28"]["cheapest_z"] == 3.0
    assert result["ma28"]["at_edge_of_tested_range"] is True
    assert result["notes"] == []


def test_exploratory_costs_are_noted(reports):
    result = run_tool(
        reports,
        "compare_costs",
        {
            "lead_time_days": 3,
            "holding_rate_percent_per_year": 30,
            "lost_sale_penalty_times_price": 0.75,
        },
    )

    assert any(note.startswith("Exploratory costs") for note in result["notes"])
    assert (
        result["lightgbm"]["total_cost_dollars"] > result["ma28"]["total_cost_dollars"]
    )


def test_list_scenarios(reports):
    result = run_tool(reports, "list_scenarios", {})

    assert result["lead_times_days"] == [1, 3, 7]
    assert result["safety_factor_z_range"] == [0.0, 3.0]
    assert result["protocol_targets_percent"] == [90.0, 95.0, 98.0]
    assert result["protocol_holding_rate_percent_per_year"] == 25.0
    assert (result["series"], result["evaluation_days"]) == (2996, 84)
    assert (result["evaluation_start"], result["evaluation_end"]) == (
        "2016-02-01",
        "2016-04-24",
    )


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("forecast_sales", {}, "Unknown tool"),
        ("stock_for_fill_rate", ["7", "90"], "must be a JSON object"),
        (
            "stock_for_fill_rate",
            {"lead_time_days": 7, "target": 90},
            "Unknown arguments",
        ),
        (
            "stock_for_fill_rate",
            {"lead_time_days": 5, "target_fill_rate_percent": 90},
            "lead_time_days",
        ),
        (
            "stock_for_fill_rate",
            {"lead_time_days": True, "target_fill_rate_percent": 90},
            "lead_time_days",
        ),
        (
            "stock_for_fill_rate",
            {"lead_time_days": 7},
            "target_fill_rate_percent is required",
        ),
        (
            "stock_for_fill_rate",
            {"lead_time_days": 7, "target_fill_rate_percent": 0},
            "above 0",
        ),
        (
            "stock_for_fill_rate",
            {"lead_time_days": 7, "target_fill_rate_percent": "high"},
            "number",
        ),
        (
            "fill_rate_for_stock",
            {"lead_time_days": 7, "average_units_on_hand": -5},
            "above 0",
        ),
        ("compare_costs", {"lead_time_days": 7, "variant": "late"}, "variant"),
        (
            "compare_costs",
            {"lead_time_days": 7, "lost_sale_penalty_times_price": 9},
            "Penalty",
        ),
    ],
)
def test_bad_calls_raise_messages_for_the_model(reports, name, arguments, message):
    with pytest.raises(ToolError, match=message):
        run_tool(reports, name, arguments)
