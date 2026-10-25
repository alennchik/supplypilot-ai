import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.inventory_metrics import ESTIMABLE
from src.inventory_whatif import (
    REPORT_DIR,
    cheapest,
    cost_table,
    curve,
    fill_rate_for_stock,
    load_reports,
    scenarios,
    stock_for_fill_rate,
    what_if,
)


@pytest.fixture(scope="module")
def reports():
    return load_reports()


def optional(value):
    return None if pd.isna(value) else float(value)


def test_stock_for_target_matches_every_published_endpoint(reports):
    endpoints = pd.read_csv(REPORT_DIR / "endpoints.csv")

    for row in endpoints.itertuples(index=False):
        frame = curve(reports, row.variant, row.lead_time, row.method)
        result = stock_for_fill_rate(frame, row.target_fill_rate)

        assert result["status"] == row.status
        assert result["lower_z"] == optional(row.lower_z)
        assert result["upper_z"] == optional(row.upper_z)
        assert result["first_reaching_z"] == optional(row.first_reaching_z)
        if row.status in ESTIMABLE:
            assert result["units"] == pytest.approx(row.inventory, rel=1e-12)
        else:
            assert result["units"] is None


def test_costs_at_protocol_settings_equal_the_published_columns(reports):
    for _, frame in reports.curves.groupby(["variant", "lead_time", "method"]):
        frame = frame.sort_values("z").reset_index(drop=True)
        for penalty in (0.25, 0.5, 1.0):
            costs = cost_table(frame, holding_rate=0.25, penalty_fraction=penalty)

            np.testing.assert_allclose(
                costs["holding_cost"], frame["holding_cost"], rtol=1e-12
            )
            np.testing.assert_allclose(
                costs["total_cost"], frame[f"total_cost_penalty_{penalty}"], rtol=1e-12
            )


def test_fill_for_stock_inverts_stock_for_target(reports):
    frame = curve(reports, "primary", 7, "ma28")
    units = stock_for_fill_rate(frame, 0.95)["units"]

    result = fill_rate_for_stock(frame, units)

    assert result["status"] == "interpolated"
    assert result["fill_rate"] == pytest.approx(0.95, abs=1e-12)
    assert (result["lower_z"], result["upper_z"]) == (2.9, 3.0)


def test_fill_for_stock_at_a_simulated_policy_is_exact(reports):
    frame = curve(reports, "primary", 3, "lightgbm")
    row = frame.iloc[10]

    result = fill_rate_for_stock(frame, row["average_on_hand_units"])

    assert result["status"] == "exact"
    assert result["fill_rate"] == row["fill_rate"]
    assert result["lower_z"] == result["upper_z"] == row["z"]


def test_fill_for_stock_outside_the_simulated_range(reports):
    frame = curve(reports, "primary", 7, "lightgbm")
    lowest = frame["average_on_hand_units"].iloc[0]
    highest = frame["average_on_hand_units"].iloc[-1]

    below = fill_rate_for_stock(frame, lowest - 1)
    above = fill_rate_for_stock(frame, highest + 1)

    assert (below["status"], below["fill_rate"], below["upper_z"]) == (
        "below_grid",
        None,
        0.0,
    )
    assert (above["status"], above["fill_rate"], above["lower_z"]) == (
        "above_grid",
        None,
        3.0,
    )
    assert below["min_units"] == lowest and below["max_units"] == highest


def test_a_curve_that_loses_stock_as_z_rises_is_refused():
    frame = pd.DataFrame(
        {"z": [0.0, 1.0], "fill_rate": [0.8, 0.9], "average_on_hand_units": [10.0, 9.0]}
    )

    with pytest.raises(ValueError, match="gain stock"):
        fill_rate_for_stock(frame, 9.5)


@pytest.mark.parametrize(
    ("totals", "edge"),
    [
        ([5.0, 4.0, 3.0], "highest_z"),
        ([3.0, 4.0, 5.0], "lowest_z"),
        ([5.0, 3.0, 4.0], None),
    ],
)
def test_cheapest_policy_flags_the_edges_of_the_grid(totals, edge):
    frame = pd.DataFrame(
        {
            "z": [0.0, 1.0, 2.0],
            "fill_rate": [0.8, 0.9, 0.95],
            "average_on_hand_units": [1, 2, 3],
        }
    )
    costs = pd.DataFrame(
        {
            "z": frame["z"],
            "holding_cost": totals,
            "penalty_cost": 0.0,
            "total_cost": totals,
        }
    )

    result = cheapest(frame, costs)

    assert result["grid_edge"] == edge
    assert result["total_cost"] == min(totals)


def test_primary_question_reports_lightgbm_as_not_reaching_the_target(reports):
    result = what_if(reports)
    lightgbm = result["methods"]["lightgbm"]["stock_for_target"]
    ma28 = result["methods"]["ma28"]["stock_for_target"]

    assert lightgbm["status"] == "not_reached"
    assert lightgbm["units"] is None
    assert lightgbm["max_fill_rate"] == pytest.approx(0.922427, abs=1e-6)
    assert ma28["status"] == "interpolated"
    assert ma28["units"] == pytest.approx(51465.062877, rel=1e-9)
    assert result["comparison"]["stock_comparable"] is False
    assert result["comparison"]["stock_difference_lightgbm_vs_ma28"] is None
    assert result["settings"]["costs_preregistered"] is True
    assert result["settings"]["primary_scenario"] is True
    assert result["reference"]["average_on_hand_units"] == pytest.approx(
        14285.1667, rel=1e-6
    )
    assert len(result["methods"]["ma28"]["policies"]) == 31
    json.dumps(result, allow_nan=False)


def test_ninety_percent_difference_equals_the_published_value(reports):
    result = what_if(reports, target_fill_rate=0.9, stock_budget=35_000)

    assert result["comparison"]["stock_difference_lightgbm_vs_ma28"] == pytest.approx(
        0.18500263451207885, rel=1e-12
    )
    assert (
        result["methods"]["ma28"]["fill_for_budget"]["fill_rate"]
        > (result["methods"]["lightgbm"]["fill_for_budget"]["fill_rate"])
    )


def test_custom_settings_are_labelled_exploratory(reports):
    result = what_if(
        reports,
        lead_time=3,
        target_fill_rate=0.93,
        holding_rate=0.3,
        penalty_fraction=0.75,
    )
    settings = result["settings"]

    assert settings["costs_preregistered"] is False
    assert settings["target_preregistered"] is False
    assert settings["primary_scenario"] is False
    assert len(settings["notes"]) == 3
    assert "exploratory" in settings["notes"][0]


def test_cheapest_policies_at_protocol_costs_sit_at_the_top_of_the_grid(reports):
    result = what_if(reports, penalty_fraction=1.0)

    for method in ("lightgbm", "ma28"):
        assert result["methods"][method]["cheapest"]["z"] == 3.0
        assert result["methods"][method]["cheapest"]["grid_edge"] == "highest_z"
    assert result["comparison"]["cheapest_cost_difference_lightgbm_vs_ma28"] > 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"variant": "late_start"}, "Unknown variant"),
        ({"lead_time": 5}, "Lead time must be one of"),
        ({"lead_time": True}, "Lead time must be one of"),
        ({"target_fill_rate": 0.0}, "Target fill rate must be above 0"),
        ({"target_fill_rate": 1.2}, "Target fill rate must be above 0"),
        ({"target_fill_rate": "high"}, "Target fill rate must be a number"),
        ({"holding_rate": -0.1}, "Holding rate must be at least 0"),
        ({"penalty_fraction": float("nan")}, "Penalty fraction must be at least 0"),
        ({"stock_budget": 0}, "Stock budget must be above 0"),
    ],
)
def test_invalid_questions_are_refused(reports, changes, message):
    with pytest.raises(ValueError, match=message):
        what_if(reports, **changes)


def test_changed_reports_are_refused(tmp_path):
    copy = tmp_path / "primary"
    shutil.copytree(REPORT_DIR, copy)
    with (copy / "curves.csv").open("a") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="curves.csv does not match checksums.json"):
        load_reports(copy)


def test_scenarios_describe_the_protocol_and_the_run(reports):
    result = scenarios(reports)

    assert result["lead_times"] == [7, 3, 1]
    assert [variant["id"] for variant in result["variants"]] == [
        "primary",
        "empty_start",
        "half_up",
    ]
    assert len(result["z_grid"]) == 31
    assert result["defaults"]["target_fill_rate"] == 0.95
    assert result["preregistered"]["penalty_fractions"] == [0.25, 0.5, 1.0]
    provenance = result["provenance"]
    assert (provenance["evaluation_start"], provenance["evaluation_end"]) == (
        "2016-02-01",
        "2016-04-24",
    )
    assert (provenance["series"], provenance["evaluated_days"]) == (2996, 84)
    assert provenance["source_commit"].startswith("7d81e23")
    json.dumps(result, allow_nan=False)
