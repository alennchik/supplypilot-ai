"""Synthetic tests for the frozen Phase 3 experiment driver."""

import numpy as np
import pytest

from src import inventory_experiment as experiment
from src.inventory_metrics import ESTIMABLE

N_SERIES = 60
WEEKLY = np.array([1.3, 1.15, 0.9, 0.82, 0.85, 0.95, 1.2])


def synthetic_case(noise_sigma=0.0, seed=0):
    """Overdispersed demand with known means; forecasts = true means times noise.

    Means and dispersion are chosen so the 95% fill target is bracketed by
    the z grid in every bootstrap resample.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 4.0, size=(N_SERIES, 1))
    days = np.arange(112 + 28)
    mean = base * WEEKLY[days % 7]
    demand = rng.poisson(rng.gamma(2.0, mean[:, :112] / 2.0))

    origins = np.arange(0, 112, 7)
    truth = np.stack([mean[:, t : t + 28] for t in origins], axis=1)
    noise = np.exp(rng.normal(0.0, noise_sigma, size=(N_SERIES, len(origins), 1)))

    clusters = np.arange(N_SERIES) // 2
    groups = {
        "demand_group": np.where(base[:, 0] > 2, "frequent", "mixed"),
        "cat_id": np.array(["FOODS", "HOBBIES", "HOUSEHOLD"])[np.arange(N_SERIES) % 3],
    }
    values = rng.uniform(1.0, 10.0, size=N_SERIES)
    return demand, truth, truth * noise, clusters, groups, values


@pytest.fixture(scope="module")
def better_lightgbm():
    demand, truth, noisy, clusters, groups, values = synthetic_case(noise_sigma=0.5)
    return experiment.run_experiment(
        demand,
        {"lightgbm": truth, "ma28": noisy},
        clusters,
        groups,
        values,
        n_resamples=200,
    )


def test_parameters_are_the_frozen_protocol():
    parameters = experiment.protocol_parameters()

    assert parameters["protocol_version"] == "1.0"
    assert parameters["z_grid"] == [round(0.1 * i, 1) for i in range(31)]
    assert parameters["z_grid"][-1] == 3.0
    assert parameters["lead_times"] == [7, 3, 1]
    assert parameters["warmup_days"] == 28
    assert parameters["targets"] == [0.90, 0.95, 0.98]
    assert parameters["variants"] == [
        ["primary", "lead_time_demand", "ceil"],
        ["empty_start", "empty", "ceil"],
        ["half_up", "lead_time_demand", "half_up"],
    ]
    assert parameters["bootstrap"]["n_resamples"] == 2000
    assert parameters["bootstrap"]["seed"] == 42


def test_every_scenario_is_reported(better_lightgbm):
    curves = better_lightgbm["curves"]

    assert len(curves) == 3 * 3 * 2 * 31
    assert set(curves.variant) == {"primary", "empty_start", "half_up"}
    assert set(curves.lead_time) == {7, 3, 1}
    assert set(curves.method) == {"lightgbm", "ma28"}
    assert (curves.evaluated_days == 84).all()
    assert len(better_lightgbm["reference"]) == 9
    assert len(better_lightgbm["endpoints"]) == 9 * 3 * 2
    assert {"holding_cost", "total_cost_penalty_1.0"} <= set(curves.columns)


def test_reference_serves_all_demand_in_the_evaluation_window(better_lightgbm):
    reference = better_lightgbm["reference"]

    assert (reference.fill_rate == 1.0).all()
    assert (reference.lost_units == 0).all()


def test_primary_endpoints_are_estimable_and_reported_with_brackets(better_lightgbm):
    endpoints = better_lightgbm["endpoints"]
    primary = endpoints.loc[
        (endpoints.variant == "primary")
        & (endpoints.lead_time == 7)
        & (endpoints.target_fill_rate == 0.95)
    ]

    assert len(primary) == 2
    assert set(primary.status) <= set(ESTIMABLE)
    assert (primary.lower_fill_rate < 0.95).all()
    assert (primary.upper_fill_rate >= 0.95).all()
    assert (primary.first_reaching_fill_rate >= 0.95).all()


def test_more_accurate_forecasts_need_less_inventory(better_lightgbm):
    primary = better_lightgbm["summary"]["primary"]

    assert primary["point_estimate"] < 0
    assert primary["verdict"] == "a_needs_less_inventory"
    assert primary["interval"] == "percentile"
    assert primary["n_resamples"] == 200


def test_summary_applies_the_starting_state_rule(better_lightgbm):
    summary = better_lightgbm["summary"]

    expected = summary["sensitivity"]["empty_start"]["verdict"] != summary["primary"]["verdict"]
    assert summary["starting_state_dependent"] is expected
    assert isinstance(summary["rounding_check_agrees"], bool)


def test_summary_carries_one_unambiguous_conclusion(better_lightgbm):
    conclusion = better_lightgbm["summary"]["conclusion"]

    assert conclusion["code"] == "a_needs_less_inventory"
    assert conclusion["statement"].startswith("LightGBM needs less inventory")


def _summary(primary, empty, half_up="practically_equivalent"):
    return {
        "primary": {"verdict": primary},
        "sensitivity": {"empty_start": {"verdict": empty}, "half_up": {"verdict": half_up}},
        "starting_state_dependent": primary != empty,
    }


def test_a_starting_state_dependence_overrides_the_primary_verdict():
    conclusion = experiment.overall_conclusion(
        _summary("a_needs_less_inventory", "inconclusive")
    )

    assert conclusion["code"] == "depends_on_starting_state"
    assert "No robust advantage" in conclusion["statement"]
    assert conclusion["primary_verdict"] == "a_needs_less_inventory"
    assert conclusion["empty_start_verdict"] == "inconclusive"


@pytest.mark.parametrize("verdict", sorted(experiment.STATEMENTS))
def test_every_verdict_has_a_plain_statement(verdict):
    conclusion = experiment.overall_conclusion(_summary(verdict, verdict))

    assert conclusion["code"] == verdict
    assert conclusion["statement"] == experiment.STATEMENTS[verdict]


def test_group_breakdowns_cover_every_group(better_lightgbm):
    group_endpoints = better_lightgbm["group_endpoints"]
    group_curves = better_lightgbm["group_curves"]

    assert set(group_endpoints.group_type) == {"demand_group", "cat_id"}
    assert set(group_endpoints.loc[group_endpoints.group_type == "cat_id", "group"]) == {
        "FOODS",
        "HOBBIES",
        "HOUSEHOLD",
    }
    assert len(group_curves) == (2 + 3) * 2 * 31


def test_a_target_met_at_the_smallest_z_makes_the_interval_unreliable():
    """When 95% is already exceeded at z = 0 in some resamples, those resamples
    are invalid and the protocol forbids declaring a winner."""
    rng = np.random.default_rng(1)
    base = rng.uniform(5.0, 8.0, size=(N_SERIES, 1))
    mean = base * WEEKLY[np.arange(140) % 7]
    demand = rng.poisson(mean[:, :112])
    truth = np.stack([mean[:, t : t + 28] for t in range(0, 112, 7)], axis=1)
    groups = {"cat_id": np.array(["FOODS"] * N_SERIES)}

    result = experiment.run_experiment(
        demand,
        {"lightgbm": truth, "ma28": truth.copy()},
        np.arange(N_SERIES),
        groups,
        np.ones(N_SERIES),
        n_resamples=200,
    )
    primary = result["summary"]["primary"]

    assert primary["invalid_resamples"] > 0
    assert primary["verdict"] in ("unreliable", "not_estimable")


def test_identical_forecasts_are_practically_equivalent():
    demand, truth, _, clusters, groups, values = synthetic_case(seed=3)

    result = experiment.run_experiment(
        demand,
        {"lightgbm": truth, "ma28": truth.copy()},
        clusters,
        groups,
        values,
        n_resamples=50,
    )

    assert result["summary"]["primary"]["point_estimate"] == 0.0
    assert result["summary"]["primary"]["verdict"] == "practically_equivalent"
    assert result["summary"]["starting_state_dependent"] is False


@pytest.mark.parametrize(
    "change, message",
    [
        ("demand", "Demand must have shape"),
        ("method", "exactly"),
        ("path_shape", "paths must have shape"),
        ("clusters", "one entry per series"),
    ],
)
def test_invalid_inputs_are_rejected(change, message):
    demand, truth, noisy, clusters, groups, values = synthetic_case()
    paths = {"lightgbm": truth, "ma28": noisy}

    if change == "demand":
        demand = demand[:, :100]
    elif change == "method":
        paths = {"lightgbm": truth}
    elif change == "path_shape":
        paths["ma28"] = noisy[:, :15]
    else:
        clusters = clusters[:-1]

    with pytest.raises(ValueError, match=message):
        experiment.run_experiment(demand, paths, clusters, groups, values, n_resamples=10)
