"""Tests for pooled Phase 3 service and inventory metrics."""

import numpy as np
import pytest

from src.inventory_metrics import (
    ABOVE_TARGET_AT_MIN_Z,
    EXACT,
    INTERPOLATED,
    NOT_REACHED,
    classify_relative_difference,
    illustrative_costs,
    interpolated_inventory_at_fill_rate,
    paired_cluster_bootstrap,
    per_series_totals,
    summarize_inventory,
    value_proxy_from_prices,
)
from src.inventory_simulation import simulate_periodic_review


@pytest.fixture
def traced():
    """The hand-traced example from the simulator tests."""
    return simulate_periodic_review(
        [[3, 1, 4, 1, 5, 9]],
        [[5, 6, 7]],
        lead_time_days=1,
        review_period_days=2,
    )


def test_summary_excludes_warm_up_days(traced):
    summary = summarize_inventory(traced, warmup_days=1)

    # Evaluated days 2..6: demand 1+4+1+5+9, sales 1+4+1+1+6.
    assert summary["evaluated_days"] == 5
    assert summary["demand_units"] == 20
    assert summary["sales_units"] == 13
    assert summary["lost_units"] == 7
    assert summary["fill_rate"] == pytest.approx(0.65)
    assert summary["stockout_series_days"] == 2
    assert summary["demand_series_days"] == 5
    assert summary["stockout_rate"] == pytest.approx(0.4)
    assert summary["average_on_hand_units"] == pytest.approx(1.0)
    assert summary["deliveries"] == 3
    assert summary["units_received"] == 13


def test_summary_without_warm_up_counts_every_day(traced):
    summary = summarize_inventory(traced, warmup_days=0)

    assert summary["demand_units"] == 23
    assert summary["lost_units"] == 10
    assert summary["fill_rate"] == pytest.approx(13 / 23)


def test_value_weighted_metrics_use_the_value_proxy():
    result = simulate_periodic_review(
        [[0, 2, 2, 0], [1, 1, 1, 1]],
        [[4, 0], [0, 0]],
        lead_time_days=0,
        review_period_days=2,
    )

    summary = summarize_inventory(result, warmup_days=0, unit_values=[2.5, 10.0])

    # Series 0 holds 4, 2, 0, 0 units; series 1 never has stock.
    assert summary["average_on_hand_units"] == pytest.approx(6 / 4)
    assert summary["average_on_hand_value"] == pytest.approx(6 * 2.5 / 4)
    assert summary["lost_sales_value"] == pytest.approx(4 * 10.0)


def test_stockout_rate_counts_only_days_with_demand():
    result = simulate_periodic_review(
        [[0, 2, 2, 0], [1, 1, 1, 1]],
        [[4, 0], [0, 0]],
        lead_time_days=0,
        review_period_days=2,
    )

    summary = summarize_inventory(result, warmup_days=0)

    # 8 series-days, but only 6 have demand; series 1 stocks out on all 4.
    assert summary["demand_series_days"] == 6
    assert summary["stockout_series_days"] == 4
    assert summary["stockout_rate"] == pytest.approx(4 / 6)


def test_per_series_totals_pool_to_the_summary():
    rng = np.random.default_rng(5)
    result = simulate_periodic_review(
        rng.poisson(2.0, size=(8, 35)),
        rng.uniform(0, 30, size=(8, 5)),
        lead_time_days=3,
    )

    totals = per_series_totals(result, warmup_days=7)
    summary = summarize_inventory(result, warmup_days=7)

    assert totals["demand_units"].shape == (8,)
    assert totals["sales_units"].sum() == summary["sales_units"]
    assert totals["on_hand_unit_days"].sum() / 28 == pytest.approx(
        summary["average_on_hand_units"]
    )
    np.testing.assert_array_equal(
        totals["sales_units"] + totals["lost_units"], totals["demand_units"]
    )


@pytest.mark.parametrize("warmup", [-1, 6, 2.5])
def test_invalid_warm_up_is_rejected(traced, warmup):
    with pytest.raises(ValueError, match="Warm-up"):
        summarize_inventory(traced, warmup_days=warmup)


def test_zero_evaluated_demand_is_rejected():
    result = simulate_periodic_review(
        [[1, 0, 0]],
        [[0]],
        lead_time_days=1,
    )

    with pytest.raises(ValueError, match="undefined"):
        summarize_inventory(result, warmup_days=1)


FILLS = [0.80, 0.90, 0.96, 0.99]
STOCK = [100.0, 140.0, 200.0, 300.0]


def test_endpoint_interpolates_between_the_bracketing_simulated_points():
    endpoint = interpolated_inventory_at_fill_rate(FILLS, STOCK, 0.93)

    assert endpoint.status == INTERPOLATED
    assert endpoint.inventory == pytest.approx(170.0)
    assert (endpoint.lower_index, endpoint.upper_index) == (1, 2)
    # The first policy that was actually run and reaches the target.
    assert endpoint.first_reaching_index == 2


def test_endpoint_reports_an_exact_simulated_hit():
    endpoint = interpolated_inventory_at_fill_rate(FILLS, STOCK, 0.90)

    assert endpoint.status == EXACT
    assert endpoint.inventory == pytest.approx(140.0)
    assert endpoint.first_reaching_index == 1

    first = interpolated_inventory_at_fill_rate([0.95, 0.97], [10.0, 12.0], 0.95)
    assert first.status == EXACT
    assert first.inventory == pytest.approx(10.0)


def test_endpoint_separates_not_reached_from_above_target_at_min_z():
    not_reached = interpolated_inventory_at_fill_rate(FILLS, STOCK, 0.995)
    above = interpolated_inventory_at_fill_rate([0.97, 0.99], [10.0, 12.0], 0.95)

    assert not_reached.status == NOT_REACHED
    assert np.isnan(not_reached.inventory)
    assert not_reached.first_reaching_index is None

    assert above.status == ABOVE_TARGET_AT_MIN_Z
    assert np.isnan(above.inventory)
    assert above.first_reaching_index == 0


def test_endpoint_uses_the_first_crossing_in_z_order():
    endpoint = interpolated_inventory_at_fill_rate(
        [0.90, 0.96, 0.95, 0.97], [10.0, 16.0, 17.0, 18.0], 0.95
    )

    assert endpoint.status == INTERPOLATED
    assert endpoint.inventory == pytest.approx(15.0)
    assert (endpoint.lower_index, endpoint.upper_index) == (0, 1)


@pytest.mark.parametrize(
    "fills, stock, target",
    [
        ([0.9], [1.0, 2.0], 0.95),
        ([], [], 0.95),
        ([1.2, 0.9], [1.0, 2.0], 0.95),
        ([0.9, np.nan], [1.0, 2.0], 0.95),
        ([0.9, 0.96], [-10.0, -4.0], 0.95),
        ([0.9, 0.96], [1.0, 2.0], 0.0),
        ([0.9, 0.96], [1.0, 2.0], 95),
    ],
)
def test_endpoint_rejects_invalid_inputs(fills, stock, target):
    with pytest.raises(ValueError):
        interpolated_inventory_at_fill_rate(fills, stock, target)


@pytest.mark.parametrize(
    "low, high, expected",
    [
        (-0.01, 0.015, "practically_equivalent"),
        (-0.015, -0.001, "practically_equivalent"),
        (-0.10, -0.03, "a_needs_less_inventory"),
        (0.03, 0.10, "b_needs_less_inventory"),
        (-0.05, 0.01, "inconclusive"),
        (-0.15, 0.12, "inconclusive"),
    ],
)
def test_classification_uses_the_whole_interval(low, high, expected):
    assert classify_relative_difference(low, high) == expected


@pytest.mark.parametrize("low, high", [(0.1, -0.1), (np.nan, 0.1)])
def test_classification_rejects_invalid_intervals(low, high):
    with pytest.raises(ValueError):
        classify_relative_difference(low, high)


def _curves(n_series=40, seed=0):
    """Per-series totals whose pooled fill is the same for any resample."""
    rng = np.random.default_rng(seed)
    demand = rng.integers(50, 100, size=n_series).astype(float)
    fills = np.array([0.85, 0.90, 0.94, 0.97, 0.99])
    sales = fills[:, None] * demand[None, :]
    on_hand = (np.arange(1, 6)[:, None] * 10.0) + rng.integers(0, 3, size=n_series)
    return demand, sales, on_hand


def test_bootstrap_of_identical_methods_is_practically_equivalent():
    demand, sales, on_hand = _curves()

    result = paired_cluster_bootstrap(
        demand_units=demand,
        sales_a=sales,
        on_hand_a=on_hand,
        sales_b=sales,
        on_hand_b=on_hand,
        evaluated_days=84,
        clusters=np.arange(40),
        n_resamples=200,
    )

    assert result["point_estimate"] == 0.0
    assert (result["ci_low"], result["ci_high"]) == (0.0, 0.0)
    assert result["verdict"] == "practically_equivalent"
    assert result["interval"] == "percentile"
    assert result["endpoint_a"].status == INTERPOLATED


def test_bootstrap_detects_uniformly_lower_inventory():
    demand, sales, on_hand = _curves()

    result = paired_cluster_bootstrap(
        demand_units=demand,
        sales_a=sales,
        on_hand_a=0.8 * on_hand,
        sales_b=sales,
        on_hand_b=on_hand,
        evaluated_days=84,
        clusters=np.arange(40),
        n_resamples=200,
    )

    assert result["point_estimate"] == pytest.approx(-0.2)
    assert result["ci_low"] == pytest.approx(-0.2)
    assert result["ci_high"] == pytest.approx(-0.2)
    assert result["verdict"] == "a_needs_less_inventory"


def test_bootstrap_keeps_each_cluster_together():
    demand, sales, on_hand_b = _curves()
    on_hand_a = on_hand_b.copy()
    # Within each pair of series, A holds 5 more units in one and 5 fewer in
    # the other, so every resample of whole pairs gives identical inventory.
    on_hand_a[:, 0::2] += 5
    on_hand_a[:, 1::2] -= 5
    arguments = {
        "demand_units": demand,
        "sales_a": sales,
        "on_hand_a": on_hand_a,
        "sales_b": sales,
        "on_hand_b": on_hand_b,
        "evaluated_days": 84,
        "n_resamples": 300,
    }

    paired = paired_cluster_bootstrap(clusters=np.arange(40) // 2, **arguments)
    separate = paired_cluster_bootstrap(clusters=np.arange(40), **arguments)

    assert (paired["ci_low"], paired["ci_high"]) == (0.0, 0.0)
    assert separate["ci_high"] - separate["ci_low"] > 0
    assert paired["n_clusters"] == 20


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    demand, sales, on_hand = _curves()
    noisy = on_hand + np.random.default_rng(9).integers(0, 4, size=on_hand.shape)
    arguments = {
        "demand_units": demand,
        "sales_a": sales,
        "on_hand_a": noisy,
        "sales_b": sales,
        "on_hand_b": on_hand,
        "evaluated_days": 84,
        "clusters": np.arange(40),
        "n_resamples": 100,
    }

    first = paired_cluster_bootstrap(**arguments)
    second = paired_cluster_bootstrap(**arguments)

    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])


def test_bootstrap_interval_is_the_95_percent_percentile_of_valid_resamples():
    demand, sales, on_hand = _curves()
    noisy = on_hand + np.random.default_rng(9).integers(0, 4, size=on_hand.shape)

    result = paired_cluster_bootstrap(
        demand_units=demand,
        sales_a=sales,
        on_hand_a=noisy,
        sales_b=sales,
        on_hand_b=on_hand,
        evaluated_days=84,
        clusters=np.arange(40),
        n_resamples=400,
    )
    deltas = result["resample_deltas"]

    assert deltas.size == 400 - result["invalid_resamples"]
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.percentile(deltas, [2.5, 97.5])
    )
    assert deltas.min() < result["ci_low"] < result["ci_high"] < deltas.max()


def test_bootstrap_counts_resamples_where_a_method_never_reaches_target():
    demand, sales, on_hand = _curves()
    low_service = 0.9 * sales  # tops out at 89.1% fill

    result = paired_cluster_bootstrap(
        demand_units=demand,
        sales_a=sales,
        on_hand_a=on_hand,
        sales_b=low_service,
        on_hand_b=on_hand,
        evaluated_days=84,
        clusters=np.arange(40),
        n_resamples=50,
    )

    assert result["endpoint_b"].status == NOT_REACHED
    assert result["invalid_resamples"] == 50
    assert result["reliable"] is False
    assert result["verdict"] == "not_estimable"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"clusters": np.arange(39)}, "one label per series"),
        ({"n_resamples": 0}, "positive integer"),
        ({"sales_b": np.zeros((4, 40))}, "same z grid"),
        ({"on_hand_a": -np.ones((5, 40))}, "nonnegative"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(change, message):
    demand, sales, on_hand = _curves()
    arguments = {
        "demand_units": demand,
        "sales_a": sales,
        "on_hand_a": on_hand,
        "sales_b": sales,
        "on_hand_b": on_hand,
        "evaluated_days": 84,
        "clusters": np.arange(40),
        "n_resamples": 10,
        **change,
    }

    with pytest.raises(ValueError, match=message):
        paired_cluster_bootstrap(**arguments)


def test_bootstrap_rejects_sales_above_demand():
    demand, sales, on_hand = _curves()

    with pytest.raises(ValueError, match="exceed demand"):
        paired_cluster_bootstrap(
            demand_units=demand,
            sales_a=sales * 1.2,
            on_hand_a=on_hand,
            sales_b=sales,
            on_hand_b=on_hand,
            evaluated_days=84,
            clusters=np.arange(40),
            n_resamples=10,
        )


def test_illustrative_costs_follow_the_documented_formula():
    totals = {
        "on_hand_unit_days": np.array([365.0, 0.0]),
        "lost_units": np.array([0.0, 4.0]),
    }

    costs = illustrative_costs(totals, [2.0, 10.0])

    # Holding: 2.0 * 365 unit-days * 0.25 / 365. Penalty: 40 * p.
    assert costs["holding_cost"] == pytest.approx(0.5)
    assert costs["penalty_cost"] == pytest.approx({0.25: 10.0, 0.5: 20.0, 1.0: 40.0})
    assert costs["total_cost"][1.0] == pytest.approx(40.5)


def test_value_proxy_is_the_median_available_price():
    # Median, not mean: the first row's mean would be 3.0.
    prices = [[1.0, np.nan, 2.0, 6.0], [2.0, 2.0, 2.0, 2.0]]

    np.testing.assert_allclose(value_proxy_from_prices(prices), [2.0, 2.0])


@pytest.mark.parametrize(
    "prices, message",
    [
        ([[1.0, 2.0], [np.nan, np.nan]], "No available price for series at rows \\[1\\]"),
        ([[0.0, 2.0]], "positive"),
        ([[np.inf, 2.0]], "positive"),
    ],
)
def test_value_proxy_stops_instead_of_dropping_series(prices, message):
    with pytest.raises(ValueError, match=message):
        value_proxy_from_prices(prices)


def test_value_proxy_names_series_without_a_price():
    prices = [[1.0, 2.0], [np.nan, np.nan]]

    with pytest.raises(ValueError, match=r"series: \['TX_2_item'\]"):
        value_proxy_from_prices(prices, series_ids=["CA_1_item", "TX_2_item"])

    with pytest.raises(ValueError, match="one entry per price row"):
        value_proxy_from_prices(prices, series_ids=["only_one"])
