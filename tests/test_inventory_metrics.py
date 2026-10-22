"""Tests for pooled Phase 3 service and inventory metrics."""

import numpy as np
import pytest

from src.inventory_metrics import (
    inventory_at_fill_rate,
    per_series_totals,
    summarize_inventory,
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


def test_inventory_at_fill_rate_interpolates_between_bracketing_points():
    fills = [0.80, 0.90, 0.96, 0.99]
    stock = [100.0, 140.0, 200.0, 300.0]

    assert inventory_at_fill_rate(fills, stock, 0.93) == pytest.approx(170.0)
    assert inventory_at_fill_rate(fills, stock, 0.90) == pytest.approx(140.0)


def test_inventory_at_fill_rate_is_nan_outside_the_grid():
    fills = [0.80, 0.90, 0.96]
    stock = [100.0, 140.0, 200.0]

    assert np.isnan(inventory_at_fill_rate(fills, stock, 0.98))
    assert np.isnan(inventory_at_fill_rate(fills, stock, 0.70))


def test_inventory_at_fill_rate_uses_the_first_crossing():
    fills = [0.90, 0.96, 0.95, 0.97]
    stock = [10.0, 16.0, 17.0, 18.0]

    assert inventory_at_fill_rate(fills, stock, 0.95) == pytest.approx(15.0)


@pytest.mark.parametrize(
    "fills, stock, target",
    [
        ([0.9], [1.0, 2.0], 0.95),
        ([], [], 0.95),
        ([1.2, 0.9], [1.0, 2.0], 0.95),
        ([0.9, np.nan], [1.0, 2.0], 0.95),
        ([0.9, 0.96], [1.0, 2.0], 0.0),
        ([0.9, 0.96], [1.0, 2.0], 95),
    ],
)
def test_inventory_at_fill_rate_rejects_invalid_inputs(fills, stock, target):
    with pytest.raises(ValueError):
        inventory_at_fill_rate(fills, stock, target)
