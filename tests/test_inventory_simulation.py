"""Tests for the periodic-review, lost-sales inventory simulator."""

import math

import numpy as np
import pytest

from src.inventory_policy import (
    initial_stock_covering_lead_time,
    perfect_foresight_protection_demand,
)
from src.inventory_simulation import (
    MAX_UNITS,
    order_quantity,
    review_times,
    simulate_periodic_review,
)


def test_review_times_start_at_origin_and_stay_inside_span():
    np.testing.assert_array_equal(review_times(112, 7), np.arange(0, 106, 7))
    assert len(review_times(112, 7)) == 16
    np.testing.assert_array_equal(review_times(6, 2), [0, 2, 4])
    np.testing.assert_array_equal(review_times(1, 7), [0])


@pytest.mark.parametrize("n_days, period", [(0, 7), (10, 0), (10.0, 7), (True, 7)])
def test_review_times_reject_invalid_days(n_days, period):
    with pytest.raises(ValueError, match="whole number of days"):
        review_times(n_days, period)


def test_hand_traced_single_series():
    """Every unit traced by hand: R = 2, L = 1, empty start."""
    result = simulate_periodic_review(
        [[3, 1, 4, 1, 5, 9]],
        [[5, 6, 7]],
        lead_time_days=1,
        review_period_days=2,
    )

    # Review 0 (end of origin day): IP 0 -> order 5, arrives day 2.
    # Day 1: nothing on hand, 3 lost.
    # Day 2: receive 5, sell 1 -> 4. Review: IP 4 -> order 2, arrives day 4.
    # Day 3: sell 4 -> 0.
    # Day 4: receive 2, sell 1 -> 1. Review: IP 1 -> order 6, arrives day 6.
    # Day 5: demand 5, sell 1, lose 4 -> 0.
    # Day 6: receive 6, demand 9, sell 6, lose 3 -> 0.
    np.testing.assert_array_equal(result.orders, [[5, 2, 6]])
    np.testing.assert_array_equal(result.inventory_position_before_order, [[0, 4, 1]])
    np.testing.assert_array_equal(result.arrival_days, [2, 4, 6])
    np.testing.assert_array_equal(result.receipts, [[0, 5, 0, 2, 0, 6]])
    np.testing.assert_array_equal(result.sales, [[0, 1, 4, 1, 1, 6]])
    np.testing.assert_array_equal(result.lost, [[3, 0, 0, 0, 4, 3]])
    np.testing.assert_array_equal(result.on_hand, [[0, 4, 0, 1, 0, 0]])


@pytest.mark.parametrize("lead_time", [0, 1, 3, 7])
def test_order_arrives_after_exactly_lead_time_days_of_demand(lead_time):
    n_days = 14
    result = simulate_periodic_review(
        np.ones((1, n_days), dtype=int),
        np.full((1, 2), 20.0),
        lead_time_days=lead_time,
        review_period_days=7,
    )

    first_receipt_day = int(np.flatnonzero(result.receipts[0])[0]) + 1

    assert first_receipt_day == lead_time + 1
    # Exactly lead_time days of demand pass before the order is usable.
    assert result.lost[0, :lead_time].sum() == lead_time
    assert result.sales[0, lead_time] == 1


def test_inventory_position_includes_outstanding_orders():
    """With L > R, the second review must count the order still in transit."""
    result = simulate_periodic_review(
        np.zeros((1, 6), dtype=int),
        [[10, 10, 10]],
        lead_time_days=3,
        review_period_days=2,
    )

    # Order 0 (10 units) arrives day 4, so it is outstanding at the day-2 review.
    np.testing.assert_array_equal(result.inventory_position_before_order, [[0, 10, 10]])
    np.testing.assert_array_equal(result.orders, [[10, 0, 0]])


def test_inventory_above_target_orders_nothing():
    result = simulate_periodic_review(
        [[1, 1, 1, 1]],
        [[5, 5]],
        lead_time_days=1,
        review_period_days=2,
        initial_on_hand=[10],
    )

    np.testing.assert_array_equal(result.orders, [[0, 0]])
    np.testing.assert_array_equal(result.on_hand, [[9, 8, 7, 6]])


def test_starting_stock_changes_the_first_order():
    """Review example: the same target of 100 units, different starting stock."""
    empty = simulate_periodic_review([[10] * 7], [[100]], lead_time_days=7)
    stocked = simulate_periodic_review(
        [[10] * 7], [[100]], lead_time_days=7, initial_on_hand=[70]
    )

    np.testing.assert_array_equal(empty.orders, [[100]])
    np.testing.assert_array_equal(stocked.orders, [[30]])


def test_lost_sales_are_not_backordered():
    result = simulate_periodic_review(
        [[4, 4, 0, 0]],
        [[3, 0]],
        lead_time_days=1,
        review_period_days=2,
    )

    # Order 3 arrives day 2: day-1 demand is lost, day 2 sells 3 of 4.
    # Nothing lost earlier is served later from the leftover stock.
    np.testing.assert_array_equal(result.sales, [[0, 3, 0, 0]])
    np.testing.assert_array_equal(result.lost, [[4, 1, 0, 0]])
    np.testing.assert_array_equal(result.on_hand, [[0, 0, 0, 0]])


def test_units_are_conserved_and_never_negative():
    rng = np.random.default_rng(7)
    demand = rng.poisson(2.0, size=(25, 56))
    targets = rng.uniform(0, 40, size=(25, len(review_times(56, 7))))
    initial = rng.integers(0, 10, size=25)

    result = simulate_periodic_review(
        demand,
        targets,
        lead_time_days=3,
        initial_on_hand=initial,
    )

    np.testing.assert_array_equal(result.sales + result.lost, demand)
    np.testing.assert_array_equal(
        initial + result.receipts.sum(axis=1) - result.sales.sum(axis=1),
        result.on_hand[:, -1],
    )
    assert (result.on_hand >= 0).all()
    assert (result.orders >= 0).all()
    assert (result.sales <= demand).all()
    # Units still in transit at the end are ordered but never received.
    assert (result.orders.sum(axis=1) >= result.receipts.sum(axis=1)).all()


def test_orders_never_depend_on_future_demand():
    rng = np.random.default_rng(11)
    demand = rng.poisson(3.0, size=(10, 42))
    targets = rng.uniform(5, 30, size=(10, len(review_times(42, 7))))

    baseline = simulate_periodic_review(demand, targets, lead_time_days=7)

    changed = demand.copy()
    changed[:, 21:] = rng.poisson(9.0, size=(10, 21))
    perturbed = simulate_periodic_review(changed, targets, lead_time_days=7)

    # Reviews at times 0, 7, 14, 21 only saw days 1..21.
    np.testing.assert_array_equal(perturbed.orders[:, :4], baseline.orders[:, :4])
    assert not np.array_equal(perturbed.orders[:, 4:], baseline.orders[:, 4:])


def test_empty_start_leaves_a_permanent_surplus_for_rare_demand():
    """Review counterexample: 10 units a day for a week, then nothing."""
    demand = np.array([[10] * 7 + [0] * 105])
    targets = perfect_foresight_protection_demand(demand, lead_time_days=7)

    result = simulate_periodic_review(demand, targets, lead_time_days=7)

    # The first order covers the 70 units lost before it arrives; with no
    # later demand they stay on the shelf after any warm-up.
    assert result.lost.sum() == 70
    assert result.on_hand[0, 27] == result.on_hand[0, 41] == result.on_hand[0, -1] == 70


def test_starting_with_lead_time_demand_removes_the_surplus():
    demand = np.array([[10] * 7 + [0] * 105])
    targets = perfect_foresight_protection_demand(demand, lead_time_days=7)

    result = simulate_periodic_review(
        demand,
        targets,
        lead_time_days=7,
        initial_on_hand=initial_stock_covering_lead_time(demand, 7),
    )

    assert result.lost.sum() == 0
    assert (result.on_hand[0, 7:] == 0).all()


@pytest.mark.parametrize("lead_time", [0, 1, 3, 7])
def test_perfect_foresight_never_loses_a_sale_after_first_delivery(lead_time):
    rng = np.random.default_rng(lead_time)
    demand = rng.poisson(1.5, size=(30, 70))
    targets = perfect_foresight_protection_demand(demand, lead_time_days=lead_time)

    result = simulate_periodic_review(demand, targets, lead_time_days=lead_time)

    assert result.lost[:, lead_time:].sum() == 0


def _reference_single_series(demand, targets, lead_time, period, initial):
    """Plain-Python simulator written separately from the vectorized engine."""
    stock = int(initial)
    in_transit = []  # (arrival_day, units)
    on_hand, lost, orders = [], [], []

    def order(target):
        position = stock + sum(units for _, units in in_transit)
        quantity = max(0, math.ceil(target - position - 1e-9))
        orders.append(quantity)
        return quantity

    in_transit.append((lead_time + 1, order(targets[0])))
    review = 1

    for day, units_demanded in enumerate(demand, start=1):
        stock += sum(units for arrival, units in in_transit if arrival == day)
        in_transit = [(arrival, units) for arrival, units in in_transit if arrival != day]

        sold = min(stock, units_demanded)
        lost.append(units_demanded - sold)
        stock -= sold
        on_hand.append(stock)

        if review < len(targets) and day == review * period:
            in_transit.append((day + lead_time + 1, order(targets[review])))
            review += 1

    return on_hand, lost, orders


@pytest.mark.parametrize("lead_time, period", [(0, 1), (1, 7), (3, 7), (7, 7), (9, 4)])
def test_matches_an_independent_plain_python_simulator(lead_time, period):
    rng = np.random.default_rng(100 + lead_time * 10 + period)
    n_series, n_days = 12, 45
    demand = rng.poisson(rng.uniform(0.2, 6.0, size=(n_series, 1)), size=(n_series, n_days))
    targets = rng.uniform(0, 50, size=(n_series, len(review_times(n_days, period))))
    initial = rng.integers(0, 15, size=n_series)

    result = simulate_periodic_review(
        demand,
        targets,
        lead_time_days=lead_time,
        review_period_days=period,
        initial_on_hand=initial,
    )

    for i in range(n_series):
        on_hand, lost, orders = _reference_single_series(
            demand[i].tolist(), targets[i].tolist(), lead_time, period, initial[i]
        )
        np.testing.assert_array_equal(result.on_hand[i], on_hand)
        np.testing.assert_array_equal(result.lost[i], lost)
        np.testing.assert_array_equal(result.orders[i], orders)


def test_series_are_simulated_independently():
    rng = np.random.default_rng(3)
    demand = rng.poisson(2.0, size=(3, 28))
    targets = rng.uniform(0, 25, size=(3, 4))

    joint = simulate_periodic_review(demand, targets, lead_time_days=2)

    for i in range(3):
        alone = simulate_periodic_review(
            demand[i : i + 1], targets[i : i + 1], lead_time_days=2
        )
        np.testing.assert_array_equal(alone.on_hand[0], joint.on_hand[i])
        np.testing.assert_array_equal(alone.orders[0], joint.orders[i])


def test_order_quantity_rounds_up_without_floating_point_noise():
    np.testing.assert_array_equal(
        order_quantity([3.2, 3.0 + 1e-12, 0.0, 0.001, 2.0], [0, 0, 0, 0, 5]),
        [4, 3, 0, 1, 0],
    )


def test_order_quantity_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        order_quantity([np.nan], [0])


def test_order_quantity_rejects_values_that_would_overflow():
    with pytest.raises(ValueError, match="supported range"):
        order_quantity([1e20], [0])


@pytest.mark.parametrize(
    "demand, targets, initial",
    [
        ([[MAX_UNITS + 1, 0]], [[1]], None),
        ([[1, 0]], [[MAX_UNITS + 1]], None),
        ([[1, 0]], [[1]], [MAX_UNITS + 1]),
    ],
)
def test_values_outside_the_supported_range_are_rejected(demand, targets, initial):
    with pytest.raises(ValueError, match="supported range"):
        simulate_periodic_review(
            demand, targets, lead_time_days=1, initial_on_hand=initial
        )


@pytest.mark.parametrize(
    "demand, targets, kwargs, message",
    [
        ([[-1, 2]], [[1]], {}, "nonnegative whole"),
        ([[1.5, 2]], [[1]], {}, "nonnegative whole"),
        ([[np.nan, 2]], [[1]], {}, "finite"),
        ([1, 2], [[1]], {}, "series-by-day"),
        ([[1, 2]], [[np.nan]], {}, "finite"),
        ([[1, 2]], [[-1]], {}, "negative"),
        ([[1, 2]], [[1, 1]], {}, "shape"),
        ([[1, 2]], [[1]], {"lead_time_days": -1}, "Lead time"),
        ([[1, 2]], [[1]], {"lead_time_days": 1.5}, "Lead time"),
        ([[1, 2]], [[1]], {"review_period_days": 0}, "Review period"),
        ([[1, 2]], [[1]], {"initial_on_hand": [-1]}, "Initial on-hand"),
        ([[1, 2]], [[1]], {"initial_on_hand": [1, 2]}, "one value per series"),
    ],
)
def test_invalid_inputs_are_rejected(demand, targets, kwargs, message):
    arguments = {"lead_time_days": 1, "review_period_days": 7, **kwargs}

    with pytest.raises(ValueError, match=message):
        simulate_periodic_review(demand, targets, **arguments)
