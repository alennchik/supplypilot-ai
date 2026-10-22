"""Tests for order-up-to targets shared by every forecasting method."""

import numpy as np
import pytest

from src.inventory_policy import (
    initial_stock_covering_lead_time,
    perfect_foresight_protection_demand,
    poisson_scaled_order_up_to,
    protection_forecast,
    protection_interval_days,
)
from src.inventory_simulation import review_times, simulate_periodic_review


@pytest.mark.parametrize(
    "lead_time, review_period, expected",
    [(7, 7, 14), (3, 7, 10), (1, 7, 8), (0, 7, 7)],
)
def test_protection_interval_is_lead_time_plus_review_period(
    lead_time, review_period, expected
):
    assert protection_interval_days(lead_time, review_period) == expected


@pytest.mark.parametrize("lead_time, review_period", [(-1, 7), (7, 0), (1.5, 7), (True, 7)])
def test_protection_interval_rejects_invalid_days(lead_time, review_period):
    with pytest.raises(ValueError):
        protection_interval_days(lead_time, review_period)


def test_protection_forecast_sums_only_the_first_p_horizons():
    paths = np.arange(1, 29, dtype=float).reshape(1, 1, 28)

    np.testing.assert_allclose(protection_forecast(paths, 14), [[105.0]])
    np.testing.assert_allclose(protection_forecast(paths, 8), [[36.0]])


def test_protection_forecast_ignores_values_beyond_the_interval():
    paths = np.ones((2, 3, 28))
    paths[:, :, 14:] = np.nan

    np.testing.assert_allclose(protection_forecast(paths, 14), np.full((2, 3), 14.0))


@pytest.mark.parametrize(
    "paths, protection, message",
    [
        (np.ones((1, 1, 7)), 8, "shorter than the protection interval"),
        (-np.ones((1, 1, 28)), 8, "negative"),
        (np.full((1, 1, 28), np.inf), 8, "finite"),
        (np.ones((1, 28)), 8, "series, reviews, horizons"),
        (np.ones((1, 1, 28)), 0, "Protection interval"),
    ],
)
def test_protection_forecast_rejects_invalid_inputs(paths, protection, message):
    with pytest.raises(ValueError, match=message):
        protection_forecast(paths, protection)


def test_poisson_scaled_order_up_to():
    forecasts = np.array([0.0, 4.0, 9.0])

    np.testing.assert_allclose(poisson_scaled_order_up_to(forecasts, 0), forecasts)
    np.testing.assert_allclose(poisson_scaled_order_up_to(forecasts, 2), [0, 8, 15])


def test_zero_forecast_gets_no_safety_stock():
    assert poisson_scaled_order_up_to([0.0], 3.0)[0] == 0.0


@pytest.mark.parametrize(
    "forecasts, z",
    [([-1.0], 1.0), ([np.nan], 1.0), ([1.0], -0.5), ([1.0], np.inf), ([1.0], True), ([1.0], "a")],
)
def test_poisson_scaled_order_up_to_rejects_invalid_inputs(forecasts, z):
    with pytest.raises(ValueError):
        poisson_scaled_order_up_to(forecasts, z)


def test_perfect_foresight_sums_the_protection_interval():
    demand = np.arange(1, 11, dtype=float).reshape(1, 10)

    # R = 7, L = 1, P = 8: reviews at times 0 and 7.
    # Time 0 covers days 1..8; time 7 covers days 8..15, truncated at day 10.
    targets = perfect_foresight_protection_demand(demand, lead_time_days=1)

    np.testing.assert_allclose(targets, [[36.0, 8 + 9 + 10]])


def test_perfect_foresight_rejects_negative_demand():
    with pytest.raises(ValueError, match="nonnegative"):
        perfect_foresight_protection_demand([[1, -1]], lead_time_days=1)


def test_perfect_foresight_rejects_fractional_demand():
    with pytest.raises(ValueError, match="whole units"):
        perfect_foresight_protection_demand([[1, 0.5]], lead_time_days=1)


def test_initial_stock_covers_exactly_the_first_lead_time_days():
    demand = [[1, 2, 3, 4], [0, 0, 5, 0]]

    np.testing.assert_array_equal(initial_stock_covering_lead_time(demand, 2), [3, 0])
    np.testing.assert_array_equal(initial_stock_covering_lead_time(demand, 0), [0, 0])
    np.testing.assert_array_equal(initial_stock_covering_lead_time(demand, 10), [10, 5])


@pytest.mark.parametrize("lead_time", [-1, 1.5])
def test_initial_stock_rejects_invalid_lead_time(lead_time):
    with pytest.raises(ValueError, match="Lead time"):
        initial_stock_covering_lead_time([[1, 2]], lead_time)


def _stock_needed_until_next_delivery(demand, lead_time, period=7):
    """Demand from tomorrow until the day before the next scheduled delivery."""
    n_days = demand.shape[1]
    arrivals = review_times(n_days, period) + lead_time + 1
    cumulative = np.concatenate(
        [np.zeros((demand.shape[0], 1)), demand.cumsum(axis=1)], axis=1
    )
    needed = np.zeros_like(demand)

    for day in range(1, n_days + 1):
        later = arrivals[arrivals > day]
        last_day = min(later[0] - 1, n_days) if later.size else n_days
        needed[:, day - 1] = cumulative[:, last_day] - cumulative[:, day]

    return needed


@pytest.mark.parametrize("lead_time", [0, 1, 3, 7])
def test_perfect_foresight_holds_the_minimum_stock_for_zero_lost_sales(lead_time):
    """With lead-time start stock, it ends each day holding exactly what must
    be sold before the next delivery: no zero-loss policy with the same
    delivery days can hold less on any day."""
    rng = np.random.default_rng(20 + lead_time)
    demand = rng.poisson(rng.uniform(0.05, 5.0, size=(60, 1)), size=(60, 112))

    result = simulate_periodic_review(
        demand,
        perfect_foresight_protection_demand(demand, lead_time_days=lead_time),
        lead_time_days=lead_time,
        initial_on_hand=initial_stock_covering_lead_time(demand, lead_time),
    )

    assert result.lost.sum() == 0
    np.testing.assert_array_equal(
        result.on_hand, _stock_needed_until_next_delivery(demand, lead_time)
    )


@pytest.mark.parametrize("lead_time, changed_days", [(7, []), (3, [109, 110, 111, 112])])
def test_unknown_demand_after_the_span_only_changes_late_reference_stock(
    lead_time, changed_days
):
    rng = np.random.default_rng(4)
    longer = rng.poisson(2.0, size=(50, 140))
    span = longer[:, :112]
    reviews = len(review_times(112, 7))
    start = initial_stock_covering_lead_time(span, lead_time)

    truncated = simulate_periodic_review(
        span,
        perfect_foresight_protection_demand(span, lead_time_days=lead_time),
        lead_time_days=lead_time,
        initial_on_hand=start,
    )
    known_future = simulate_periodic_review(
        span,
        perfect_foresight_protection_demand(longer, lead_time_days=lead_time)[
            :, :reviews
        ],
        lead_time_days=lead_time,
        initial_on_hand=start,
    )

    differs = np.flatnonzero((truncated.on_hand != known_future.on_hand).any(axis=0))

    assert (differs + 1).tolist() == changed_days
    np.testing.assert_array_equal(truncated.sales, known_future.sales)
