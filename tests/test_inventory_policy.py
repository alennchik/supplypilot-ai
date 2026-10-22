"""Tests for order-up-to targets shared by every forecasting method."""

import numpy as np
import pytest

from src.inventory_policy import (
    perfect_foresight_protection_demand,
    poisson_scaled_order_up_to,
    protection_forecast,
    protection_interval_days,
)


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
