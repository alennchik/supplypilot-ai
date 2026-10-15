"""Synthetic tests for leakage-free rolling-origin backtesting."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    build_sales_matrix,
    make_forecast_windows,
    run_backtesting,
    select_eligible_series,
)

EVAL_END = "2016-04-24"
FIRST_ORIGIN = pd.Timestamp("2016-01-03")


@pytest.fixture
def sales():
    dates = pd.date_range(end=EVAL_END, periods=140, freq="D")

    return pd.DataFrame(
        {
            "series_id": np.repeat(["A", "B"], len(dates)),
            "date": np.tile(dates, 2),
            "quantity": np.concatenate(
                [
                    np.arange(140) % 7,
                    np.full(140, 2, dtype=int),
                ]
            ),
        }
    )


def test_four_windows_include_leap_day():
    windows = make_forecast_windows(EVAL_END)

    assert [w.origin.strftime("%Y-%m-%d") for w in windows] == [
        "2016-01-03",
        "2016-01-31",
        "2016-02-28",
        "2016-03-27",
    ]

    assert windows[2].start == pd.Timestamp("2016-02-29")
    assert windows[2].end == pd.Timestamp("2016-03-27")

    for window in windows:
        assert (window.end - window.start).days + 1 == 28


def test_matrix_keeps_zeros_and_missing_dates(sales):
    sales = sales.loc[~((sales.series_id == "B") & (sales.date == FIRST_ORIGIN))]

    matrix = build_sales_matrix(sales)

    assert matrix.dtypes.eq(np.float32).all()
    assert np.isnan(matrix.loc["B", FIRST_ORIGIN])
    assert matrix.loc["A", pd.Timestamp("2015-12-07")] == 0


def test_ineligible_series_excluded_using_only_first_origin(sales):
    sales = sales.loc[~((sales.series_id == "B") & (sales.date == FIRST_ORIGIN))]

    selected = select_eligible_series(
        build_sales_matrix(sales),
        FIRST_ORIGIN,
    )

    assert selected.tolist() == ["A"]

    predictions = run_backtesting(sales, EVAL_END)

    assert predictions.series_id.unique().tolist() == ["A"]


def test_predictions_have_correct_shape_and_values(sales):
    result = run_backtesting(sales, EVAL_END)

    assert len(result) == 2 * 4 * 28 * 3

    assert result.columns.tolist() == [
        "series_id",
        "origin",
        "date",
        "model",
        "actual",
        "predicted",
    ]

    assert not result.duplicated(["series_id", "origin", "date", "model"]).any()

    assert result.date.min() == pd.Timestamp("2016-01-04")
    assert result.date.max() == pd.Timestamp(EVAL_END)

    assert set(result.model) == {
        "naive",
        "seasonal_naive",
        "moving_average_28",
    }

    assert result.loc[result.series_id.eq("B"), "predicted"].eq(2).all()


def test_future_changes_do_not_change_first_origin_predictions(sales):
    before = run_backtesting(sales, EVAL_END)

    changed = sales.copy()
    changed.loc[changed.date > FIRST_ORIGIN, "quantity"] = 999

    after = run_backtesting(changed, EVAL_END)

    columns = ["series_id", "date", "model", "predicted"]

    before_first = before.loc[before.origin.eq(FIRST_ORIGIN), columns].reset_index(
        drop=True
    )

    after_first = after.loc[after.origin.eq(FIRST_ORIGIN), columns].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        before_first,
        after_first,
    )

    assert not before.loc[before.origin.eq(FIRST_ORIGIN), "actual"].equals(
        after.loc[after.origin.eq(FIRST_ORIGIN), "actual"]
    )


def test_later_origin_uses_newly_available_actuals(sales):
    before = run_backtesting(sales, EVAL_END)

    changed = sales.copy()
    changed.loc[
        (changed.series_id == "A") & (changed.date == pd.Timestamp("2016-01-31")),
        "quantity",
    ] = 99

    after = run_backtesting(changed, EVAL_END)

    def second_origin_naive(result):
        return result.loc[
            result.series_id.eq("A")
            & result.origin.eq(pd.Timestamp("2016-01-31"))
            & result.model.eq("naive"),
            "predicted",
        ]

    assert (second_origin_naive(before) != second_origin_naive(after)).all()


def test_missing_target_date_is_error(sales):
    sales = sales.loc[
        ~((sales.series_id == "A") & (sales.date == pd.Timestamp("2016-02-29")))
    ]

    with pytest.raises(ValueError, match="Missing observations"):
        run_backtesting(sales, EVAL_END)


def test_missing_end_date_is_error(sales):
    sales = sales.loc[sales.date < pd.Timestamp(EVAL_END)]

    with pytest.raises(ValueError, match="Missing calendar dates"):
        run_backtesting(sales, EVAL_END)


def test_duplicate_key_is_error(sales):
    sales = pd.concat(
        [sales, sales.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_sales_matrix(sales)


@pytest.mark.parametrize(
    "quantity",
    [-1, np.nan, 0.5, 2**24 + 1],
)
def test_invalid_quantities_are_rejected(sales, quantity):
    changed = sales.copy()
    changed["quantity"] = changed["quantity"].astype(float)
    changed.loc[0, "quantity"] = quantity

    with pytest.raises(ValueError):
        build_sales_matrix(changed)


def test_no_eligible_series_fails(sales):
    sales = sales.loc[~sales.date.eq(FIRST_ORIGIN)]

    with pytest.raises(ValueError, match="No series"):
        run_backtesting(sales, EVAL_END)


def test_insufficient_global_history_fails(sales):
    sales = sales.loc[sales.date > FIRST_ORIGIN - pd.Timedelta(days=25)]

    with pytest.raises(
        ValueError,
        match="Not enough calendar history",
    ):
        run_backtesting(sales, EVAL_END)
