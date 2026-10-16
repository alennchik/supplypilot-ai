"""Synthetic evidence for the locked Cycle 2 aggregation and decision rules."""

import numpy as np
import pandas as pd
import pytest

from src.cycle2_metrics import (
    CYCLE2_CANDIDATES,
    FROZEN_CYCLE1_ITERATION,
    FROZEN_CYCLE1_NAME,
    FROZEN_CYCLE1_WAPE,
    aggregation_metrics,
    align_forecasts,
    select_cycle2,
    validate_scored,
)
from src.ml_validation import V2_ORIGIN


@pytest.fixture
def scored():
    return pd.DataFrame(
        [
            {
                "series_id": series,
                "origin": V2_ORIGIN,
                "date": V2_ORIGIN + pd.Timedelta(days=horizon),
                "horizon": horizon,
                "target": 10.0,
                "prediction": 10.0,
            }
            for series in ("A", "B")
            for horizon in range(1, 29)
        ]
    )


def summary(wape8=79.0, wape16=78.0):
    return pd.DataFrame(
        [
            {
                "candidate": name,
                "objective": "tweedie",
                "n_origins": n_origins,
                "v2_wape": wape8 if n_origins == 8 else wape16,
                "best_iteration": 123 if n_origins == 8 else 456,
            }
            for name, n_origins in CYCLE2_CANDIDATES.items()
        ]
    )


def test_validation_and_metrics_do_not_mutate_input(scored):
    original = scored.copy(deep=True)
    assert validate_scored(scored) is None
    results = aggregation_metrics(scored)
    assert set(results) == {
        "daily_wape",
        "weekly_wape",
        "period_28_wape",
        "mae",
        "rmse",
        "week_1_wape",
        "week_2_wape",
        "week_3_wape",
        "week_4_wape",
    }
    assert all(value == 0 for value in results.values())
    pd.testing.assert_frame_equal(scored, original)


def test_opposite_daily_errors_cancel_within_one_series_week(scored):
    scored.loc[
        (scored["series_id"] == "A") & (scored["horizon"] == 1), "prediction"
    ] = 20.0
    scored.loc[
        (scored["series_id"] == "A") & (scored["horizon"] == 2), "prediction"
    ] = 0.0
    results = aggregation_metrics(scored)
    assert results["daily_wape"] == pytest.approx(100 * 20 / 560)
    assert results["mae"] == pytest.approx(20 / 56)
    assert results["rmse"] == pytest.approx(np.sqrt(200 / 56))
    assert results["weekly_wape"] == 0.0
    assert results["period_28_wape"] == 0.0
    assert all(results[f"week_{week}_wape"] == 0.0 for week in range(1, 5))


def test_opposite_series_errors_never_cancel(scored):
    scored.loc[scored["series_id"].eq("A"), "prediction"] = 20.0
    scored.loc[scored["series_id"].eq("B"), "prediction"] = 0.0
    results = aggregation_metrics(scored)
    assert results["daily_wape"] == 100.0
    assert results["weekly_wape"] == 100.0
    assert results["period_28_wape"] == 100.0
    assert all(results[f"week_{week}_wape"] == 100.0 for week in range(1, 5))


def test_four_week_buckets_and_period_aggregation(scored):
    scored["prediction"] = ((scored["horizon"] - 1) // 7) * 10.0
    results = aggregation_metrics(scored)
    for week, expected in enumerate((100.0, 0.0, 100.0, 200.0), start=1):
        assert results[f"week_{week}_wape"] == expected
    assert results["daily_wape"] == 100.0
    assert results["weekly_wape"] == 100.0
    assert results["period_28_wape"] == 50.0


def test_cancellation_between_weeks_occurs_only_in_period_metric(scored):
    scored.loc[scored["horizon"].between(1, 7), "prediction"] = 20.0
    scored.loc[scored["horizon"].between(8, 14), "prediction"] = 0.0
    results = aggregation_metrics(scored)
    assert results["daily_wape"] == 50.0
    assert results["weekly_wape"] == 50.0
    assert results["period_28_wape"] == 0.0


def test_wape_is_pooled_by_units_and_uses_unrounded_predictions(scored):
    scored.loc[scored["series_id"].eq("A"), "target"] = 1.0
    scored.loc[scored["series_id"].eq("B"), "target"] = 100.0
    scored["prediction"] = scored["target"]
    scored.loc[scored["series_id"].eq("A"), "prediction"] = 1.25
    expected = 100 * (28 * 0.25) / (28 * 101)
    results = aggregation_metrics(scored)
    assert results["daily_wape"] == pytest.approx(expected)
    assert results["weekly_wape"] == pytest.approx(expected)
    assert results["period_28_wape"] == pytest.approx(expected)


def test_zero_demand_series_is_valid_when_pooled_demand_is_positive(scored):
    scored.loc[scored["series_id"].eq("A"), ["target", "prediction"]] = 0.0
    assert aggregation_metrics(scored)["daily_wape"] == 0.0


def test_zero_pooled_actual_demand_fails_explicitly(scored):
    scored["target"] = 0.0
    with pytest.raises(ValueError, match="daily_wape.*pooled actual sales are zero"):
        aggregation_metrics(scored)


@pytest.mark.parametrize("week", range(1, 5))
def test_individual_zero_demand_week_fails_explicitly(scored, week):
    scored.loc[scored["horizon"].between(7 * week - 6, 7 * week), "target"] = 0.0
    with pytest.raises(ValueError, match=f"week_{week}_wape.*pooled actual"):
        aggregation_metrics(scored)


def test_aggregation_is_independent_of_input_order(scored):
    scored["prediction"] = scored["horizon"] / 3
    assert aggregation_metrics(scored) == aggregation_metrics(
        scored.sample(frac=1, random_state=37)
    )


@pytest.mark.parametrize("column", ["target", "prediction"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf, "invalid", True])
def test_rejects_invalid_scored_values(scored, column, value):
    scored[column] = scored[column].astype(object)
    scored.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_scored(scored)


@pytest.mark.parametrize("value", [0, 29, 1.5, np.nan, np.inf, "bad", True])
def test_rejects_invalid_horizons(scored, value):
    scored["horizon"] = scored["horizon"].astype(object)
    scored.loc[0, "horizon"] = value
    with pytest.raises(ValueError):
        validate_scored(scored)


@pytest.mark.parametrize(
    "column", ["series_id", "origin", "date", "target", "prediction", "horizon"]
)
def test_rejects_missing_required_columns(scored, column):
    with pytest.raises(ValueError, match="Missing scored"):
        validate_scored(scored.drop(columns=column))


def test_rejects_empty_and_duplicate_column_frames(scored):
    with pytest.raises(ValueError, match="nonempty"):
        validate_scored(scored.iloc[:0])
    duplicated = pd.concat([scored, scored[["target"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_scored(duplicated)


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_rejects_missing_or_nonstring_ids(scored, value):
    scored["series_id"] = scored["series_id"].astype(object)
    scored.loc[0, "series_id"] = value
    with pytest.raises(ValueError):
        validate_scored(scored)


@pytest.mark.parametrize("column", ["origin", "date"])
@pytest.mark.parametrize("kind", ["invalid", "missing", "time", "timezone"])
def test_rejects_invalid_date_types(scored, column, kind):
    scored[column] = scored[column].astype(object)
    original = scored.loc[0, column]
    value = {
        "invalid": "not-a-date",
        "missing": pd.NaT,
        "time": original + pd.Timedelta(hours=1),
        "timezone": original.tz_localize("UTC"),
    }[kind]
    scored.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_scored(scored)


def test_rejects_wrong_origin_and_wrong_target_dates(scored):
    wrong_origin = scored.copy()
    wrong_origin["origin"] = V2_ORIGIN + pd.Timedelta(days=28)
    with pytest.raises(ValueError, match="fixed V2 origin"):
        validate_scored(wrong_origin)
    scored.loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="origin plus horizon"):
        validate_scored(scored)


def test_rejects_duplicate_keys_and_incomplete_individual_series(scored):
    duplicate = pd.concat([scored, scored.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_scored(duplicate)
    with pytest.raises(ValueError, match="all 28"):
        validate_scored(scored.iloc[1:])


def test_date_strings_are_valid_and_alignment_preserves_diagnostics(scored):
    forecasts = scored.sample(frac=1, random_state=42)
    forecasts["date"] = forecasts["date"].dt.strftime("%Y-%m-%d")
    forecasts["origin"] = forecasts["origin"].dt.strftime("%Y-%m-%d")
    forecasts["prediction"] = forecasts["prediction"] + 0.25
    forecasts["diagnostic"] = "calendar"
    before = forecasts.copy(deep=True)
    aligned = align_forecasts(scored, forecasts)
    assert aligned["series_id"].tolist() == scored["series_id"].tolist()
    assert aligned["prediction"].eq(10.25).all()
    assert aligned["diagnostic"].eq("calendar").all()
    pd.testing.assert_frame_equal(forecasts, before)


def test_alignment_requires_identical_complete_key_set(scored):
    changed = scored.copy()
    changed.loc[changed["series_id"].eq("A"), "series_id"] = "C"
    with pytest.raises(ValueError, match="identical V2 key set"):
        align_forecasts(scored, changed)
    with pytest.raises(ValueError, match="identical V2 key set"):
        align_forecasts(scored, scored.loc[scored["series_id"].eq("A")])


def test_alignment_compares_targets_exactly_without_tolerance(scored):
    changed = scored.copy()
    changed.loc[0, "target"] = np.nextafter(changed.loc[0, "target"], np.inf)
    with pytest.raises(ValueError, match="identical V2 target"):
        align_forecasts(scored, changed)


@pytest.mark.parametrize("which", ["reference", "forecasts"])
def test_alignment_validates_both_inputs(scored, which):
    invalid = scored.iloc[1:]
    with pytest.raises(ValueError, match="all 28"):
        align_forecasts(
            invalid if which == "reference" else scored,
            invalid if which == "forecasts" else scored,
        )


def test_retains_frozen_cycle1_when_neither_candidate_qualifies():
    decision = select_cycle2(summary(80.0, 81.0))
    assert decision["candidate"] == FROZEN_CYCLE1_NAME
    assert decision["best_iteration"] == FROZEN_CYCLE1_ITERATION
    assert decision["v2_wape"] == FROZEN_CYCLE1_WAPE
    assert decision["adopted_cycle2"] is False


def test_exactly_one_point_improvement_does_not_qualify():
    threshold = FROZEN_CYCLE1_WAPE - 1.0
    decision = select_cycle2(summary(threshold, threshold))
    assert decision["candidate"] == FROZEN_CYCLE1_NAME
    assert decision["strict_adoption_threshold"] == threshold
    below = np.nextafter(threshold, -np.inf)
    assert select_cycle2(summary(below, threshold))["n_origins"] == 8
    assert select_cycle2(summary(threshold, below))["n_origins"] == 16


@pytest.mark.parametrize(
    "wape8,wape16,expected",
    [
        (79.0, 80.0, 8),
        (80.0, 79.0, 16),
        (78.0, 78.0, 8),
        (79.0, 78.0, 8),
        (78.5, 78.0, 8),
        (77.0, 78.0, 8),
        (79.1, 78.0, 16),
    ],
)
def test_adoption_and_fewer_origins_complexity_rule(wape8, wape16, expected):
    decision = select_cycle2(summary(wape8, wape16))
    assert decision["n_origins"] == expected
    assert decision["adopted_cycle2"] is True
    assert decision["best_iteration"] == (123 if expected == 8 else 456)


def test_complexity_tolerance_boundary_is_inclusive():
    assert select_cycle2(summary(79.0, 78.0))["n_origins"] == 8
    above = np.nextafter(79.0, np.inf)
    assert select_cycle2(summary(above, 78.0))["n_origins"] == 16


def test_nonqualifying_eight_origins_never_wins_complexity_tie():
    threshold = FROZEN_CYCLE1_WAPE - 1
    decision = select_cycle2(summary(threshold, threshold - 0.1))
    assert decision["n_origins"] == 16


def test_selection_ignores_diagnostics_and_is_order_independent():
    records = summary(79.1, 78.0)
    records["weekly_wape"] = [0.0, 999.0]
    records["period_28_wape"] = [0.0, np.nan]
    records["christmas_excluded_wape"] = [0.0, 999.0]
    assert select_cycle2(records)["n_origins"] == 16
    assert select_cycle2(records) == select_cycle2(records.iloc[::-1])


def test_selection_does_not_mutate_summary():
    records = summary()
    original = records.copy(deep=True)
    select_cycle2(records)
    pd.testing.assert_frame_equal(records, original)


@pytest.mark.parametrize(
    "column", ["candidate", "n_origins", "best_iteration", "v2_wape"]
)
def test_selection_requires_columns(column):
    with pytest.raises(ValueError, match="selection columns"):
        select_cycle2(summary().drop(columns=column))


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "wrong_name"])
def test_selection_requires_exact_two_candidates(kind):
    records = summary()
    if kind == "missing":
        records = records.iloc[:1]
    elif kind == "extra":
        records = pd.concat([records, records.iloc[[0]]], ignore_index=True)
    elif kind == "duplicate":
        records.loc[1, "candidate"] = records.loc[0, "candidate"]
    else:
        records.loc[0, "candidate"] = "tweedie_8_origins"
    with pytest.raises(ValueError, match="exact two"):
        select_cycle2(records)


@pytest.mark.parametrize(
    "column,value",
    [
        ("v2_wape", np.nan),
        ("v2_wape", np.inf),
        ("v2_wape", -1),
        ("v2_wape", "bad"),
        ("v2_wape", True),
        ("best_iteration", 0),
        ("best_iteration", 2001),
        ("best_iteration", 2.5),
        ("best_iteration", np.inf),
        ("best_iteration", True),
        ("n_origins", 16),
        ("n_origins", 8.5),
        ("objective", "poisson"),
    ],
)
def test_selection_rejects_invalid_candidate_metadata(column, value):
    records = summary()
    records[column] = records[column].astype(object)
    records.loc[0, column] = value
    with pytest.raises(ValueError):
        select_cycle2(records)


def test_selection_rejects_duplicate_columns():
    records = summary()
    with pytest.raises(ValueError, match="duplicate Cycle 2 selection"):
        select_cycle2(pd.concat([records, records[["v2_wape"]]], axis=1))


def test_selection_accepts_tree_cap_and_no_objective_column():
    records = summary().drop(columns="objective")
    records["best_iteration"] = 2000
    assert select_cycle2(records)["best_iteration"] == 2000
