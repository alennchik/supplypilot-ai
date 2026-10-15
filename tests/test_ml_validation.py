"""Tests for fixed, chronological ML validation windows."""

import pandas as pd
import pytest

from src.ml_validation import (
    build_validation_actuals,
    validation_windows,
)


@pytest.fixture
def sample_sales():
    """Two eligible series and one series introduced too late."""
    dates = pd.date_range("2015-10-01", "2016-01-03")

    records = []

    for series_id in ("A", "B", "C"):
        for day, date in enumerate(dates):
            # C has fewer than 28 days of history at V1.
            # It becomes eligible later, but must stay excluded.
            if series_id == "C" and date < pd.Timestamp("2015-11-01"):
                continue

            records.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "quantity": day % 7,
                }
            )

    return pd.DataFrame(records)


def test_validation_windows_are_fixed():
    windows = validation_windows()

    assert len(windows) == 2

    assert windows[0].origin == pd.Timestamp("2015-11-08")
    assert windows[0].start == pd.Timestamp("2015-11-09")
    assert windows[0].end == pd.Timestamp("2015-12-06")

    assert windows[1].origin == pd.Timestamp("2015-12-06")
    assert windows[1].start == pd.Timestamp("2015-12-07")
    assert windows[1].end == pd.Timestamp("2016-01-03")

    for window in windows:
        assert (window.end - window.start).days + 1 == 28


def test_fixed_validation_cohort(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    # A and B have enough history at V1; C does not.
    assert set(actuals["series_id"]) == {"A", "B"}

    # Two eligible series, two windows, 28 days each.
    assert len(actuals) == 2 * 2 * 28

    cohorts = [set(group["series_id"]) for _, group in actuals.groupby("origin")]

    assert cohorts == [{"A", "B"}, {"A", "B"}]


def test_validation_dates_and_horizons(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    assert actuals["horizon"].between(1, 28).all()

    expected_dates = actuals["origin"] + pd.to_timedelta(actuals["horizon"], unit="D")

    pd.testing.assert_series_equal(
        actuals["date"],
        expected_dates,
        check_names=False,
    )

    assert not actuals.duplicated(["series_id", "origin", "horizon"]).any()


def test_actuals_match_original_sales(sample_sales):
    actuals = build_validation_actuals(sample_sales)

    matched = actuals.merge(
        sample_sales,
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    assert matched["quantity"].notna().all()
    assert (matched["actual"] == matched["quantity"]).all()


def test_missing_future_target_raises_error(sample_sales):
    broken = sample_sales.loc[
        ~(
            (sample_sales["series_id"] == "A")
            & (sample_sales["date"] == pd.Timestamp("2015-12-25"))
        )
    ].copy()

    with pytest.raises(ValueError, match="Missing observations"):
        build_validation_actuals(broken)


def test_missing_v2_history_raises_error(sample_sales):
    broken = sample_sales.loc[
        ~(
            (sample_sales["series_id"] == "B")
            & (sample_sales["date"] == pd.Timestamp("2015-12-03"))
        )
    ].copy()

    with pytest.raises(ValueError, match="Missing observations"):
        build_validation_actuals(broken)


def test_no_eligible_series_raises_error(sample_sales):
    recent_only = sample_sales.loc[
        sample_sales["date"] >= pd.Timestamp("2015-11-01")
    ].copy()

    with pytest.raises(ValueError, match="Not enough calendar history"):
        build_validation_actuals(recent_only)


def test_validation_feature_sets_keep_fixed_v1_cohort(monkeypatch):
    """V2 must not introduce additional series beyond the V1 cohort."""
    import src.ml_validation as validation

    original_series = ["series_A", "series_B"]
    windows = validation.validation_windows()
    keys = ["series_id", "origin", "date", "horizon"]

    actual_rows = []

    for window in windows:
        for series_id in original_series:
            for horizon in range(1, 29):
                actual_rows.append(
                    {
                        "series_id": series_id,
                        "origin": window.origin,
                        "date": window.origin + pd.Timedelta(days=horizon),
                        "horizon": horizon,
                        "actual": float(horizon),
                    }
                )

    actuals = pd.DataFrame(actual_rows)

    monkeypatch.setattr(
        validation,
        "build_validation_actuals",
        lambda sales: actuals.copy(),
    )

    observed_origins = []

    def fake_inference(sales, metadata, origin):
        observed_origins.append(origin)

        series_ids = original_series.copy()

        if origin == validation.V2_ORIGIN:
            series_ids.append("new_series_C")

        rows = []

        for series_id in series_ids:
            for horizon in range(1, 29):
                rows.append(
                    {
                        "series_id": series_id,
                        "origin": origin,
                        "date": origin + pd.Timedelta(days=horizon),
                        "horizon": horizon,
                        "historical_feature": 10.0,
                    }
                )

        return pd.DataFrame(rows)

    monkeypatch.setattr(
        validation,
        "build_ml_inference_examples",
        fake_inference,
    )

    v1, v2 = validation.build_validation_feature_sets(
        sales=pd.DataFrame(),
        metadata=pd.DataFrame(),
    )

    assert observed_origins == [
        validation.V1_ORIGIN,
        validation.V2_ORIGIN,
    ]

    for frame, window in zip((v1, v2), windows, strict=True):
        assert len(frame) == 56
        assert set(frame["series_id"]) == set(original_series)
        assert frame["origin"].eq(window.origin).all()
        assert frame["target"].notna().all()
        assert frame["historical_feature"].eq(10.0).all()
        assert not frame.duplicated(keys).any()

        assert (frame["target"].to_numpy() == frame["horizon"].to_numpy()).all()

    assert "new_series_C" not in set(v2["series_id"])


@pytest.mark.parametrize("missing_from", ["features", "targets"])
def test_validation_feature_sets_reject_missing_rows(monkeypatch, missing_from):
    """Missing V2 predictions or targets must never be ignored."""
    import src.ml_validation as validation

    rows = []

    for window in validation.validation_windows():
        for horizon in range(1, 29):
            rows.append(
                {
                    "series_id": "series_A",
                    "origin": window.origin,
                    "date": window.origin + pd.Timedelta(days=horizon),
                    "horizon": horizon,
                    "actual": float(horizon),
                }
            )

    actuals = pd.DataFrame(rows)

    if missing_from == "targets":
        actuals = actuals.loc[
            ~(actuals["origin"].eq(validation.V2_ORIGIN) & actuals["horizon"].eq(28))
        ].copy()

    def fake_actuals(sales):
        return actuals.copy()

    def fake_inference(sales, metadata, origin):
        examples = pd.DataFrame(
            [
                {
                    "series_id": "series_A",
                    "origin": origin,
                    "date": origin + pd.Timedelta(days=horizon),
                    "horizon": horizon,
                    "historical_feature": 10.0,
                }
                for horizon in range(1, 29)
            ]
        )

        if missing_from == "features" and origin == validation.V2_ORIGIN:
            examples = examples.loc[examples["horizon"] != 28].copy()

        return examples

    monkeypatch.setattr(validation, "build_validation_actuals", fake_actuals)
    monkeypatch.setattr(validation, "build_ml_inference_examples", fake_inference)

    with pytest.raises(
        ValueError,
        match="Validation features and targets do not match",
    ):
        validation.build_validation_feature_sets(
            sales=pd.DataFrame(),
            metadata=pd.DataFrame(),
        )


def test_validation_features_ignore_sales_after_forecast_origins():
    """Changing future labels must never change historical features."""
    import src.ml_validation as validation
    from src.ml_dataset import METADATA_COLUMNS

    dates = pd.date_range("2015-09-01", "2016-01-03", freq="D")
    series_ids = ["series_A", "series_B"]

    sales = pd.DataFrame(
        [
            {
                "series_id": series_id,
                "date": date,
                "quantity": float((day_number + series_number) % 7 + 1),
            }
            for series_number, series_id in enumerate(series_ids)
            for day_number, date in enumerate(dates)
        ]
    )

    metadata = pd.DataFrame(
        {
            "series_id": series_ids,
            **{column: [f"{column}_A", f"{column}_B"] for column in METADATA_COLUMNS},
        }
    )

    original_v1, original_v2 = validation.build_validation_feature_sets(sales, metadata)

    changed_sales = sales.copy()

    mask = changed_sales["series_id"].eq("series_A") & changed_sales["date"].eq(
        pd.Timestamp("2015-12-20")
    )
    changed_sales.loc[mask, "quantity"] += 100.0

    changed_v1, changed_v2 = validation.build_validation_feature_sets(
        changed_sales, metadata
    )

    # V1 and V2 feature rows must be completely unchanged.
    pd.testing.assert_frame_equal(
        original_v1.drop(columns="target"),
        changed_v1.drop(columns="target"),
    )
    pd.testing.assert_frame_equal(
        original_v2.drop(columns="target"),
        changed_v2.drop(columns="target"),
    )

    # V1 labels precede the modified date.
    pd.testing.assert_series_equal(
        original_v1["target"],
        changed_v1["target"],
    )

    # December 20 is inside the V2 forecast window.
    assert not original_v2["target"].equals(changed_v2["target"])

    assert len(original_v1) == 2 * 28
    assert len(original_v2) == 2 * 28
