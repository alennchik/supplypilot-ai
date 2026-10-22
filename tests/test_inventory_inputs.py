"""Synthetic tests for Phase 3C inputs that do not need LightGBM."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.backtesting import run_backtesting
from src.inventory_inputs import (
    FIRST_REVIEW,
    ISSUE28_ARTIFACTS,
    ISSUE28_ORIGINS,
    MA28_MODEL,
    REVIEW_ORIGINS,
    check_paths_match,
    cohort_sha256,
    demand_matrix,
    file_sha256,
    frozen_cohort,
    ma28_paths,
    paths_from_predictions,
    simulated_days,
    value_proxies,
    verify_input_files,
    verify_issue28_artifacts,
)

SERIES = ["CA_food", "TX_hobby", "WI_house"]


@pytest.fixture
def sales():
    dates = pd.date_range("2015-08-01", "2016-04-24")
    rows = [
        {
            "series_id": name,
            "date": date,
            "quantity": (day * (i + 3) + day // 5) % 17,
            "price": 1.0 + i + (day % 3) * 0.25,
            "price_available": True,
        }
        for i, name in enumerate(SERIES)
        for day, date in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def test_review_calendar_matches_the_frozen_protocol():
    assert len(REVIEW_ORIGINS) == 16
    assert REVIEW_ORIGINS[0] == pd.Timestamp("2016-01-03")
    assert REVIEW_ORIGINS[-1] == pd.Timestamp("2016-04-17")
    assert all(origin.day_name() == "Sunday" for origin in REVIEW_ORIGINS)
    assert set(ISSUE28_ORIGINS) <= set(REVIEW_ORIGINS)

    days = simulated_days()
    assert len(days) == 112
    assert (days[0], days[-1]) == (pd.Timestamp("2016-01-04"), pd.Timestamp("2016-04-24"))


def test_cohort_hash_uses_the_issue28_canonical_form():
    expected = hashlib.sha256(b"series_id\na\nb\nc\n").hexdigest()

    assert cohort_sha256(["c", "a", "b"]) == expected
    assert cohort_sha256(["a", "b", "c"]) == expected


@pytest.fixture
def artifacts(tmp_path):
    for name in ISSUE28_ARTIFACTS:
        (tmp_path / name).write_bytes(f"content of {name}".encode())

    manifest = {
        "artifact_sha256": {
            name: file_sha256(tmp_path / name) for name in ISSUE28_ARTIFACTS
        },
        "cohort_sha256": cohort_sha256(SERIES),
        "execution": {"inputs": {}},
    }
    (tmp_path / "preflight.json").write_text(json.dumps(manifest))
    return tmp_path


def test_artifacts_with_recorded_hashes_are_accepted(artifacts):
    manifest = verify_issue28_artifacts(artifacts)

    assert set(manifest["artifact_sha256"]) == set(ISSUE28_ARTIFACTS)


def test_a_changed_artifact_is_rejected(artifacts):
    (artifacts / "model.txt").write_bytes(b"retrained")

    with pytest.raises(ValueError, match="artifact changed: model.txt"):
        verify_issue28_artifacts(artifacts)


def test_a_missing_artifact_is_rejected(artifacts):
    (artifacts / "lookups.parquet").unlink()

    with pytest.raises(ValueError, match="Missing Issue #28 artifact"):
        verify_issue28_artifacts(artifacts)


def test_input_files_must_match_recorded_hashes(tmp_path):
    data = tmp_path / "sales.parquet"
    data.write_bytes(b"original")
    manifest = {"execution": {"inputs": {"sales": {"sha256": file_sha256(data)}}}}

    verify_input_files(manifest, {"sales": data})

    data.write_bytes(b"edited")
    with pytest.raises(ValueError, match="Input differs"):
        verify_input_files(manifest, {"sales": data})

    with pytest.raises(ValueError, match="No recorded"):
        verify_input_files(manifest, {"calendar": data})


def test_frozen_cohort_checks_count_and_hash():
    manifest = {"cohort_sha256": cohort_sha256(SERIES)}
    lookups = pd.DataFrame({"series_id": ["WI_house", "CA_food", "TX_hobby"]})

    cohort = frozen_cohort(lookups, manifest, expected=3)
    assert cohort.tolist() == sorted(SERIES)

    with pytest.raises(ValueError, match="expected 4"):
        frozen_cohort(lookups, manifest, expected=4)

    with pytest.raises(ValueError, match="hash"):
        frozen_cohort(lookups, {"cohort_sha256": "other"}, expected=3)

    with pytest.raises(ValueError, match="unique"):
        frozen_cohort(
            pd.DataFrame({"series_id": ["a", "a", "b"]}), manifest, expected=3
        )


def test_demand_matrix_covers_the_simulated_days(sales):
    demand = demand_matrix(sales, SERIES)
    days = simulated_days()

    assert demand.shape == (3, 112)
    assert demand.dtype == np.int64
    expected = (
        sales.loc[sales.series_id == "TX_hobby"].set_index("date").quantity.loc[days]
    )
    np.testing.assert_array_equal(demand[1], expected.to_numpy())


def test_demand_matrix_rejects_missing_days(sales):
    gap = sales.drop(
        sales.index[(sales.series_id == "CA_food") & (sales.date == "2016-02-10")]
    )

    with pytest.raises(ValueError, match="Missing sales"):
        demand_matrix(gap, SERIES)


def test_ma28_paths_match_the_phase1_backtest_exactly(sales):
    phase1 = run_backtesting(sales[["series_id", "date", "quantity"]], "2016-04-24")
    reference = paths_from_predictions(
        phase1.loc[phase1.model == MA28_MODEL],
        SERIES,
        ISSUE28_ORIGINS,
        value_column="predicted",
    )

    paths = ma28_paths(sales, SERIES, ISSUE28_ORIGINS)

    assert check_paths_match(paths, reference, tolerance=0.0, label="MA28") == 0.0
    np.testing.assert_array_equal(paths, reference)


def test_ma28_paths_use_only_sales_through_each_origin(sales):
    origin = REVIEW_ORIGINS[5]
    baseline = ma28_paths(sales, SERIES, REVIEW_ORIGINS)

    changed = sales.copy()
    changed.loc[changed.date > origin, "quantity"] = 99
    perturbed = ma28_paths(changed, SERIES, REVIEW_ORIGINS)

    np.testing.assert_array_equal(perturbed[:, :6], baseline[:, :6])
    assert not np.array_equal(perturbed[:, 6:], baseline[:, 6:])
    assert baseline.shape == (3, 16, 28)


def _long_predictions():
    rows = [
        {"series_id": s, "origin": o, "horizon": h, "predicted": float(h)}
        for s in SERIES
        for o in ISSUE28_ORIGINS
        for h in range(1, 29)
    ]
    return pd.DataFrame(rows)


def test_paths_from_predictions_places_every_value():
    paths = paths_from_predictions(
        _long_predictions(), SERIES, ISSUE28_ORIGINS, value_column="predicted"
    )

    assert paths.shape == (3, 4, 28)
    np.testing.assert_array_equal(paths[2, 3], np.arange(1, 29, dtype=float))


@pytest.mark.parametrize(
    "damage, message",
    [
        ("missing", "Expected"),
        ("duplicate", "Duplicate"),
        ("horizon", "horizons"),
        ("negative", "nonnegative"),
    ],
)
def test_paths_from_predictions_rejects_damage(damage, message):
    frame = _long_predictions()

    if damage == "missing":
        frame = frame.iloc[1:]
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif damage == "horizon":
        frame.loc[0, "horizon"] = 29
    else:
        frame.loc[0, "predicted"] = -1.0

    with pytest.raises(ValueError, match=message):
        paths_from_predictions(frame, SERIES, ISSUE28_ORIGINS, value_column="predicted")


def test_check_paths_match_enforces_the_tolerance():
    reference = np.ones((2, 4, 28))

    assert check_paths_match(reference + 5e-10, reference, tolerance=1e-9, label="x") < 1e-9

    with pytest.raises(ValueError, match="differ from the reference"):
        check_paths_match(reference + 1e-6, reference, tolerance=1e-9, label="LightGBM")

    with pytest.raises(ValueError, match="shapes differ"):
        check_paths_match(np.ones((2, 3, 28)), reference, tolerance=1e-9, label="x")


def test_value_proxies_use_the_median_available_price(sales):
    window = pd.date_range(end=FIRST_REVIEW, periods=28)
    data = sales.copy()
    unavailable = (data.series_id == "CA_food") & (data.date == window[0])
    data.loc[unavailable, "price_available"] = False

    values = value_proxies(data, SERIES)

    for i, name in enumerate(SERIES):
        rows = data.loc[(data.series_id == name) & data.date.isin(window)]
        rows = rows.loc[rows.price_available]
        assert values[i] == pytest.approx(rows.price.median())


def test_value_proxies_stop_when_a_series_has_no_price(sales):
    window = pd.date_range(end=FIRST_REVIEW, periods=28)
    data = sales.copy()
    data.loc[(data.series_id == "WI_house") & data.date.isin(window), "price"] = np.nan

    with pytest.raises(ValueError, match="No available price"):
        value_proxies(data, SERIES)
