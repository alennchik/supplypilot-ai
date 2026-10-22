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
    PUBLISHED_REVEAL_MANIFEST,
    REVIEW_ORIGINS,
    check_paths_match,
    cohort_sha256,
    demand_matrix,
    file_sha256,
    frozen_cohort,
    ma28_paths,
    paths_from_predictions,
    simulated_days,
    snapshot_verified,
    value_proxies,
    verified_snapshots,
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


INPUT_NAMES = ("sales", "metadata", "calendar", "baselines")


def write_run(folder, label):
    """A self-consistent Issue #28 run: artifacts, inputs and a preflight.json."""
    artifacts = folder / "artifacts"
    data = folder / "data"
    artifacts.mkdir(parents=True)
    data.mkdir()

    for name in ISSUE28_ARTIFACTS:
        (artifacts / name).write_bytes(f"{label}: content of {name}".encode())

    inputs = {}
    for name in INPUT_NAMES:
        path = data / f"{name}.bin"
        path.write_bytes(f"{label}: {name} data".encode())
        inputs[name] = path

    manifest = {
        "artifact_sha256": {
            name: file_sha256(artifacts / name) for name in ISSUE28_ARTIFACTS
        },
        "cohort_sha256": cohort_sha256(SERIES),
        "execution": {
            "inputs": {name: {"sha256": file_sha256(path)} for name, path in inputs.items()}
        },
    }
    (artifacts / "preflight.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return artifacts, inputs


def write_published_record(path, artifacts):
    """The committed reveal manifest that anchors one specific run."""
    path.write_text(
        json.dumps(
            {
                "preflight_sha256": file_sha256(artifacts / "preflight.json"),
                "prediction_sha256": file_sha256(artifacts / "predictions.parquet"),
            }
        )
    )
    return path


@pytest.fixture
def run(tmp_path):
    artifacts, inputs = write_run(tmp_path / "original", "original run")
    published = write_published_record(tmp_path / "published.json", artifacts)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    return artifacts, inputs, published, snapshots


def test_a_run_traced_to_the_published_record_is_snapshotted(run):
    artifacts, inputs, published, folder = run

    manifest, snapshots, provenance = verified_snapshots(
        artifacts, inputs, folder, published
    )

    assert set(manifest["artifact_sha256"]) == set(ISSUE28_ARTIFACTS)
    assert set(snapshots) == {"preflight.json", *ISSUE28_ARTIFACTS, *INPUT_NAMES}
    for name in ISSUE28_ARTIFACTS:
        assert snapshots[name].read_bytes() == (artifacts / name).read_bytes()
    assert snapshots["sales"].read_bytes() == inputs["sales"].read_bytes()
    assert provenance["published_manifest_sha256"] == file_sha256(published)
    assert provenance["input_sha256"]["calendar"] == file_sha256(inputs["calendar"])
    assert set(provenance["artifact_sha256"]) == set(ISSUE28_ARTIFACTS)


def test_a_different_self_consistent_run_is_rejected(run, tmp_path):
    _, _, published, folder = run
    other, other_inputs = write_run(tmp_path / "other", "another run")

    with pytest.raises(ValueError, match="not the one behind the published"):
        verified_snapshots(other, other_inputs, folder, published)


def test_a_preflight_whose_prediction_hash_differs_from_the_published_one_is_rejected(
    run,
):
    artifacts, inputs, published, folder = run
    record = json.loads(published.read_text())
    record["prediction_sha256"] = "0" * 64
    published.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="prediction hash differs"):
        verified_snapshots(artifacts, inputs, folder, published)


def test_a_missing_published_record_is_rejected(run, tmp_path):
    artifacts, inputs, _, folder = run

    with pytest.raises(ValueError, match="Missing published"):
        verified_snapshots(artifacts, inputs, folder, tmp_path / "absent.json")


def test_a_changed_artifact_is_rejected(run):
    artifacts, inputs, published, folder = run
    (artifacts / "model.txt").write_bytes(b"retrained")

    with pytest.raises(ValueError, match="artifact changed: model.txt"):
        verified_snapshots(artifacts, inputs, folder, published)


def test_a_missing_artifact_is_rejected(run):
    artifacts, inputs, published, folder = run
    (artifacts / "lookups.parquet").unlink()

    with pytest.raises(ValueError, match="Missing file"):
        verified_snapshots(artifacts, inputs, folder, published)


def test_a_changed_input_is_rejected(run):
    artifacts, inputs, published, folder = run
    inputs["sales"].write_bytes(b"edited sales")

    with pytest.raises(ValueError, match="Input differs from the Issue #28 run: sales"):
        verified_snapshots(artifacts, inputs, folder, published)


def test_an_input_without_a_recorded_hash_is_rejected(run):
    artifacts, inputs, published, folder = run

    with pytest.raises(ValueError, match="No recorded Issue #28 hash"):
        verified_snapshots(
            artifacts, {**inputs, "extra": inputs["sales"]}, folder, published
        )


def test_a_snapshot_keeps_the_verified_bytes_after_the_original_changes(tmp_path):
    original = tmp_path / "sales.parquet"
    original.write_bytes(b"verified bytes")
    copy = snapshot_verified(
        original, file_sha256(original), tmp_path / "snapshot", mismatch="changed"
    )

    original.write_bytes(b"edited later")

    assert copy.read_bytes() == b"verified bytes"

    with pytest.raises(ValueError, match="changed"):
        snapshot_verified(
            original, "0" * 64, tmp_path / "second", mismatch="changed"
        )
    assert not (tmp_path / "second").exists()


def test_the_committed_published_record_anchors_the_chain():
    record = json.loads(PUBLISHED_REVEAL_MANIFEST.read_text())

    assert record["preflight_sha256"] == (
        "1c3bafd437e173042650065c20b4054d33163d97372413c21338e5c993783371"
    )
    assert record["prediction_sha256"] == (
        "5b5317ef50bf2fcde8e635ac4380cf4f12d164c37b21e409d714cb2254ee0211"
    )


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


@pytest.mark.parametrize("tolerance", [np.nan, -1e-9, np.inf])
def test_check_paths_match_rejects_an_invalid_tolerance(tolerance):
    reference = np.ones((2, 4, 28))

    with pytest.raises(ValueError, match="Tolerance"):
        check_paths_match(reference + 100, reference, tolerance=tolerance, label="x")


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

    with pytest.raises(ValueError, match=r"No available price for series: \['WI_house'\]"):
        value_proxies(data, SERIES)
