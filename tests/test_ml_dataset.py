import itertools

import numpy as np
import pandas as pd
import pytest

from src.ml_dataset import (
    build_ml_training_dataset,
    select_training_origins,
)


def make_data(days=100):
    dates = pd.date_range("2015-01-01", periods=days)

    sales = pd.DataFrame(
        {
            "series_id": np.repeat(["one", "two"], days),
            "date": np.tile(dates, 2),
            "quantity": np.r_[
                np.arange(days) % 7,
                np.zeros(days),
            ].astype(int),
        }
    )

    metadata = pd.DataFrame(
        {
            "series_id": ["two", "one"],
            "store_id": ["S2", "S1"],
            "dept_id": ["D2", "D1"],
            "cat_id": ["HOBBIES", "FOODS"],
        }
    )

    return sales, metadata, dates


def test_origins_relative_to_cutoff():
    cutoff = pd.Timestamp("2016-01-01")
    origins = select_training_origins(cutoff, 8)

    assert len(origins) == 8
    assert origins[-1] == cutoff - pd.Timedelta(days=28)
    assert origins[0] == cutoff - pd.Timedelta(days=77)

    assert all((b - a).days == 7 for a, b in itertools.pairwise(origins))

    later = select_training_origins(cutoff + pd.Timedelta(days=56), 8)

    assert later[-1] == origins[-1] + pd.Timedelta(days=56)


def test_origins_input_validation():
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            select_training_origins("2016-01-01", invalid)

    with pytest.raises(ValueError, match="calendar date"):
        select_training_origins("2016-01-01T13:00:00")


def test_generate_28_horizons_and_metadata():
    sales, metadata, dates = make_data()

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=2)

    assert len(result) == 2 * 2 * 28
    assert not result.duplicated(["series_id", "origin", "horizon"]).any()

    assert sorted(result["horizon"].unique()) == list(range(1, 29))

    assert (
        result["date"]
        == result["origin"] + pd.to_timedelta(result["horizon"], unit="D")
    ).all()

    assert (result["date"] <= dates[-1]).all()

    one = result.loc[
        (result["series_id"] == "one") & (result["origin"] == result["origin"].min())
    ]

    assert (one["store_id"] == "S1").all()

    actual = sales.loc[
        (sales["series_id"] == "one") & sales["date"].isin(one["date"]),
        "quantity",
    ]

    assert one["target"].tolist() == actual.tolist()

    two = result.loc[result["series_id"] == "two"]
    assert (two["target"] == 0).all()
    assert (two["zero_fraction_28"] == 1).all()


def test_future_change_does_not_modify_earlier_features():
    sales, metadata, dates = make_data()
    cutoff = dates[-1]

    original = build_ml_training_dataset(sales, metadata, cutoff, n_origins=2)

    first_origin = original["origin"].min()

    changed = sales.copy()
    changed.loc[changed["date"] > first_origin, "quantity"] = 123

    modified = build_ml_training_dataset(changed, metadata, cutoff, n_origins=2)

    first_rows = original["origin"] == first_origin

    feature_columns = [
        column for column in original.columns if column not in {"target", "date"}
    ]

    pd.testing.assert_frame_equal(
        original.loc[first_rows, feature_columns].reset_index(drop=True),
        modified.loc[first_rows, feature_columns].reset_index(drop=True),
    )

    assert not original.loc[first_rows, "target"].equals(
        modified.loc[first_rows, "target"]
    )


def test_training_cutoff_cannot_exceed_data():
    sales, metadata, dates = make_data()

    with pytest.raises(ValueError, match="cutoff extends"):
        build_ml_training_dataset(
            sales,
            metadata,
            dates[-1] + pd.Timedelta(days=1),
        )


def test_ineligible_short_history_skipped():
    sales, metadata, dates = make_data()

    sales = sales.loc[~((sales["series_id"] == "two") & (sales["date"] < dates[80]))]

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=1)

    assert set(result["series_id"]) == {"one"}


def test_missing_target_for_eligible_raises():
    sales, metadata, dates = make_data()

    sales = sales.loc[~((sales["series_id"] == "one") & (sales["date"] == dates[-2]))]

    with pytest.raises(ValueError, match="Missing target"):
        build_ml_training_dataset(sales, metadata, dates[-1], n_origins=1)


def test_missing_recent_history_excluded():
    sales, metadata, dates = make_data()
    origin = dates[-29]

    sales = sales.loc[~((sales["series_id"] == "two") & (sales["date"] == origin))]

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=1)

    assert set(result["series_id"]) == {"one"}


def test_missing_long_window_is_not_zero():
    sales, metadata, dates = make_data()
    origin = dates[-29]

    sales = sales.loc[
        ~(
            (sales["series_id"] == "one")
            & (sales["date"] == origin - pd.Timedelta(days=40))
        )
    ]

    result = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=1)

    one = result.loc[result["series_id"] == "one"]

    assert one["rolling_mean_56"].isna().all()
    assert one["rolling_mean_28"].notna().all()


def test_metadata_must_match_sales():
    sales, metadata, dates = make_data()

    with pytest.raises(ValueError, match="same series IDs"):
        build_ml_training_dataset(
            sales,
            metadata.loc[metadata["series_id"] == "one"],
            dates[-1],
            n_origins=1,
        )

    duplicates = pd.concat(
        [metadata, metadata.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_ml_training_dataset(sales, duplicates, dates[-1], n_origins=1)


def test_rows_after_training_cutoff_do_not_change_training_data():
    sales, metadata, dates = make_data()
    cutoff = dates[-10]

    original = build_ml_training_dataset(sales, metadata, cutoff, n_origins=2)

    changed = sales.copy()
    changed.loc[changed["date"] > cutoff, "quantity"] = 900

    modified = build_ml_training_dataset(changed, metadata, cutoff, n_origins=2)

    pd.testing.assert_frame_equal(modified, original)


def test_cutoff_moves_weekly_training_origins():
    sales, metadata, dates = make_data()

    first = build_ml_training_dataset(sales, metadata, dates[-8], n_origins=1)

    second = build_ml_training_dataset(sales, metadata, dates[-1], n_origins=1)

    assert second["origin"].unique()[0] - first["origin"].unique()[0] == pd.Timedelta(
        days=7
    )

    assert first["date"].max() <= dates[-8]
    assert second["date"].max() <= dates[-1]
