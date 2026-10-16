"""Calendar-only validation and the seven locked target-date features."""

import numpy as np
import pandas as pd
import pytest

from src.calendar_features import (
    CALENDAR_CATEGORICAL_FEATURES,
    CALENDAR_FEATURE_NAMES,
    CALENDAR_NUMERIC_FEATURES,
    attach_calendar_features,
    build_calendar_features,
    validate_calendar,
)


def calendar_table(start="2015-12-07", end="2016-01-03"):
    dates = pd.date_range(start, end)
    return pd.DataFrame(
        {
            "date": dates,
            "event_name_1": pd.Series([None] * len(dates), dtype=object),
            "event_type_1": pd.Series([None] * len(dates), dtype=object),
            "event_name_2": pd.Series([None] * len(dates), dtype=object),
            "event_type_2": pd.Series([None] * len(dates), dtype=object),
            "snap_CA": np.ones(len(dates), dtype=np.int8),
            "snap_TX": np.zeros(len(dates), dtype=np.int8),
            "snap_WI": np.arange(len(dates)) % 2,
        }
    )


def example_rows():
    origin = pd.Timestamp("2015-12-06")
    return pd.DataFrame(
        [
            {
                "series_id": state,
                "store_id": f"{state}_1",
                "origin": origin,
                "horizon": horizon,
                "date": origin + pd.Timedelta(days=horizon),
            }
            for state in ("CA", "TX", "WI")
            for horizon in range(1, 29)
        ]
    )


def test_exact_feature_names_order_types_and_month_boundary():
    examples = example_rows()
    result = build_calendar_features(examples, calendar_table())
    assert (
        tuple(result.columns)
        == CALENDAR_FEATURE_NAMES
        == (
            "target_month",
            "target_day_of_month",
            "target_is_event",
            "target_event_type_1",
            "target_event_type_2",
            "target_is_christmas",
            "target_is_snap",
        )
    )
    assert len(CALENDAR_NUMERIC_FEATURES) == 5
    assert len(CALENDAR_CATEGORICAL_FEATURES) == 2
    assert result.index.equals(examples.index)
    assert (result["target_month"] == examples["date"].dt.month).all()
    assert (result["target_day_of_month"] == examples["date"].dt.day).all()
    assert result.loc[27, "target_month"] == 1
    assert result.loc[27, "target_day_of_month"] == 3
    assert not {"weekday", "state_id", "event_name_1", "event_name_2"} & set(
        result.columns
    )


def test_both_event_slots_and_explicit_none_level():
    calendar = calendar_table()
    calendar.loc[0, ["event_name_1", "event_type_1"]] = ["One", "National"]
    calendar.loc[1, ["event_name_2", "event_type_2"]] = ["Two", "Sporting"]
    calendar.loc[2, ["event_name_1", "event_type_1"]] = ["One", "Cultural"]
    calendar.loc[2, ["event_name_2", "event_type_2"]] = ["Two", "Religious"]
    calendar.loc[3, ["event_name_1", "event_type_1"]] = ["  ", ""]
    result = build_calendar_features(example_rows(), calendar)
    assert result["target_is_event"].iloc[:5].tolist() == [1, 1, 1, 0, 0]
    assert result["target_event_type_1"].iloc[:4].tolist() == [
        "National",
        "None",
        "Cultural",
        "None",
    ]
    assert result["target_event_type_2"].iloc[:4].tolist() == [
        "None",
        "Sporting",
        "Religious",
        "None",
    ]
    assert not result.loc[:, list(CALENDAR_CATEGORICAL_FEATURES)].isna().any().any()
    renamed = calendar.copy()
    renamed.loc[0, "event_name_1"] = "Different identity"
    pd.testing.assert_frame_equal(
        result, build_calendar_features(example_rows(), renamed)
    )


def test_calendar_only_extraction_may_have_categorical_event_columns():
    calendar = calendar_table()
    calendar.loc[0, ["event_name_1", "event_type_1"]] = ["Example", "National"]
    expected = build_calendar_features(example_rows(), calendar)
    for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        calendar[column] = calendar[column].astype("category")
    pd.testing.assert_frame_equal(
        expected, build_calendar_features(example_rows(), calendar)
    )


@pytest.mark.parametrize(
    "column", ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
)
def test_nonstring_event_slots_rejected(column):
    calendar = calendar_table()
    calendar.loc[0, column] = 17
    with pytest.raises(ValueError, match="strings or missing"):
        build_calendar_features(example_rows(), calendar)


@pytest.mark.parametrize("slot", [1, 2])
@pytest.mark.parametrize("only", ["name", "type"])
def test_incomplete_event_slot_rejected(slot, only):
    calendar = calendar_table()
    calendar.loc[0, f"event_{only}_{slot}"] = "Present"
    with pytest.raises(ValueError, match=f"slot {slot}"):
        build_calendar_features(example_rows(), calendar)


def test_christmas_exact_date_independent_of_event_names():
    calendar = calendar_table("2015-12-07", "2016-12-25")
    examples = example_rows()
    result = build_calendar_features(examples, calendar)
    expected = examples["date"].eq(pd.Timestamp("2015-12-25")).astype(np.int8)
    pd.testing.assert_series_equal(
        result["target_is_christmas"], expected, check_names=False
    )
    assert result["target_is_christmas"].sum() == 3
    assert result["target_is_event"].sum() == 0


def test_snap_by_store_prefix_and_validated_explicit_state():
    examples = example_rows()
    calendar = calendar_table()
    by_prefix = build_calendar_features(examples, calendar)
    for state, expected in (("CA", [1] * 28), ("TX", [0] * 28), ("WI", [0, 1] * 14)):
        assert (
            by_prefix.loc[examples["series_id"].eq(state), "target_is_snap"].tolist()
            == expected
        )
    metadata = examples.loc[:, ["series_id", "store_id"]].drop_duplicates()
    metadata["state_id"] = metadata["series_id"]
    pd.testing.assert_frame_equal(
        by_prefix, build_calendar_features(examples, calendar, metadata)
    )
    assert "state_id" not in attach_calendar_features(examples, calendar, metadata)


@pytest.mark.parametrize("state", ["NY", None, "ca", ""])
def test_invalid_explicit_state_rejected(state):
    examples = example_rows()
    examples["state_id"] = examples["series_id"]
    examples.loc[0, "state_id"] = state
    with pytest.raises(ValueError, match="state_id|metadata"):
        build_calendar_features(examples, calendar_table())


def test_inconsistent_explicit_state_rejected_even_if_all_snap_values_match():
    examples = example_rows()
    metadata = examples.loc[:, ["series_id", "store_id"]].drop_duplicates()
    metadata["state_id"] = metadata["series_id"]
    metadata.loc[metadata["series_id"].eq("CA"), "state_id"] = "TX"
    with pytest.raises(ValueError, match="Inconsistent state_id"):
        build_calendar_features(examples, calendar_table(), metadata)


@pytest.mark.parametrize("column", ["state_id", "store_id"])
def test_nullable_example_mapping_cannot_bypass_metadata_consistency(column):
    examples = example_rows()
    examples["state_id"] = examples["series_id"]
    metadata = examples.loc[:, ["series_id", "store_id", "state_id"]].drop_duplicates()
    examples[column] = examples[column].astype("string")
    examples.loc[0, column] = pd.NA
    with pytest.raises(ValueError, match="Inconsistent"):
        build_calendar_features(examples, calendar_table(), metadata)


@pytest.mark.parametrize("source", ["examples", "metadata"])
@pytest.mark.parametrize("invalid", [None, "", "  ", 17])
def test_series_ids_must_be_nonempty_strings(source, invalid):
    examples = example_rows()
    metadata = examples.loc[:, ["series_id", "store_id"]].drop_duplicates()
    selected = examples if source == "examples" else metadata
    selected["series_id"] = selected["series_id"].astype(object)
    selected.iloc[0, selected.columns.get_loc("series_id")] = invalid
    with pytest.raises(ValueError, match="series IDs must be nonempty strings"):
        build_calendar_features(examples, calendar_table(), metadata)


def test_duplicate_metadata_columns_rejected():
    examples = example_rows()
    metadata = examples.loc[:, ["series_id", "store_id"]].drop_duplicates()
    metadata = pd.concat([metadata, metadata[["store_id"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate calendar metadata columns"):
        build_calendar_features(examples, calendar_table(), metadata)


@pytest.mark.parametrize("store", ["NY_1", "S1", "CA", "CA_", None])
def test_unsupported_store_mapping_rejected(store):
    examples = example_rows()
    examples.loc[examples["series_id"].eq("CA"), "store_id"] = store
    with pytest.raises(ValueError, match="store-to-state"):
        build_calendar_features(examples, calendar_table())


@pytest.mark.parametrize("state", ["CA", "TX", "WI"])
@pytest.mark.parametrize("invalid", [None, -1, 2, 0.5, "invalid", np.inf])
def test_all_snap_columns_require_binary_values(state, invalid):
    calendar = calendar_table()
    calendar[f"snap_{state}"] = calendar[f"snap_{state}"].astype(object)
    calendar.loc[0, f"snap_{state}"] = invalid
    with pytest.raises(ValueError, match=f"snap_{state}.*binary"):
        build_calendar_features(example_rows(), calendar)


def test_calendar_requires_unique_complete_dates_and_preserves_inputs():
    examples = example_rows().iloc[::-1].copy()
    examples.index = np.arange(len(examples)) * 3
    calendar = calendar_table().iloc[::-1].copy()
    expected_examples, expected_calendar = examples.copy(), calendar.copy()
    result = build_calendar_features(examples, calendar)
    assert result.index.equals(examples.index)
    assert result["target_day_of_month"].tolist() == examples["date"].dt.day.tolist()
    pd.testing.assert_frame_equal(examples, expected_examples)
    pd.testing.assert_frame_equal(calendar, expected_calendar)
    with pytest.raises(ValueError, match="Duplicate calendar dates"):
        build_calendar_features(examples, pd.concat([calendar, calendar.iloc[[0]]]))
    with pytest.raises(ValueError, match="Missing calendar coverage"):
        build_calendar_features(examples, calendar.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate calendar columns"):
        validate_calendar(pd.concat([calendar, calendar[["date"]]], axis=1))


@pytest.mark.parametrize(
    "invalid",
    ["not-a-date", "2015-12-32", "2015-12-07T01:00:00", "2015-12-07T00:00:00Z", None],
)
def test_invalid_calendar_date_rejected(invalid):
    calendar = calendar_table()
    calendar["date"] = calendar["date"].astype(object)
    calendar.loc[0, "date"] = invalid
    with pytest.raises(ValueError, match="calendar dates"):
        build_calendar_features(example_rows(), calendar)


def test_missing_calendar_columns_and_empty_calendar_rejected():
    with pytest.raises(ValueError, match="Missing required calendar columns"):
        build_calendar_features(
            example_rows(), calendar_table().drop(columns="event_type_2")
        )
    with pytest.raises(ValueError, match="Calendar cannot be empty"):
        build_calendar_features(example_rows(), calendar_table().iloc[:0])


def test_sunday_origins_make_weekday_redundant_and_other_origins_rejected():
    examples = example_rows()
    assert (examples["date"].dt.dayofweek == (examples["horizon"] - 1) % 7).all()
    shifted = examples.copy()
    shifted["origin"] += pd.Timedelta(days=1)
    shifted["date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Sunday"):
        build_calendar_features(shifted, calendar_table("2015-12-08", "2016-01-04"))


@pytest.mark.parametrize("horizon", [0, 29, 1.5, np.nan, True, "invalid"])
def test_horizon_validation(horizon):
    examples = example_rows()
    examples["horizon"] = examples["horizon"].astype(object)
    examples.loc[0, "horizon"] = horizon
    with pytest.raises(ValueError, match="integer from 1 to 28"):
        build_calendar_features(examples, calendar_table())


def test_target_date_alignment_duplicate_keys_and_feature_overwrite_rejected():
    examples = example_rows()
    wrong = examples.copy()
    wrong.loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match=r"origin \+ horizon"):
        build_calendar_features(wrong, calendar_table())
    with pytest.raises(ValueError, match="Duplicate calendar example keys"):
        build_calendar_features(
            pd.concat([examples, examples.iloc[[0]]]), calendar_table()
        )
    once = attach_calendar_features(examples, calendar_table())
    with pytest.raises(ValueError, match="already exist"):
        attach_calendar_features(once, calendar_table())
