"""Tests for deterministic M5 sampling."""

import pandas as pd
import pytest

from src.m5_sampling import sample_series


@pytest.fixture
def metadata():
    """Create 12 synthetic series across three groups."""
    rows = []

    groups = [
        ("CA_1", "FOODS", 6),
        ("CA_1", "HOBBIES", 3),
        ("TX_1", "FOODS", 3),
    ]

    for store_id, cat_id, count in groups:
        for i in range(count):
            item_id = f"{cat_id}_{i:03d}"

            rows.append(
                {
                    "id": f"{item_id}_{store_id}_validation",
                    "item_id": item_id,
                    "dept_id": f"{cat_id}_1",
                    "cat_id": cat_id,
                    "store_id": store_id,
                    "state_id": store_id[:2],
                }
            )

    return pd.DataFrame(rows)


def test_exact_sample_size(metadata):
    result = sample_series(metadata, sample_size=6)

    assert len(result) == 6
    assert result["series_id"].nunique() == 6


def test_proportional_allocation(metadata):
    result = sample_series(metadata, sample_size=6)

    counts = result.groupby(["store_id", "cat_id"]).size()

    assert counts[("CA_1", "FOODS")] == 3
    assert counts[("CA_1", "HOBBIES")] == 2
    assert counts[("TX_1", "FOODS")] == 1


def test_reproducible_after_row_shuffle(metadata):
    first = sample_series(metadata, sample_size=6, seed=42)

    shuffled = metadata.sample(
        frac=1,
        random_state=123,
    )

    second = sample_series(
        shuffled,
        sample_size=6,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_output_schema(metadata):
    result = sample_series(metadata, sample_size=6)

    assert result.columns.tolist() == [
        "series_id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    assert result["series_id"].is_unique


@pytest.mark.parametrize("sample_size", [0, -1, 13, True])
def test_invalid_sample_size(metadata, sample_size):
    with pytest.raises(ValueError, match="Invalid sample size"):
        sample_series(metadata, sample_size=sample_size)


def test_missing_metadata_column(metadata):
    invalid = metadata.drop(columns=["cat_id"])

    with pytest.raises(
        ValueError,
        match="Missing metadata columns",
    ):
        sample_series(invalid, sample_size=6)


def test_missing_metadata_value(metadata):
    invalid = metadata.copy()
    invalid.loc[0, "store_id"] = None

    with pytest.raises(
        ValueError,
        match="Metadata contains missing values",
    ):
        sample_series(invalid, sample_size=6)


def test_duplicate_series_id(metadata):
    invalid = pd.concat(
        [metadata, metadata.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate series IDs",
    ):
        sample_series(invalid, sample_size=6)
