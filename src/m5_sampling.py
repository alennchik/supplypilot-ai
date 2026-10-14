"""Deterministic stratified sampling of Walmart M5 series."""

import hashlib

import pandas as pd

METADATA_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def _stable_hash(series_id: str, seed: int) -> str:
    """Generate a stable ranking key for a series."""
    value = f"{seed}:{series_id}".encode()
    return hashlib.sha256(value).hexdigest()


def sample_series(
    metadata: pd.DataFrame,
    sample_size: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample series proportionally across store/category groups."""
    missing = set(METADATA_COLUMNS) - set(metadata.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if metadata[METADATA_COLUMNS].isna().any().any():
        raise ValueError("Metadata contains missing values")

    if metadata["id"].duplicated().any():
        raise ValueError("Duplicate series IDs found")

    if (
        not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or not 0 < sample_size <= len(metadata)
    ):
        raise ValueError("Invalid sample size")

    # Count series in each store/category group.
    counts = metadata.groupby(
        ["store_id", "cat_id"],
        sort=True,
    ).size()

    total = len(metadata)
    quotas = {}
    remainders = []

    # Calculate proportional quotas using integer arithmetic.
    for group, count in counts.items():
        numerator = int(count) * sample_size
        base, remainder = divmod(numerator, total)

        quotas[group] = base
        remainders.append((remainder, group))

    # Distribute any remaining places deterministically.
    remaining = sample_size - sum(quotas.values())

    for _, group in sorted(
        remainders,
        key=lambda entry: (-entry[0], entry[1]),
    )[:remaining]:
        quotas[group] += 1

    selected = []

    # Rank series by stable hash within each group.
    for group, frame in metadata.groupby(
        ["store_id", "cat_id"],
        sort=True,
    ):
        ranked = frame.assign(
            _sample_hash=frame["id"].map(
                lambda series_id: _stable_hash(series_id, seed)
            )
        )

        ranked = ranked.sort_values(["_sample_hash", "id"])

        selected.append(ranked.head(quotas[group]))

    result = pd.concat(selected, ignore_index=True)

    result = (
        result[METADATA_COLUMNS]
        .rename(columns={"id": "series_id"})
        .sort_values("series_id")
        .reset_index(drop=True)
    )

    return result
