"""Generate the reproducible M5 series sample."""

from pathlib import Path

import pandas as pd

from src.m5_sampling import METADATA_COLUMNS, sample_series

SOURCE_PATH = Path("data/processed/sales_train_validation.parquet")
SERIES_PATH = Path("data/processed/series.parquet")
SAMPLE_IDS_PATH = Path("reports/sample_series.csv")

SAMPLE_SIZE = 3000
SEED = 42


def main() -> None:
    """Generate and save sampled series metadata and IDs."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source Parquet not found: {SOURCE_PATH}")

    metadata = pd.read_parquet(
        SOURCE_PATH,
        columns=METADATA_COLUMNS,
    )

    sampled = sample_series(
        metadata,
        sample_size=SAMPLE_SIZE,
        seed=SEED,
    )

    sampled_ids = sampled[["series_id"]]

    # Protect an existing sample manifest from accidental changes.
    if SAMPLE_IDS_PATH.exists():
        existing_ids = pd.read_csv(
            SAMPLE_IDS_PATH,
            dtype={"series_id": "string"},
        )
        if not existing_ids.equals(sampled_ids.astype({"series_id": "string"})):
            raise ValueError("Existing sample manifest differs from generated IDs")

    SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)

    sampled.to_parquet(
        SERIES_PATH,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    sampled_ids.to_csv(SAMPLE_IDS_PATH, index=False)

    print(f"Selected series: {len(sampled)}")
    print(f"Seed: {SEED}")
    print(f"Series metadata: {SERIES_PATH}")
    print(f"Sample manifest: {SAMPLE_IDS_PATH}")


if __name__ == "__main__":
    main()
