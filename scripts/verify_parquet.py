"""Verify that M5 CSV and Parquet contain identical values."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data/raw/sales_train_validation.csv"
PARQUET_PATH = (
    PROJECT_ROOT / "data/processed/sales_train_validation.parquet"
)

BATCH_SIZE = 250


def main() -> None:
    """Compare CSV and Parquet without loading both into memory."""
    parquet_file = pq.ParquetFile(PARQUET_PATH)

    csv_columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    parquet_columns = parquet_file.schema_arrow.names

    if csv_columns != parquet_columns:
        raise ValueError("CSV and Parquet columns do not match")

    total_rows = 0

    # Each Parquet batch is compared against the same CSV rows.
    with pd.read_csv(CSV_PATH, chunksize=BATCH_SIZE) as reader:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            parquet_chunk = batch.to_pandas()
            csv_chunk = reader.get_chunk(batch.num_rows)

            try:
                pd.testing.assert_frame_equal(
                    csv_chunk.reset_index(drop=True),
                    parquet_chunk.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
            except AssertionError as error:
                raise ValueError(
                    f"Data mismatch near row {total_rows:,}"
                ) from error

            total_rows += batch.num_rows

        # Make sure the CSV has no additional rows.
        try:
            reader.get_chunk(1)
        except StopIteration:
            pass
        else:
            raise ValueError("CSV contains additional rows")

    if total_rows != parquet_file.metadata.num_rows:
        raise ValueError("Parquet row count mismatch")

    print("Verification PASSED!")
    print(f"Identical rows: {total_rows:,}")
    print(f"Identical columns: {len(csv_columns):,}")
    print("All identifiers and sales values match exactly.")


if __name__ == "__main__":
    main()
