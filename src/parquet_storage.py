"""Convert M5 wide-format sales CSV files to Parquet."""

import os
import tempfile
from pathlib import Path

import pandas as pd

ID_COLUMNS = (
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)

MIN_SALES = 0
MAX_INT16 = 32767


def convert_sales_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
) -> tuple[int, int]:
    """Convert an M5 wide CSV to Parquet without losing sales values."""
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    # Inspect the schema without loading the full dataset.
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    missing = [col for col in ID_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"Missing identifier columns: {missing}")

    day_columns = [col for col in columns if col.startswith("d_") and col[2:].isdigit()]
    if not day_columns:
        raise ValueError("No daily sales columns found")

    unexpected = [
        col for col in columns if col not in ID_COLUMNS and col not in day_columns
    ]
    if unexpected:
        raise ValueError(f"Unexpected columns: {unexpected}")

    # Use int32 initially so values can be validated before
    # narrowing them to int16.
    dtypes = {col: "string" for col in ID_COLUMNS}
    dtypes.update({col: "int32" for col in day_columns})

    try:
        df = pd.read_csv(csv_path, dtype=dtypes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid or missing sales values in CSV. "
            "Daily sales must be non-missing integers "
            "within the supported range."
        ) from exc

    if df.empty:
        raise ValueError("Sales dataset is empty")

    if df[list(ID_COLUMNS)].isna().any().any():
        raise ValueError("Identifier columns contain missing values")

    min_sales = int(df[day_columns].min().min())
    max_sales = int(df[day_columns].max().max())

    if min_sales < MIN_SALES or max_sales > MAX_INT16:
        raise ValueError(
            f"Sales values must be between {MIN_SALES} and "
            f"{MAX_INT16}; found min={min_sales}, max={max_sales}"
        )

    # Safe to convert: the range has already been checked.
    df[day_columns] = df[day_columns].astype("int16")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file before replacing the final file.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{parquet_path.stem}-",
        suffix=".parquet.tmp",
        dir=parquet_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        df.to_parquet(
            temporary_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        os.replace(temporary_path, parquet_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return len(df), len(df.columns)
