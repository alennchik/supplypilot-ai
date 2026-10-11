
from pathlib import Path

import pandas as pd


def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Load sales data from a CSV file."""
    return pd.read_csv(path)


def reshape_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert M5 daily sales columns into rows."""
    id_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    day_columns = [
        col for col in df.columns if col.startswith("d_")
    ]

    missing_columns = [
        col for col in id_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not day_columns:
        raise ValueError("No daily sales columns found")

    return df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="day",
        value_name="sales",
    )

def process_sales_in_chunks(path: str | Path, chunk_size: int = 100):
    """Read and reshape sales data in smaller chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        yield reshape_sales_data(chunk)