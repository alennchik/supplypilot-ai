"""Build unified daily sales Parquet from sampled Walmart M5 data."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.m5_unified import build_unified_batch

SOURCE_PATH = Path("data/processed/sales_train_validation.parquet")
SERIES_PATH = Path("data/processed/series.parquet")
CALENDAR_PATH = Path("data/raw/calendar.csv")
PRICES_PATH = Path("data/raw/sell_prices.csv")
OUTPUT_PATH = Path("data/processed/sales.parquet")

SOURCE_BATCH_SIZE = 1000
PRICE_CHUNK_SIZE = 250_000

OUTPUT_SCHEMA = pa.schema(
    [
        ("series_id", pa.string()),
        ("date", pa.timestamp("us")),
        ("quantity", pa.int32()),
        ("price", pa.float64()),
        ("price_available", pa.bool_()),
    ]
)


def load_sampled_prices(series: pd.DataFrame) -> pd.DataFrame:
    """Load prices only for selected item-store combinations."""
    keys = series[["store_id", "item_id"]].drop_duplicates()
    matched_chunks = []

    reader = pd.read_csv(
        PRICES_PATH,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        chunksize=PRICE_CHUNK_SIZE,
    )

    for chunk in reader:
        matched = chunk.merge(
            keys,
            on=["store_id", "item_id"],
            how="inner",
            validate="many_to_one",
        )

        if not matched.empty:
            matched_chunks.append(matched)

    if not matched_chunks:
        raise ValueError("No prices found for sampled series")

    prices = pd.concat(matched_chunks, ignore_index=True)

    price_keys = ["store_id", "item_id", "wm_yr_wk"]

    if prices.duplicated(price_keys).any():
        raise ValueError("Duplicate weekly price keys")

    print(f"Loaded sampled price records: {len(prices):,}")

    return prices


def write_batch(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    writer: pq.ParquetWriter,
) -> int:
    """Convert and write one batch of selected series."""
    batch_keys = sales[["store_id", "item_id"]].drop_duplicates()

    batch_prices = prices.merge(
        batch_keys,
        on=["store_id", "item_id"],
        how="inner",
        validate="many_to_one",
    )

    unified = build_unified_batch(
        sales=sales,
        calendar=calendar,
        prices=batch_prices,
    )

    if unified.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate series-date observations")

    table = pa.Table.from_pandas(
        unified,
        schema=OUTPUT_SCHEMA,
        preserve_index=False,
    )

    writer.write_table(table)

    return len(unified)


def main() -> None:
    """Convert sampled M5 sales into a unified Parquet dataset."""
    for path in [
        SOURCE_PATH,
        SERIES_PATH,
        CALENDAR_PATH,
        PRICES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    series = pd.read_parquet(SERIES_PATH)

    if series["series_id"].isna().any():
        raise ValueError("Missing sampled series IDs")

    if series["series_id"].duplicated().any():
        raise ValueError("Duplicate sampled series IDs")

    selected_ids = set(series["series_id"])

    calendar = pd.read_csv(
        CALENDAR_PATH,
        usecols=["d", "date", "wm_yr_wk"],
    )

    prices = load_sampled_prices(series)

    source = pq.ParquetFile(SOURCE_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=".sales_",
        suffix=".parquet",
        dir=OUTPUT_PATH.parent,
    )
    os.close(fd)

    temporary_path = Path(temporary_name)

    seen_ids = set()
    total_rows = 0
    total_series = 0

    try:
        with pq.ParquetWriter(
            temporary_path,
            schema=OUTPUT_SCHEMA,
            compression="snappy",
        ) as writer:
            for record_batch in source.iter_batches(
                batch_size=SOURCE_BATCH_SIZE,
            ):
                source_batch = record_batch.to_pandas()

                selected = source_batch.loc[
                    source_batch["id"].isin(selected_ids)
                ].copy()

                if selected.empty:
                    continue

                batch_ids = set(selected["id"])

                if selected["id"].duplicated().any():
                    raise ValueError("Duplicate source series IDs")

                if seen_ids.intersection(batch_ids):
                    raise ValueError("Repeated series across batches")

                seen_ids.update(batch_ids)

                rows_written = write_batch(
                    sales=selected,
                    calendar=calendar,
                    prices=prices,
                    writer=writer,
                )

                total_rows += rows_written
                total_series += len(selected)

                print(f"Processed {total_series:,} series, {total_rows:,} daily rows")

            if seen_ids != selected_ids:
                missing = selected_ids - seen_ids
                raise ValueError(f"Missing {len(missing)} sampled series in source")

        # Replace the final output only after successful conversion.
        os.replace(temporary_path, OUTPUT_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)

    print("\nUnified dataset completed!")
    print(f"Series processed: {total_series:,}")
    print(f"Daily observations: {total_rows:,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
