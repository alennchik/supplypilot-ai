"""Convert Walmart M5 sales CSV to wide-format Parquet."""

import argparse
from pathlib import Path

from src.parquet_storage import convert_sales_to_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data/raw/sales_train_validation.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/processed/sales_train_validation.parquet"
)


def main() -> None:
    """Run CSV-to-Parquet conversion from the command line."""
    parser = argparse.ArgumentParser(
        description="Convert M5 wide-format sales CSV to Parquet."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated Parquet file.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("Converting M5 sales data...")

    rows, columns = convert_sales_to_parquet(
        args.input,
        args.output,
    )

    size_mb = args.output.stat().st_size / (1024 ** 2)

    print("Conversion completed successfully!")
    print(f"Rows: {rows:,}")
    print(f"Columns: {columns:,}")
    print(f"Parquet file size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    main()
