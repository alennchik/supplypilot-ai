"""Generate a reproducible ML training dataset from unified M5 Parquet files."""

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from src.ml_dataset import build_ml_training_dataset, select_training_origins

DEFAULT_CUTOFF = "2015-11-08"  # M5 d_1745, before validation windows


def _atomic_write(path: Path, writer) -> None:
    """Replace one output only after its temporary file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)

    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _peak_process_memory_mib() -> float:
    """Return process peak RSS in MiB on macOS or Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 2)


def run_experiment(
    sales_path: Path,
    metadata_path: Path,
    output_path: Path,
    manifest_path: Path,
    cutoff: str = DEFAULT_CUTOFF,
    n_origins: int = 8,
) -> dict:
    """Build, validate, and save training data and its experiment manifest."""
    sales_path = Path(sales_path)
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    paths = [sales_path, metadata_path, output_path, manifest_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input and output file paths must be distinct")

    origins = select_training_origins(cutoff, n_origins)
    cutoff_date = pd.Timestamp(cutoff)
    started = time.perf_counter()

    # Load only the columns needed for historical demand features.
    sales = pd.read_parquet(
        sales_path,
        columns=["series_id", "date", "quantity"],
    )
    metadata = pd.read_parquet(
        metadata_path,
        columns=["series_id", "store_id", "dept_id", "cat_id"],
    )

    result = build_ml_training_dataset(
        sales=sales,
        metadata=metadata,
        training_cutoff=cutoff_date,
        n_origins=n_origins,
    )

    if result.empty:
        raise ValueError("No training examples were generated")

    if result["target"].isna().any():
        raise ValueError("Generated training labels contain missing values")

    if not (result["date"] <= cutoff_date).all():
        raise ValueError("Generated dataset contains targets after cutoff")

    _atomic_write(
        output_path,
        lambda temporary: result.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        ),
    )

    counts = result.groupby("origin", sort=True)["series_id"].nunique()
    elapsed = round(time.perf_counter() - started, 2)

    manifest = {
        "training_cutoff": cutoff_date.date().isoformat(),
        "n_origins": n_origins,
        "origins": [date.date().isoformat() for date in origins],
        "forecast_horizon_days": 28,
        "rows": len(result),
        "unique_series": int(result["series_id"].nunique()),
        "eligible_series_by_origin": {
            date.date().isoformat(): int(count) for date, count in counts.items()
        },
        "feature_columns": [
            name
            for name in result.columns
            if name
            not in {
                "series_id",
                "origin",
                "date",
                "horizon",
                "store_id",
                "dept_id",
                "cat_id",
                "target",
            }
        ],
        "target_sum": float(result["target"].to_numpy(dtype=np.float64).sum()),
        "output_file": str(output_path),
        "output_size_mib": round(output_path.stat().st_size / (1024**2), 2),
        "elapsed_seconds": elapsed,
        "peak_process_rss_mib": _peak_process_memory_mib(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }

    _atomic_write(
        manifest_path,
        lambda temporary: temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    )

    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sales",
        type=Path,
        default=Path("data/processed/sales.parquet"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/series.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ml_training.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/ml_training_manifest.json"),
    )
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--origins", type=int, default=8)

    args = parser.parse_args(argv)

    manifest = run_experiment(
        sales_path=args.sales,
        metadata_path=args.metadata,
        output_path=args.output,
        manifest_path=args.manifest,
        cutoff=args.cutoff,
        n_origins=args.origins,
    )

    print(f"Training cutoff: {manifest['training_cutoff']}")
    print(f"Weekly origins: {manifest['n_origins']}")
    print(f"Training rows: {manifest['rows']:,}")
    print(f"Unique series: {manifest['unique_series']:,}")
    print(f"Peak process RSS: {manifest['peak_process_rss_mib']:.2f} MiB")
    print(f"Elapsed: {manifest['elapsed_seconds']:.2f} s")
    print(f"Saved: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
