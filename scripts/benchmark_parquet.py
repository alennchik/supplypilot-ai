"""Benchmark CSV and Parquet loading performance for Walmart M5."""

import argparse
import json
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/raw/sales_train_validation.csv"
PARQUET_PATH = ROOT / "data/processed/sales_train_validation.parquet"
REPORT_PATH = ROOT / "reports/parquet_benchmark.json"

ID_COLUMNS = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id",
]

CASES = [
    "csv_full",
    "parquet_full",
    "csv_recent",
    "parquet_recent",
]


def get_columns():
    """Find the full schema and the last 365 days."""
    columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    days = [c for c in columns if c.startswith("d_")]

    if len(days) < 365:
        raise ValueError("Dataset has fewer than 365 days")

    recent = ID_COLUMNS + days[-365:]
    return columns, recent


def peak_memory_mib():
    """Return peak process RSS in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if sys.platform == "darwin":
        return peak / (1024 ** 2)

    return peak / 1024


def worker(case):
    """Run one timed load in an isolated process."""
    columns, recent = get_columns()

    is_csv = case.startswith("csv_")
    is_recent = case.endswith("_recent")

    selected = recent if is_recent else columns

    start = time.perf_counter()

    if is_csv:
        dtypes = {
            col: "string" if col in ID_COLUMNS else "int16"
            for col in selected
        }
        frame = pd.read_csv(
            CSV_PATH,
            usecols=selected,
            dtype=dtypes,
        )
    else:
        frame = pd.read_parquet(
            PARQUET_PATH,
            columns=selected,
            engine="pyarrow",
        )

    elapsed = time.perf_counter() - start

    result = {
        "case": case,
        "seconds": elapsed,
        "peak_memory_mib": peak_memory_mib(),
        "rows": len(frame),
        "columns": len(frame.columns),
    }

    print(json.dumps(result))


def mac_info():
    """Get additional macOS hardware information."""
    if sys.platform != "darwin":
        return {}

    result = {}

    for key, name in [
        ("hw.model", "hardware_model"),
        ("hw.memsize", "ram_bytes"),
    ]:
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", key],
                text=True,
            ).strip()
            result[name] = int(value) if key == "hw.memsize" else value
        except (OSError, subprocess.CalledProcessError, ValueError):
            result[name] = None

    return result


def main():
    """Run repeated benchmarks and save the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker)
        return

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    if not CSV_PATH.is_file() or not PARQUET_PATH.is_file():
        parser.error("Run M5 conversion before benchmarking")

    samples = {case: [] for case in CASES}
    rng = random.Random(42)

    for repetition in range(args.repeats):
        order = CASES.copy()
        rng.shuffle(order)

        print(f"Repetition {repetition + 1}/{args.repeats}")

        for case in order:
            print(f"  Running {case}...", flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                     "scripts.benchmark_parquet",
                     "--worker",
                     case,
                ],
                cwd=ROOT,
                capture_output=True,
                 text=True,
                 check=True,
                 timeout=30,
            )

            result = json.loads(completed.stdout)
            samples[case].append(result)

            print(
                f"  {case}: {result['seconds']:.3f}s, "
                f"{result['peak_memory_mib']:.1f} MiB"
            )

    summary = {}

    for case, results in samples.items():
        summary[case] = {
            "median_seconds": statistics.median(
                r["seconds"] for r in results
            ),
            "median_peak_memory_mib": statistics.median(
                r["peak_memory_mib"] for r in results
            ),
            "samples": results,
        }

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            **mac_info(),
        },
        "repetitions": args.repeats,
        "methodology": (
            "Independent processes; explicit int16 CSV types; "
            "median times; process peak RSS including imports; "
            "OS filesystem caches not cleared."
        ),
        "files": {
            "csv_bytes": CSV_PATH.stat().st_size,
            "parquet_bytes": PARQUET_PATH.stat().st_size,
        },
        "summary": summary,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nBenchmark saved to {REPORT_PATH}")

    for suffix in ["full", "recent"]:
        csv_time = summary[f"csv_{suffix}"]["median_seconds"]
        parquet_time = summary[f"parquet_{suffix}"]["median_seconds"]

        print(f"{suffix} speedup: {csv_time / parquet_time:.2f}x")

    csv_size = report["files"]["csv_bytes"]
    parquet_size = report["files"]["parquet_bytes"]
    print(f"Storage reduction: {(1 - parquet_size / csv_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
