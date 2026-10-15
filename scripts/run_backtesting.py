"""Run the reproducible 28-day M5 forecasting benchmark."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from src.backtesting import make_forecast_windows, run_backtesting
from src.evaluation import build_demand_profile, evaluate_forecasts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_END_DATE = "2016-04-24"

REPORT_NAMES = {
    "overall": "m5_baselines_overall.csv",
    "by_window": "m5_baselines_by_window.csv",
    "by_category": "m5_baselines_by_category.csv",
    "by_sparsity": "m5_baselines_by_sparsity.csv",
}


def write_atomic(destination: Path, writer) -> None:
    """Replace a single output only after writing succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)

    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_experiment(
    sales_path: Path,
    series_path: Path,
    predictions_path: Path,
    reports_dir: Path,
    end_date: str = DEFAULT_END_DATE,
) -> dict:
    """Generate baseline predictions, grouped metrics, and run metadata."""
    print("Loading daily sales and series metadata...")

    sales = pd.read_parquet(
        sales_path,
        columns=["series_id", "date", "quantity"],
    )
    series = pd.read_parquet(
        series_path,
        columns=["series_id", "cat_id"],
    )

    windows = make_forecast_windows(end_date)
    first_origin = windows[0].origin

    print("Running leakage-free rolling-origin backtesting...")
    predictions = run_backtesting(
        sales,
        evaluation_end_date=end_date,
    )

    # Only series accepted by the backtesting engine enter evaluation.
    eligible_series = pd.Index(predictions["series_id"].unique())

    print("Building historical demand groups...")
    profile = build_demand_profile(
        sales,
        eligible_series,
        first_origin,
    )

    print("Calculating forecast accuracy...")
    results = evaluate_forecasts(
        predictions,
        series,
        profile,
    )

    group_summary = (
        profile.groupby("demand_group", sort=True)
        .agg(
            series_count=("series_id", "size"),
            mean_zero_fraction=("zero_fraction", "mean"),
            mean_observed_days=("observed_days", "mean"),
        )
        .reset_index()
    )
    group_summary["series_percentage"] = (
        group_summary["series_count"] / len(profile) * 100
    )

    manifest = {
        "evaluation_end_date": windows[-1].end.date().isoformat(),
        "horizon_days": 28,
        "window_count": len(windows),
        "eligible_series": len(eligible_series),
        "excluded_series": (int(sales["series_id"].nunique()) - len(eligible_series)),
        "prediction_rows": len(predictions),
        "models": sorted(predictions["model"].unique().tolist()),
        "demand_lookback_days": 365,
        "demand_group_method": "fixed_zero_fraction_thresholds",
        "demand_group_thresholds": {
            "frequent": "zero_fraction < 0.25",
            "mixed": "0.25 <= zero_fraction <= 0.75",
            "sparse": "zero_fraction > 0.75",
        },
        "windows": [
            {
                "origin": window.origin.date().isoformat(),
                "start": window.start.date().isoformat(),
                "end": window.end.date().isoformat(),
            }
            for window in windows
        ],
    }

    # Finish calculations before publishing output files.
    print("Saving predictions and reports...")

    write_atomic(
        predictions_path,
        lambda path: predictions.to_parquet(path, index=False),
    )

    for key, filename in REPORT_NAMES.items():
        write_atomic(
            reports_dir / filename,
            lambda path, frame=results[key]: frame.to_csv(path, index=False),
        )

    write_atomic(
        reports_dir / "m5_demand_group_summary.csv",
        lambda path: group_summary.to_csv(path, index=False),
    )

    write_atomic(
        reports_dir / "m5_backtesting_manifest.json",
        lambda path: path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        ),
    )

    print("\n=== OVERALL BASELINE RESULTS ===")
    print(
        results["overall"].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n=== DEMAND GROUPS ===")
    print(group_summary.to_string(index=False))

    print(f"\nEligible series: {len(eligible_series):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Reports saved to: {reports_dir}")

    return manifest


def main(argv=None) -> None:
    """Parse command-line arguments and execute the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run 28-day rolling-origin forecasting benchmarks."
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sales.parquet",
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=PROJECT_ROOT / "data/processed/series.parquet",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "data/processed/m5_backtest_predictions.parquet",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
    )

    args = parser.parse_args(argv)

    run_experiment(
        sales_path=args.sales,
        series_path=args.series,
        predictions_path=args.predictions,
        reports_dir=args.reports_dir,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
