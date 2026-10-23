"""Draw the Phase 3 fill-rate-versus-stock chart from the published reports.

    .venv/bin/python -m scripts.plot_inventory_tradeoff

Reads reports/phase3_inventory/primary/ (curves, reference and endpoints) and
writes a light and a dark SVG to reports/phase3_inventory/figures/. It reads
only the published reports: nothing is simulated or recomputed.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.inventory_plots import THEMES, plot_fill_rate_vs_stock

REPORT_DIR = Path("reports/phase3_inventory/primary")
FIGURE_DIR = Path("reports/phase3_inventory/figures")
SOURCE_NOTE = (
    "Source: reports/phase3_inventory/primary "
    "(one registered run of the frozen Phase 3 protocol v1.0)"
)


def figure_paths(figure_dir: Path, lead_time: int) -> dict[str, Path]:
    return {
        theme: figure_dir / f"fill_rate_vs_stock_l{lead_time}_{theme}.svg"
        for theme in THEMES
    }


def draw_figures(report_dir: Path, figure_dir: Path, lead_time: int = 7) -> dict:
    curves = pd.read_csv(report_dir / "curves.csv")
    reference = pd.read_csv(report_dir / "reference.csv")
    endpoints = pd.read_csv(report_dir / "endpoints.csv")

    paths = figure_paths(figure_dir, lead_time)
    for theme, path in paths.items():
        plot_fill_rate_vs_stock(
            curves,
            reference,
            endpoints,
            path,
            theme=theme,
            lead_time=lead_time,
            source_note=SOURCE_NOTE,
        )

    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--lead-time", type=int, default=7)
    args = parser.parse_args(argv)

    for path in draw_figures(args.report_dir, args.figure_dir, args.lead_time).values():
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
