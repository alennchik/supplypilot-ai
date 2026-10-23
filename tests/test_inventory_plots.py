import pandas as pd
import pytest

from scripts.plot_inventory_tradeoff import draw_figures, figure_paths
from src.inventory_plots import THEMES, plot_fill_rate_vs_stock


def make_reports():
    rows = []
    for method, start_fill, start_stock in (
        ("lightgbm", 0.78, 21_000),
        ("ma28", 0.83, 23_000),
    ):
        for step, z in enumerate((0.0, 1.0, 2.0, 3.0)):
            rows.append(
                {
                    "variant": "primary",
                    "lead_time": 7,
                    "method": method,
                    "z": z,
                    "series_count": 3,
                    "evaluated_days": 84,
                    "fill_rate": start_fill + 0.045 * step,
                    "average_on_hand_units": start_stock + 9_000 * step,
                }
            )
    curves = pd.DataFrame(rows)
    reference = pd.DataFrame(
        [
            {
                "variant": "primary",
                "lead_time": 7,
                "method": "perfect_foresight",
                "fill_rate": 1.0,
                "average_on_hand_units": 14_000.0,
            }
        ]
    )
    endpoints = pd.DataFrame(
        [
            {
                "variant": "primary",
                "lead_time": 7,
                "target_fill_rate": target,
                "method": method,
                "status": status,
                "inventory": inventory,
                "relative_difference_lightgbm_vs_ma28": difference,
            }
            for target, method, status, inventory, difference in (
                (0.9, "lightgbm", "interpolated", 39_000.0, 0.18),
                (0.9, "ma28", "interpolated", 33_050.0, 0.18),
                (0.95, "lightgbm", "not_reached", None, None),
                (0.95, "ma28", "interpolated", 50_000.0, None),
            )
        ]
    )
    return curves, reference, endpoints


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_chart_is_written_and_reproducible(tmp_path, theme):
    curves, reference, endpoints = make_reports()
    first, second = tmp_path / "first.svg", tmp_path / "second.svg"

    plot_fill_rate_vs_stock(curves, reference, endpoints, first, theme=theme)
    plot_fill_rate_vs_stock(curves, reference, endpoints, second, theme=theme)

    assert b"<svg" in first.read_bytes()
    assert first.read_bytes() == second.read_bytes()


def test_chart_draws_when_no_method_reaches_the_comparison_target(tmp_path):
    curves, reference, endpoints = make_reports()
    endpoints.loc[endpoints.target_fill_rate == 0.9, "status"] = "not_reached"
    output = tmp_path / "chart.png"

    plot_fill_rate_vs_stock(curves, reference, endpoints, output)

    assert output.stat().st_size > 0


def test_unknown_theme_is_refused(tmp_path):
    curves, reference, endpoints = make_reports()
    output = tmp_path / "chart.svg"

    with pytest.raises(ValueError, match="Theme must be one of"):
        plot_fill_rate_vs_stock(curves, reference, endpoints, output, theme="sepia")

    assert not output.exists()


def test_a_missing_method_is_refused(tmp_path):
    curves, reference, endpoints = make_reports()
    output = tmp_path / "chart.svg"

    with pytest.raises(ValueError, match="No ma28 curve"):
        plot_fill_rate_vs_stock(
            curves[curves.method == "lightgbm"], reference, endpoints, output
        )

    assert not output.exists()


def test_a_missing_reference_is_refused(tmp_path):
    curves, reference, endpoints = make_reports()

    with pytest.raises(ValueError, match="perfect-foresight row"):
        plot_fill_rate_vs_stock(
            curves, reference.iloc[0:0], endpoints, tmp_path / "chart.svg"
        )


def test_script_writes_both_themes_from_the_report_files(tmp_path):
    report_dir, figure_dir = tmp_path / "primary", tmp_path / "figures"
    report_dir.mkdir()
    curves, reference, endpoints = make_reports()
    curves.to_csv(report_dir / "curves.csv", index=False)
    reference.to_csv(report_dir / "reference.csv", index=False)
    endpoints.to_csv(report_dir / "endpoints.csv", index=False)

    paths = draw_figures(report_dir, figure_dir)

    assert paths == figure_paths(figure_dir, 7)
    assert all(path.read_bytes().count(b"<svg") == 1 for path in paths.values())
