"""Charts of the Phase 3 inventory results, drawn from the published reports.

Series colors come from a categorical palette validated for color-vision
deficiency, with separate steps for light and dark backgrounds. Text always
uses neutral ink, never a series color: each series is identified by its
legend entry and a direct label next to its line.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

METHODS = ("lightgbm", "ma28")
METHOD_LABELS = {"lightgbm": "LightGBM", "ma28": "MA28"}
ESTIMABLE = ("exact", "interpolated")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "lightgbm": "#2a78d6",
        "ma28": "#eb6834",
        "reference": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "lightgbm": "#3987e5",
        "ma28": "#d95926",
        "reference": "#c3c2b7",
    },
}

# DejaVu Sans ships with Matplotlib, so every machine draws the same glyphs.
# A fixed hash salt and no timestamp make the same data give the same SVG bytes.
STYLE = {
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",
    "svg.hashsalt": "supplypilot-phase3",
}

LINE_WIDTH = 1.75
MARKER_SIZE = 7
RING_WIDTH = 1.6


def _rows(frame: pd.DataFrame, variant: str, lead_time: int) -> pd.DataFrame:
    return frame[(frame["variant"] == variant) & (frame["lead_time"] == lead_time)]


def _curve(curves, method: str, variant: str, lead_time: int) -> pd.DataFrame:
    rows = _rows(curves, variant, lead_time)
    rows = rows[rows["method"] == method].sort_values("z")

    if rows.empty:
        raise ValueError(
            f"No {method} curve for variant {variant!r} and lead time {lead_time}"
        )

    return rows


def _endpoint(endpoints, method: str, variant: str, lead_time: int, target: float):
    rows = _rows(endpoints, variant, lead_time)
    rows = rows[
        (rows["method"] == method) & np.isclose(rows["target_fill_rate"], target)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one {method} endpoint at {target:.0%} for variant "
            f"{variant!r} and lead time {lead_time}"
        )

    return rows.iloc[0]


def _reference(reference, variant: str, lead_time: int):
    rows = _rows(reference, variant, lead_time)

    if len(rows) != 1:
        raise ValueError(
            f"Expected one perfect-foresight row for variant {variant!r} "
            f"and lead time {lead_time}"
        )

    return rows.iloc[0]


def _thousands(value, _position) -> str:
    return "0" if value == 0 else f"{value / 1000:g}K"


def _percent(value, _position) -> str:
    return f"{value:.0%}"


def _reached(endpoint) -> bool:
    return endpoint["status"] in ESTIMABLE


def plot_fill_rate_vs_stock(
    curves: pd.DataFrame,
    reference: pd.DataFrame,
    endpoints: pd.DataFrame,
    output_path: str | Path,
    *,
    theme: str = "light",
    variant: str = "primary",
    lead_time: int = 7,
    target: float = 0.95,
    comparison_target: float = 0.90,
    source_note: str | None = None,
) -> None:
    """Save the fill-rate-versus-stock trade-off of both methods for one scenario.

    Each method's line joins its simulated policies in z order. The chart marks
    the target fill rate, the interpolated stock both methods need for the
    comparison fill rate, and the perfect-foresight reference. The file format
    follows the extension of output_path.
    """
    if theme not in THEMES:
        raise ValueError(f"Theme must be one of {tuple(THEMES)}")

    lines = {method: _curve(curves, method, variant, lead_time) for method in METHODS}
    at_target = {
        method: _endpoint(endpoints, method, variant, lead_time, target)
        for method in METHODS
    }
    compared = {
        method: _endpoint(endpoints, method, variant, lead_time, comparison_target)
        for method in METHODS
    }
    oracle = _reference(reference, variant, lead_time)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with matplotlib.rc_context(STYLE):
        fig = _draw(
            THEMES[theme],
            lines,
            at_target,
            compared,
            oracle,
            lead_time=lead_time,
            target=target,
            comparison_target=comparison_target,
            source_note=source_note,
        )
        fig.savefig(
            output_path,
            facecolor=fig.get_facecolor(),
            metadata={"Date": None} if output_path.suffix == ".svg" else None,
        )


def _draw(
    ink,
    lines,
    at_target,
    compared,
    oracle,
    *,
    lead_time,
    target,
    comparison_target,
    source_note,
) -> Figure:
    fig = Figure(figsize=(10, 6), facecolor=ink["surface"])
    ax = fig.add_axes((0.075, 0.14, 0.9, 0.66), facecolor=ink["surface"])

    first = lines["lightgbm"]
    series = int(first["series_count"].iloc[0])
    days = int(first["evaluated_days"].iloc[0])

    fig.text(
        0.075,
        0.935,
        "LightGBM needed more stock than MA28 for the same fill rate",
        color=ink["primary"],
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.855,
        f"Simulated weekly replenishment with lost sales · {lead_time}-day lead time · "
        f"{series:,} Walmart M5 series · {days} evaluation days\n"
        f"Order-up-to level F + z·√F; each line runs from z = {first['z'].min():g} "
        f"(left) to z = {first['z'].max():.1f} (right)",
        color=ink["secondary"],
        fontsize=10.5,
        linespacing=1.5,
    )

    # Recessive frame: horizontal hairlines only, no top or right spine.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=ink["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink["muted"], labelsize=9.5, length=0, pad=6)

    ax.xaxis.set_major_locator(MultipleLocator(10_000))
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_xlabel(
        "Average units on hand, all series combined",
        color=ink["secondary"],
        fontsize=10.5,
    )
    ax.set_ylabel(
        "Fill rate (share of demand served)", color=ink["secondary"], fontsize=10.5
    )

    lowest = min(line["fill_rate"].min() for line in lines.values())
    widest = max(line["average_on_hand_units"].max() for line in lines.values())
    gap = widest * 0.018
    ax.set_ylim(np.floor(lowest * 20) / 20 - 0.01, 1.015)
    ax.set_xlim(0, widest * 1.22)

    ax.axhline(target, color=ink["muted"], linewidth=0.9, zorder=2)
    ax.text(
        widest * 0.012,
        target + 0.003,
        f"{target:.0%} fill (primary target)",
        color=ink["muted"],
        fontsize=9,
        va="bottom",
    )

    ring = {"markeredgecolor": ink["surface"], "markeredgewidth": RING_WIDTH}

    for method, line in lines.items():
        x = line["average_on_hand_units"].to_numpy()
        y = line["fill_rate"].to_numpy()
        ax.plot(
            x,
            y,
            color=ink[method],
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        ax.plot(
            x[[0, -1]],
            y[[0, -1]],
            "o",
            color=ink[method],
            markersize=MARKER_SIZE,
            zorder=4,
            **ring,
        )

        # Direct label at the top end of the line, clear of the target line.
        name = METHOD_LABELS[method]
        if _reached(at_target[method]):
            ax.text(
                x[-1],
                y[-1] + 0.008,
                f"{name} · reaches {target:.0%}",
                color=ink["primary"],
                fontsize=10,
                ha="right",
                va="bottom",
            )
        else:
            ax.text(
                x[-1] + gap,
                y[-1],
                f"{name} · max {y.max():.1%}",
                color=ink["primary"],
                fontsize=10,
                ha="left",
                va="center",
            )

    if all(_reached(compared[method]) for method in METHODS):
        stock = {method: compared[method]["inventory"] for method in METHODS}
        ax.plot(
            [stock["ma28"], stock["lightgbm"]],
            [comparison_target, comparison_target],
            color=ink["secondary"],
            linewidth=1.2,
            solid_capstyle="round",
            zorder=4,
        )
        for method in METHODS:
            ax.plot(
                stock[method],
                comparison_target,
                "o",
                color=ink[method],
                markersize=MARKER_SIZE - 1,
                zorder=5,
                **ring,
            )
        difference = compared["lightgbm"]["relative_difference_lightgbm_vs_ma28"]
        direction = "more" if difference >= 0 else "less"
        ax.text(
            max(stock.values()) + gap,
            comparison_target - 0.004,
            f"LightGBM needs {abs(difference):.1%} {direction} stock\n"
            f"for {comparison_target:.0%} fill ({stock['lightgbm']:,.0f} vs "
            f"{stock['ma28']:,.0f} units)",
            color=ink["secondary"],
            fontsize=9.5,
            ha="left",
            va="top",
            linespacing=1.4,
        )

    ax.plot(
        oracle["average_on_hand_units"],
        oracle["fill_rate"],
        "o",
        color=ink["reference"],
        markersize=MARKER_SIZE,
        zorder=4,
        **ring,
    )
    ax.text(
        oracle["average_on_hand_units"] + gap,
        oracle["fill_rate"],
        "Perfect foresight · all demand served with "
        f"{oracle['average_on_hand_units']:,.0f} units",
        color=ink["secondary"],
        fontsize=9.5,
        ha="left",
        va="center",
    )

    handles = [
        Line2D(
            [], [], color=ink[method], linewidth=LINE_WIDTH, label=METHOD_LABELS[method]
        )
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=ink["reference"],
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            label="Perfect foresight (reference)",
            **ring,
        )
    )
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        labelcolor=ink["secondary"],
        handlelength=2.2,
    )

    if source_note:
        fig.text(0.075, 0.02, source_note, color=ink["muted"], fontsize=8.5)

    return fig
