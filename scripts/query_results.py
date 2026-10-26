"""Query the published Phase 3 results with SQL.

    .venv/bin/python -m scripts.query_results
    .venv/bin/python -m scripts.query_results --target 0.95 --holding-rate 0.3 --penalty 0.75
    .venv/bin/python -m scripts.query_results --sql "SELECT method, max(fill_rate) FROM policy GROUP BY method"
    .venv/bin/python -m scripts.query_results --export phase3_results.sqlite

Loads the published reports into an in-memory SQLite database, after checking
them against checksums.json, and prints the queries in sql/ for one starting
condition: the stock each forecaster needs for a target fill rate, the cheapest
policy at given costs, the post-hoc equal-stock gap and the data checks.
--sql runs a read-only query of your own, and --export saves the database to a
file for any SQLite client. Nothing is simulated again.
"""

import argparse
import math
import sqlite3
import sys

from src.inventory_whatif import VARIANT_LABELS, load_reports
from src.results_sql import export, load_database, rows_as_dicts, run

METHODS = ("lightgbm", "ma28")


def _bounded(low: float, high: float, *, open_low: bool = False):
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        inside = (low < value if open_low else low <= value) and value <= high
        if not math.isfinite(value) or not inside:
            bracket = "(" if open_low else "["
            raise argparse.ArgumentTypeError(f"must be in {bracket}{low:g}, {high:g}]")
        return value

    return parse


def table(headers: list[str], rows: list[list]) -> str:
    """Rows as aligned text: the first column on the left, the others on the right."""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in cells))
        for index, header in enumerate(headers)
    ]

    def line(values):
        return "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(values, widths, strict=True))
        ).rstrip()

    return "\n".join(
        [line(headers), line(["-" * width for width in widths]), *map(line, cells)]
    )


def days(lead_time: int) -> str:
    return "1 day" if lead_time == 1 else f"{lead_time} days"


def percent(value: float) -> str:
    """90% rather than 90.0%, but 92.5% when it needs the decimal."""
    return (
        f"{value:.0%}"
        if math.isclose(value * 100, round(value * 100))
        else f"{value:.1%}"
    )


def signed_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def by_lead_time(rows: list[dict], variant: str) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        if row["variant"] == variant:
            grouped.setdefault(row["lead_time"], {})[row["method"]] = row
    return grouped


def stock_section(database, variant: str, target: float) -> str:
    rows = by_lead_time(run(database, "stock_for_target", target=target), variant)

    def units(row):
        if row["stock_units"] is None:
            return f"not reached (best {row['best_fill_rate']:.1%})"
        return f"{row['stock_units']:,.0f}"

    body = [
        [
            days(lead_time),
            *(units(pair[m]) for m in METHODS),
            signed_percent(pair["lightgbm"]["lightgbm_vs_ma28"]),
        ]
        for lead_time, pair in rows.items()
    ]
    title = f"Average stock for a {percent(target)} fill rate, {VARIANT_LABELS[variant].lower()}"
    return (
        f"{title}\n{table(['Lead time', 'LightGBM', 'MA28', 'LightGBM vs MA28'], body)}"
    )


def cost_section(database, variant: str, holding_rate: float, penalty: float) -> str:
    rows = by_lead_time(
        run(database, "cheapest_policy", holding_rate=holding_rate, penalty=penalty),
        variant,
    )
    edge = False

    def cost(row):
        nonlocal edge
        edge = edge or row["grid_edge"] is not None
        star = "*" if row["grid_edge"] else ""
        return f"${row['total_cost']:,.0f} at z = {row['z']:.1f}{star}"

    body = [
        [
            days(lead_time),
            *(cost(pair[m]) for m in METHODS),
            signed_percent(pair["lightgbm"]["lightgbm_vs_ma28"]),
        ]
        for lead_time, pair in rows.items()
    ]
    title = (
        f"Cheapest policy at {percent(holding_rate)} a year and {penalty:g}× the sell price, "
        f"{VARIANT_LABELS[variant].lower()}"
    )
    text = (
        f"{title}\n{table(['Lead time', 'LightGBM', 'MA28', 'LightGBM vs MA28'], body)}"
    )
    if edge:
        text += (
            "\n* at the edge of the tested z range; the true minimum may lie outside it"
        )
    return text


def gap_section(database, variant: str) -> str:
    body = [
        [
            row["slice"],
            days(row["lead_time"]),
            f"{row['smallest_gap_points']:.1f}–{row['largest_gap_points']:.1f}",
        ]
        for row in run(database, "equal_stock_gap")
        if row["variant"] == variant
    ]
    title = "MA28 minus LightGBM fill rate at equal stock, in points (post hoc)"
    return f"{title}\n{table(['Slice', 'Lead time', 'Gap'], body)}"


def checks_section(database) -> str:
    problems = run(database, "data_checks")
    if not problems:
        return "Data checks: all passed."
    lines = [f"- {row['problem']}: {row['location']}" for row in problems]
    return "Data checks: problems found\n" + "\n".join(lines)


def exploratory_notes(
    protocol: dict, variant: str, target: float, holding_rate: float, penalty: float
) -> list:
    notes = []
    if variant != "primary":
        notes.append(
            f"The {VARIANT_LABELS[variant].lower()} is a robustness check; the primary comparison "
            f"is the primary start at a {protocol['primary_lead_time']}-day lead time."
        )
    if not any(
        math.isclose(target, value, abs_tol=1e-9) for value in protocol["targets"]
    ):
        notes.append(
            "The target fill rate is exploratory; the protocol reported 90%, 95% and 98%."
        )
    cost_ok = math.isclose(
        holding_rate, protocol["holding_rate_per_year"], abs_tol=1e-9
    ) and any(
        math.isclose(penalty, value, abs_tol=1e-9)
        for value in protocol["penalty_fractions"]
    )
    if not cost_ok:
        notes.append(
            "The cost settings are exploratory; the protocol fixed holding at "
            f"{protocol['holding_rate_per_year']:.0%} a year and penalties at "
            f"{', '.join(f'{value:g}' for value in protocol['penalty_fractions'])}× the sell price."
        )
    return notes


def run_sql(database, sql: str) -> int:
    try:
        rows = rows_as_dicts(database.execute(sql))
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("(no rows)")
        return 0
    headers = list(rows[0])
    body = [
        [
            f"{value:.6g}" if isinstance(value, float) else value
            for value in row.values()
        ]
        for row in rows
    ]
    print(table(headers, body))
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANT_LABELS), default="primary")
    parser.add_argument("--target", type=_bounded(0, 1, open_low=True), default=0.9)
    parser.add_argument("--holding-rate", type=_bounded(0, 1), default=0.25)
    parser.add_argument("--penalty", type=_bounded(0, 5), default=0.5)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--sql", help="run one read-only SQL statement and print its rows"
    )
    action.add_argument(
        "--export", metavar="PATH", help="save the database to a new SQLite file"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    database = load_database()
    try:
        if args.sql:
            return run_sql(database, args.sql)
        if args.export:
            try:
                path = export(database, args.export)
            except FileExistsError as exc:
                print(exc, file=sys.stderr)
                return 2
            print(
                f"Wrote {path}: tables policy, group_policy, reference, published_endpoint."
            )
            return 0

        sections = [
            stock_section(database, args.variant, args.target),
            cost_section(database, args.variant, args.holding_rate, args.penalty),
            gap_section(database, args.variant),
            checks_section(database),
        ]
        notes = exploratory_notes(
            load_reports().protocol,
            args.variant,
            args.target,
            args.holding_rate,
            args.penalty,
        )
        print("\n\n".join(sections + (["\n".join(notes)] if notes else [])))
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
