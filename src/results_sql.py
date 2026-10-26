"""SQL over the published Phase 3 results.

load_database() checks the published report files against checksums.json and
loads them into an in-memory SQLite database, which is read-only from then on.
run() executes one of the named queries in sql/.

The queries re-derive the protocol's endpoints, the cheapest policy at given
costs and the post-hoc equal-stock reading in SQL. That makes them a second
implementation of the same rules, which tests/test_results_sql.py checks against
the Python engine and the published numbers. Nothing here re-runs the
simulation.
"""

import csv
import io
import sqlite3
from pathlib import Path

from src.inventory_whatif import REPORT_DIR, read_verified

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
QUERIES = ("stock_for_target", "cheapest_policy", "equal_stock_gap", "data_checks")
GROUP_TYPES = {"demand_group": "demand_group", "cat_id": "category"}


def _z_step(text: str) -> int:
    return round(float(text) * 10)


def _optional(text: str) -> float | None:
    return float(text) if text.strip() else None


def _group_type(text: str) -> str:
    if text not in GROUP_TYPES:
        raise ValueError(f"Unknown group type in group_curves.csv: {text!r}")
    return GROUP_TYPES[text]


# Each table: the published file it comes from, then (column, CSV field, conversion).
TABLES = {
    "policy": (
        "curves.csv",
        [
            ("variant", "variant", str),
            ("lead_time", "lead_time", int),
            ("method", "method", str),
            ("z_step", "z", _z_step),
            ("z", "z", float),
            ("evaluated_days", "evaluated_days", int),
            ("demand_units", "demand_units", int),
            ("sales_units", "sales_units", int),
            ("lost_units", "lost_units", int),
            ("fill_rate", "fill_rate", float),
            ("average_on_hand_units", "average_on_hand_units", float),
            ("average_on_hand_value", "average_on_hand_value", float),
            ("lost_sales_value", "lost_sales_value", float),
        ],
    ),
    "group_policy": (
        "group_curves.csv",
        [
            ("group_type", "group_type", _group_type),
            ("group_name", "group", str),
            ("series_count", "series_count", int),
            ("method", "method", str),
            ("z_step", "z", _z_step),
            ("fill_rate", "fill_rate", float),
            ("average_on_hand_units", "average_on_hand_units", float),
        ],
    ),
    "reference": (
        "reference.csv",
        [
            ("variant", "variant", str),
            ("lead_time", "lead_time", int),
            ("fill_rate", "fill_rate", float),
            ("average_on_hand_units", "average_on_hand_units", float),
        ],
    ),
    "published_endpoint": (
        "endpoints.csv",
        [
            ("variant", "variant", str),
            ("lead_time", "lead_time", int),
            ("target_fill_rate", "target_fill_rate", float),
            ("method", "method", str),
            ("status", "status", str),
            ("stock_units", "inventory", _optional),
            ("lower_z", "lower_z", _optional),
            ("upper_z", "upper_z", _optional),
            ("lightgbm_vs_ma28", "relative_difference_lightgbm_vs_ma28", _optional),
        ],
    ),
}


def load_database(report_dir=REPORT_DIR) -> sqlite3.Connection:
    """The published results in an in-memory SQLite database, read-only once loaded.

    Raises ValueError when a report file does not match checksums.json, and
    sqlite3.IntegrityError when a row breaks a constraint in sql/schema.sql.
    """
    files = read_verified(report_dir, [source for source, _ in TABLES.values()])
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript((SQL_DIR / "schema.sql").read_text())

    with connection:
        for table, (source, columns) in TABLES.items():
            records = csv.DictReader(io.StringIO(files[source].decode("utf-8")))
            rows = [
                tuple(convert(record[field]) for _, field, convert in columns)
                for record in records
            ]
            names = ", ".join(column for column, _, _ in columns)
            slots = ", ".join("?" for _ in columns)
            # Table and column names come from TABLES above; every value is bound.
            connection.executemany(
                f"INSERT INTO {table} ({names}) VALUES ({slots})", rows
            )

    connection.execute("PRAGMA query_only = ON")
    return connection


def run(connection: sqlite3.Connection, name: str, **parameters) -> list[dict]:
    """The rows of one named query in sql/, as dictionaries."""
    if name not in QUERIES:
        raise ValueError(f"Unknown query {name!r}; choose one of {', '.join(QUERIES)}")
    sql = (SQL_DIR / f"{name}.sql").read_text()
    return rows_as_dicts(connection.execute(sql, parameters))


def rows_as_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    """A cursor's rows as dictionaries keyed by column name."""
    names = [column[0] for column in cursor.description or ()]
    return [dict(zip(names, row, strict=True)) for row in cursor]


def export(connection: sqlite3.Connection, path) -> Path:
    """Write the database to a new file that any SQLite client can open."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose a new file name")
    copy = sqlite3.connect(target)
    try:
        connection.backup(copy)
    finally:
        copy.close()
    return target
