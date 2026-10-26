import shutil
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from scripts import query_results
from src.inventory_whatif import (
    REPORT_DIR,
    cheapest,
    cost_table,
    curve,
    load_reports,
    stock_for_fill_rate,
)
from src.results_sql import QUERIES, SQL_DIR, export, load_database, rows_as_dicts, run

GROUP_TYPES = {"demand group": "demand_group", "category": "cat_id"}


@pytest.fixture(scope="module")
def database():
    connection = load_database()
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def reports():
    return load_reports()


def writable_copy() -> sqlite3.Connection:
    copy = sqlite3.connect(":memory:")
    with closing(load_database()) as source:
        source.backup(copy)
    return copy


def same(value, expected, rel=1e-12):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        return value is None
    return value == pytest.approx(expected, rel=rel)


def test_loads_every_published_row(database):
    tables = ("policy", "group_policy", "reference", "published_endpoint")
    counts = {
        table: database.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in tables
    }
    assert counts == {
        "policy": 558,
        "group_policy": 372,
        "reference": 9,
        "published_endpoint": 54,
    }


def test_every_named_query_has_a_file():
    assert all((SQL_DIR / f"{name}.sql").is_file() for name in QUERIES)


def test_data_checks_find_no_problems(database):
    assert run(database, "data_checks") == []


def test_data_checks_catch_a_broken_curve():
    with closing(writable_copy()) as copy:
        copy.execute(
            "DELETE FROM policy WHERE variant = 'primary' AND lead_time = 7 "
            "AND method = 'ma28' AND z_step = 30"
        )
        copy.execute(
            "UPDATE policy SET fill_rate = 0.5 WHERE variant = 'primary' AND lead_time = 3 "
            "AND method = 'lightgbm' AND z_step = 10"
        )
        copy.execute("UPDATE reference SET fill_rate = 0.99 WHERE lead_time = 1")
        problems = {row["problem"] for row in run(copy, "data_checks")}

    assert problems == {
        "curve does not have the 31 policies from z = 0 to 3.0",
        "fill rate is not sales divided by demand",
        "fill rate falls, or stock does not rise, as z rises",
        "perfect foresight loses sales",
    }


# variant, lead_time, method, z_step, z, evaluated_days, demand_units, sales_units, lost_units,
# fill_rate, average_on_hand_units, average_on_hand_value, lost_sales_value
FIRST_POLICY = ("primary", 7, "lightgbm", 0, 0.0, 84, 100, 80, 20, 0.8, 10.0, 20.0, 5.0)
NEXT_POLICY = (
    "primary",
    7,
    "lightgbm",
    1,
    0.1,
    84,
    100,
    85,
    15,
    0.85,
    11.0,
    22.0,
    3.75,
)


def test_schema_accepts_valid_rows():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript((SQL_DIR / "schema.sql").read_text())
        insert = f"INSERT INTO policy VALUES ({', '.join('?' * len(FIRST_POLICY))})"
        connection.executemany(insert, [FIRST_POLICY, NEXT_POLICY])
        assert connection.execute("SELECT count(*) FROM policy").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({8: 30}, "sales and lost sales add up to more than demand"),
        ({9: 1.5}, "fill rate above 1"),
        ({2: "prophet"}, "unknown forecaster"),
        ({3: 31, 4: 3.1}, "z beyond the grid"),
        ({4: 0.2}, "z that does not match its step"),
        ({10: -1.0}, "negative stock"),
        ({3: 0, 4: 0.0}, "the same policy twice"),
    ],
)
def test_schema_rejects_impossible_rows(change, why):
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript((SQL_DIR / "schema.sql").read_text())
        insert = f"INSERT INTO policy VALUES ({', '.join('?' * len(FIRST_POLICY))})"
        connection.execute(insert, FIRST_POLICY)
        row = list(NEXT_POLICY)
        for index, value in change.items():
            row[index] = value
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert, row)


def test_database_is_read_only(database):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        database.execute("DELETE FROM policy")


def test_refuses_reports_that_changed(tmp_path):
    folder = tmp_path / "primary"
    shutil.copytree(REPORT_DIR, folder)
    path = folder / "group_curves.csv"
    path.write_text(path.read_text().replace("frequent", "Frequent", 1))

    with pytest.raises(
        ValueError, match=r"group_curves\.csv does not match checksums\.json"
    ):
        load_database(folder)


def test_unknown_query_is_refused(database):
    with pytest.raises(ValueError, match="Unknown query"):
        run(database, "schema")


@pytest.mark.parametrize("target", [0.9, 0.95, 0.98])
def test_stock_for_target_matches_the_published_endpoints(database, target):
    published = {
        (row["variant"], row["lead_time"], row["method"]): row
        for row in rows_as_dicts(
            database.execute(
                "SELECT * FROM published_endpoint WHERE target_fill_rate = ?", (target,)
            )
        )
    }
    rows = run(database, "stock_for_target", target=target)

    assert len(rows) == len(published) == 18
    for row in rows:
        expected = published[(row["variant"], row["lead_time"], row["method"])]
        assert row["status"] == expected["status"]
        assert same(row["stock_units"], expected["stock_units"])
        assert row["lower_z"] == expected["lower_z"]
        assert row["upper_z"] == expected["upper_z"]
        assert same(row["lightgbm_vs_ma28"], expected["lightgbm_vs_ma28"], rel=1e-9)


def test_stock_for_target_matches_the_engine_at_other_targets(database, reports):
    statuses = set()
    for target in (0.8, 0.85, 0.925, 0.97, 0.99):
        for row in run(database, "stock_for_target", target=target):
            frame = curve(reports, row["variant"], row["lead_time"], row["method"])
            expected = stock_for_fill_rate(frame, target)
            statuses.add(row["status"])
            assert row["status"] == expected["status"]
            assert same(row["stock_units"], expected["units"])
            assert row["lower_z"] == expected["lower_z"]
            assert row["upper_z"] == expected["upper_z"]
            # pandas and float() may parse a CSV value one unit in the last place apart.
            assert row["best_fill_rate"] == pytest.approx(
                expected["max_fill_rate"], rel=1e-12
            )

    assert {"interpolated", "above_target_at_min_z", "not_reached"} <= statuses


def test_stock_gap_at_90_percent_is_the_headline(database):
    lightgbm = {
        row["lead_time"]: row
        for row in run(database, "stock_for_target", target=0.9)
        if row["variant"] == "primary" and row["method"] == "lightgbm"
    }

    assert {days: round(row["stock_units"]) for days, row in lightgbm.items()} == {
        7: 39157,
        3: 27567,
        1: 25283,
    }
    assert {
        days: round(100 * row["lightgbm_vs_ma28"], 1) for days, row in lightgbm.items()
    } == {
        7: 18.5,
        3: 10.3,
        1: 7.1,
    }


@pytest.mark.parametrize(
    ("holding_rate", "penalty"),
    [(0.25, 0.25), (0.25, 0.5), (0.25, 1.0), (0.3, 0.75), (0.0, 2.0), (0.6, 0.0)],
)
def test_cheapest_policy_matches_the_engine(database, reports, holding_rate, penalty):
    rows = run(database, "cheapest_policy", holding_rate=holding_rate, penalty=penalty)

    assert len(rows) == 18
    for row in rows:
        frame = curve(reports, row["variant"], row["lead_time"], row["method"])
        costs = cost_table(frame, holding_rate=holding_rate, penalty_fraction=penalty)
        expected = cheapest(frame, costs)
        assert row["z"] == expected["z"]
        assert row["total_cost"] == pytest.approx(expected["total_cost"], rel=1e-12)
        assert row["grid_edge"] == expected["grid_edge"]


@pytest.mark.parametrize("penalty", [0.25, 0.5, 1.0])
def test_cheapest_policy_uses_the_published_cost_columns(database, reports, penalty):
    for row in run(database, "cheapest_policy", holding_rate=0.25, penalty=penalty):
        frame = curve(reports, row["variant"], row["lead_time"], row["method"])
        published = frame[f"total_cost_penalty_{penalty}"]
        assert row["total_cost"] == pytest.approx(published.min(), rel=1e-9)


def test_cheapest_policy_at_the_protocol_costs_is_the_headline(database):
    rows = {
        row["method"]: row
        for row in run(database, "cheapest_policy", holding_rate=0.25, penalty=0.5)
        if row["variant"] == "primary" and row["lead_time"] == 7
    }

    assert round(rows["lightgbm"]["total_cost"]) == 44877
    assert round(rows["ma28"]["total_cost"]) == 33559
    assert round(100 * rows["lightgbm"]["lightgbm_vs_ma28"], 1) == 33.7
    assert rows["lightgbm"]["grid_edge"] == rows["ma28"]["grid_edge"] == "highest_z"


def numpy_gap(lightgbm: pd.DataFrame, ma28: pd.DataFrame) -> tuple[float, float, int]:
    """The same reading with numpy: both curves at every simulated stock level they share."""
    stock_a, fill_a = (
        lightgbm["average_on_hand_units"].to_numpy(),
        lightgbm["fill_rate"].to_numpy(),
    )
    stock_b, fill_b = (
        ma28["average_on_hand_units"].to_numpy(),
        ma28["fill_rate"].to_numpy(),
    )
    low, high = max(stock_a[0], stock_b[0]), min(stock_a[-1], stock_b[-1])
    levels = np.unique(np.concatenate([stock_a, stock_b]))
    levels = levels[(levels >= low) & (levels <= high)]
    gap = 100 * (
        np.interp(levels, stock_b, fill_b) - np.interp(levels, stock_a, fill_a)
    )
    return float(gap.min()), float(gap.max()), len(levels)


def test_equal_stock_gap_matches_numpy(database):
    curves = pd.read_csv(REPORT_DIR / "curves.csv")
    groups = pd.read_csv(REPORT_DIR / "group_curves.csv")
    rows = run(database, "equal_stock_gap")

    assert len(rows) == 9 + 6
    for row in rows:
        if row["slice"] == "all series":
            points = curves[
                (curves["variant"] == row["variant"])
                & (curves["lead_time"] == row["lead_time"])
            ]
        else:
            kind, name = row["slice"].split(": ")
            points = groups[
                (groups["group_type"] == GROUP_TYPES[kind]) & (groups["group"] == name)
            ]
        ordered = points.sort_values("z")
        low, high, count = numpy_gap(
            ordered[ordered["method"] == "lightgbm"],
            ordered[ordered["method"] == "ma28"],
        )
        assert row["stock_levels"] == count
        assert row["smallest_gap_points"] == pytest.approx(low, abs=1e-9)
        assert row["largest_gap_points"] == pytest.approx(high, abs=1e-9)


def test_equal_stock_gap_matches_the_readme(database):
    gaps = {
        (row["slice"], row["lead_time"]): (
            round(row["smallest_gap_points"], 1),
            round(row["largest_gap_points"], 1),
        )
        for row in run(database, "equal_stock_gap")
        if row["variant"] == "primary"
    }

    assert gaps[("all series", 7)] == (2.1, 2.7)
    assert gaps[("all series", 3)] == (1.2, 1.6)
    assert gaps[("all series", 1)] == (1.0, 1.2)
    assert gaps[("category: FOODS", 7)] == (2.7, 3.5)
    assert gaps[("demand group: frequent", 7)] == (2.6, 3.0)
    assert gaps[("category: HOUSEHOLD", 7)] == (0.4, 0.9)
    # MA28 serves more demand at every shared stock level, in every slice.
    assert all(low > 0 for low, _ in gaps.values())


def test_export_writes_a_database_any_client_can_open(database, tmp_path):
    path = export(database, tmp_path / "results.sqlite")

    with closing(sqlite3.connect(path)) as copy:
        assert copy.execute("SELECT count(*) FROM policy").fetchone()[0] == 558
    with pytest.raises(FileExistsError):
        export(database, path)


def test_report_prints_the_headline_tables(capsys):
    assert query_results.main([]) == 0
    out = capsys.readouterr().out

    for expected in (
        "39,157",
        "33,044",
        "+18.5%",
        "$44,877",
        "+33.7%",
        "2.1–2.7",
        "all passed",
    ):
        assert expected in out
    assert "exploratory" not in out


def test_report_labels_exploratory_settings(capsys):
    argv = ["--variant", "empty_start", "--target", "0.93", "--penalty", "2"]
    assert query_results.main(argv) == 0
    out = capsys.readouterr().out

    assert "Average stock for a 93% fill rate, empty start" in out
    assert "robustness check" in out
    assert "target fill rate is exploratory" in out
    assert "cost settings are exploratory" in out


def test_sql_option_runs_read_only_queries(capsys):
    query = "SELECT method, count(*) AS policies FROM policy GROUP BY method ORDER BY method"
    assert query_results.main(["--sql", query]) == 0
    out = capsys.readouterr().out
    assert "lightgbm" in out
    assert "279" in out

    assert query_results.main(["--sql", "DELETE FROM policy"]) == 2
    assert "readonly" in capsys.readouterr().err


def test_export_option(tmp_path, capsys):
    path = tmp_path / "results.sqlite"
    assert query_results.main(["--export", str(path)]) == 0
    assert path.is_file()

    assert query_results.main(["--export", str(path)]) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--target", "1.5"],
        ["--target", "0"],
        ["--penalty", "-1"],
        ["--holding-rate", "nan"],
    ],
)
def test_report_rejects_settings_out_of_range(argv):
    with pytest.raises(SystemExit):
        query_results.main(argv)
