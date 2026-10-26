-- Tables for the published Phase 3 results: protocol v1.0, one registered run.
--
-- src/results_sql.py fills them from reports/phase3_inventory/primary/ after checking each file
-- against checksums.json, then makes the database read-only. Rules about a single row are the
-- constraints below; rules that compare rows are in data_checks.sql.

-- One simulated replenishment policy: a forecaster in one scenario at one safety factor z
-- (curves.csv). A scenario is a starting condition (variant) and a supplier lead time.
CREATE TABLE policy (
    variant               TEXT    NOT NULL,
    lead_time             INTEGER NOT NULL CHECK (lead_time > 0),
    method                TEXT    NOT NULL CHECK (method IN ('lightgbm', 'ma28')),
    z_step                INTEGER NOT NULL CHECK (z_step BETWEEN 0 AND 30),  -- z in tenths
    z                     REAL    NOT NULL CHECK (abs(z - z_step / 10.0) < 1e-9),
    evaluated_days        INTEGER NOT NULL CHECK (evaluated_days > 0),
    demand_units          INTEGER NOT NULL CHECK (demand_units > 0),
    sales_units           INTEGER NOT NULL CHECK (sales_units >= 0),
    lost_units            INTEGER NOT NULL CHECK (lost_units >= 0),
    fill_rate             REAL    NOT NULL CHECK (fill_rate BETWEEN 0 AND 1),
    average_on_hand_units REAL    NOT NULL CHECK (average_on_hand_units >= 0),
    average_on_hand_value REAL    NOT NULL CHECK (average_on_hand_value >= 0),  -- at sell price
    lost_sales_value      REAL    NOT NULL CHECK (lost_sales_value >= 0),       -- at sell price
    PRIMARY KEY (variant, lead_time, method, z_step),
    CHECK (sales_units + lost_units = demand_units)
);

-- The same policies summed over a demand group or a product category (group_curves.csv). The
-- protocol publishes these for the primary start at the 7-day lead time only.
CREATE TABLE group_policy (
    group_type            TEXT    NOT NULL CHECK (group_type IN ('demand_group', 'category')),
    group_name            TEXT    NOT NULL,
    series_count          INTEGER NOT NULL CHECK (series_count > 0),
    method                TEXT    NOT NULL CHECK (method IN ('lightgbm', 'ma28')),
    z_step                INTEGER NOT NULL CHECK (z_step BETWEEN 0 AND 30),
    fill_rate             REAL    NOT NULL CHECK (fill_rate BETWEEN 0 AND 1),
    average_on_hand_units REAL    NOT NULL CHECK (average_on_hand_units >= 0),
    PRIMARY KEY (group_type, group_name, method, z_step)
);

-- Perfect foresight knows future demand, so it shows the least stock that loses no sales on this
-- delivery schedule. A reference point, not a forecasting method (reference.csv).
CREATE TABLE reference (
    variant               TEXT    NOT NULL,
    lead_time             INTEGER NOT NULL CHECK (lead_time > 0),
    fill_rate             REAL    NOT NULL CHECK (fill_rate BETWEEN 0 AND 1),
    average_on_hand_units REAL    NOT NULL CHECK (average_on_hand_units >= 0),
    PRIMARY KEY (variant, lead_time)
);

-- The stock for each preregistered target as the Python protocol code published it
-- (endpoints.csv). Kept so that stock_for_target.sql can be checked against it.
CREATE TABLE published_endpoint (
    variant               TEXT    NOT NULL,
    lead_time             INTEGER NOT NULL,
    target_fill_rate      REAL    NOT NULL CHECK (target_fill_rate > 0 AND target_fill_rate <= 1),
    method                TEXT    NOT NULL CHECK (method IN ('lightgbm', 'ma28')),
    status                TEXT    NOT NULL
        CHECK (status IN ('exact', 'interpolated', 'above_target_at_min_z', 'not_reached')),
    stock_units           REAL    CHECK (stock_units IS NULL OR stock_units >= 0),
    lower_z               REAL,
    upper_z               REAL,
    lightgbm_vs_ma28      REAL,
    PRIMARY KEY (variant, lead_time, target_fill_rate, method),
    CHECK ((status IN ('exact', 'interpolated')) = (stock_units IS NOT NULL))
);

-- Every curve as a list of points, overall and by group, so one query can read them all alike.
CREATE VIEW curve_point AS
SELECT 'all series' AS slice, variant, lead_time, method, z_step, fill_rate,
       average_on_hand_units AS stock_units
FROM policy
UNION ALL
SELECT replace(group_type, '_', ' ') || ': ' || group_name, 'primary', 7, method, z_step,
       fill_rate, average_on_hand_units
FROM group_policy;
