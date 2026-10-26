-- Average stock each forecaster needs to reach a target fill rate, in every scenario.
--
-- Parameter :target, a fill rate in (0, 1].
--
-- Same rule as the protocol (interpolated_inventory_at_fill_rate in src/inventory_metrics.py):
-- take the first policy, in z order, whose fill rate reaches the target, and interpolate linearly
-- between it and the policy before it. The result is an estimate between two simulated policies.
-- If the first policy on the grid already beats the target there is nothing to interpolate from,
-- and if no policy reaches it the target is not reached on the tested range.
WITH step AS (
    SELECT variant, lead_time, method, z_step, z, fill_rate,
           average_on_hand_units AS stock_units,
           LAG(z) OVER curve AS previous_z,
           LAG(fill_rate) OVER curve AS previous_fill_rate,
           LAG(average_on_hand_units) OVER curve AS previous_stock_units
    FROM policy
    WINDOW curve AS (PARTITION BY variant, lead_time, method ORDER BY z_step)
),
first_reaching AS (
    SELECT *,
           row_number() OVER (PARTITION BY variant, lead_time, method ORDER BY z_step) AS position
    FROM step
    WHERE fill_rate >= :target
),
curve AS (
    SELECT variant, lead_time, method, max(fill_rate) AS best_fill_rate
    FROM policy
    GROUP BY variant, lead_time, method
),
endpoint AS (
    SELECT c.variant, c.lead_time, c.method, c.best_fill_rate,
           CASE
               WHEN f.z_step IS NULL THEN 'not_reached'
               WHEN f.fill_rate = :target THEN 'exact'
               WHEN f.previous_z IS NULL THEN 'above_target_at_min_z'
               ELSE 'interpolated'
           END AS status,
           CASE
               WHEN f.fill_rate = :target THEN f.stock_units
               WHEN f.previous_z IS NOT NULL THEN
                   f.previous_stock_units
                   + (:target - f.previous_fill_rate)
                     * (f.stock_units - f.previous_stock_units)
                     / (f.fill_rate - f.previous_fill_rate)
           END AS stock_units,
           CASE WHEN f.fill_rate = :target THEN f.z ELSE f.previous_z END AS lower_z,
           f.z AS upper_z
    FROM curve AS c
    LEFT JOIN first_reaching AS f
        ON f.variant = c.variant AND f.lead_time = c.lead_time AND f.method = c.method
       AND f.position = 1
)
SELECT variant, lead_time, :target AS target_fill_rate, method, status, stock_units,
       lower_z, upper_z, best_fill_rate,
       max(CASE WHEN method = 'lightgbm' THEN stock_units END) OVER scenario
           / max(CASE WHEN method = 'ma28' THEN stock_units END) OVER scenario - 1
           AS lightgbm_vs_ma28
FROM endpoint
WINDOW scenario AS (PARTITION BY variant, lead_time)
ORDER BY variant <> 'primary', variant, lead_time DESC, method;
