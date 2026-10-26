-- The cheapest simulated policy for each forecaster, in every scenario, at given costs.
--
-- Parameters: :holding_rate, the yearly holding cost as a share of the sell price, and :penalty,
-- the cost of each lost sale as a multiple of its sell price.
--
--     holding cost   = average stock value x evaluated days x :holding_rate / 365
--     lost-sale cost = lost-sales value x :penalty
--
-- At the protocol's settings (25% a year; penalties of 0.25, 0.5 and 1) this reproduces the cost
-- columns published in curves.csv. Ties go to the lower z. grid_edge marks a cheapest policy at
-- the first or last z of the grid, where the true minimum may lie outside the tested range.
WITH cost AS (
    SELECT variant, lead_time, method, z_step, z, fill_rate, average_on_hand_units,
           average_on_hand_value * evaluated_days * :holding_rate / 365.0 AS holding_cost,
           lost_sales_value * :penalty AS lost_sale_cost
    FROM policy
),
ranked AS (
    SELECT *,
           holding_cost + lost_sale_cost AS total_cost,
           row_number() OVER (
               PARTITION BY variant, lead_time, method
               ORDER BY holding_cost + lost_sale_cost, z_step
           ) AS position,
           min(z_step) OVER curve AS first_step,
           max(z_step) OVER curve AS last_step
    FROM cost
    WINDOW curve AS (PARTITION BY variant, lead_time, method)
)
SELECT variant, lead_time, method, z, total_cost, holding_cost, lost_sale_cost, fill_rate,
       average_on_hand_units,
       CASE z_step WHEN first_step THEN 'lowest_z' WHEN last_step THEN 'highest_z' END
           AS grid_edge,
       max(CASE WHEN method = 'lightgbm' THEN total_cost END) OVER scenario
           / max(CASE WHEN method = 'ma28' THEN total_cost END) OVER scenario - 1
           AS lightgbm_vs_ma28
FROM ranked
WHERE position = 1
WINDOW scenario AS (PARTITION BY variant, lead_time)
ORDER BY variant <> 'primary', variant, lead_time DESC, method;
