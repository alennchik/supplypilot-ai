-- Checks that compare rows. Each row returned is a problem, so an empty result means the loaded
-- results are consistent. Checks on a single row are constraints in schema.sql.
WITH ordered AS (
    SELECT variant, lead_time, method, z_step, fill_rate, average_on_hand_units,
           lag(fill_rate) OVER curve AS previous_fill_rate,
           lag(average_on_hand_units) OVER curve AS previous_stock_units
    FROM policy
    WINDOW curve AS (PARTITION BY variant, lead_time, method ORDER BY z_step)
)
SELECT 'curve does not have the 31 policies from z = 0 to 3.0' AS problem,
       variant || ', ' || lead_time || '-day, ' || method AS location
FROM policy
GROUP BY variant, lead_time, method
HAVING count(*) <> 31 OR min(z_step) <> 0 OR max(z_step) <> 30

UNION ALL
SELECT 'fill rate is not sales divided by demand',
       variant || ', ' || lead_time || '-day, ' || method || ', z = ' || z
FROM policy
WHERE abs(fill_rate - 1.0 * sales_units / demand_units) > 1e-12

UNION ALL
SELECT 'fill rate falls, or stock does not rise, as z rises',
       variant || ', ' || lead_time || '-day, ' || method || ', z step ' || z_step
FROM ordered
WHERE fill_rate < previous_fill_rate OR average_on_hand_units <= previous_stock_units

UNION ALL
SELECT 'the two forecasters face different demand or days', variant || ', ' || lead_time || '-day'
FROM policy
GROUP BY variant, lead_time
HAVING count(DISTINCT demand_units) > 1 OR count(DISTINCT evaluated_days) > 1

UNION ALL
SELECT 'scenario has no perfect-foresight reference', p.variant || ', ' || p.lead_time || '-day'
FROM (SELECT DISTINCT variant, lead_time FROM policy) AS p
LEFT JOIN reference AS r ON r.variant = p.variant AND r.lead_time = p.lead_time
WHERE r.variant IS NULL

UNION ALL
SELECT 'perfect foresight loses sales', variant || ', ' || lead_time || '-day'
FROM reference
WHERE fill_rate < 1

UNION ALL
SELECT 'group curve does not have 31 policies', group_type || ': ' || group_name || ', ' || method
FROM group_policy
GROUP BY group_type, group_name, method
HAVING count(*) <> 31;
