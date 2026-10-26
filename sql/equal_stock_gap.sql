-- How much more demand MA28 serves than LightGBM with the same average stock, overall and by
-- demand group and category. A post-hoc, descriptive reading of the published curves: the
-- protocol did not preregister it.
--
-- Each curve is read as straight lines between its simulated policies. The gap between two such
-- curves is made of straight lines too, so its smallest and largest values over the stock range
-- both curves cover sit at a simulated stock level of one curve or the other. The query reads both
-- curves at every one of those levels.
WITH segment AS (
    -- each pair of neighbouring policies on a curve
    SELECT slice, variant, lead_time, method,
           stock_units AS stock_from, fill_rate AS fill_from,
           lead(stock_units) OVER curve AS stock_to,
           lead(fill_rate) OVER curve AS fill_to
    FROM curve_point
    WINDOW curve AS (PARTITION BY slice, variant, lead_time, method ORDER BY z_step)
),
covered AS (
    -- the stock range each curve covers
    SELECT slice, variant, lead_time, method,
           min(stock_units) AS lowest, max(stock_units) AS highest
    FROM curve_point
    GROUP BY slice, variant, lead_time, method
),
shared AS (
    -- the stock range both curves cover
    SELECT slice, variant, lead_time, max(lowest) AS lowest, min(highest) AS highest
    FROM covered
    GROUP BY slice, variant, lead_time
    HAVING count(*) = 2
),
level AS (
    -- every simulated stock level of either curve inside that shared range
    SELECT DISTINCT p.slice, p.variant, p.lead_time, p.stock_units
    FROM curve_point AS p
    JOIN shared AS s
        ON s.slice = p.slice AND s.variant = p.variant AND s.lead_time = p.lead_time
    WHERE p.stock_units BETWEEN s.lowest AND s.highest
),
reading AS (
    -- each curve's fill rate at each level; a level on a joint between two segments matches
    -- both, with the same value, so max() keeps one
    SELECT l.slice, l.variant, l.lead_time, l.stock_units, g.method,
           max(g.fill_from + (l.stock_units - g.stock_from) * (g.fill_to - g.fill_from)
               / (g.stock_to - g.stock_from)) AS fill_rate
    FROM level AS l
    JOIN segment AS g
        ON g.slice = l.slice AND g.variant = l.variant AND g.lead_time = l.lead_time
       AND l.stock_units BETWEEN g.stock_from AND g.stock_to
    GROUP BY l.slice, l.variant, l.lead_time, l.stock_units, g.method
),
gap AS (
    SELECT slice, variant, lead_time, stock_units,
           max(CASE WHEN method = 'ma28' THEN fill_rate END)
           - max(CASE WHEN method = 'lightgbm' THEN fill_rate END) AS ma28_minus_lightgbm
    FROM reading
    GROUP BY slice, variant, lead_time, stock_units
)
SELECT slice, variant, lead_time,
       count(*) AS stock_levels,
       min(stock_units) AS stock_from,
       max(stock_units) AS stock_to,
       100 * min(ma28_minus_lightgbm) AS smallest_gap_points,
       100 * max(ma28_minus_lightgbm) AS largest_gap_points
FROM gap
GROUP BY slice, variant, lead_time
ORDER BY slice <> 'all series', slice, variant <> 'primary', variant, lead_time DESC;
