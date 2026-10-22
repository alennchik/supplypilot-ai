# Phase 3 inventory simulation protocol

**Status: DRAFT v0.1. Not frozen.** The maintainer reviews this document, edits it if needed, confirms the decisions in Section 15 on the Phase 3A issue, and merges it. That merge freezes it as v1.0. No inventory metric may be computed on real M5 data before the freeze. Any later change gets a new version number and is disclosed in the final report.

## 1. Research question

The frozen Issue #28 benchmark gave a mixed result. LightGBM had lower daily pooled WAPE than MA28 (73.50% vs. 74.38%), but MA28 had lower weekly WAPE, 28-day WAPE and RMSE. Inventory decisions depend on cumulative demand over a replenishment cycle, so forecast error alone does not say which method gives better ordering decisions.

> Under identical replenishment rules and the same historical demand, does replacing MA28 forecasts with the frozen LightGBM forecasts give a better trade-off between service (fill rate) and inventory held?

A result in either direction, or no detectable difference, is a valid outcome.

## 2. Fixed inputs inherited from Issue #28

| Input | Rule |
|---|---|
| Cohort | The frozen 2,996-series cohort: the `series_id` values in the local Issue #28 artifact `lookups.parquet`. Its `cohort_sha256` must equal the value in the local `preflight.json`, which is also transcribed in `reports/ml_final_evaluation/primary/PUBLICATION_SUMMARY.md`. |
| LightGBM | The saved frozen model `cycle2_tweedie_8_origins_calendar` (1,743 trees, training labels through 2016-01-03), loaded from `data/processed/ml_final_evaluation/primary/model.txt` and `model.json`. It is **never refitted**. If either file is missing, or its hash differs from `artifact_sha256` in the local `preflight.json`, the run stops. |
| MA28 | `src/forecast_baselines.moving_average_forecast`, unchanged. |
| Data | `data/processed/sales.parquet`, `data/processed/series.parquet`, `data/raw/calendar.csv`, with the input hashes recorded in the local `preflight.json` (also transcribed in `PUBLICATION_SUMMARY.md`). |

## 3. Simulation calendar

- **First review (t0):** Sunday 2016-01-03, at end of day.
- **Simulated demand days:** 2016-01-04 through 2016-04-24 (112 days). This is the Issue #28 test span and the last day of the M5 data used here.
- **Reviews:** every 7 days at end of Sunday, 2016-01-03 through 2016-04-17 (16 reviews). Each review date is also a forecast origin.
- **Warm-up:** 2016-01-04 through 2016-01-31 (28 days) are simulated but excluded from every metric.
- **Evaluation window:** 2016-02-01 through 2016-04-24 (84 days, 12 weeks). This window is identical for every method, lead time and safety-stock setting.

## 4. Daily event order

For each series, on each simulated day `d`:

1. Every order due on day `d` arrives before demand.
2. Demand `D_d` is the observed M5 sales quantity. Units sold are `min(on_hand, D_d)`. The rest is **lost**: no backorders, no substitution.
3. End-of-day on-hand is recorded.
4. If `d` is a review day, the inventory position is computed as `IP = on_hand + outstanding orders`. The order is `Q = max(0, ceil(S - IP))`, using a 1e-9 tolerance so floating-point noise never adds a unit. The order arrives at the start of day `d + L + 1`.

The first review, at t0, happens before the first simulated day.

So lead time `L` means that exactly `L` days of demand pass between placing and receiving an order. An order placed at review `t` has to cover demand from `t + 1` through `t + L + R`, until the next order arrives. That span is the **protection interval** `P = L + R`.

## 5. Scenarios

| Parameter | Values |
|---|---|
| Review period `R` | 7 days (Sunday reviews) |
| Lead time `L` | **7 (primary)**, 3, 1, so `P` = 14, 10, 8 days |
| Safety factor `z` | 0 to 3.0 in steps of 0.25 (13 values) |
| Methods | LightGBM, MA28, plus a perfect-foresight reference (Section 7) |

All `P` values are within the 28-day forecast horizon. The grid is fixed in advance: no value is added, removed or chosen after looking at Phase 3 results.

## 6. Order-up-to level

At review `t`, each method sums its own daily forecasts over the protection interval:

```
F(i, t) = sum of forecasts for horizons 1..P made at origin t for series i
S(i, t) = F(i, t) + z * sqrt(F(i, t))
```

- **LightGBM:** daily predictions from `predict_at_origin()` at origin `t`, using sales through `t` and future-known calendar/SNAP features only.
- **MA28:** `P` times the mean of the last 28 days of sales through `t`.

**Why `sqrt(F)` and not the standard deviation of residuals:** demand here is mostly small counts, and for Poisson counts the standard deviation of `P`-day demand is the square root of its mean. One transparent formula applies equally to both methods and needs no fitted error model. Residual-based calibration on the validation period would not be fair. The frozen model's training labels cover 2015-10-19 through 2016-01-03, so its validation-period residuals are in-sample, while MA28's are out-of-sample.

Real demand is more variable than Poisson. Because every `z` is evaluated, a scale error shared by all series only moves points along each curve. Series-specific overdispersion remains a limitation, and it applies to both methods equally.

`z` is a lever that traces each method's trade-off curve. It is **not** a service-level guarantee, so `z = 1.645` must not be described as "95% service". Service is measured, never assumed. The `inventory.service_level: 0.95` value in `config.yaml` is not used by Phase 3.

**Integer units:** `ceil(S - IP)` raises the inventory position to the smallest whole number at or above `S`. This is the discrete analogue of a base-stock quantile. One consequence matters most for sparse series: any strictly positive forecast keeps at least one unit on hand, while a zero forecast stocks nothing. A round-half-up variant is run as a robustness check (Section 10).

## 7. Perfect-foresight reference

The reference uses `S(i, t)` = the **actual** demand over the protection interval, with `z = 0`. Days past 2016-04-24 count as zero; they only affect demand after the evaluation window. Once its first order has arrived, it never loses a sale. Its inventory shows how much stock weekly reviews require even with perfect knowledge. It is a bound for interpretation, not a feasible method, and it is the only place where future demand is used.

## 8. Initial state and end of span

- Every series starts with zero on-hand and no outstanding orders, identically for all methods and settings.
- The empty start mechanically distorts the first cycles: demand is lost before the first delivery, and the first order can leave a small overhang. The 28-day warm-up covers at least `L + 2R` days for every lead time in the grid, so the evaluation window starts after this distortion is gone.
- There is no terminal valuation. Metrics use evaluation days only, and orders arriving after 2016-04-24 never count.

## 9. Information rules

- At review `t`, a decision may use only that method's forecast from origin `t` and the simulated inventory position. Future demand is never used, except by the labelled perfect-foresight reference.
- Both forecasters see the **external** M5 sales history, not the simulated sales. Simulated stockouts do not censor what the forecasters see. This isolates forecast quality but leaves out the feedback loop of a real store.
- The frozen model is not refitted after 2016-01-03. It sees new sales through its features at every origin, but its learned relationships stay those of the October 2015 – January 2016 training period; for example, target months 2–4 were never seen in training. This may handicap LightGBM later in the span. It is accepted as a conservative choice and disclosed.
- These windows were already exposed in Phase 1 and Phase 2. Every Phase 3 parameter is fixed in this document before any Phase 3 run, and nothing is tuned on Phase 3 outcomes.

## 10. Metrics

Each metric is pooled over the cohort and the 84 evaluation days, for every combination of method, `L` and `z`:

| Metric | Definition |
|---|---|
| Fill rate | units sold / units demanded |
| Lost units | units demanded − units sold |
| Stockout rate | series-days with a lost sale / series-days with positive demand |
| Average on-hand | mean over evaluation days of total end-of-day units on hand |
| Deliveries | number of positive receipts on evaluation days |
| Units received | units received on evaluation days |

Secondary, descriptive:

- **Value-weighted** on-hand and lost sales, using each series' median sell price over the 28 days through 2016-01-03 as a value proxy. Sell price is not cost.
- **Illustrative cost:** a holding rate of 25% per year of the value proxy, plus a lost-sale penalty of 0.25, 0.5 or 1.0 times the value proxy. M5 has no cost data, so these figures are never presented as Walmart dollars.

## 11. Primary endpoint and decision rule

**Primary endpoint (`L = 7`):** the average on-hand units each method needs to reach a pooled **95% fill rate**. It is linearly interpolated between the two adjacent `z` values whose fill rates first bracket 95%, in `z` order. The comparison is

```
delta = (inventory_LightGBM - inventory_MA28) / inventory_MA28
```

with a 95% **paired bootstrap** confidence interval: 2,000 resamples of series with replacement, the same resample for both methods, seed 42.

- LightGBM is better if the whole interval is below 0. MA28 is better if it is entirely above 0. Otherwise there is no evidence of a difference.
- An absolute `delta` under 2% is described as practically negligible, even if the interval excludes 0.
- If either method does not reach 95% within the grid, the primary endpoint is reported as "not estimable" and the full curves are the result.
- Bootstrap resamples in which either endpoint is not estimable are counted and reported. If they exceed 1% of resamples, the interval is reported as unreliable.

Secondary analyses are descriptive and have no decision rule:

- full trade-off curves for every `L`, with the perfect-foresight point;
- the same endpoint at 90% and 98% fill rate;
- breakdowns by demand group (frequent, mixed, sparse, using `build_demand_profile` at 2016-01-03) and by category;
- value-weighted metrics and the illustrative cost grid;
- robustness checks: round-half-up ordering, and a 42-day warm-up.

The bootstrap resamples series, not days. Shocks shared across series are not modelled, so the intervals may be too narrow, and the report must say so.

## 12. Consistency checks before any inventory metric

1. LightGBM forecasts at the four Issue #28 origins (2016-01-03, 01-31, 02-28, 03-27) must match the saved Issue #28 predictions within 1e-9. The predictions file is `predictions.parquet`, and its hash is `prediction_sha256` in the public reveal manifest.
2. MA28 forecasts at those origins must match the Phase 1 predictions in `data/processed/m5_backtest_predictions.parquet` exactly.
3. The cohort, model and data hashes must match the recorded ones.
4. Every series and review must have finite, nonnegative forecasts for horizons `1..P`.

If any check fails, the run stops without writing results.

## 13. Assumptions and limitations

- Observed sales are treated as demand. Demand censored by Walmart's own historical stockouts is not recovered.
- Lost sales only: no backorders, substitution or cross-store transfers.
- Deterministic lead times, unlimited supplier capacity, no minimum order quantities or case packs, no shelf capacity, no perishability, no fixed ordering cost.
- Each product-store series is simulated independently, as a single echelon.
- Results describe a 2,996-series sample over 12 evaluation weeks in early 2016, not the full M5 population or production use.

## 14. Implementation stages

| Stage | Content | Needs real data |
|---|---|---|
| 3A | This protocol, reviewed and frozen | No |
| 3B | `src/inventory_simulation.py`, `src/inventory_policy.py`, `src/inventory_metrics.py`, with hand-traced tests (implemented alongside this draft) | No |
| 3C | Weekly LightGBM/MA28/reference protection forecasts and the Section 12 checks | Model artifacts + data |
| 3D | `scripts/run_inventory_simulation.py`, bootstrap and reports in `reports/phase3_inventory/`, never overwriting existing results | Model artifacts + data |

## 15. Decisions for the maintainer to confirm before freezing

1. `ceil` ordering as primary, with round-half-up only as a robustness check.
2. `S = F + z·sqrt(F)` with the 13-value `z` grid (0 to 3.0, step 0.25). The fine step keeps linear-interpolation error small at the 95% endpoint, and the extra runs cost well under a second each.
3. Zero initial stock, a 28-day warm-up, and evaluation from 2016-02-01 to 2016-04-24.
4. `L` ∈ {7 primary, 3, 1}, with `R = 7` and Sunday reviews.
5. Primary endpoint: inventory at 95% fill for `L = 7`, 2,000 paired bootstrap resamples, seed 42, 2% negligibility band.
6. Inclusion of the perfect-foresight reference.
7. Illustrative cost parameters (25% per year holding rate; penalty 0.25, 0.5 or 1.0 times the value proxy).
