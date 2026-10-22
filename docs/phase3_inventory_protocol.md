# Phase 3 inventory simulation protocol

**Status: DRAFT v0.3 (revised after two reviews). Not frozen.**

To freeze it, the maintainer:

1. confirms the decisions in Section 15 on the Phase 3A issue (#31);
2. adds a separate commit that changes this status line to `FROZEN v1.0`, with the date and a link to the confirmation;
3. merges that commit.

No inventory metric may be computed on real M5 data before that merge. Any later change gets a new version number and is disclosed in the final report.

## 1. Research question

The frozen Issue #28 benchmark gave a mixed result. LightGBM had lower daily pooled WAPE than MA28 (73.50% vs. 74.38%), but MA28 had lower weekly WAPE, 28-day WAPE and RMSE. Inventory decisions depend on cumulative demand over a replenishment cycle, so forecast error alone does not say which method gives better ordering decisions.

> Under identical replenishment rules and the same historical demand, does replacing MA28 forecasts with the frozen LightGBM forecasts give a better trade-off between fill rate and inventory held?

A result in either direction, or no detectable difference, is a valid outcome. The answer holds for the policy family and assumptions below, not for every inventory system.

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
- **Warm-up:** 2016-01-04 through 2016-01-31 (28 days) are simulated but excluded from every metric. This is a chosen assumption. It does not guarantee that every effect of the starting state is gone (Section 8).
- **Evaluation window:** 2016-02-01 through 2016-04-24 (84 days, 12 weeks). This window is identical for every method, lead time, safety-stock setting and sensitivity run.

## 4. Daily event order

For each series, on each simulated day `d`:

1. Every order due on day `d` arrives before demand.
2. Demand `D_d` is the observed M5 sales quantity. Units sold are `min(on_hand, D_d)`. The rest is **lost**: no backorders, no substitution.
3. End-of-day on-hand is recorded.
4. If `d` is a review day, the inventory position is computed as `IP = on_hand + outstanding orders`. The order is `Q = max(0, ceil(S - IP))`, using a 1e-9 tolerance so floating-point noise never adds a unit. The order arrives at the start of day `d + L + 1`.

The first review, at t0, happens before the first simulated day.

**Lead-time convention:** an order is placed on Sunday evening, then exactly `L` days of demand pass, and the delivery is available the next morning. For `L = 7`, demand on Monday through Sunday is served from existing stock, and the delivery is on the shelf the following Monday.

An order placed at review `t` has to cover demand from `t + 1` through `t + L + R`, until the next order arrives. That span is the **protection interval** `P = L + R`.

## 5. Scenarios

| Parameter | Values |
|---|---|
| Review period `R` | 7 days (Sunday reviews) |
| Lead time `L` | **7 (primary)**, 3, 1, so `P` = 14, 10, 8 days |
| Safety factor `z` | 0 to 3.0 in steps of 0.1 (31 values) |
| Methods | LightGBM, MA28, plus the perfect-foresight policy reference (Section 7) |

All `P` values are within the 28-day forecast horizon. The grid is fixed in advance: no value is added, removed or chosen after looking at Phase 3 results. A finer grid shortens the gap between simulated policies, but the endpoint is still an interpolation (Section 11).

## 6. Order-up-to level

At review `t`, each method sums its own daily forecasts over the protection interval:

```
F(i, t) = sum of forecasts for horizons 1..P made at origin t for series i
S(i, t) = F(i, t) + z * sqrt(F(i, t))
```

- **LightGBM:** daily predictions from `predict_at_origin()` at origin `t`, using sales through `t` and future-known calendar/SNAP features only.
- **MA28:** `P` times the mean of the last 28 days of sales through `t`.

**Why `sqrt(F)`:** demand here is mostly small counts, and for Poisson counts the standard deviation of `P`-day demand is the square root of its mean. It is a simple heuristic that needs no fitted error model. Residual-based calibration on the validation period would be unfair: the frozen model's training labels cover 2015-10-19 through 2016-01-03, so its validation-period residuals are in-sample, while MA28's are out-of-sample.

**What it does not guarantee:** both methods use the same rule, but the rule acts on different forecasts. Their errors, and sparse demand, can interact with it differently. Real demand is also more variable than Poisson. A scale error shared by all series only moves points along each curve, within the range the grid covers. Series-specific mismatches remain. The experiment therefore compares the two forecasters **within this policy family**. It is not a universal ranking of forecasters for every inventory system.

`z` is a lever that traces each method's trade-off curve. It is **not** a service-level guarantee, so `z = 1.645` must not be described as "95% service". Service is measured, never assumed. The `inventory.service_level: 0.95` value in `config.yaml` is not used by Phase 3.

**Integer units:** `ceil(S - IP)` raises the inventory position to the smallest whole number at or above `S`. One consequence matters most for sparse series: any strictly positive forecast keeps the **inventory position** (on hand plus on order) at one unit or more. The shelf itself can still be empty while that unit is in transit. A zero forecast orders nothing. A round-half-up variant is run as a robustness check (Section 11).

## 7. Perfect-foresight policy reference

The reference uses `S(i, t)` = the **actual** demand over the protection interval, with `z = 0`. It is a reference with future information, not a feasible method. It is the only method whose order-up-to levels use future demand. The primary starting stock (Section 8) also uses future demand, for every method.

**What it shows, under stated conditions:** with the primary starting stock (Section 8), the reference ends every day holding exactly the demand that must be served before the next scheduled delivery. Any policy with the same review days and lead time that loses no sales must hold at least that much on every day. Under these conditions it is therefore the minimum average stock compatible with a 100% fill rate, and a test checks this property day by day.

It does not bound policies with a different delivery schedule, such as daily deliveries. With an empty start it is not minimal either: the first order would include demand lost before the first delivery.

**After the span:** demand after 2016-04-24 is unknown and counts as zero for the reference. Sales inside the span never change. For `L = 7`, no delivery that arrives inside the span is affected. For `L = 3` and `L = 1`, the reference holds less stock on the last 4 and 6 days of the span. It does not stock for demand after the span, while the forecasting methods do.

## 8. Initial state and end of span

- **Primary start:** each series begins with on-hand stock equal to its observed demand over the first `L` simulated days, with no outstanding orders. The value is identical for every method and `z`.
- **This start uses future demand.** It is never a forecasting input, but it does enter decisions. The starting stock is part of the inventory position at the first reviews, so it changes the first orders. For example, with a target of 100 units, an empty start orders 100, while a start of 70 orders 30. No store could set up this stock in advance without knowing future demand, so it is an idealized experimental condition. It is the same for every method, but that does not guarantee it affects every method equally.
- **Why not an empty start:** with zero stock, the first order also covers demand that is lost before it arrives. For a rare-demand series that surplus can remain for the whole window. Example: with 10 units a day for a week, then nothing, and `L = 7`, an empty start leaves 70 units on the shelf to the end of the span, after any warm-up. The primary start removes this surplus by construction.
- **Sensitivity:** the empty start is also run on the same evaluation window, with the same endpoint and bootstrap, and reported next to the primary results. Starting effects cannot be ruled out in general. If the empty-start verdict (Section 11) differs from the primary verdict, the result is reported as dependent on the starting state, and the primary result is not presented as a robust advantage of either method.
- **End of span:** there is no terminal valuation. Metrics use evaluation days only, and orders arriving after 2016-04-24 never count.

## 9. Information rules

- At review `t`, a decision may use only that method's forecast from origin `t` and the simulated inventory position. Forecasts never use future demand. Future demand enters the simulation in only two places: the perfect-foresight reference's targets, and the primary starting stock (Section 8), which changes the first orders of every method.
- Both forecasters see the **external** M5 sales history, not the simulated sales. Simulated stockouts do not censor what the forecasters see. This isolates forecast quality but leaves out the feedback loop of a real store.
- The frozen model is not refitted after 2016-01-03. It sees new sales through its features at every origin, but its learned relationships stay those of the October 2015 – January 2016 training period; for example, target months 2–4 were never seen in training. This may handicap LightGBM later in the span. It is accepted as a conservative choice and disclosed.
- These windows were already exposed in Phase 1 and Phase 2. Every Phase 3 parameter is fixed in this document before any Phase 3 run, and nothing is tuned on Phase 3 outcomes.

## 10. Metrics

Each metric is pooled over the cohort and the 84 evaluation days, for every combination of method, `L` and `z`:

| Metric | Definition |
|---|---|
| Fill rate | units sold / units demanded: the share of demanded **units** served. The data has no customers. |
| Lost units | units demanded − units sold |
| Stockout rate | series-days with a lost sale / series-days with positive demand |
| Average on-hand | mean over evaluation days of total end-of-day units on hand |
| Deliveries | number of positive receipts on evaluation days |
| Units received | units received on evaluation days |

**Value proxy:** `v_i` is the median of series `i`'s available daily sell prices over the 28 days through 2016-01-03, in US dollars per unit. If a series has no available price in that window, the run stops and names the series. It is never dropped silently or given a zero value. Sell price is not cost.

Secondary, descriptive:

- **Value-weighted metrics:** `sum_i v_i × on_hand_unit_days_i / 84` (average stock value) and `sum_i v_i × lost_units_i` (lost sales value).
- **Illustrative cost**, over the 84 evaluation days:
  - holding: `sum_i v_i × on_hand_unit_days_i × 0.25 / 365`;
  - penalty: `sum_i v_i × lost_units_i × p`, for `p` in {0.25, 0.5, 1.0}.

  Results are reported in illustrative dollars at sell price, never as Walmart costs.

## 11. Primary endpoint, uncertainty and interpretation

**Primary endpoint (`L = 7`): interpolated inventory at a 95% fill rate.** On each method's curve, in `z` order, the endpoint is linearly interpolated between the two simulated points that first bracket 95%. It is an estimate between two simulated policies, not a policy that was itself simulated, and integer ordering can make the true curve step-like. Every report shows:

- the endpoint status: `exact`, `interpolated`, `above_target_at_min_z` (the smallest `z` already exceeds 95%), or `not_reached`;
- both bracketing simulated points (`z`, fill rate, average on-hand);
- the first simulated `z` that reaches at least 95%, as a policy that was actually run.

The comparison is

```
delta = (inventory_LightGBM - inventory_MA28) / inventory_MA28
```

It is computed only when both endpoints are `exact` or `interpolated`. Otherwise the primary endpoint is "not estimable", the two statuses are reported, and the full curves are the result.

**Uncertainty: paired cluster bootstrap.**

- **Clusters:** whole items (`item_id`) are resampled with replacement, so all stores of the same product stay together. The same resample is used for both methods.
- **Resamples:** 2,000, seed 42, with a 95% **percentile** interval.
- **Inside each resample:** pooled fill rates and average on-hand are recomputed for every `z` from per-series totals, and both endpoints are re-interpolated.
- **Invalid resamples:** a resample is invalid if its evaluated demand is zero, either endpoint is not estimable, or the MA28 endpoint is zero. Invalid resamples are counted and reported. The interval uses valid resamples only, and it is labelled unreliable when invalid resamples exceed 1%.

**Interpretation:**

- **Practically equivalent:** the whole interval lies inside (−2%, +2%). A small point estimate alone is not evidence of equivalence: for example, −1% with an interval of −15% to +12% is inconclusive.
- **LightGBM needs less inventory:** the whole interval is below 0 and not inside the band.
- **MA28 needs less inventory:** the whole interval is above 0 and not inside the band.
- Otherwise the result is **inconclusive**.
- **Starting-state check:** if the empty-start verdict differs from the primary verdict, the conclusion is reported as dependent on the starting state (Section 8).
- The interval describes uncertainty from resampling items within this cohort and this 12-week period. Store-wide and calendar shocks shared across items are not captured, and the conclusion does not transfer to other years, stores or policy families.

Secondary analyses are descriptive and have no decision rule:

- full trade-off curves for every `L`, with the perfect-foresight point;
- the same endpoint at 90% and 98% fill rate;
- breakdowns by demand group (frequent, mixed, sparse, using `build_demand_profile` at 2016-01-03) and by category;
- value-weighted metrics and the illustrative cost grid;
- robustness checks: the empty start (Section 8), and round-half-up ordering.

## 12. Consistency checks before any inventory metric

1. LightGBM forecasts at the four Issue #28 origins (2016-01-03, 01-31, 02-28, 03-27) must match the saved Issue #28 predictions within 1e-9. The predictions file is `predictions.parquet`, and its hash is `prediction_sha256` in the public reveal manifest.
2. MA28 forecasts at those origins must match the Phase 1 predictions in `data/processed/m5_backtest_predictions.parquet` exactly.
3. The cohort, model and data hashes must match the recorded ones.
4. Every series and review must have finite, nonnegative forecasts for horizons `1..P`.
5. Every series must have a value proxy (Section 10).

If any check fails, the run stops without writing results.

## 13. Assumptions and limitations

- Observed sales are treated as demand. Demand censored by Walmart's own historical stockouts is not recovered.
- Lost sales only: no backorders, substitution or cross-store transfers.
- Deterministic lead times, unlimited supplier capacity, no minimum order quantities or case packs, no shelf capacity, no perishability, no fixed ordering cost.
- Each product-store series is simulated independently, as a single echelon.
- The primary scenario starts from an idealized stock equal to the actual demand of the first `L` days. A real store could not set this up without knowing future demand, and it influences the first orders.
- The comparison is between forecasters inside one policy family (`S = F + z·sqrt(F)`, weekly review, lost sales).
- Results describe a 2,996-series sample over 12 evaluation weeks in early 2016, not the full M5 population or production use. They are simulation results under these assumptions, not measured savings of a real business.

## 14. Implementation stages

| Stage | Content | Needs real data |
|---|---|---|
| 3A | This protocol, reviewed and frozen | No |
| 3B | `src/inventory_simulation.py`, `src/inventory_policy.py`, `src/inventory_metrics.py`: simulator, ordering rule, starting stock, reference, metrics, endpoint statuses, paired cluster bootstrap, costs and value proxy, with tests (implemented alongside this draft) | No |
| 3C | Weekly LightGBM/MA28/reference protection forecasts and the Section 12 checks | Model artifacts + data |
| 3D | `scripts/run_inventory_simulation.py` and reports in `reports/phase3_inventory/`, never overwriting existing results | Model artifacts + data |

## 15. Decisions for the maintainer to confirm before freezing

1. `ceil` ordering as primary, with round-half-up only as a robustness check.
2. `S = F + z·sqrt(F)` with `z` from 0 to 3.0 in steps of 0.1 (31 values).
3. Starting stock equal to each series' demand over the first `L` days, the same for all methods. It is idealized: it uses future demand and changes the first orders. An empty-start sensitivity run gets the same endpoint and bootstrap; if its verdict differs, the result is reported as dependent on the starting state. A 28-day warm-up, and evaluation from 2016-02-01 to 2016-04-24.
4. `L` ∈ {7 primary, 3, 1}, with `R = 7` and Sunday reviews.
5. Primary endpoint: interpolated inventory at 95% fill for `L = 7`, reported with its status, both bracketing points and the first simulated point at or above 95%.
6. Paired item-cluster percentile bootstrap: 2,000 resamples, seed 42. Equivalence is claimed only if the whole interval is inside ±2%.
7. Inclusion of the perfect-foresight policy reference, with the conditions in Section 7.
8. Value proxy and illustrative cost as defined in Section 10 (25% per year holding; penalty 0.25, 0.5 or 1.0 times the value proxy).

## Changes from v0.2

- **Starting stock:** the claim that no decision sees it is withdrawn. It uses future demand and changes the first orders through the inventory position, and this is now described in Sections 7, 8, 9 and 13. A preregistered rule is added: if the empty-start verdict differs from the primary one, the result is reported as dependent on the starting state.

## Changes from v0.1

- **Start:** the empty start is replaced by starting stock equal to the first-`L`-day demand, because an empty start can leave a surplus that never sells. The empty start is kept as a sensitivity run. The claim that the warm-up removes all start effects is withdrawn.
- **Reference:** renamed the perfect-foresight *policy reference*. Its minimum-stock property is now stated with its conditions and tested. The effect of unknown demand after the span is described exactly.
- **Endpoint:** renamed *interpolated* inventory at 95% fill, with explicit statuses (including above-target-at-minimum-`z`), bracketing points, and the first simulated policy reaching the target. The `z` step goes from 0.25 to 0.1.
- **Bootstrap and decision rule:** item clusters instead of single series; a percentile interval; full recomputation inside each resample; rules for invalid resamples; equivalence requires the whole interval inside ±2%.
- **Wording:** the claim that the safety-stock rule affects both methods equally is withdrawn. Rounding up raises the inventory position, not necessarily the shelf. Fill rate is defined in units.
- **Definitions:** an exact cost formula with units, a stop rule for missing prices, and an explicit freeze step.
