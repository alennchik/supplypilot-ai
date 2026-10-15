# SupplyPilot AI

[![CI](https://github.com/alennchik/supplypilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/alennchik/supplypilot-ai/actions/workflows/ci.yml)

**Demand Forecasting & Inventory Planning | In Development**

SupplyPilot AI is a Python-based supply chain analytics project for forecasting retail demand and, in later phases, supporting data-driven inventory replenishment decisions. It uses the [Walmart M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) to build a reproducible data pipeline and evaluate forecasting methods on historical product-store sales.

**Current status (October 2026):** The Phase 1 data pipeline and frozen 28-day baseline evaluation are complete. Phase 2 now includes origin-relative demand features, a reproducible supervised ML dataset, and an **initial trained global LightGBM model** for direct 28-day forecasting. Model training, save/reload, and synthetic integration checks work locally. **LightGBM forecast accuracy has not yet been evaluated on validation or final-test windows**; objective/hyperparameter comparisons, inventory optimization, and the dashboard remain future work.

## Highlights

- **58,327,370** original daily sales values verified during wide-format CSV-to-Parquet conversion.
- **3,000** reproducibly selected product-store series; **4,550,960** unified daily observations.
- **2,996** eligible series evaluated over **four 28-day windows**, generating **1,006,656 predictions** across three models.
- **Best tested baseline:** 28-day moving average, **74.38% pooled WAPE**, versus **86.98%** for seasonal naive and **97.96%** for naive.
- **216 locally passing automated tests**; Ruff lint checks pass. GitHub Actions CI checks changes on pull requests and on `main`.
- Wide Parquet storage is **73.2% smaller** than CSV, with **17.94× faster** median full-table loading in the measured local benchmark (not a universal performance guarantee).
- **668,808 cutoff-bounded supervised ML rows** generated across eight weekly training origins, 28 forecast horizons, and 2,989 unique eligible series; 2.80 s generation time and 850.81 MiB process peak RSS on the local Mac benchmark.
- **Initial LightGBM training completed** on those 668,808 examples: 4.96 s model fitting, 451.70 MiB peak process RSS, and a successful 28-horizon save/reload smoke test on a MacBook Air M2 (8 GB RAM). Forecasting accuracy has not yet been measured.

## Current Capabilities

### Data Engineering

- Validate M5 identifiers and sales quantities, including null, negative, and integer-range checks.
- Convert the 30,490-series, 1,913-day wide M5 CSV to Apache Parquet using PyArrow.
- Process large datasets in batches and use temporary files followed by atomic replacement for individual outputs.
- Verify converted Parquet values against the original CSV.
- Deterministically sample 3,000 product-store series using stratification and SHA-256 ranking.
- Join sales with calendar dates and weekly observed prices to construct a unified daily sales table.
- Keep product metadata in a separate series table; preserve observed zero sales and distinguish them from missing history.

### Exploratory Analysis and Feature Engineering

- Analyze aggregate sales trends, top-selling products, and zero-sales behavior.
- Generate lag-1, lag-7, and lag-28 features and historical 7-day/28-day moving averages.
- Use chronological train/validation splits and avoid current-day target leakage in the earlier one-step feature pipeline.
- Compute additional origin-relative demand lags (1, 7, 14, 28), historical rolling averages, sums, standard deviations, and zero-sales fractions for direct 28-day ML examples.
- Build supervised rows with forecast horizons 1–28 and labels strictly at or before a configurable training cutoff; preserve missing-history values as `NaN`.

### Demand Forecasting and Evaluation

- Preserve preliminary one-step-ahead naive and seasonal-naive forecasting checks.
- Implement **direct 28-day** naive, seasonal naive (7 days), and moving average (28 days) baselines.
- Evaluate four non-overlapping chronological forecast windows with the same eligible product-store series.
- Calculate overall, per-window, per-category, and per-demand-sparsity MAE, RMSE, and **pooled WAPE**.
- Generate reproducible prediction files, CSV reports, and an experiment manifest.
- Generate reproducible ML training Parquet and a JSON dataset manifest; train a global LightGBM regressor across all 28 forecast horizons, save/load the model and category vocabulary, and record a model-training manifest.
- Predict nonnegative daily demand from **already-prepared origin-relative feature rows**; end-to-end inference from a new raw sales-history file and out-of-sample ML accuracy assessment are not yet implemented.

### Quality and Automation

- Pytest unit and integration tests, including future-data leakage, calendar boundaries, missing data, duplicate keys, reproducibility, and failed-write protection.
- Ruff linting and formatting; pinned dependencies and GitHub Actions CI.
- Source datasets, generated Parquet predictions, and trained model binaries excluded from version control; compact experiment reports can be tracked.

## Getting Started

### 1. Clone and install

Python 3.13 is the tested development environment.

```bash
git clone https://github.com/alennchik/supplypilot-ai.git
cd supplypilot-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

### macOS setup for LightGBM

On macOS, LightGBM may require the OpenMP runtime. Install it using Homebrew:

```bash
brew install libomp
```

Then install the Python dependencies as described above. The initial model was trained and tested on a MacBook Air M2 (8 GB RAM) using Python 3.13.1, LightGBM 4.7.0, and scikit-learn 1.9.1.

On Linux or Windows, follow the platform-specific LightGBM installation requirements. Homebrew is not required on those systems.

```
On macOS with Homebrew, LightGBM may need the OpenMP runtime (`brew install libomp`), as used in the local training run. The training command currently uses the Unix `resource` module to report peak RSS and is **tested on macOS**, not native Windows; use a Unix-like environment for reproducing its resource report.

### 2. Download the M5 input files

Download the data from the [M5 Forecasting – Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place the following files in `data/raw/`:

```text
data/raw/sales_train_validation.csv
data/raw/calendar.csv
data/raw/sell_prices.csv
```

The repo does **not** bundle the source data. Raw and processed datasets are ignored by Git.

### 3. Build the datasets

Run from the repository root:

```bash
python -m scripts.convert_to_parquet
python -m scripts.verify_parquet
python -m scripts.sample_m5
python -m scripts.build_m5_unified
```

This produces `data/processed/sales_train_validation.parquet`, `series.parquet`, and `sales.parquet` (among other intermediate files). The fixed sampled series IDs are tracked in `reports/sample_series.csv`; the sampling script checks against this manifest to detect accidental sample changes.

### 4. Run the 28-day forecasting benchmark

```bash
python -m scripts.run_backtesting
```

The default M5 experiment ends on **2016-04-24**, and writes a detailed prediction Parquet file to `data/processed/` and summary reports to `reports/`. Use `python -m scripts.run_backtesting --help` for custom input/output paths and evaluation end date. Changing the end date changes the evaluation protocol; use the fixed default to reproduce the published results.

### 5. Build the Phase 2 ML training dataset

After preparing the unified daily data in step 3, run:

```bash
python -m scripts.build_ml_dataset --cutoff 2015-11-08 --origins 8
```

This generates `data/processed/ml_training.parquet` (ignored by Git) and `reports/ml_training_manifest.json` (version-controlled). No LightGBM dependency is needed to generate the dataset. The fixed cutoff and weekly origins reproduce the documented experiment; run `python -m scripts.build_ml_dataset --help` for optional paths and origin count.

### 6. Train the initial global LightGBM model

After generating `data/processed/ml_training.parquet`, run from the repository root:

```bash
python -m scripts.train_ml_model
```

The default run fits one CPU LightGBM regression model (100 boosting rounds, two threads, seed 42) to examples with labels through **2015-11-08**. It writes `models/global_lightgbm.txt` (LightGBM trees), `models/global_lightgbm.json` (training-time categorical vocabulary and feature schema), and `reports/ml_model_training_manifest.json` (configuration, runtime, environment, and training counts). The `models/` directory is excluded from Git; the report is intended to be version-controlled. You can inspect options with `python -m scripts.train_ml_model --help`.

This is a **training and persistence smoke test**, not a validation score. The script checks the training cutoff and reload consistency on one prepared 28-day forecast path. It does not run the later validation or final-test evaluation.

### 7. Run quality checks

```bash
python -m ruff check .
python -m pytest -q
```

The documented local run passed **216 tests** with one known Pandas `RuntimeWarning` from a deliberately invalid-value test. A successful local run does not itself mean CI has passed for an unmerged pull request.

## M5 Parquet Storage and Performance

The wide-format converter preserves **30,490 product-store series**, **1,919 columns** (six identifiers plus 1,913 daily values), and **58,327,370** historical sales observations. It validates quantities before storing them in a compact integer format and uses temporary-file replacement to protect a valid output from unsuccessful conversion.

Verification on the full source dataset found matching identifiers and sales values in the original CSV and converted Parquet output.

### Local performance benchmark

Measured on macOS with Python 3.13.1, Pandas 3.0.6, and PyArrow 26.0.0. Three runs per loading scenario; values shown are medians.

| Metric | CSV | Parquet | Result |
|---|---:|---:|---|
| Full-table loading | 1.810 s | 0.101 s | **17.94× faster** |
| IDs plus last 365 days | 0.666 s | 0.041 s | **16.14× faster** |
| Full-load peak memory | 368.6 MiB | 410.6 MiB | Higher with Parquet |
| Recent-load peak memory | 171.6 MiB | 176.4 MiB | Higher with Parquet |
| File size | ~114 MiB | 30.62 MiB | **73.2% smaller** |

The timing scenarios load either all 1,919 columns or the six identifier columns plus the last 365 sales days. CSV uses explicit data types; Parquet is read through Pandas/PyArrow. Each timed load runs in a separate process; execution order is randomized with a fixed seed. Operating-system filesystem caches were not cleared. Peak memory includes Python and library overhead.

Parquet improved disk usage and load time **but did not reduce measured peak memory**. These results describe this environment, dataset, and benchmark, not guaranteed speedups elsewhere.

Reproduce the benchmark with:

```bash
python -m scripts.benchmark_parquet --repeats 3
```

Detailed runs and environment information are in [`reports/parquet_benchmark.json`](reports/parquet_benchmark.json).

## Reproducible M5 Sampling and Unified Daily Data

### Sampling methodology

Exactly **3,000 of 30,490** product-store series were selected using metadata alone, without inspecting demand values. The algorithm allocates sample sizes proportionally across the **30 `store_id × cat_id` strata** by the largest-remainder method, breaking ties deterministically, then ranks series using SHA-256 with seed `42`.

For each of ten stores, the selected sample contains:

| Category | Series per store |
|---|---:|
| FOODS | 141 |
| HOBBIES | 56 |
| HOUSEHOLD | 103 |
| **Total** | **300** |

The full list of chosen IDs is stored in [`reports/sample_series.csv`](reports/sample_series.csv) for reproducibility.

### Output schema

| File | Contents |
|---|---|
| `data/processed/sales_train_validation.parquet` | Verified original wide-format M5 sales |
| `data/processed/series.parquet` | `series_id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| `data/processed/sales.parquet` | `series_id`, `date`, `quantity`, `price`, `price_available` |

The unified pipeline maps M5 `d_1`–`d_1913` columns to real dates and joins weekly observed prices on `store_id`, `item_id`, and `wm_yr_wk`. It removes each series' days **before the first observed price** as a proxy for available history—not as proof of a product launch. Observed zero-sales days are retained. Missing later prices would remain null with `price_available = False`, although **none occurred after trimming in this particular sample**.

An observed price does not prove inventory availability, so missing prices and stockouts must not be conflated. Baseline forecasts in this phase use **only historical sales quantities**, not future prices or calendar/event features.

### Local data validation

| Metric | Result |
|---|---:|
| Selected product-store series | 3,000 |
| Unified daily rows | 4,550,960 |
| Zero-sales rows | 2,724,948 (about 60%) |
| Missing prices after trimming | 0 |
| Average available history | Approximately 1,517 days per series |

Checks covered unique series/date pairs, consecutive dates within each series, valid quantities, and price-availability consistency. Independently comparing all **4,550,960 retained daily quantities** to the original M5 wide table found no mismatches.

## Direct 28-Day Forecasting and Rolling-Origin Backtesting

The primary benchmark evaluates **28-day-ahead paths predicted at a fixed origin**, not one-step forecasts that can use actual sales from earlier days inside the test period.

### Baseline models

| Model | Prediction for the next 28 days |
|---|---|
| **Naive** | Repeat the last observed daily quantity for every forecast day |
| **Seasonal naive (7)** | Repeat the last seven observed daily quantities four times |
| **Moving average (28)** | Repeat the average daily quantity from the previous 28 observed days |

These models are implemented in `src/forecast_baselines.py`. Earlier one-step baselines remain in `src/baselines.py` as separate preliminary checks.

### Evaluation protocol

- **Dataset:** 3,000 sampled Walmart M5 product-store series; daily dates through **2016-04-24**.
- **Horizon:** 28 calendar days per origin, with **four non-overlapping final-evaluation windows**.
- **Eligibility:** at least 28 consecutive observed historical days at the **first** evaluation origin (`2016-01-03`); **2,996 eligible**, **4 excluded**. This cohort stays fixed for all models and windows.
- **No leakage:** each forecast receives only historical demand available through its origin. Earlier completed windows become historical data at later origins; future actual sales never enter the same origin's forecast inputs.
- **Missingness:** absence of an observation is `NaN`, not zero. Missing target observations for an already eligible series raise an error instead of silently removing the series.
- **Comparison:** all three models use identical eligible series, dates, forecast origins, and actual quantities.

| Window | Forecast origin | 28-day prediction period |
|---|---|---|
| 1 | 2016-01-03 | 2016-01-04 to 2016-01-31 |
| 2 | 2016-01-31 | 2016-02-01 to 2016-02-28 |
| 3 | 2016-02-28 | 2016-02-29 to 2016-03-27 |
| 4 | 2016-03-27 | 2016-03-28 to 2016-04-24 |

Each model has **335,552** evaluated daily predictions (`2,996 × 4 × 28`), for **1,006,656 total prediction rows**. The combined actual demand over those 335,552 target observations is **482,952 units** (the same target set for every model).

### Error metrics

- **MAE:** mean absolute daily error, in units sold.
- **RMSE:** root mean squared daily error, in units sold; penalizes larger misses more.
- **Pooled WAPE:** `100 × sum(abs(actual − predicted)) / sum(actual)` over **all observations in the reported group**. It is **not** an average of per-series WAPE percentages.
- **Undefined WAPE:** if a group's total actual demand is zero, WAPE is reported as missing (`NaN`), not zero.

All three metrics are calculated through the existing implementations in `src/metrics.py` using numerical arrays converted to `float64`.

**WAPE is an error metric, not an accuracy percentage.** Since actual demand is nonnegative, an **always-zero** forecast would have exactly **100% WAPE** when total actual demand is positive. This is a mathematical reference point, **not a separately executed model in the benchmark**.

### Overall results

| Model | MAE ↓ | RMSE ↓ | Pooled WAPE ↓ |
|---|---:|---:|---:|
| **Moving average (28)** | **1.071** | **2.600** | **74.38%** |
| Seasonal naive (7) | 1.252 | 3.096 | 86.98% |
| Naive | 1.410 | 3.499 | 97.96% |

**Finding:** the 28-day moving average has the lowest MAE, RMSE, and pooled WAPE of the three tested baselines on this fixed M5 evaluation. Its pooled WAPE is **23.58 percentage points lower than naive**, a **24.1% relative reduction in aggregate absolute error**. This is a baseline benchmark, not evidence of production-ready forecast quality.

### Performance by demand intermittency

Demand groups were defined using each eligible series' proportion of zero-sales days in **at most the 365 days ending at the first forecast origin (`2016-01-03`)**. Series with shorter history use their available observed days (minimum observed: 58). Group definitions were fixed **before looking at forecasting errors**:

- **Frequent:** fewer than 25% zero-sales days.
- **Mixed:** 25%–75% zero-sales days, inclusive.
- **Sparse:** more than 75% zero-sales days.

| Demand group | Series | MA28 WAPE ↓ | Seasonal naive WAPE ↓ | Naive WAPE ↓ |
|---|---:|---:|---:|---:|
| Frequent | 334 (11.15%) | **51.92%** | 63.32% | 74.31% |
| Mixed | 1,584 (52.87%) | **88.48%** | 103.22% | 114.31% |
| Sparse | 1,078 (35.98%) | **115.25%** | 122.81% | 133.10% |

**Finding:** the moving average wins in all three demand groups, but even its error exceeds the always-zero **100% WAPE reference** for sparse demand. Forecasting rarely sold items remains a significant problem. That observation does **not** justify tuning a model or choosing a special strategy using these final-test results.

Frequent series make up only **11.15% of series** but account for approximately **45.04% of target demand** in this evaluation. Sparse series account for **35.98% of series** but only about **8.84% of target demand**. This helps explain why overall pooled WAPE is closer to the errors of higher-volume groups.

### Performance by category

| Category | Series | MA28 WAPE ↓ | Seasonal naive WAPE ↓ | Naive WAPE ↓ |
|---|---:|---:|---:|---:|
| FOODS | 1,406 | **68.32%** | 79.64% | 90.13% |
| HOUSEHOLD | 1,030 | **80.25%** | 95.25% | 105.78% |
| HOBBIES | 560 | **102.84%** | 118.74% | 134.17% |

The moving average is best among the tested methods in all three categories. HOBBIES has the highest relative error; these tables alone do not establish why.

### Stability across test windows

| Prediction period | MA28 WAPE ↓ | Seasonal naive WAPE ↓ | Naive WAPE ↓ |
|---|---:|---:|---:|
| Jan 4–31, 2016 | **79.04%** | 92.89% | 100.59% |
| Feb 1–28, 2016 | **72.28%** | 83.88% | 97.79% |
| Feb 29–Mar 27, 2016 | **73.64%** | 85.12% | 95.54% |
| Mar 28–Apr 24, 2016 | **72.96%** | 86.58% | 98.18% |

The ranking is consistent across all four windows. The first window has the highest MA28 WAPE; its lower total actual demand also contributes to the larger relative error. Causes such as holidays have **not** been established by this analysis.

### Reproduce the forecasting benchmark

After completing the data preparation steps, run:

```bash
python -m scripts.run_backtesting
```

The script loads `sales.parquet` and `series.parquet`, generates the direct 28-day predictions, creates historical demand groups, evaluates all models using existing metrics, and saves:

| Output | Version-controlled? | Purpose |
|---|---|---|
| `data/processed/m5_backtest_predictions.parquet` | No | Detailed actual and predicted demand by series, origin, date, and model |
| `reports/m5_baselines_overall.csv` | Yes | Overall model metrics |
| `reports/m5_baselines_by_window.csv` | Yes | Metrics per forecast origin |
| `reports/m5_baselines_by_category.csv` | Yes | Metrics by product category |
| `reports/m5_baselines_by_sparsity.csv` | Yes | Metrics by demand group |
| `reports/m5_demand_group_summary.csv` | Yes | Demand-group sizes and pre-test zero-sale fractions |
| `reports/m5_backtesting_manifest.json` | Yes | Evaluation windows, models, cohort size, and grouping rules |

Reports are recomputable from the M5 inputs. The script uses temporary files and atomic replacement for **each output**, but the collection of reports is not one all-or-nothing transaction. Tests use small synthetic Parquet files, so CI does not require the large M5 dataset.

### Limitations and evaluation integrity

- The evaluation covers a **deterministic 3,000-series sample**, not the full 30,490-series M5 population.
- A 28-day average is a simple constant daily forecast; it does not model promotions, holidays, trends, or uncertainty.
- Demand is often intermittent; around 60% of unified daily rows have zero sales. WAPE above 100% is possible, especially for low-volume groups.
- We have already inspected results for the four final-evaluation windows. **Do not use them for selecting future LightGBM features, objectives, or hyperparameters.** Future model development must rely on separate, earlier validation windows.
- Planned Phase 2 tuning windows are **`d_1746`–`d_1773`** and **`d_1774`–`d_1801`**, with origins at `d_1745` and `d_1773`. Determine eligibility using the history available at **each tuning origin**, not at the later final-test origin.
- Forecast accuracy is not the same as inventory effectiveness; stockout costs, service levels, and lead times have not yet been modeled.

## Phase 2: Leakage-Safe ML Features, Training Data, and Initial LightGBM

This phase creates training examples for a **global, direct multi-horizon LightGBM forecasting model**. The dataset generator, initial model-training pipeline, and save/reload tests are implemented. The model has been **trained locally**; **hyperparameter comparison, independent validation, and final ML evaluation have not yet been run**.

### Feature design

Each example has a unique `(series_id, origin, horizon)` key, where `horizon` ranges from 1 to 28. `date = origin + horizon`, and `target` is the observed unit sales on that future date. Historical demand features are calculated **as of the forecast origin**, rather than the target date.

- **Demand lags:** sales at the origin, plus lag 1, 7, 14, and 28 days relative to the origin.
- **Rolling features:** mean demand over 7, 28, and 56 days; standard deviation over 7 and 28 days; sum over 7 and 28 days; and zero-sales fractions over 28 and 56 days.
- **Known identifiers:** `store_id`, `dept_id`, and `cat_id`; `item_id` is intentionally excluded from this initial dataset.
- **Forecast feature:** integer `horizon` (1–28), used by the initial LightGBM model to share parameters across all forecast days.
- **Missingness:** a missing historical observation is `NaN`, not a zero sale. A series needs 28 consecutive observed days up to its origin; longer features can remain missing if their full lookback is unavailable.

The initial dataset **does not include future sales or future prices**. Calendar/event features and additional categorical features may be considered later, using earlier validation windows only.

### Training cutoff and leakage protection

For a chosen `training_cutoff`, the generator selects the most recent **N** weekly origins whose entire 28-day target path is observed by that cutoff. It recalculates series eligibility for every origin. Every target must satisfy `date <= training_cutoff`; missing targets for eligible series cause an error rather than silently removing rows.

The **documented initial training experiment** fixes the cutoff at **2015-11-08 (`d_1745`)**, uses **eight weekly origins (2015-08-23 through 2015-10-11)**, and predicts horizons 1–28. This is earlier than the two planned model-tuning target windows and the four previously inspected final-test windows. For future objective and hyperparameter comparisons, use validation origins 2015-11-08 and 2015-12-06, with target periods `d_1746`–`d_1773` and `d_1774`–`d_1801`, respectively; do not tune using final-test errors.

### Reproduce ML training-data generation

```bash
python -m scripts.build_ml_dataset --cutoff 2015-11-08 --origins 8
```

The generator reads `data/processed/sales.parquet` and `data/processed/series.parquet` created earlier, and saves:

| Output | Tracked in Git? | Contents |
|---|---|---|
| `data/processed/ml_training.parquet` | No | 28-day-horizon examples and numeric demand features |
| `reports/ml_training_manifest.json` | Yes | Cutoff, origin dates, row counts, feature names, generation runtime, environment, and peak process RSS |

The individual output files use temporary-file replacement; writing the Parquet and manifest **is not an atomic transaction across both files**. Tests exercise small synthetic Parquet inputs, so CI does not require the large Walmart M5 source dataset.

### Local Walmart M5 dataset results

| Metric | Result |
|---|---:|
| Sampled series available | 3,000 |
| Training cutoff | 2015-11-08 (`d_1745`) |
| Weekly training origins | 8 |
| Forecast horizons per series and origin | 28 |
| Generated supervised rows | **668,808** |
| Unique eligible series across all origins | **2,989** |
| Eligible series per origin | 2,984–2,989 |
| Missing target values | **0** |
| Duplicate `(series_id, origin, horizon)` keys | **0** |
| Targets after training cutoff | **0** |
| Local generation time | **2.80 s** |
| Peak process RSS | **850.81 MiB** |

Eligible series increase slightly across weekly origins as more history becomes available. This dataset-generation benchmark was measured on a MacBook Air M2 (8 GB RAM) with Python 3.13.1; runtime and process peak RSS vary by environment and **do not predict LightGBM training memory usage**. The 216 passing tests establish current checks, not a guarantee against all forms of leakage or forecasting error.

### Initial global LightGBM model — completed training smoke test

The initial model uses `src/ml_model.py` and `scripts/train_ml_model.py` to fit **one direct multi-horizon LightGBM regressor** on the prepared dataset. Each training row corresponds to a `(series_id, origin, horizon)` example with an observed future target, and all demand features are calculated as of that row's origin.

**Model inputs and leakage boundary:** 14 historical demand features, numeric `horizon` (1–28), and the categorical `store_id`, `dept_id`, and `cat_id` — **18 predictors in total**. Target sales, `series_id`, origin and target dates, and unrelated future columns are **not** included in the model feature matrix. Categorical levels are learned from the training data and stored with the model; unseen categories become missing values. Predictions are clipped to zero if the regression model yields a negative value.

**Initial fixed configuration:** LightGBM `LGBMRegressor` with regression objective, 100 estimators, learning rate `0.05`, 31 leaves, minimum 20 samples per leaf, seed `42`, two CPU threads, `deterministic=True`, and `force_col_wise=True`. These are initial fixed parameters **not selected using final-test errors**. The model uses training labels no later than `2015-11-08` and has not yet been evaluated on held-out target periods.

| Local training result | Value |
|---|---:|
| Training cutoff | 2015-11-08 (`d_1745`) |
| Training rows | **668,808** |
| Unique training series | **2,989** |
| Weekly training origins | 8 (2015-08-23 through 2015-10-11) |
| Forecast horizon | 1–28 days |
| Model fit time | **4.96 s** |
| Total training-script execution | **5.42 s** |
| Peak process RSS | **451.70 MiB** |
| Saved LightGBM model file | 282 KiB (approximately 0.28 MiB) |
| Reload smoke test | **PASS** — 28 finite, nonnegative predictions; loaded and in-memory predictions match |
| Project tests | **216 passed** (1 unrelated existing warning) |

These measurements were obtained locally on a **MacBook Air M2 (8 GB RAM)** with Python 3.13.1, LightGBM 4.7.0, scikit-learn 1.9.1, and Pandas 3.0.6. Fit time excludes Parquet reading and artifact serialization; total time includes those steps. Peak RSS is the measured **process** peak, not the Mac's total system memory use. No claims about universal performance or ML forecast quality follow from these measurements.

**Reproduce the training:**

```bash
python -m scripts.build_ml_dataset --cutoff 2015-11-08 --origins 8
python -m scripts.train_ml_model
```

| Training artifact | Tracked in Git? | Purpose |
|---|---|---|
| `data/processed/ml_training.parquet` | No | Origin-relative supervised training examples |
| `models/global_lightgbm.txt` | No | Serialized fitted LightGBM booster |
| `models/global_lightgbm.json` | No | Predictor schema and learned category vocabulary |
| `reports/ml_model_training_manifest.json` | Yes | Model configuration, environment, dataset summary, runtime, memory, and smoke-test result |

The training script validates the `(series_id, origin, horizon)` keys, 28-day paths, target dates, and cutoff; loads the saved model and checks the same complete 28-day prediction path against the in-memory model. Tests cover saved-schema compatibility, invalid targets/horizons, unseen categories, feature inclusion and exclusion, duplicates, incomplete paths, and saved-model reload. The output files are written directly and **are not a multi-file atomic transaction**.

**Not yet implemented:** automated construction of new prediction feature rows from raw unseen sales histories, ML evaluation on earlier validation windows, objective/hyperparameter comparison, and the frozen four-window final ML benchmark. The Phase 1 WAPE numbers above apply **only to the three baseline models**, not LightGBM. For model selection, use the earlier validation periods described above and keep final-test windows out of tuning.

## Preliminary One-Step-Ahead Forecasting (Legacy Sanity Check)

The earlier experiment uses **aggregate** daily sales from 20 M5 product-store series over a 28-day validation period. It is a separate one-step-ahead exercise: actual sales from previous validation dates may be available when predicting later validation dates.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 16.86 | 20.33 | 43.91% |
| Seasonal naive | 19.00 | 23.99 | 49.49% |

These results are **not comparable** to the direct 28-day product-store-level benchmark above: the aggregation, sample size, and information available at prediction time differ. They are retained only as an early pipeline sanity check.

Run the legacy experiment with:

```bash
python -m scripts.evaluate_baselines
```

## Exploratory Data Analysis

### Daily sales trends

![Daily Sales Trends](reports/daily_sales.png)

### Top-selling products

![Top-Selling Products](reports/top_products.png)

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3.13 |
| Processing | Pandas, NumPy |
| Parquet storage | PyArrow, Apache Parquet |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Modeling | LightGBM, scikit-learn |
| Testing and linting | Pytest, Ruff |
| Collaboration and automation | Git, GitHub, GitHub Actions |

**Planned:** validated and tuned LightGBM forecasting; replenishment policies, safety stock, and inventory simulation; a Streamlit dashboard and customer CSV imports. DuckDB/SQL, PostgreSQL, a backend API, and an LLM assistant are possible future additions, not current features.

## Project Roadmap

### Completed — Phase 0: Setup and quality

- [x] Project structure, pinned dependencies, Git workflow, and GitHub Actions CI.
- [x] Initial M5 validation, exploratory analysis, time-series features, and one-step forecasting sanity checks.

### Completed — Phase 1: Data foundation and forecasting evaluation

- [x] Wide-format M5 CSV-to-Parquet conversion, verification, and performance benchmarking.
- [x] Deterministic stratified sampling and unified daily sales/price data.
- [x] Three direct 28-day forecasting baselines and four rolling-origin test windows.
- [x] Common-series eligibility, leakage checks, and pooled performance metrics.
- [x] Performance breakdowns by window, category, and historically defined demand sparsity.
- [x] Reproducible full-sample experiment script and version-controlled summary reports.

### In progress — Phase 2: Machine learning

- [x] Implement leakage-safe lag and rolling features for multi-step ML forecasts.
- [x] Build and validate a supervised 28-day training dataset with weekly origins and a reproducible M5 generation script.
- [x] Implement and train an initial global LightGBM regression model; verify model persistence and add synthetic integration tests.
- [ ] Train and compare LightGBM objectives and hyperparameters **using earlier tuning windows only**.
- [ ] Compare a finalized ML model with the frozen Phase 1 baselines on the same final-test protocol.
- [ ] Analyze forecast limitations and uncertainty without tuning to test results.

### Planned — Phase 3: Inventory optimization

- [ ] Define and document inventory, lead-time, and cost assumptions absent from M5.
- [ ] Implement replenishment rules, safety-stock policies, and simulation.
- [ ] Compare service levels, stockouts, and inventory costs.

### Planned — Phase 4: Product interface

- [ ] Build an interactive Streamlit dashboard.
- [ ] Support user-provided CSVs through a standardized input format.
- [ ] Display forecasts, error analysis, and inventory recommendations.

## Data and Scope Notes

The Walmart M5 dataset provides historical unit sales, calendar information, and observed selling prices for product-store series across U.S. Walmart stores. It does **not** provide all operational inputs needed for inventory planning (such as reliable on-hand inventory, supplier lead times, or stockout costs). Later inventory experiments will need explicit, documented assumptions.

This project currently demonstrates **data engineering, reproducible baseline evaluation, supervised ML dataset preparation, and initial LightGBM training with tested model persistence**. A trained ML model exists, but **its forecasting accuracy and operational usefulness have not yet been validated**; no inventory optimizer has been implemented. Claims about inventory cost savings or real-world replenishment improvements are out of scope until those systems are built and evaluated.