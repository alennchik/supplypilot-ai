# SupplyPilot AI

[![CI](https://github.com/alennchik/supplypilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/alennchik/supplypilot-ai/actions/workflows/ci.yml)

**Demand Forecasting & Inventory Planning | In Development**

SupplyPilot AI is a Python-based supply chain analytics project designed to forecast retail product demand and support data-driven inventory replenishment decisions.

The project uses the Walmart M5 Forecasting dataset to develop and evaluate demand forecasting methods. Its long-term goal is to connect reliable forecasts with inventory planning, stockout risk analysis, and actionable replenishment recommendations.

**Current status:** Data engineering, exploratory analysis, time-series feature engineering, and preliminary forecasting baselines are implemented. Wide-format Apache Parquet storage, data integrity verification, and performance benchmarking have been implemented. Machine learning models and inventory optimization are planned.

## Current Features

### Data Engineering

- Load and validate Walmart M5 sales data.
- Transform wide-format sales data into long-format observations.
- Process large datasets using memory-efficient, chunk-based ingestion.
- Support analysis of approximately **58.3 million historical daily sales observations** across 1,913 days.
- Convert wide-format M5 sales data from CSV to Apache Parquet using PyArrow.
- Validate sales values before conversion to compact `int16` storage.
- Preserve the original identifiers, columns, and sales values.
- Use temporary files and atomic replacement to protect completed Parquet outputs.
- Verify converted data against the original CSV.
- Benchmark full-table and selected-column loading performance.

### Exploratory Data Analysis

- Calculate total, average, and maximum sales.
- Identify top-selling products.
- Analyze daily sales trends.
- Identify products with frequent zero-sales observations.
- Generate sales analytics visualizations.

### Forecasting Preprocessing

- Prepare chronological time-series data.
- Generate lag features for 1, 7, and 28 days.
- Calculate 7-day and 28-day rolling averages using previous observations.
- Create chronological training and validation splits.
- Prevent current-day target leakage in the existing one-step-ahead features.

### Baseline Demand Forecasting

- Implement naive forecasts based on previous-day sales.
- Implement seasonal naive forecasts based on sales from seven days earlier.
- Evaluate forecasting performance using **MAE, RMSE, and WAPE**.
- Compare forecasting baselines on historical validation data.

### Testing and Code Quality

- Automated tests using Pytest.
- Static analysis and linting using Ruff.
- Reproducible dependency installation with pinned package versions.
- GitHub Actions CI for automated testing and linting on pull requests and pushes to `main`.
- Unit tests covering Parquet conversion, data validation, integer overflow, and safe writes.
- Reproducible performance benchmark scripts and saved measurement results.

## Exploratory Data Analysis

### Daily Sales Trends

Historical daily sales patterns from the Walmart M5 dataset.

![Daily Sales Trends](reports/daily_sales.png)

### Top-Selling Products

Visualization of the highest-selling products in the analyzed data.

![Top-Selling Products](reports/top_products.png)

## M5 Parquet Storage & Performance

SupplyPilot AI supports converting the Walmart M5 sales dataset from CSV to wide-format Apache Parquet using Pandas and PyArrow.

The conversion preserves all **30,490 product-store series and 1,919 columns**, including 1,913 daily sales columns.

Sales values are validated before conversion to `int16`. The converter writes to a temporary file before replacing the output, preventing a failed conversion from overwriting an existing valid Parquet file.

The converted dataset is verified against the original CSV to ensure identical identifiers and sales values.

### Run the Conversion

Place `sales_train_validation.csv` in `data/raw/`, then run:

```bash
python -m scripts.convert_to_parquet
```

The output is saved to:

```text
data/processed/sales_train_validation.parquet
```

Verify that the original CSV and converted Parquet data match:

```bash
python -m scripts.verify_parquet
```

Verification performed on the full Walmart M5 sales dataset:

- **Rows verified:** 30,490
- **Columns verified:** 1,919
- **Historical sales observations:** 58,327,370
- **Result:** All identifiers and sales values matched exactly.

### Performance Benchmark

CSV and Parquet loading performance was measured locally on macOS using Python 3.13.1, Pandas 3.0.6, and PyArrow 26.0.0.

Each loading scenario was repeated three times, with results reported as medians.

| Metric | CSV | Parquet | Improvement |
|---|---:|---:|---:|
| Full-table loading time | 1.810 s | 0.101 s | **17.94× faster** |
| IDs + last 365 days loading | 0.666 s | 0.041 s | **16.14× faster** |
| Full-load peak memory | 368.6 MiB | 410.6 MiB | Higher for Parquet |
| Recent-load peak memory | 171.6 MiB | 176.4 MiB | Higher for Parquet |
| File size | ~114 MiB | 30.62 MiB | **73.2% smaller** |

**Key result:** Apache Parquet reduced storage requirements by 73.2% and provided approximately 18× faster full-table loading and 16× faster recent-data loading in the local benchmark.

Parquet did not reduce peak memory consumption in these experiments. Full-table loading used more peak memory than the CSV equivalent.

### Benchmark Methodology

The benchmark evaluates two scenarios:

1. **Full-table loading:** Read all 1,919 columns.
2. **Recent-history loading:** Read the six identifier columns and the last 365 days of sales data.

For fairer comparisons:

- CSV loading uses explicit column types.
- Parquet uses PyArrow through Pandas.
- Each timed load runs in a separate Python process.
- Each scenario is repeated three times.
- The benchmark reports median execution time and peak process memory.
- Peak memory measurements include Python and library overhead.
- Operating system filesystem caches are not cleared between runs.
- Scenario execution order is shuffled using a fixed random seed.

The results represent performance on the tested environment and dataset. They are not universal performance guarantees.

To reproduce the benchmark after converting the dataset:

```bash
python -m scripts.benchmark_parquet --repeats 3
```

Detailed measurements, individual runs, package versions, and environment information are stored in:

```text
reports/parquet_benchmark.json
```

Raw and converted datasets are excluded from Git version control.

## Reproducible M5 Sampling & Unified Daily Data

SupplyPilot AI provides a reproducible data preparation pipeline that transforms Walmart M5 sales, calendar, and pricing data into a standardized daily time-series format for future forecasting models.

### Sampling Strategy

The pipeline selects exactly **3,000 product-store series** from the original 30,490 M5 series.

Sampling is stratified by `store_id × cat_id` across 30 groups, with allocation proportional to group size using the largest-remainder method. Within each group, series are ranked deterministically using SHA-256 and a fixed seed of `42`.

Sampling uses only product and store metadata, not historical sales statistics.

For each of the 10 stores, the sample contains:

| Category | Selected Series |
|---|---:|
| FOODS | 141 |
| HOBBIES | 56 |
| HOUSEHOLD | 103 |
| **Total per store** | **300** |

The exact selected IDs are version-controlled in `reports/sample_series.csv`.

### Required Data

Download the Walmart M5 data from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place these files in `data/raw/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

The raw datasets and generated Parquet files are excluded from Git.

### Reproduce the Pipeline

Run these commands from the project root after installing the dependencies:

```bash
python -m scripts.convert_to_parquet
python -m scripts.sample_m5
python -m scripts.build_m5_unified
```

### Output Files

| File | Description |
|---|---|
| `reports/sample_series.csv` | Version-controlled manifest of 3,000 selected series IDs |
| `data/processed/series.parquet` | Product-store metadata for sampled series |
| `data/processed/sales.parquet` | Unified daily sales observations |

The metadata file contains:

`series_id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

The daily sales file contains:

`series_id`, `date`, `quantity`, `price`, `price_available`

Product-store attributes are kept separate from daily observations to avoid unnecessary duplication.

### Data Preparation Rules

- Original `d_1` through `d_1913` identifiers are mapped to actual calendar dates.
- Weekly selling prices are joined using `store_id`, `item_id`, and `wm_yr_wk`.
- Days before the first observed price are excluded for each series.
- The first observed price is treated as a proxy for the beginning of available history, not a verified product launch date.
- Observed zero-sales days remain in the dataset.
- Missing prices after the first observed price are retained as null, with `price_available = False`.
- Calendar events and SNAP indicators are reserved for later feature engineering.

Price availability indicates whether a price was recorded, not whether inventory was physically available.

### Dataset Validation Results

The pipeline was executed on the sampled Walmart M5 data and produced:

| Metric | Result |
|---|---:|
| Sampled series | 3,000 |
| Daily observations | 4,550,960 |
| Zero-sales observations | 2,724,948 |
| Missing prices after trimming | 0 |
| Average series history | Approximately 1,517 days |

The generated dataset passed local integrity checks for unique series-date pairs, consecutive dates, valid quantities, and consistency between price values and availability indicators.

An independent local reconciliation also confirmed that all **4,550,960 retained sales quantities** match the original M5 sales data.

The pipeline uses batched processing and PyArrow Parquet writing to avoid holding the entire unified dataset in memory at once.

## Preliminary Forecasting Results (One-Step-Ahead)

The initial forecasting benchmark evaluated aggregate daily sales from a sample of **20 Walmart M5 product-store series** over a 28-day validation period.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 16.86 | 20.33 | 43.91% |
| Seasonal Naive | 19.00 | 23.99 | 49.49% |

On this initial sample, the naive model achieved lower forecasting errors than the seasonal naive model.

**Important limitations:**

- These are preliminary **one-step-ahead** results, not fixed-origin 28-day forecasts.
- Actual sales from preceding validation days are available when making subsequent predictions.
- The evaluation uses an aggregate of only 20 product-store series and is not representative of the full M5 dataset.
- These results should be treated as an initial sanity check, not a final measure of forecasting performance.

The next forecasting milestones will focus on fixed-origin 28-day forecasting baselines, rolling-origin backtesting, and machine learning models.

## Technology Stack

### Currently Used

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Data Storage:** Apache Parquet, PyArrow
- **Visualization:** Matplotlib
- **Configuration:** PyYAML
- **Testing:** Pytest
- **Code Quality:** Ruff
- **Version Control:** Git, GitHub
- **Continuous Integration:** GitHub Actions

### Planned

- **Data Analytics:** DuckDB / SQL
- **Machine Learning:** LightGBM
- **Inventory Planning:** Replenishment policies, safety stock, simulation
- **Application Interface:** Streamlit
- **User Data Support:** CSV import and standardized forecasting inputs

Additional capabilities, including PostgreSQL integration, a backend API, and an LLM-powered assistant, may be explored after the core forecasting and inventory planning system is functional.

## Project Roadmap

### Completed

- [x] Project configuration and development environment
- [x] Walmart M5 data loading and validation
- [x] Memory-efficient, chunk-based sales processing
- [x] Exploratory data analysis and visualizations
- [x] Time-series feature engineering
- [x] Chronological training and validation splitting
- [x] Naive and seasonal naive forecasting
- [x] Initial MAE, RMSE, and WAPE evaluation
- [x] Automated unit tests and linting
- [x] GitHub Actions CI setup
- [x] M5 wide-format CSV-to-Parquet conversion
- [x] Sales value validation and safe Parquet writes
- [x] Full-dataset CSV/Parquet integrity verification
- [x] CSV vs. Parquet performance benchmarking
- [x] Complete Parquet documentation, PR review, and CI verification
- [x] Implement reproducible product-store sampling
- [x] Create a unified long-format data representation with real calendar dates
- [x] Incorporate price availability and handle pre-price-observation periods

### Next Milestones

- [ ] Develop fixed-origin 28-day forecasting baselines.
- [ ] Evaluate forecasts across multiple rolling-origin backtesting windows.
- [ ] Train and evaluate a LightGBM forecasting model.
- [ ] Analyze model performance across product categories and demand patterns.
- [ ] Estimate forecast uncertainty for inventory planning.
- [ ] Implement inventory replenishment policies and simulation.
- [ ] Compare replenishment strategies using service-level and inventory metrics.
- [ ] Build an interactive Streamlit dashboard with user data upload.
- [ ] Publish reproducible experimental results and project documentation.

## Dataset

SupplyPilot AI uses the **Walmart M5 Forecasting – Accuracy** dataset, containing historical daily retail sales across multiple Walmart stores, product categories, and U.S. states.

**Dataset source:**

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download the dataset from Kaggle and place the following file in the project's `data/raw/` directory:

```text
data/raw/sales_train_validation.csv
```

The sales dataset contains:

- 30,490 product-store time series.
- 1,913 days of historical sales.
- Approximately 58.3 million daily sales observations.
- Product, department, category, store, and state identifiers.

Raw datasets and generated Parquet files are excluded from Git version control because of their size.

Future forecasting development will also use `calendar.csv` and `sell_prices.csv` for calendar, event, and pricing features.

The long-term architecture will separate data sources from the forecasting engine, allowing future customer datasets to be transformed into a common internal format.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/alennchik/supplypilot-ai.git
cd supplypilot-ai
```

### 2. Create a Virtual Environment

Python 3.13 is recommended. Verify your installed version:

```bash
python3 --version
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate using the appropriate script from `.venv\Scripts\`.

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

Download `sales_train_validation.csv` from the Kaggle dataset link above and place it in:

```text
data/raw/sales_train_validation.csv
```

### 5. Convert Sales Data to Parquet

```bash
python -m scripts.convert_to_parquet
```

This creates the optimized wide-format Parquet file in `data/processed/`.

Verify data integrity:

```bash
python -m scripts.verify_parquet
```

Optionally reproduce the storage performance benchmark:

```bash
python -m scripts.benchmark_parquet --repeats 3
```

### 6. Run the Baseline Evaluation

```bash
python -m scripts.evaluate_baselines
```

This runs the initial one-step-ahead baseline evaluation. It is not the final fixed-origin 28-day forecasting benchmark.

## Testing and Code Quality

Run the automated tests:

```bash
python -m pytest -q
```

Run Ruff:

```bash
python -m ruff check .
```

The project has 109 locally passing automated tests in the verified Python 3.13 environment, including 10 tests for Parquet conversion and validation.

GitHub Actions runs these checks automatically on pull requests and pushes to `main`.

The CI badge at the top reflects the workflow status on `main`. New development-branch changes must pass their own pull request checks before merging.

## Development Status

**Active development — October 2026**

SupplyPilot AI currently provides a tested data engineering foundation, preliminary demand forecasting methods, and an optimized storage workflow with validated performance measurements.

The immediate development priorities are:

1. **Reliable forecasting evaluation:** Fixed-origin 28-day forecasting baselines and rolling-origin backtesting using the completed unified M5 dataset.
2. **Machine learning:** LightGBM forecasting compared against baseline models using consistent evaluation windows.
3. **Inventory planning:** Forecast-driven replenishment, uncertainty estimation, and simulation of inventory outcomes.
4. **Application interface:** Interactive analytics and support for uploading user-provided sales data.

The project prioritizes measured forecasting quality, transparent evaluation, and reproducible results over adding unnecessary features.

**Note:** Inventory levels, supplier lead times, and other operational inputs not available in the M5 dataset will require clearly documented simulation assumptions.