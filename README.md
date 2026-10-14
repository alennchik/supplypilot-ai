# SupplyPilot AI

![CI](https://github.com/alennchik/supplypilot-ai/actions/workflows/ci.yml/badge.svg)

**Demand Forecasting & Inventory Planning | In Development**

SupplyPilot AI is a Python-based supply chain analytics project designed to forecast retail product demand and support data-driven inventory replenishment decisions.

The project uses the Walmart M5 Forecasting dataset to develop and evaluate demand forecasting methods. Its long-term goal is to connect reliable forecasts with inventory planning, stockout risk analysis, and actionable replenishment recommendations.

**Current status:** Data engineering, exploratory analysis, time-series feature engineering, and initial forecasting baselines are implemented. Machine learning models and inventory optimization are planned.

## Current Features

### Data Engineering

- Load and validate Walmart M5 sales data.
- Transform wide-format sales data into long-format observations.
- Process large datasets using memory-efficient, chunk-based ingestion.
- Support analysis of approximately **58.3 million historical daily sales observations** across 1,913 days.

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
- GitHub Actions CI for automated testing and linting on pull requests.

## Exploratory Data Analysis

### Daily Sales Trends

Historical daily sales patterns from the Walmart M5 dataset.

![Daily Sales Trends](reports/daily_sales.png)

### Top-Selling Products

Visualization of the highest-selling products in the analyzed data.

![Top-Selling Products](reports/top_products.png)

## Preliminary Forecasting Results (One-Step-Ahead)

The initial benchmark evaluated aggregate daily sales from a sample of **20 Walmart M5 product-store series** over a 28-day validation period.

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

The next forecasting milestone will implement product-store-level, fixed-origin 28-day evaluation across multiple rolling-origin backtesting windows.

## Technology Stack

### Currently Used

- **Language:** Python 3.13
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Configuration:** PyYAML
- **Testing:** Pytest
- **Code Quality:** Ruff
- **Version Control:** Git, GitHub
- **Continuous Integration:** GitHub Actions

### Planned

- **Data Storage:** Apache Parquet
- **Data Analytics:** DuckDB / SQL
- **Machine Learning:** LightGBM
- **Inventory Planning:** Replenishment policies, safety stock, simulation
- **Application Interface:** Streamlit

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

### Next Milestones

- [ ] Convert retail sales data to Parquet and benchmark processing performance.
- [ ] Implement reproducible product-store sampling.
- [ ] Develop fixed-origin 28-day forecasting baselines.
- [ ] Evaluate forecasts across multiple rolling-origin backtesting windows.
- [ ] Train and evaluate a LightGBM forecasting model.
- [ ] Analyze model performance across product categories and demand patterns.
- [ ] Estimate forecast uncertainty for inventory planning.
- [ ] Implement inventory replenishment policies and simulation.
- [ ] Compare replenishment strategies using service-level and inventory metrics.
- [ ] Build an interactive Streamlit dashboard.
- [ ] Publish reproducible experimental results and project documentation.

## Dataset

SupplyPilot AI uses the **Walmart M5 Forecasting – Accuracy** dataset, which contains historical daily retail sales across multiple Walmart stores, product categories, and U.S. states.

**Dataset source:**

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download the dataset from Kaggle and place the following file in the project's `data/raw/` directory:

`data/raw/sales_train_validation.csv`

Raw datasets are excluded from Git version control because of their size.

Future forecasting development will also use `calendar.csv` and `sell_prices.csv` for calendar, event, and pricing features.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/alennchik/supplypilot-ai.git
cd supplypilot-ai
```

### 2. Create a virtual environment

Python 3.13 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `sales_train_validation.csv` from the Kaggle dataset link above and place it in:

```text
data/raw/sales_train_validation.csv
```

### 5. Run the baseline evaluation

```bash
python -m scripts.evaluate_baselines
```

This runs the initial one-step-ahead baseline evaluation. It is not the final 28-day forecasting benchmark.

## Testing and Code Quality

Run the automated tests:

```bash
python -m pytest -q
```

Run Ruff:

```bash
python -m ruff check .
```

The project currently has **74 passing automated tests** in the verified local Python 3.13 environment.

GitHub Actions is configured to run these checks automatically on pull requests and pushes to `main`.

## Development Status

**Active development — October 2026**

SupplyPilot AI currently provides a tested data engineering and preliminary demand forecasting foundation.

The immediate development priorities are:

1. **Reliable 28-day forecasting:** Product-store evaluation, multiple backtesting windows, and reproducible metrics.
2. **Machine learning:** LightGBM forecasting compared against statistical baselines.
3. **Inventory planning:** Forecast-driven replenishment, uncertainty estimation, and simulation of inventory outcomes.

The project will prioritize measured forecasting quality, transparent evaluation, and reproducible results over adding unnecessary features.

**Note:** Inventory levels, supplier lead times, and other operational inputs not available in the M5 dataset will require clearly documented simulation assumptions.