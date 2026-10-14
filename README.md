# SupplyPilot AI

**AI-Powered Supply Chain Forecasting & Inventory Planning — In Development**

SupplyPilot AI is a Python-based supply chain analytics project designed to help businesses forecast product demand, optimize inventory decisions, and interpret operational data using machine learning and AI.

The project uses the Walmart M5 Forecasting dataset as a development and evaluation benchmark.

## Current Features

### Data Engineering
- Load and validate Walmart M5 sales data.
- Transform wide-format sales data into long-format observations.
- Process large datasets using memory-efficient chunk-based ingestion.
- Analyze approximately 58.3 million historical sales observations.

### Exploratory Data Analysis
- Calculate total, average, and maximum sales.
- Identify top-selling products.
- Analyze daily sales trends across 1,913 days.
- Identify products with frequent zero-sales observations.
- Generate sales analytics visualizations.

### Forecasting Preprocessing
- Prepare chronological time-series data.
- Generate lag features (1, 7, and 28 days).
- Calculate rolling averages (7 and 28 days).
- Create chronological training and validation splits.
- Prevent current-day target leakage in forecasting features.

### Baseline Demand Forecasting
- Naive forecasting using previous-day sales.
- Seasonal naive forecasting using sales from seven days earlier.
- Evaluate forecasts using MAE, RMSE, and WAPE.
- Compare baseline performance using historical validation data.

## Initial Forecasting Results

The first benchmark evaluated aggregate daily sales from a sample of 20 Walmart M5 product-store series over a 28-day validation period.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive | 16.86 | 20.33 | 43.91% |
| Seasonal Naive | 19.00 | 23.99 | 49.49% |

The naive model outperformed the seasonal naive model on this initial sample.

These are preliminary, one-step-ahead forecasting results, not full-dataset or multi-step forecasting benchmarks.

## Technology Stack

**Currently Used**
- Python
- Pandas and NumPy
- Matplotlib
- PyYAML
- Pytest
- Ruff
- Git and GitHub

**Planned**
- Scikit-learn
- LightGBM or XGBoost
- PostgreSQL and SQL
- LLM API integration
- Streamlit dashboard
- Backend API

## Project Roadmap

- [x] Project configuration and development environment
- [x] Walmart M5 data loading and validation
- [x] Memory-efficient sales processing
- [x] Exploratory data analysis and visualizations
- [x] Time-series feature engineering
- [x] Chronological training/validation splitting
- [x] Naive and seasonal naive forecasting
- [x] MAE, RMSE, and WAPE evaluation
- [ ] Product-store-level forecasting evaluation
- [ ] Machine learning demand forecasting
- [ ] PostgreSQL data storage and SQL integration
- [ ] Inventory optimization and replenishment recommendations
- [ ] LLM-powered planning assistant
- [ ] Interactive dashboard
- [ ] Deployment and end-to-end evaluation

## Dataset

SupplyPilot AI uses the Walmart M5 Forecasting dataset, which contains historical daily retail sales across multiple stores, product categories, and states.

Raw dataset files are excluded from Git version control.

## Running the Project

Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the automated tests:

```bash
python -m pytest -v
```

Check code quality:

```bash
ruff check src/ tests/ scripts/
```

After placing the M5 validation sales CSV in `data/raw/`, run the baseline evaluation:

```bash
python -m scripts.evaluate_baselines
```

## Development Status

**Active development.**

The initial data engineering, exploratory analytics, forecasting preprocessing, and baseline evaluation components are implemented.

The next milestones focus on broader forecasting evaluation, machine learning models, database integration, inventory optimization, and an AI-powered planning interface.
