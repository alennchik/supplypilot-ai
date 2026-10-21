"""Frozen Issue #28 protocol, blind alignment, and explicitly invoked scoring."""

import numpy as np
import pandas as pd

from scripts.train_ml_model import validate_training_dataset
from src.backtesting import build_sales_matrix, run_backtesting, select_eligible_series
from src.cycle2_metrics import _dates, _numeric
from src.evaluation import build_demand_profile
from src.ml_dataset import build_ml_training_dataset, select_training_origins
from src.ml_inference import build_ml_inference_examples
from src.ml_model import (
    CYCLE2_FEATURE_COLUMNS,
    predict_global_model,
    train_global_model,
)

MODEL = "cycle2_tweedie_8_origins_calendar"
BASELINES = ("moving_average_28", "seasonal_naive", "naive")
METHODS = (MODEL, *BASELINES)
CUTOFF = pd.Timestamp("2016-01-03")
ORIGINS = tuple(
    pd.to_datetime(["2016-01-03", "2016-01-31", "2016-02-28", "2016-03-27"])
)
TRAINING_ORIGINS = tuple(pd.date_range("2015-10-18", "2015-12-06", freq="7D"))
TREES = 1743
EXPECTED_SERIES = 2996
KEYS = ["series_id", "origin", "date", "horizon"]
LIMITATIONS = [
    (
        "Training targets span Oct 19, 2015–Jan 3, 2016; test months 2–4 and "
        "possibly event types are unseen. Trees do not extrapolate smooth month effects; "
        "unseen categorical levels encode as missing."
    ),
    (
        "Phase 1 baseline results already exposed these windows: this is a frozen ML "
        "benchmark, not a project-wide untouched holdout."
    ),
    (
        "Cycle 2 reused V2; selection optimism is possible. Its adoption threshold "
        "does not establish statistical significance."
    ),
    (
        "WAPE is forecast error, not classification accuracy or evidence of inventory "
        "cost savings, service levels, or production readiness."
    ),
]


def protocol() -> dict:
    return {
        "issue": 28,
        "model": MODEL,
        "training_cutoff": str(CUTOFF.date()),
        "training_origins": [str(d.date()) for d in TRAINING_ORIGINS],
        "evaluation_origins": [str(d.date()) for d in ORIGINS],
        "feature_columns": list(CYCLE2_FEATURE_COLUMNS),
        "parameters": {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "n_estimators": TREES,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": 2,
            "deterministic": True,
            "force_col_wise": True,
        },
        "early_stopping": False,
        "fits_per_run": 1,
        "methods": list(METHODS),
        "series": EXPECTED_SERIES,
        "daily_rows_per_method": EXPECTED_SERIES * 112,
        "total_predictions": EXPECTED_SERIES * 112 * 4,
        "unseen_category_policy": "missing",
        "limitations": LIMITATIONS,
    }


def validate_paths(frame, cohort) -> pd.DataFrame:
    """Validate a complete frozen key grid, without computing any errors."""
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Empty forecasts or duplicate columns")
    if not set(KEYS).issubset(frame):
        raise ValueError("Missing forecast keys")
    out = frame.copy()
    if out[KEYS].isna().any().any():
        raise ValueError("Missing forecast keys")
    if not out.series_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Invalid series identifiers")
    out["origin"] = _dates(out.origin, "origin")
    out["date"] = _dates(out.date, "date")
    h = _numeric(out.horizon, "horizon")
    if not np.isin(h, np.arange(1, 29)).all():
        raise ValueError("Invalid horizon")
    out["horizon"] = h.astype(int)
    cohort = pd.Index(cohort)
    if cohort.empty or cohort.has_duplicates or cohort.isna().any():
        raise ValueError("Invalid cohort")
    if set(out.series_id) != set(cohort) or set(out.origin) != set(ORIGINS):
        raise ValueError("Frozen cohort/origins mismatch")
    if not out.date.eq(out.origin + pd.to_timedelta(out.horizon, unit="D")).all():
        raise ValueError("Target dates must equal origin plus horizon")
    if out.duplicated(KEYS).any() or len(out) != len(cohort) * 112:
        raise ValueError("Duplicate or missing forecast keys")
    for col in ("actual", "predicted"):
        if col in out:
            numbers = _numeric(out[col], col)
            if (numbers < 0).any():
                raise ValueError(f"Invalid nonnegative {col}")
            out[col] = numbers
    return out.sort_values(KEYS).reset_index(drop=True)


def align_methods(forecasts, cohort) -> pd.DataFrame:
    """Strictly align all methods and actuals; no metrics or residuals."""
    required = {*KEYS, "model", "actual", "predicted"}
    if not required.issubset(forecasts) or set(forecasts.model) != set(METHODS):
        raise ValueError("All four methods and scored columns are required")
    frames = []
    reference = None
    for method in METHODS:
        frame = validate_paths(forecasts.loc[forecasts.model == method], cohort)
        if reference is not None and not frame[[*KEYS, "actual"]].equals(reference):
            raise ValueError("Methods have different keys or actuals")
        reference = frame[[*KEYS, "actual"]]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fit_frozen_model(training):
    validate_training_dataset(training, str(CUTOFF.date()))
    if set(training.origin) != set(TRAINING_ORIGINS):
        raise ValueError("Incorrect frozen training origins")
    if select_training_origins(CUTOFF, 8) != TRAINING_ORIGINS:
        raise ValueError("Training origin implementation drift")
    model = train_global_model(
        training,
        n_estimators=TREES,
        n_jobs=2,
        objective="tweedie",
        calendar_features=True,
    )
    if (
        model.booster.num_trees() != TREES
        or model.booster.current_iteration() != TREES
        or model.best_iteration != TREES
        or model.validation_mae_history is not None
        or model.feature_columns != CYCLE2_FEATURE_COLUMNS
        or model.objective != "tweedie"
    ):
        raise ValueError(
            "Frozen model must realize exactly 1743 trees without early stopping"
        )
    return model


def verified_phase1_baselines(sales, baselines, cohort):
    """Compare supplied forecasts to independent Phase 1 recomputation, not errors.

    run_backtesting only generates the original three baseline forecasts. It
    does not call the Phase 1 accuracy/reporting pipeline. No actual-versus-
    predicted differences, metrics, or comparison rows are displayed here.
    """
    required = {"series_id", "origin", "date", "model", "actual", "predicted"}
    if not required.issubset(baselines) or set(baselines.model) != set(BASELINES):
        raise ValueError("Exactly three unchanged Phase 1 baselines required")
    supplied = baselines.copy()
    if "horizon" not in supplied:
        supplied["horizon"] = (
            _dates(supplied.date, "date") - _dates(supplied.origin, "origin")
        ).dt.days
    reference = run_backtesting(sales, ORIGINS[-1] + pd.Timedelta(days=28))
    reference["horizon"] = (reference.date - reference.origin).dt.days
    columns = [*KEYS, "actual", "predicted"]
    verified = []
    for method in BASELINES:
        expected = validate_paths(reference.loc[reference.model == method], cohort)
        actual = validate_paths(supplied.loc[supplied.model == method], cohort)
        if not actual[columns].equals(expected[columns]):
            raise ValueError(
                f"Baseline differs from unchanged Phase 1 definitions: {method}"
            )
        verified.append(actual[[*columns, "model"]])
    return pd.concat(verified, ignore_index=True)


def structural_preflight(
    sales, metadata, calendar, baselines, *, expected_series=EXPECTED_SERIES
):
    """Fit once, predict four times, validate structure. Never calculate accuracy.

    expected_series is a synthetic-test seam; the CLI always uses 2,996.
    Only training labels reach fitting; origin builders receive truncated sales.
    """
    matrix = build_sales_matrix(sales)
    cohort = select_eligible_series(matrix, CUTOFF)
    if len(cohort) != expected_series:
        raise ValueError("Wrong Phase 1 cohort count")
    baseline = verified_phase1_baselines(sales, baselines, cohort)
    history = sales.loc[sales.date <= CUTOFF].copy()
    training_metadata = metadata.loc[metadata.series_id.isin(history.series_id)]
    training = build_ml_training_dataset(
        history, training_metadata, CUTOFF, 8, calendar=calendar
    )
    model = fit_frozen_model(training)
    predicted = []
    for origin in ORIGINS:
        origin_history = sales.loc[sales.date <= origin]
        origin_metadata = metadata.loc[
            metadata.series_id.isin(origin_history.series_id)
        ]
        examples = build_ml_inference_examples(
            origin_history, origin_metadata, origin, calendar=calendar
        )
        examples = examples.loc[examples.series_id.isin(cohort)]
        predicted.append(predict_global_model(model, examples))
    ml = validate_paths(pd.concat(predicted, ignore_index=True), cohort)
    ml = ml.rename(columns={"prediction": "predicted"})
    ml["model"] = MODEL
    # Recover labels only after fitting and prediction. No comparison is displayed.
    labels = sales.rename(columns={"quantity": "actual"})[
        ["series_id", "date", "actual"]
    ]
    ml = ml.merge(labels, on=["series_id", "date"], how="left", validate="many_to_one")
    aligned = align_methods(pd.concat([ml, baseline], ignore_index=True), cohort)
    profile = build_demand_profile(history, cohort, CUTOFF)
    lookups = metadata[["series_id", "cat_id"]].merge(
        profile, on="series_id", validate="one_to_one"
    )
    lookups = lookups.loc[lookups.series_id.isin(cohort)]
    if (
        set(lookups.series_id) != set(cohort)
        or lookups.cat_id.isna().any()
        or not lookups.cat_id.isin(["FOODS", "HOUSEHOLD", "HOBBIES"]).all()
    ):
        raise ValueError("Missing or invalid frozen category metadata")
    checks = {
        "status": "passed_blind_structural_preflight",
        "series_count": len(cohort),
        "rows_per_method": len(ml),
        "total_predictions": len(aligned),
        "realized_trees": model.booster.num_trees(),
        "fits": 1,
        "training_rows": len(training),
        "training_series": int(training.series_id.nunique()),
        "training_label_max": str(training.date.max().date()),
        "training_label_min": str(training.date.min().date()),
        "category_levels": model.category_levels,
        "checks": [
            "complete_unique_keys",
            "fixed_cohort",
            "dates_and_horizons",
            "identical_actuals",
            "finite_nonnegative_predictions",
            "training_cutoff",
            "exact_tree_count",
            "exact_phase1_baseline_reproduction",
        ],
        "accuracy_calculated": False,
    }
    return model, aligned, lookups, checks


def pooled_metrics(frame):
    """Compute pooled ratios, aggregating within series before absolute errors."""
    actual = frame.actual.to_numpy(dtype=float)
    prediction = frame.predicted.to_numpy(dtype=float)
    total = float(actual.sum())
    result = {
        "series_count": int(frame.series_id.nunique()),
        "observations": len(frame),
        "total_actual": total,
        "mae": float(np.abs(actual - prediction).mean()),
        "rmse": float(np.sqrt(np.square(actual - prediction).mean())),
    }
    for name, group in [
        ("daily_wape", None),
        ("weekly_wape", ["series_id", "origin", "horizon_week"]),
        ("day28_wape", ["series_id", "origin"]),
    ]:
        pairs = (
            frame
            if group is None
            else frame.groupby(group)[["actual", "predicted"]].sum()
        )
        result[name] = (
            100 * float((pairs.actual - pairs.predicted).abs().sum()) / total
            if total
            else np.nan
        )
    if any(np.isinf(value) for value in result.values()):
        raise ValueError("Metric overflow")
    return result


def reveal_metrics(forecasts, lookups):
    """Explicit reveal boundary; caller must verify the saved preflight artifacts."""
    cohort = lookups.series_id
    aligned = align_methods(forecasts, cohort)
    if lookups.series_id.duplicated().any():
        raise ValueError("Duplicate diagnostic lookup")
    enriched = aligned.merge(lookups, on="series_id", validate="many_to_one")
    if (
        enriched[["cat_id", "demand_group"]].isna().any().any()
        or not enriched.demand_group.isin(["frequent", "mixed", "sparse"]).all()
    ):
        raise ValueError("Invalid diagnostic lookup")
    enriched["horizon_week"] = (enriched.horizon - 1) // 7 + 1
    reports = {}
    for name, fields in [
        ("overall", ["model"]),
        ("by_window", ["model", "origin"]),
        ("by_category", ["model", "cat_id"]),
        ("by_demand_group", ["model", "demand_group"]),
        ("by_horizon_week", ["model", "horizon_week"]),
    ]:
        rows = []
        for key, frame in enriched.groupby(fields, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append(
                {**dict(zip(fields, key, strict=True)), **pooled_metrics(frame)}
            )
        reports[name] = pd.DataFrame(rows)
        if name == "by_horizon_week":
            # A week slice cannot supply a complete 28-day metric.
            reports[name] = reports[name].drop(columns="day28_wape")
    return reports
