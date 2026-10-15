"""Run preregistered chronological LightGBM validation experiments."""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_ml_model import validate_training_dataset
from src.forecast_baselines import FORECAST_HORIZON, moving_average_forecast
from src.metrics import wape
from src.ml_dataset import build_ml_training_dataset, select_training_origins
from src.ml_model import predict_global_model, train_global_model
from src.ml_validation import (
    TRAINING_CUTOFF,
    V1_ORIGIN,
    V2_ORIGIN,
    build_validation_feature_sets,
)

TRAINING_ORIGIN_COUNTS = (8, 16, 32)
OBJECTIVES = ("regression", "poisson", "tweedie")
MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100
N_JOBS = 2


@dataclass(frozen=True)
class Candidate:
    """One preregistered LightGBM experiment configuration."""

    objective: str
    n_origins: int

    @property
    def name(self) -> str:
        return f"{self.objective}_{self.n_origins}_origins"


def candidate_grid() -> tuple[Candidate, ...]:
    """Return the exact nine configurations in a deterministic order."""
    return tuple(
        Candidate(objective=objective, n_origins=n_origins)
        for objective in OBJECTIVES
        for n_origins in TRAINING_ORIGIN_COUNTS
    )


def validate_protocol_dates() -> None:
    """Ensure fixed training and validation periods are chronological."""
    if TRAINING_CUTOFF != V1_ORIGIN:
        raise ValueError("V1 origin must equal the training cutoff")
    if (V2_ORIGIN - V1_ORIGIN).days != 28:
        raise ValueError("V1 and V2 must be exactly 28 days apart")


def prepare_training_dataset(
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate: Candidate,
) -> pd.DataFrame:
    """Prepare one candidate using only preregistered training targets."""
    validate_protocol_dates()
    if candidate not in candidate_grid():
        raise ValueError("Candidate is not in the preregistered grid")
    training = build_ml_training_dataset(
        sales=sales,
        metadata=metadata,
        training_cutoff=TRAINING_CUTOFF,
        n_origins=candidate.n_origins,
    )
    # Reuse the existing Phase 2 training-data safety checks.
    validate_training_dataset(
        training,
        cutoff=str(TRAINING_CUTOFF.date()),
    )
    expected_origins = set(
        select_training_origins(
            TRAINING_CUTOFF,
            n_origins=candidate.n_origins,
        )
    )
    actual_origins = set(pd.to_datetime(training["origin"]))
    if actual_origins != expected_origins:
        raise ValueError("Training origins do not match the fixed protocol")
    if training["date"].max() > TRAINING_CUTOFF:
        raise ValueError("Training targets exceed the fixed cutoff")
    return training


def validate_validation_sets(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> None:
    """Reject swapped, incomplete, or mismatched validation windows."""
    required = {"series_id", "origin", "date", "horizon", "target"}
    cohorts = []
    for name, frame, expected_origin in (
        ("V1", v1, V1_ORIGIN),
        ("V2", v2, V2_ORIGIN),
    ):
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"{name} is empty or missing required columns")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} contains missing values")
        if not pd.to_datetime(frame["origin"]).eq(expected_origin).all():
            raise ValueError(f"{name} has an incorrect forecast origin")
        horizons = pd.to_numeric(frame["horizon"], errors="raise")
        if not horizons.between(1, FORECAST_HORIZON).all():
            raise ValueError(f"{name} has invalid horizons")
        dates = pd.to_datetime(frame["date"])
        expected_dates = expected_origin + pd.to_timedelta(horizons, unit="D")
        if not dates.eq(expected_dates).all():
            raise ValueError(f"{name} has incorrect forecast dates")
        if frame.duplicated(["series_id", "origin", "horizon"]).any():
            raise ValueError(f"{name} contains duplicate forecast keys")
        counts = frame.groupby("series_id")["horizon"].nunique()
        if not counts.eq(FORECAST_HORIZON).all():
            raise ValueError(f"{name} has incomplete forecast horizons")
        targets = pd.to_numeric(frame["target"], errors="raise")
        values = targets.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} contains invalid targets")
        cohorts.append(set(frame["series_id"]))
    if cohorts[0] != cohorts[1]:
        raise ValueError("V1 and V2 must contain the same fixed cohort")


def run_candidate(
    candidate: Candidate,
    training: pd.DataFrame,
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Train with V1 early stopping and evaluate on untouched V2."""
    validate_protocol_dates()
    if candidate not in candidate_grid():
        raise ValueError("Candidate is not in the preregistered grid")
    validate_training_dataset(
        training,
        cutoff=str(TRAINING_CUTOFF.date()),
    )
    validate_validation_sets(v1, v2)
    expected_origins = set(
        select_training_origins(
            TRAINING_CUTOFF,
            n_origins=candidate.n_origins,
        )
    )
    if set(pd.to_datetime(training["origin"])) != expected_origins:
        raise ValueError("Incorrect training origins for candidate")
    # Only V1 targets may influence early stopping.
    model = train_global_model(
        training,
        objective=candidate.objective,
        n_estimators=MAX_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        n_jobs=N_JOBS,
        validation_data=v1,
    )
    # V2 targets are excluded from the prediction input.
    predictions = predict_global_model(
        model,
        v2.drop(columns="target"),
    )
    keys = ["series_id", "origin", "date", "horizon"]
    scored = v2[keys + ["target"]].merge(
        predictions,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scored["_merge"].eq("both").all():
        raise ValueError("V2 predictions do not match validation targets")
    scored = scored.drop(columns="_merge")
    v2_wape = wape(
        scored["target"].to_numpy(),
        scored["prediction"].to_numpy(),
    )
    result = {
        "candidate": candidate.name,
        "objective": candidate.objective,
        "n_origins": candidate.n_origins,
        "best_iteration": model.best_iteration,
        "v2_wape": v2_wape,
    }
    return result, scored


def run_all_candidates(
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate the exact nine candidates sequentially.
    V1 selects best_iteration via early stopping.
    V2 is used only for scoring model configurations.
    """
    validate_protocol_dates()
    # Prepare validation features and labels once for all candidates.
    v1, v2 = build_validation_feature_sets(sales, metadata)
    validate_validation_sets(v1, v2)
    results = []
    forecasts = {}
    for candidate in candidate_grid():
        # Only one training dataset and model are fitted at a time.
        training = prepare_training_dataset(
            sales=sales,
            metadata=metadata,
            candidate=candidate,
        )
        result, scored = run_candidate(
            candidate=candidate,
            training=training,
            v1=v1,
            v2=v2,
        )
        results.append(result)
        forecasts[candidate.name] = scored
        del training, scored
        gc.collect()
    summary = pd.DataFrame(results)
    expected_names = {candidate.name for candidate in candidate_grid()}
    if len(summary) != 9 or set(summary["candidate"]) != expected_names:
        raise ValueError("Incomplete nine-candidate validation experiment")
    return summary, forecasts


def select_best_candidate(summary: pd.DataFrame) -> dict:
    """Select the winner using the preregistered V2 WAPE rule."""
    required = {
        "candidate",
        "objective",
        "n_origins",
        "best_iteration",
        "v2_wape",
    }
    if not required.issubset(summary.columns):
        raise ValueError("Missing candidate selection columns")
    expected = {candidate.name for candidate in candidate_grid()}
    if (
        len(summary) != len(expected)
        or summary["candidate"].duplicated().any()
        or set(summary["candidate"]) != expected
    ):
        raise ValueError("Selection requires the exact nine candidates")
    records = summary.copy()
    scores = pd.to_numeric(records["v2_wape"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Candidate WAPE values must be finite")
    if (scores < 0).any():
        raise ValueError("Candidate WAPE values must be nonnegative")
    records["v2_wape"] = scores
    for row in records.itertuples(index=False):
        expected_name = f"{row.objective}_{row.n_origins}_origins"
        if row.candidate != expected_name:
            raise ValueError("Candidate metadata does not match its name")
    objective_priority = {
        "regression": 0,
        "poisson": 1,
        "tweedie": 2,
    }
    best_wape = records["v2_wape"].min()
    # WAPE is expressed in percentage points (e.g., 45.0).
    eligible = records.loc[records["v2_wape"] <= best_wape + 1.0].copy()
    eligible["objective_priority"] = eligible["objective"].map(objective_priority)
    winner = eligible.sort_values(
        ["n_origins", "objective_priority", "candidate"],
        kind="stable",
    ).iloc[0]
    return {
        "candidate": winner["candidate"],
        "objective": winner["objective"],
        "n_origins": int(winner["n_origins"]),
        "best_iteration": int(winner["best_iteration"]),
        "v2_wape": float(winner["v2_wape"]),
        "best_observed_v2_wape": float(best_wape),
    }


def save_validation_results(
    summary: pd.DataFrame,
    output_dir: Path = Path("reports"),
    ma28_wape: float | None = None,
) -> tuple[Path, Path]:
    """Persist candidate scores and the preregistered selection."""
    winner = select_best_candidate(summary)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ml_validation_results.csv"
    selection_path = output_dir / "ml_validation_selection.json"
    manifest = {
        "protocol": {
            "training_cutoff": str(TRAINING_CUTOFF.date()),
            "v1_origin": str(V1_ORIGIN.date()),
            "v2_origin": str(V2_ORIGIN.date()),
            "training_origin_counts": list(TRAINING_ORIGIN_COUNTS),
            "objectives": list(OBJECTIVES),
            "max_estimators": MAX_ESTIMATORS,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "tweedie_variance_power": 1.5,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "n_jobs": N_JOBS,
            "selection_metric": "V2 pooled WAPE",
            "wape_tolerance_percentage_points": 1.0,
            "tie_breaking": [
                "fewer training origins",
                "regression before poisson before tweedie",
            ],
        },
        "selected_candidate": winner,
    }
    if ma28_wape is not None:
        if not np.isfinite(ma28_wape) or ma28_wape < 0:
            raise ValueError("MA28 WAPE must be finite and nonnegative")

        manifest["baseline"] = {
            "name": "MA28",
            "v2_wape": float(ma28_wape),
            "selected_lightgbm_v2_wape": winner["v2_wape"],
            "lightgbm_beats_ma28": winner["v2_wape"] < ma28_wape,
        }
    summary.to_csv(results_path, index=False)
    selection_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return results_path, selection_path


def build_v2_horizon_diagnostics(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate diagnostic errors for each V2 forecast horizon."""
    required = {"horizon", "target", "prediction"}

    if scored.empty or not required.issubset(scored.columns):
        raise ValueError("Missing V2 diagnostic data")

    if scored[list(required)].isna().any().any():
        raise ValueError("V2 diagnostics contain missing values")

    values = scored[["target", "prediction"]].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("V2 diagnostics must be finite")

    if (values < 0).any():
        raise ValueError("V2 values must be nonnegative")

    if set(scored["horizon"]) != set(range(1, FORECAST_HORIZON + 1)):
        raise ValueError("Expected all 28 forecast horizons")

    results = []

    for horizon, group in scored.groupby("horizon", sort=True):
        actual = group["target"].to_numpy(dtype=np.float64)
        predicted = group["prediction"].to_numpy(dtype=np.float64)

        errors = actual - predicted

        results.append(
            {
                "horizon": int(horizon),
                "n_rows": len(group),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "wape": float(wape(actual, predicted)),
            }
        )

    return pd.DataFrame(results)


def build_v2_breakdown_diagnostics(
    scored: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Category and Christmas sensitivity diagnostics for selected V2."""
    required = {"series_id", "date", "target", "prediction"}
    if scored.empty or not required.issubset(scored.columns):
        raise ValueError("Missing V2 forecast columns")

    categories = metadata[["series_id", "cat_id"]].copy()
    if categories["series_id"].duplicated().any():
        raise ValueError("Duplicate series_id in metadata")

    enriched = scored.merge(
        categories,
        on="series_id",
        how="left",
        validate="many_to_one",
    )
    if enriched["cat_id"].isna().any():
        raise ValueError("Missing category for V2 series")

    values = enriched[["target", "prediction"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid V2 targets or predictions")

    def metrics(frame: pd.DataFrame) -> dict:
        actual = frame["target"].to_numpy(dtype=np.float64)
        predicted = frame["prediction"].to_numpy(dtype=np.float64)
        errors = actual - predicted
        return {
            "n_rows": len(frame),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "wape": float(wape(actual, predicted)),
        }

    category_rows = [
        {"cat_id": category, **metrics(group)}
        for category, group in enriched.groupby("cat_id", sort=True)
    ]
    category_report = pd.DataFrame(category_rows)

    dates = pd.to_datetime(enriched["date"])
    christmas = dates.isin(pd.to_datetime(["2015-12-24", "2015-12-25", "2015-12-26"]))

    holiday_rows = []
    for label, subset in (
        ("all_v2", enriched),
        ("excluding_dec_24_26", enriched.loc[~christmas]),
        ("dec_24_26_only", enriched.loc[christmas]),
    ):
        if subset.empty:
            raise ValueError(f"Empty holiday diagnostic: {label}")
        holiday_rows.append({"subset": label, **metrics(subset)})

    holiday_report = pd.DataFrame(holiday_rows)
    return category_report, holiday_report


def main(argv: list[str] | None = None) -> int:
    """Run the fixed validation protocol only with explicit confirmation."""
    parser = argparse.ArgumentParser(
        description="Run preregistered LightGBM V1/V2 validation."
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path("data/processed/sales.parquet"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/series.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train and evaluate all nine models.",
    )

    args = parser.parse_args(argv)
    validate_protocol_dates()

    if not args.sales.is_file():
        parser.error(f"Sales file not found: {args.sales}")
    if not args.metadata.is_file():
        parser.error(f"Metadata file not found: {args.metadata}")

    print("LightGBM validation protocol")
    print(f"Sales: {args.sales}")
    print(f"Metadata: {args.metadata}")
    print(f"Training cutoff: {TRAINING_CUTOFF.date()}")
    print(f"V1 origin: {V1_ORIGIN.date()}")
    print(f"V2 origin: {V2_ORIGIN.date()}")
    print(f"Candidates: {len(candidate_grid())}")

    if not args.execute:
        print("DRY RUN: no training performed.")
        print("Pass --execute to run the experiments.")
        return 0

    sales = pd.read_parquet(
        args.sales,
        columns=["series_id", "date", "quantity"],
    )
    metadata = pd.read_parquet(
        args.metadata,
        columns=["series_id", "store_id", "dept_id", "cat_id"],
    )

    # Exclude the later final-test period from the validation experiment.
    validation_end = V2_ORIGIN + pd.Timedelta(days=FORECAST_HORIZON)
    sales = sales.loc[pd.to_datetime(sales["date"]) <= validation_end].copy()

    if sales.empty or metadata.empty:
        raise ValueError("Sales and metadata cannot be empty")

    print("Starting nine sequential LightGBM experiments...")
    summary, forecasts = run_all_candidates(sales=sales, metadata=metadata)

    # Selection uses only the preregistered V2 LightGBM scores.
    winner = select_best_candidate(summary)

    # Compare MA28 against the exact same V2 rows as the selected LightGBM.
    ma28_wape, ma28_scored = score_ma28_on_v2(
        sales=sales,
        v2=forecasts[winner["candidate"]],
    )

    print(f"MA28 V2 WAPE: {ma28_wape:.4f}%")
    print(f"LightGBM V2 WAPE: {winner['v2_wape']:.4f}%")
    print(f"V2 evaluation rows: {len(ma28_scored):,}")

    if winner["v2_wape"] < ma28_wape:
        print("Selected LightGBM outperforms MA28 on V2.")
    else:
        print("Selected LightGBM does not outperform MA28 on V2.")

    results_path, selection_path = save_validation_results(
        summary=summary,
        output_dir=args.output_dir,
        ma28_wape=ma28_wape,
    )

    # Diagnostics for the selected LightGBM model only.
    # These metrics do not influence model selection.
    selected_forecast = forecasts[winner["candidate"]]

    horizon_diagnostics = build_v2_horizon_diagnostics(selected_forecast)

    horizons_path = args.output_dir / "ml_validation_horizons.csv"

    horizon_diagnostics.to_csv(
        horizons_path,
        index=False,
    )

    print(f"Horizon diagnostics saved: {horizons_path}")

    category_report, holiday_report = build_v2_breakdown_diagnostics(
        selected_forecast,
        metadata,
    )

    categories_path = args.output_dir / "ml_validation_categories.csv"
    holiday_path = args.output_dir / "ml_validation_holiday_sensitivity.csv"

    category_report.to_csv(categories_path, index=False)
    holiday_report.to_csv(holiday_path, index=False)

    print(f"Category diagnostics saved: {categories_path}")
    print(f"Holiday diagnostics saved: {holiday_path}")

    print(f"Results saved: {results_path}")
    print(f"Selection saved: {selection_path}")
    print(f"Selected candidate: {winner['candidate']}")
    print(f"Selected V2 WAPE: {winner['v2_wape']:.4f}%")
    return 0


def score_ma28_on_v2(
    sales: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Evaluate MA28 on exactly the V2 LightGBM evaluation rows."""
    required_sales = {"series_id", "date", "quantity"}
    required_v2 = {"series_id", "origin", "date", "horizon", "target"}
    if not required_sales.issubset(sales.columns):
        raise ValueError("Missing sales columns for MA28")
    if not required_v2.issubset(v2.columns):
        raise ValueError("Missing V2 columns for MA28")
    if v2.empty or v2[list(required_v2)].isna().any().any():
        raise ValueError("V2 must contain complete evaluation rows")
    if not pd.to_datetime(v2["origin"]).eq(V2_ORIGIN).all():
        raise ValueError("MA28 evaluation must use V2 origin")
    cohort = set(v2["series_id"])
    history_start = V2_ORIGIN - pd.Timedelta(days=27)
    history = sales.loc[
        sales["series_id"].isin(cohort),
        ["series_id", "date", "quantity"],
    ].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.loc[history["date"].between(history_start, V2_ORIGIN)]
    if history.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate sales dates in MA28 history")
    if len(history) != len(cohort) * 28:
        raise ValueError("Each V2 series needs 28 historical days")
    predictions = {}
    for series_id, group in history.groupby("series_id"):
        ordered = group.sort_values("date")
        values = ordered["quantity"].to_numpy()
        # Reuse the existing, tested baseline implementation.
        predictions[series_id] = float(moving_average_forecast(values)[0])
    scored = v2[["series_id", "origin", "date", "horizon", "target"]].copy()
    scored["prediction"] = scored["series_id"].map(predictions)
    if scored["prediction"].isna().any():
        raise ValueError("Missing MA28 predictions for V2 series")
    baseline_wape = wape(
        scored["target"].to_numpy(),
        scored["prediction"].to_numpy(),
    )
    return float(baseline_wape), scored


if __name__ == "__main__":
    raise SystemExit(main())
