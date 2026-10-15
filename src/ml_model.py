"""Global direct multi-horizon demand model built on LightGBM."""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml_features import DEMAND_FEATURE_NAMES

CATEGORICAL_FEATURES = ("store_id", "dept_id", "cat_id")
NUMERIC_FEATURES = ("horizon", *DEMAND_FEATURE_NAMES)
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
OUTPUT_COLUMNS = ("series_id", "origin", "date", "horizon")


@dataclass
class GlobalDemandModel:
    booster: lgb.Booster
    category_levels: dict[str, list[str]]


def _prepare_features(
    data: pd.DataFrame,
    category_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Use a fixed feature schema; encode unfamiliar categories as missing."""
    missing = set(FEATURE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if data.empty:
        raise ValueError("Model input cannot be empty")

    features = pd.DataFrame(index=range(len(data)))

    for name in NUMERIC_FEATURES:
        values = pd.to_numeric(data[name], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )

        if np.isinf(values).any():
            raise ValueError(f"Non-finite numeric feature: {name}")

        if name == "horizon":
            if (
                not np.isfinite(values).all()
                or ((values < 1) | (values > 28) | (values != np.floor(values))).any()
            ):
                raise ValueError("horizon must be an integer from 1 to 28")
            features[name] = values.astype(np.int8)
        else:
            features[name] = values.astype(np.float32)

    learned_levels: dict[str, list[str]] = {}

    for name in CATEGORICAL_FEATURES:
        values = data[name].reset_index(drop=True)

        if (
            values.isna().any()
            or not values.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all()
        ):
            raise ValueError(f"Invalid categorical values: {name}")

        if category_levels is None:
            levels = sorted(values.unique().tolist())
        else:
            levels = category_levels[name]

        learned_levels[name] = list(levels)
        features[name] = pd.Categorical(
            values.where(values.isin(levels)), categories=levels
        )

    return features.loc[:, list(FEATURE_COLUMNS)], learned_levels


def train_global_model(
    training_data: pd.DataFrame,
    *,
    n_estimators: int = 100,
    n_jobs: int = 2,
) -> GlobalDemandModel:
    """Fit one regression model across all forecast horizons."""
    if "target" not in training_data.columns:
        raise ValueError("Training data must include target")

    features, levels = _prepare_features(training_data)

    target = pd.to_numeric(training_data["target"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )

    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Training targets must be finite and nonnegative")

    regressor = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    regressor.fit(
        features,
        target,
        categorical_feature=list(CATEGORICAL_FEATURES),
    )

    return GlobalDemandModel(regressor.booster_, levels)


def predict_global_model(
    model: GlobalDemandModel,
    examples: pd.DataFrame,
) -> pd.DataFrame:
    """Predict nonnegative sales for supplied forecast examples."""
    missing = set(OUTPUT_COLUMNS) - set(examples.columns)
    if missing:
        raise ValueError(f"Missing prediction identifiers: {sorted(missing)}")

    features, _ = _prepare_features(examples, model.category_levels)

    predictions = np.asarray(
        model.booster.predict(features, validate_features=True),
        dtype=np.float64,
    )

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned non-finite predictions")

    result = examples.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True).copy()
    result["prediction"] = np.maximum(predictions, 0.0)

    return result


def save_global_model(model: GlobalDemandModel, path: Path) -> None:
    """Save LightGBM trees and category vocabulary as adjacent files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.booster.save_model(str(path))

    metadata = {
        "schema_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "category_levels": model.category_levels,
    }

    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_global_model(path: Path) -> GlobalDemandModel:
    """Reload trees and the exact training-time category vocabulary."""
    path = Path(path)

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

    if metadata.get("schema_version") != 1 or metadata.get("feature_columns") != list(
        FEATURE_COLUMNS
    ):
        raise ValueError("Incompatible saved ML feature schema")

    category_levels = metadata["category_levels"]

    if set(category_levels) != set(CATEGORICAL_FEATURES):
        raise ValueError("Incompatible saved ML categorical schema")

    return GlobalDemandModel(
        lgb.Booster(model_file=str(path)),
        category_levels,
    )
