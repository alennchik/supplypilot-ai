"""Phase 3C LightGBM forecast paths from the frozen Issue #28 model.

The model is loaded from its saved artifacts and never refitted. Each origin
uses exactly the Issue #28 inference path: sales through the origin, metadata
for those series, future-known calendar features, cohort filter, predict.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory_inputs import REVIEW_ORIGINS, paths_from_predictions
from src.ml_inference import build_ml_inference_examples
from src.ml_model import GlobalDemandModel, load_global_model, predict_global_model

FROZEN_TREES = 1743


def load_frozen_model(artifacts_dir) -> GlobalDemandModel:
    """Load the saved frozen model and confirm it is the Issue #28 model."""
    model = load_global_model(Path(artifacts_dir) / "model.txt")

    if (
        model.booster.num_trees() != FROZEN_TREES
        or model.booster.current_iteration() != FROZEN_TREES
        or model.objective != "tweedie"
        or not model.calendar_features
    ):
        raise ValueError(
            "Saved model is not the frozen 1,743-tree calendar Tweedie model"
        )

    return model


def lightgbm_paths(
    model: GlobalDemandModel,
    sales: pd.DataFrame,
    metadata: pd.DataFrame,
    calendar: pd.DataFrame,
    cohort,
    origins=REVIEW_ORIGINS,
) -> np.ndarray:
    """Daily LightGBM forecasts, shape (series, origins, 28)."""
    cohort = pd.Index(cohort)
    frames = []

    for origin in origins:
        origin = pd.Timestamp(origin)
        history = sales.loc[sales["date"] <= origin]
        history_metadata = metadata.loc[metadata["series_id"].isin(history["series_id"])]
        examples = build_ml_inference_examples(
            history, history_metadata, origin, calendar=calendar
        )
        examples = examples.loc[examples["series_id"].isin(cohort)]
        frames.append(predict_global_model(model, examples))

    predictions = pd.concat(frames, ignore_index=True)

    return paths_from_predictions(
        predictions, cohort, origins, value_column="prediction"
    )
