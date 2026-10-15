"""Reporting contracts for preregistered ML validation (no model training)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_ml_validation import (
    build_v1_learning_curves,
    candidate_grid,
    save_validation_results,
    select_best_candidate,
)


def _complete_summary() -> pd.DataFrame:
    rows = []
    for candidate in candidate_grid():
        rows.append(
            {
                "candidate": candidate.name,
                "objective": candidate.objective,
                "n_origins": candidate.n_origins,
                "best_iteration": 2,
                "v2_wape": 80.0 if candidate.name == "tweedie_8_origins" else 82.0,
                "v2_mae": 0.8,
                "v2_rmse": 1.9,
                "v1_best_mae": 0.6,
                "training_rows": 100,
                "fit_seconds": 1.25,
                "v1_mae_history": [0.9, 0.6, 0.65],
            }
        )
    return pd.DataFrame(rows)


def test_v1_curves_have_exact_nine_candidates_and_best_marker():
    summary = _complete_summary()
    curves = build_v1_learning_curves(summary)
    assert len(curves) == 27
    assert curves["candidate"].nunique() == 9
    assert curves.groupby("candidate")["is_best_iteration"].sum().eq(1).all()
    assert curves.loc[curves["is_best_iteration"], "iteration"].eq(2).all()
    assert curves["v1_clipped_mae"].between(0, 1).all()
    assert select_best_candidate(summary)["candidate"] == "tweedie_8_origins"


def test_bad_v1_history_is_rejected():
    summary = _complete_summary()
    summary.at[0, "v1_mae_history"] = [0.9, float("nan")]
    with pytest.raises(ValueError, match="Invalid V1 history"):
        build_v1_learning_curves(summary)


def test_candidate_csv_keeps_metrics_not_raw_histories(tmp_path):
    summary = _complete_summary()
    results_path, selection_path = save_validation_results(
        summary, output_dir=tmp_path, ma28_wape=84.0,
        experiment_seconds=12.34, experiment_peak_rss_mib=1024.0,
    )
    saved = pd.read_csv(results_path)
    assert len(saved) == 9
    assert "v1_mae_history" not in saved.columns
    required = {"v2_mae", "v2_rmse", "v1_best_mae", "training_rows", "fit_seconds"}
    assert required.issubset(saved.columns)
    assert np.isfinite(saved["v2_wape"]).all()
    manifest = json.loads(selection_path.read_text())
    assert manifest["selected_candidate"]["candidate"] == "tweedie_8_origins"
    assert manifest["resource_usage"]["process_cumulative_peak_rss_mib"] == 1024.0
