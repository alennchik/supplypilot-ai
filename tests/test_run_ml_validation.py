"""Tests for the preregistered LightGBM validation protocol."""

import pytest

import scripts.run_ml_validation as runner


def test_candidate_grid_has_exactly_nine_configurations():
    """The candidate search space must match the preregistration."""
    candidates = runner.candidate_grid()

    expected = {
        (objective, n_origins)
        for objective in ("regression", "poisson", "tweedie")
        for n_origins in (8, 16, 32)
    }

    actual = {(candidate.objective, candidate.n_origins) for candidate in candidates}

    assert len(candidates) == 9
    assert len(actual) == 9
    assert actual == expected

    names = [candidate.name for candidate in candidates]
    assert len(names) == len(set(names))


def test_training_and_validation_dates_are_fixed():
    """V1 and V2 must follow the fixed chronological protocol."""
    runner.validate_protocol_dates()

    assert str(runner.TRAINING_CUTOFF.date()) == "2015-11-08"
    assert str(runner.V1_ORIGIN.date()) == "2015-11-08"
    assert str(runner.V2_ORIGIN.date()) == "2015-12-06"

    assert runner.MAX_ESTIMATORS == 2000
    assert runner.EARLY_STOPPING_ROUNDS == 100
    assert runner.N_JOBS == 2


def test_protocol_rejects_modified_validation_origin(monkeypatch):
    """Changing the spacing between V1 and V2 must fail."""
    import pandas as pd

    monkeypatch.setattr(
        runner,
        "V2_ORIGIN",
        runner.V2_ORIGIN + pd.Timedelta(days=1),
    )

    with pytest.raises(ValueError, match="28 days apart"):
        runner.validate_protocol_dates()


@pytest.mark.parametrize("n_origins", [8, 16, 32])
def test_prepare_training_dataset_respects_cutoff(n_origins):
    """Training datasets must use exact origins and no future targets."""
    import pandas as pd

    from src.ml_dataset import select_training_origins

    dates = pd.date_range(
        "2015-01-01",
        "2016-01-03",
        freq="D",
    )

    sales = pd.DataFrame(
        [
            {
                "series_id": series_id,
                "date": date,
                "quantity": float((day_number + series_number) % 7 + 1),
            }
            for series_number, series_id in enumerate(["A", "B"])
            for day_number, date in enumerate(dates)
        ]
    )

    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "store_id": ["CA_1", "CA_1"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
        }
    )

    candidate = runner.Candidate(
        objective="regression",
        n_origins=n_origins,
    )

    training = runner.prepare_training_dataset(
        sales=sales,
        metadata=metadata,
        candidate=candidate,
    )

    expected_origins = set(
        select_training_origins(
            runner.TRAINING_CUTOFF,
            n_origins=n_origins,
        )
    )

    assert len(training) == 2 * n_origins * 28
    assert set(training["origin"]) == expected_origins
    assert training["date"].max() == runner.TRAINING_CUTOFF
    assert training["date"].le(runner.TRAINING_CUTOFF).all()
    assert training["target"].notna().all()
    assert not training.duplicated(["series_id", "origin", "horizon"]).any()


@pytest.mark.parametrize(
    ("objective", "n_origins"),
    [
        ("regression", 4),
        ("poisson", 12),
        ("invalid_objective", 8),
    ],
)
def test_prepare_training_rejects_unregistered_candidates(objective, n_origins):
    """Only the nine preregistered candidates may enter training."""
    import pandas as pd

    candidate = runner.Candidate(
        objective=objective,
        n_origins=n_origins,
    )

    with pytest.raises(
        ValueError,
        match="Candidate is not in the preregistered grid",
    ):
        runner.prepare_training_dataset(
            sales=pd.DataFrame(),
            metadata=pd.DataFrame(),
            candidate=candidate,
        )


def make_validation_frames():
    """Create complete synthetic V1 and V2 forecast observations."""
    import pandas as pd

    frames = []

    for origin in (runner.V1_ORIGIN, runner.V2_ORIGIN):
        rows = []

        for series_id in ("series_A", "series_B"):
            for horizon in range(1, 29):
                rows.append(
                    {
                        "series_id": series_id,
                        "origin": origin,
                        "date": origin + pd.Timedelta(days=horizon),
                        "horizon": horizon,
                        "target": float(horizon),
                    }
                )

        frames.append(pd.DataFrame(rows))

    return frames[0], frames[1]


def test_validation_sets_accept_valid_windows():
    v1, v2 = make_validation_frames()

    runner.validate_validation_sets(v1, v2)


def test_validation_sets_reject_swapped_windows():
    v1, v2 = make_validation_frames()

    with pytest.raises(ValueError, match="incorrect forecast origin"):
        runner.validate_validation_sets(v2, v1)


def test_validation_sets_reject_missing_horizon():
    v1, v2 = make_validation_frames()

    v2 = v2.loc[~(v2["series_id"].eq("series_A") & v2["horizon"].eq(28))].copy()

    with pytest.raises(ValueError, match="incomplete forecast horizons"):
        runner.validate_validation_sets(v1, v2)


def test_validation_sets_reject_different_cohorts():
    v1, v2 = make_validation_frames()

    v2.loc[
        v2["series_id"].eq("series_B"),
        "series_id",
    ] = "series_C"

    with pytest.raises(ValueError, match="same fixed cohort"):
        runner.validate_validation_sets(v1, v2)


def test_run_candidate_uses_only_v1_for_early_stopping(monkeypatch):
    """V2 targets must never be passed into model training."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from src.ml_dataset import select_training_origins
    from src.ml_model import FEATURE_COLUMNS

    candidate = runner.Candidate(
        objective="regression",
        n_origins=8,
    )

    training_rows = []

    for origin in select_training_origins(
        runner.TRAINING_CUTOFF,
        n_origins=8,
    ):
        for horizon in range(1, 29):
            row = {column: 1.0 for column in FEATURE_COLUMNS}
            row.update(
                {
                    "series_id": "series_A",
                    "origin": origin,
                    "date": origin + pd.Timedelta(days=horizon),
                    "horizon": horizon,
                    "target": float(horizon),
                }
            )
            training_rows.append(row)

    training = pd.DataFrame(training_rows)
    v1, v2 = make_validation_frames()

    calls = []

    def fake_train(
        training_data,
        *,
        objective,
        n_estimators,
        early_stopping_rounds,
        n_jobs,
        validation_data,
    ):
        calls.append("train")

        assert training_data is training
        assert validation_data is v1
        assert validation_data is not v2

        assert objective == "regression"
        assert n_estimators == 2000
        assert early_stopping_rounds == 100
        assert n_jobs == 2

        assert training_data["date"].max() <= runner.TRAINING_CUTOFF
        assert validation_data["origin"].eq(runner.V1_ORIGIN).all()

        return SimpleNamespace(best_iteration=42)

    def fake_predict(model, examples):
        calls.append("predict")

        assert model.best_iteration == 42
        assert "target" not in examples.columns
        assert examples["origin"].eq(runner.V2_ORIGIN).all()

        keys = ["series_id", "origin", "date", "horizon"]
        predictions = examples[keys].copy()

        # Different prediction for each horizon.
        predictions["prediction"] = predictions["horizon"].to_numpy(dtype=float) * 0.5

        # Deliberately shuffle predictions to test key-based alignment.
        return predictions.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

    monkeypatch.setattr(runner, "train_global_model", fake_train)
    monkeypatch.setattr(runner, "predict_global_model", fake_predict)

    result, scored = runner.run_candidate(
        candidate,
        training,
        v1,
        v2,
    )

    assert calls == ["train", "predict"]
    assert result["candidate"] == "regression_8_origins"
    assert result["best_iteration"] == 42

    expected_wape = 50.0

    assert result["v2_wape"] == pytest.approx(expected_wape)

    assert len(scored) == len(v2)
    np.testing.assert_allclose(
        scored["prediction"].to_numpy(),
        scored["horizon"].to_numpy(dtype=float) * 0.5,
    )
    assert scored["target"].notna().all()


def test_run_all_candidates_executes_exact_grid_sequentially(monkeypatch):
    """Run every registered candidate once in fixed order."""
    import pandas as pd

    expected_v1, expected_v2 = make_validation_frames()
    calls = []
    active = False

    def fake_validation_features(sales, metadata):
        calls.append(("validation", None))
        return expected_v1, expected_v2

    def fake_prepare(sales, metadata, candidate):
        calls.append(("prepare", candidate.name))
        return pd.DataFrame({"candidate": [candidate.name]})

    def fake_run(candidate, training, v1, v2):
        nonlocal active

        assert not active, "Candidates must not overlap"
        active = True

        try:
            assert v1 is expected_v1
            assert v2 is expected_v2
            assert training["candidate"].iloc[0] == candidate.name

            calls.append(("run", candidate.name))

            result = {
                "candidate": candidate.name,
                "objective": candidate.objective,
                "n_origins": candidate.n_origins,
                "best_iteration": 42,
                "v2_wape": 50.0,
            }

            scored = pd.DataFrame(
                {
                    "series_id": ["series_A"],
                    "prediction": [1.0],
                }
            )

            return result, scored
        finally:
            active = False

    monkeypatch.setattr(
        runner, "build_validation_feature_sets", fake_validation_features
    )
    monkeypatch.setattr(runner, "prepare_training_dataset", fake_prepare)
    monkeypatch.setattr(runner, "run_candidate", fake_run)

    summary, forecasts = runner.run_all_candidates(
        sales=pd.DataFrame(),
        metadata=pd.DataFrame(),
    )

    expected = [candidate.name for candidate in runner.candidate_grid()]

    assert summary["candidate"].tolist() == expected
    assert list(forecasts) == expected
    assert len(summary) == 9

    assert calls == [
        ("validation", None),
        *[entry for name in expected for entry in (("prepare", name), ("run", name))],
    ]


def make_candidate_summary():
    """Create all nine candidates with synthetic V2 scores."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "objective": candidate.objective,
                "n_origins": candidate.n_origins,
                "best_iteration": 42,
                "v2_wape": 70.0,
            }
            for candidate in runner.candidate_grid()
        ]
    )


def test_selection_picks_true_minimum_outside_tolerance():
    """A model more than 1 WAPE point better must win."""
    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 41.1

    winner = runner.select_best_candidate(summary)

    assert winner["candidate"] == "tweedie_32_origins"
    assert winner["v2_wape"] == pytest.approx(40.0)


def test_selection_prefers_fewer_origins_within_tolerance():
    """Within 1 percentage point, fewer origins take priority."""
    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 40.8

    winner = runner.select_best_candidate(summary)

    assert winner["candidate"] == "regression_8_origins"
    assert winner["v2_wape"] == pytest.approx(40.8)
    assert winner["best_observed_v2_wape"] == pytest.approx(40.0)


def test_selection_uses_objective_priority():
    """With equal origins, regression beats poisson and tweedie."""
    summary = make_candidate_summary()

    for objective, score in (
        ("regression", 40.8),
        ("poisson", 40.4),
        ("tweedie", 40.0),
    ):
        summary.loc[
            summary["candidate"].eq(f"{objective}_8_origins"),
            "v2_wape",
        ] = score

    winner = runner.select_best_candidate(summary)

    assert winner["candidate"] == "regression_8_origins"


def test_selection_rejects_missing_candidate():
    """Selection cannot proceed with incomplete experiment results."""
    summary = make_candidate_summary().iloc[:-1].copy()

    with pytest.raises(ValueError, match="exact nine candidates"):
        runner.select_best_candidate(summary)


def test_selection_includes_exactly_one_point_tolerance():
    """A candidate exactly 1.0 WAPE point above the minimum qualifies."""
    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 41.0

    winner = runner.select_best_candidate(summary)

    assert winner["candidate"] == "regression_8_origins"
    assert winner["v2_wape"] == pytest.approx(41.0)


@pytest.mark.parametrize("invalid_wape", [float("nan"), float("inf"), -1.0])
def test_selection_rejects_invalid_wape(invalid_wape):
    """Invalid model scores cannot be used for selection."""
    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = invalid_wape

    with pytest.raises(ValueError, match="WAPE values"):
        runner.select_best_candidate(summary)


def test_save_validation_results_round_trip(tmp_path):
    """Saved CSV and JSON must preserve scores and selected protocol."""
    import json

    import pandas as pd

    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 40.8

    results_path, selection_path = runner.save_validation_results(
        summary,
        output_dir=tmp_path,
    )

    assert results_path.exists()
    assert selection_path.exists()

    saved_results = pd.read_csv(results_path)

    pd.testing.assert_frame_equal(
        saved_results,
        summary,
        check_dtype=False,
    )

    manifest = json.loads(selection_path.read_text(encoding="utf-8"))

    protocol = manifest["protocol"]
    winner = manifest["selected_candidate"]

    assert protocol["training_cutoff"] == "2015-11-08"
    assert protocol["v1_origin"] == "2015-11-08"
    assert protocol["v2_origin"] == "2015-12-06"

    assert protocol["training_origin_counts"] == [8, 16, 32]
    assert protocol["objectives"] == [
        "regression",
        "poisson",
        "tweedie",
    ]

    assert protocol["max_estimators"] == 2000
    assert protocol["early_stopping_rounds"] == 100
    assert protocol["wape_tolerance_percentage_points"] == 1.0

    assert winner["candidate"] == "regression_8_origins"
    assert winner["best_iteration"] == 42
    assert winner["v2_wape"] == pytest.approx(40.8)
    assert winner["best_observed_v2_wape"] == pytest.approx(40.0)


def test_cli_dry_run_never_starts_training(tmp_path, monkeypatch, capsys):
    """CLI without --execute must never train or read datasets."""
    sales_path = tmp_path / "sales.parquet"
    metadata_path = tmp_path / "series.parquet"

    # Files exist, but deliberately contain no Parquet data.
    sales_path.touch()
    metadata_path.touch()

    def forbidden_training(*args, **kwargs):
        raise AssertionError("Dry run must not start training")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Dry run must not read Parquet data")

    monkeypatch.setattr(runner, "run_all_candidates", forbidden_training)
    monkeypatch.setattr(runner.pd, "read_parquet", forbidden_read)

    result = runner.main(
        [
            "--sales",
            str(sales_path),
            "--metadata",
            str(metadata_path),
        ]
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "Candidates: 9" in output
    assert "DRY RUN: no training performed." in output
    assert "Pass --execute" in output


def test_real_lightgbm_candidate_end_to_end(monkeypatch):
    """Run a small real LightGBM fit without touching M5 data."""
    import numpy as np
    import pandas as pd

    from src.ml_validation import build_validation_feature_sets

    dates = pd.date_range(
        "2015-01-01",
        "2016-01-03",
        freq="D",
    )

    sales = pd.DataFrame(
        [
            {
                "series_id": series_id,
                "date": date,
                "quantity": float((day_number + series_number * 3) % 11 + 1),
            }
            for series_number, series_id in enumerate(
                ["series_A", "series_B", "series_C"]
            )
            for day_number, date in enumerate(dates)
        ]
    )

    metadata = pd.DataFrame(
        {
            "series_id": ["series_A", "series_B", "series_C"],
            "store_id": ["CA_1", "CA_1", "CA_2"],
            "dept_id": ["FOODS_1", "FOODS_1", "FOODS_2"],
            "cat_id": ["FOODS", "FOODS", "FOODS"],
        }
    )

    candidate = runner.Candidate(
        objective="regression",
        n_origins=8,
    )

    # Shorten training ONLY inside this synthetic test.
    monkeypatch.setattr(runner, "MAX_ESTIMATORS", 30)
    monkeypatch.setattr(runner, "EARLY_STOPPING_ROUNDS", 5)

    training = runner.prepare_training_dataset(sales, metadata, candidate)
    v1, v2 = build_validation_feature_sets(sales, metadata)

    result, scored = runner.run_candidate(candidate, training, v1, v2)

    assert result["candidate"] == "regression_8_origins"
    assert 1 <= result["best_iteration"] <= 30

    assert len(scored) == 3 * 28
    assert scored["prediction"].notna().all()
    assert np.isfinite(scored["prediction"]).all()
    assert scored["prediction"].ge(0).all()

    assert np.isfinite(result["v2_wape"])
    assert result["v2_wape"] >= 0


def test_ma28_v2_scores_exact_cohort_without_future_leakage():
    """MA28 uses only the 28 days available at the V2 origin."""
    import pandas as pd

    origin = pd.Timestamp("2015-12-06")
    history_dates = pd.date_range(
        origin - pd.Timedelta(days=27),
        origin,
    )

    sales_rows = []
    v2_rows = []

    for series_id, historical_quantity in (
        ("series_A", 2.0),
        ("series_B", 6.0),
    ):
        for date in history_dates:
            sales_rows.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "quantity": historical_quantity,
                }
            )

        for horizon in range(1, 29):
            future_date = origin + pd.Timedelta(days=horizon)

            # Deliberately extreme future sales.
            # MA28 must never use these values.
            sales_rows.append(
                {
                    "series_id": series_id,
                    "date": future_date,
                    "quantity": 9999.0,
                }
            )

            v2_rows.append(
                {
                    "series_id": series_id,
                    "origin": origin,
                    "date": future_date,
                    "horizon": horizon,
                    "target": 10.0,
                }
            )

    sales = pd.DataFrame(sales_rows)
    v2 = pd.DataFrame(v2_rows)

    baseline_wape, scored = runner.score_ma28_on_v2(sales, v2)

    assert len(scored) == 56
    assert scored["prediction"].notna().all()

    predictions_a = scored.loc[scored["series_id"].eq("series_A"), "prediction"]
    predictions_b = scored.loc[scored["series_id"].eq("series_B"), "prediction"]

    assert predictions_a.eq(2.0).all()
    assert predictions_b.eq(6.0).all()

    assert baseline_wape == pytest.approx(60.0)

    keys = ["series_id", "origin", "date", "horizon"]

    pd.testing.assert_frame_equal(
        scored[keys].reset_index(drop=True),
        v2[keys].reset_index(drop=True),
    )


def test_ma28_rejects_missing_history_day():
    """Each evaluated series must have all 28 history days."""
    import pandas as pd

    origin = pd.Timestamp("2015-12-06")

    sales = pd.DataFrame(
        {
            "series_id": ["series_A"] * 27,
            "date": pd.date_range("2015-11-10", periods=27),
            "quantity": [5.0] * 27,
        }
    )

    v2 = pd.DataFrame(
        {
            "series_id": ["series_A"] * 28,
            "origin": [origin] * 28,
            "date": pd.date_range("2015-12-07", periods=28),
            "horizon": range(1, 29),
            "target": [10.0] * 28,
        }
    )

    with pytest.raises(ValueError, match="28 historical days"):
        runner.score_ma28_on_v2(sales, v2)


def test_save_validation_results_includes_ma28(tmp_path):
    """Persist the MA28 comparison without changing model selection."""
    import json

    summary = make_candidate_summary()

    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0

    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 40.8

    _, selection_path = runner.save_validation_results(
        summary,
        output_dir=tmp_path,
        ma28_wape=42.5,
    )

    manifest = json.loads(selection_path.read_text(encoding="utf-8"))

    winner = manifest["selected_candidate"]
    baseline = manifest["baseline"]

    assert winner["candidate"] == "regression_8_origins"

    assert baseline["name"] == "MA28"
    assert baseline["v2_wape"] == pytest.approx(42.5)
    assert baseline["selected_lightgbm_v2_wape"] == pytest.approx(40.8)
    assert baseline["lightgbm_beats_ma28"] is True


def test_cli_execute_saves_results_without_real_training(tmp_path, monkeypatch, capsys):
    """Test the complete CLI path with mocked experiments."""
    import json

    import pandas as pd

    sales_path = tmp_path / "sales.parquet"
    metadata_path = tmp_path / "series.parquet"
    output_dir = tmp_path / "reports"

    origin = pd.Timestamp("2015-12-06")

    dates = pd.date_range("2015-11-09", "2016-01-05")

    sales = pd.DataFrame(
        {
            "series_id": ["series_A"] * len(dates),
            "date": dates,
            "quantity": [5.0 if date <= origin else 10.0 for date in dates],
        }
    )
    sales.to_parquet(sales_path, index=False)

    metadata = pd.DataFrame(
        {
            "series_id": ["series_A"],
            "store_id": ["CA_1"],
            "dept_id": ["FOODS_1"],
            "cat_id": ["FOODS"],
        }
    )
    metadata.to_parquet(metadata_path, index=False)

    summary = make_candidate_summary()
    summary.loc[
        summary["candidate"].eq("tweedie_32_origins"),
        "v2_wape",
    ] = 40.0
    summary.loc[
        summary["candidate"].eq("regression_8_origins"),
        "v2_wape",
    ] = 40.8

    v2 = pd.DataFrame(
        {
            "series_id": ["series_A"] * 28,
            "origin": [origin] * 28,
            "date": pd.date_range("2015-12-07", periods=28),
            "horizon": range(1, 29),
            "target": [10.0] * 28,
            "prediction": [6.0] * 28,
        }
    )

    def fake_run_all_candidates(sales, metadata):
        # The final test period must not reach experiments.
        assert sales["date"].max() == pd.Timestamp("2016-01-03")
        assert len(metadata) == 1

        return summary, {"regression_8_origins": v2.copy()}

    monkeypatch.setattr(
        runner,
        "run_all_candidates",
        fake_run_all_candidates,
    )

    result = runner.main(
        [
            "--sales",
            str(sales_path),
            "--metadata",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
            "--execute",
        ]
    )

    assert result == 0

    results_path = output_dir / "ml_validation_results.csv"
    selection_path = output_dir / "ml_validation_selection.json"

    assert results_path.exists()
    assert selection_path.exists()
    assert len(pd.read_csv(results_path)) == 9

    horizons_path = output_dir / "ml_validation_horizons.csv"

    assert horizons_path.exists()

    horizons = pd.read_csv(horizons_path)

    assert len(horizons) == 28
    assert horizons["horizon"].tolist() == list(range(1, 29))
    assert horizons["n_rows"].eq(1).all()

    assert horizons["mae"].eq(4.0).all()
    assert horizons["rmse"].eq(4.0).all()
    assert horizons["wape"].eq(40.0).all()

    # Check category and Christmas diagnostic reports.
    categories_path = output_dir / "ml_validation_categories.csv"
    holiday_path = output_dir / "ml_validation_holiday_sensitivity.csv"

    assert categories_path.exists()
    assert holiday_path.exists()

    categories = pd.read_csv(categories_path)

    assert len(categories) == 1
    assert categories.iloc[0]["cat_id"] == "FOODS"
    assert categories.iloc[0]["n_rows"] == 28
    assert categories.iloc[0]["mae"] == pytest.approx(4.0)
    assert categories.iloc[0]["wape"] == pytest.approx(40.0)

    holidays = pd.read_csv(holiday_path).set_index("subset")

    assert set(holidays.index) == {
        "all_v2",
        "excluding_dec_24_26",
        "dec_24_26_only",
    }

    assert holidays.loc["all_v2", "n_rows"] == 28
    assert holidays.loc["excluding_dec_24_26", "n_rows"] == 25
    assert holidays.loc["dec_24_26_only", "n_rows"] == 3

    assert holidays["mae"].eq(4.0).all()
    assert holidays["wape"].eq(40.0).all()

    manifest = json.loads(selection_path.read_text(encoding="utf-8"))

    assert manifest["selected_candidate"]["candidate"] == "regression_8_origins"
    assert manifest["baseline"]["v2_wape"] == pytest.approx(50.0)
    assert manifest["baseline"]["lightgbm_beats_ma28"] is True

    output = capsys.readouterr().out

    assert "MA28 V2 WAPE: 50.0000%" in output
    assert "Selected candidate: regression_8_origins" in output


def test_v2_horizon_diagnostics_calculates_correct_metrics():
    """Verify MAE, RMSE and WAPE for all 28 horizons."""
    import numpy as np
    import pandas as pd

    rows = []

    for horizon in range(1, 29):
        for target in (10.0, 20.0):
            if horizon == 1:
                prediction = target * 0.8
            else:
                prediction = target

            rows.append(
                {
                    "horizon": horizon,
                    "target": target,
                    "prediction": prediction,
                }
            )

    scored = pd.DataFrame(rows)

    diagnostics = runner.build_v2_horizon_diagnostics(scored)

    assert len(diagnostics) == 28
    assert diagnostics["horizon"].tolist() == list(range(1, 29))
    assert diagnostics["n_rows"].eq(2).all()

    first_day = diagnostics.iloc[0]

    assert first_day["mae"] == pytest.approx(3.0)
    assert first_day["rmse"] == pytest.approx(np.sqrt(10.0))
    assert first_day["wape"] == pytest.approx(20.0)

    remaining_days = diagnostics.iloc[1:]

    assert remaining_days["mae"].eq(0.0).all()
    assert remaining_days["rmse"].eq(0.0).all()
    assert remaining_days["wape"].eq(0.0).all()


def test_v2_horizon_diagnostics_rejects_missing_horizon():
    """Reject diagnostics missing one of the 28 forecast days."""
    import pandas as pd

    scored = pd.DataFrame(
        {
            "horizon": range(1, 28),
            "target": [10.0] * 27,
            "prediction": [8.0] * 27,
        }
    )

    with pytest.raises(ValueError, match="28 forecast horizons"):
        runner.build_v2_horizon_diagnostics(scored)


def test_v2_category_and_holiday_diagnostics():
    """Check category metrics and Christmas sensitivity."""
    import pandas as pd

    rows = []

    for series_id, target, prediction in (
        ("A", 10.0, 8.0),
        ("B", 20.0, 16.0),
    ):
        for horizon in range(1, 29):
            rows.append(
                {
                    "series_id": series_id,
                    "date": pd.Timestamp("2015-12-06") + pd.Timedelta(days=horizon),
                    "target": target,
                    "prediction": prediction,
                }
            )

    scored = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    categories, holidays = runner.build_v2_breakdown_diagnostics(scored, metadata)

    assert len(categories) == 2
    assert categories["n_rows"].eq(28).all()
    assert categories["wape"].eq(20.0).all()

    by_subset = holidays.set_index("subset")

    assert by_subset.loc["all_v2", "n_rows"] == 56
    assert by_subset.loc["excluding_dec_24_26", "n_rows"] == 50
    assert by_subset.loc["dec_24_26_only", "n_rows"] == 6

    assert by_subset["wape"].eq(20.0).all()
    assert by_subset.loc["all_v2", "mae"] == pytest.approx(3.0)
