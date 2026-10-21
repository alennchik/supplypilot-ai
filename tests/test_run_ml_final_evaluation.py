"""Synthetic artifact/CLI integration tests; no real M5 inputs are accessed."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import run_ml_final_evaluation as runner
from src import ml_final_evaluation as final
from tests.test_final_evaluation import (
    fake_model,
    make_synthetic_inputs,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    sales, metadata, calendar, baseline = make_synthetic_inputs()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_SERIES", 3)
    for filename in ("README.md", "requirements.txt", "config.yaml"):
        (tmp_path / filename).write_text("synthetic source")
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"series_id": metadata.series_id}).to_csv(
        reports / "sample_series.csv", index=False
    )
    (reports / "m5_backtesting_manifest.json").write_text("{}")
    adoption = reports / "cycle2_calendar/first_run/protocol_selection.json"
    adoption.parent.mkdir(parents=True)
    adoption.write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "candidate": final.MODEL,
                    "best_iteration": 1743,
                    "n_origins": 8,
                    "adopted_cycle2": True,
                }
            }
        )
    )
    inputs = tmp_path / "synthetic_inputs"
    inputs.mkdir()
    sales.to_parquet(inputs / "sales.parquet", index=False)
    metadata.to_parquet(inputs / "metadata.parquet", index=False)
    calendar.to_csv(inputs / "calendar.csv", index=False)
    baseline.to_parquet(inputs / "baselines.parquet", index=False)
    monkeypatch.setattr(runner, "validate_sample_metadata", lambda *a: None)

    def execution(paths):
        return {
            "source_commit": "synthetic-commit",
            "source_dirty": True,
            "dependencies": {},
            "inputs": {
                name: {"path": str(path), "sha256": runner.file_sha256(path)}
                for name, path in paths.items()
            },
        }

    monkeypatch.setattr(runner, "execution_metadata", execution)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **k: "synthetic-commit"
    )

    def predict(model, examples):
        result = examples[final.KEYS].copy()
        result["prediction"] = 1.0
        return result

    monkeypatch.setattr(final, "train_global_model", lambda *a, **k: fake_model())
    monkeypatch.setattr(final, "predict_global_model", predict)
    original = final.structural_preflight
    monkeypatch.setattr(
        runner, "structural_preflight", lambda *a: original(*a, expected_series=3)
    )

    def save(model, path):
        path.write_text("synthetic trees")
        path.with_suffix(".json").write_text("{}")

    monkeypatch.setattr(runner, "save_global_model", save)
    monkeypatch.setattr(runner, "load_global_model", lambda *a: fake_model())
    output = reports / "ml_final_evaluation/primary"
    artifacts = tmp_path / "data/processed/ml_final_evaluation/primary"
    args = [
        "--output-dir",
        str(output),
        "--artifacts-dir",
        str(artifacts),
        "--sales",
        str(inputs / "sales.parquet"),
        "--metadata",
        str(inputs / "metadata.parquet"),
        "--calendar",
        str(inputs / "calendar.csv"),
        "--baselines",
        str(inputs / "baselines.parquet"),
    ]
    return args, output, artifacts, inputs


def test_default_is_blind_dry_run(monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("Dry run performed I/O or evaluation")

    monkeypatch.setattr(runner.pd, "read_parquet", forbidden)
    monkeypatch.setattr(runner.pd, "read_csv", forbidden)
    monkeypatch.setattr(runner, "run_preflight", forbidden)
    monkeypatch.setattr(runner, "run_reveal", forbidden)
    assert runner.main([]) == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_preflight_then_reveal_no_retraining(sandbox, monkeypatch, capsys):
    args, output, artifacts, _ = sandbox

    def forbidden(*a, **k):
        pytest.fail("Blind preflight called scoring")

    original = runner.reveal_metrics
    monkeypatch.setattr(runner, "reveal_metrics", forbidden)
    assert runner.main(["--preflight", *args]) == 0
    captured = capsys.readouterr().out
    assert "No accuracy metrics calculated" in captured
    assert not (output / "reveal").exists()
    manifest = json.loads((output / "preflight.json").read_text())
    assert manifest["structural_checks"]["accuracy_calculated"] is False
    assert set(manifest["artifact_sha256"]) == set(runner.ARTIFACT_NAMES)
    before = runner.file_sha256(artifacts / "predictions.parquet")
    monkeypatch.setattr(runner, "reveal_metrics", original)
    monkeypatch.setattr(runner, "structural_preflight", forbidden)
    monkeypatch.setattr(final, "train_global_model", forbidden)
    assert runner.main(["--reveal", *args]) == 0
    assert before == runner.file_sha256(artifacts / "predictions.parquet")
    assert len(pd.read_csv(output / "reveal/overall.csv")) == 4
    assert (
        json.loads((output / "reveal/manifest.json").read_text())["retrained"] is False
    )
    with pytest.raises(ValueError, match="already revealed"):
        runner.main(["--reveal", *args])


@pytest.mark.parametrize(
    "changed",
    [
        "predictions.parquet",
        "model.txt",
        "model.json",
        "lookups.parquet",
        "source",
        "input",
        "artifact_manifest",
        "report_manifest",
    ],
)
def test_reveal_refuses_changed_inputs(sandbox, changed):
    args, output, artifacts, inputs = sandbox
    runner.main(["--preflight", *args])
    if changed == "source":
        (runner.ROOT / "README.md").write_text("changed")
    elif changed == "input":
        (inputs / "calendar.csv").write_text("changed")
    elif changed == "artifact_manifest":
        (artifacts / "preflight.json").write_text("{}")
    elif changed == "report_manifest":
        manifest = json.loads((output / "preflight.json").read_text())
        manifest["protocol"]["early_stopping"] = True
        (output / "preflight.json").write_text(json.dumps(manifest))
    else:
        (artifacts / changed).write_text("changed")
    with pytest.raises(ValueError):
        runner.main(["--reveal", *args])
    assert not (output / "reveal_started.json").exists()
    assert not (output / "reveal").exists()


def test_failed_preflight_write_publishes_no_success(sandbox, monkeypatch):
    args, output, artifacts, _ = sandbox

    def fail(*a, **k):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(runner.pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk failure"):
        runner.main(["--preflight", *args])
    assert not output.exists() and not artifacts.exists()
    assert not list(artifacts.parent.glob(".pending-*"))


@pytest.fixture
def interrupted_publication(sandbox, monkeypatch):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    calculate = Mock(wraps=runner.reveal_metrics)
    monkeypatch.setattr(runner, "reveal_metrics", calculate)
    original_write = Path.write_bytes
    writes = []

    def interrupted_write(path, payload):
        if path.parent.parent == output and path.parent.name.startswith(".pending-"):
            assert (output / "score_completed.json").is_file()
            assert (artifacts / "scored_result/checksums.json").is_file()
            writes.append(path.name)
            if len(writes) == 2:
                raise OSError("synthetic score publication failure")
        return original_write(path, payload)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "write_bytes", interrupted_write)
        with pytest.raises(OSError, match="score publication failure"):
            runner.main(["--reveal", *args])
    assert len(writes) == 2
    assert calculate.call_count == 1
    assert (output / "reveal_failure.json").exists()
    assert (output / "score_completed.json").exists()
    assert not (output / "reveal").exists()
    assert not list(output.glob(".pending-*"))
    return sandbox, calculate


def test_failed_reveal_is_disclosed_and_not_retried_as_fresh(interrupted_publication):
    (args, output, _, _), calculate = interrupted_publication
    with pytest.raises(FileExistsError):
        runner.main(["--reveal", *args])
    assert calculate.call_count == 1
    assert (output / "reveal_failure.json").is_file()


def test_publication_only_recovers_exact_bytes_without_scoring_or_training(
    interrupted_publication, monkeypatch, capsys
):
    (args, output, artifacts, _), calculate = interrupted_publication
    expected = {
        name: (artifacts / "scored_result" / name).read_bytes()
        for name in runner.SCORED_REPORTS
    }
    started_hash = runner.file_sha256(output / "reveal_started.json")

    def forbidden(*a, **k):
        pytest.fail(
            "Publication-only recovery attempted scoring, prediction I/O, or training"
        )

    for module, name in (
        (runner, "reveal_metrics"),
        (final, "reveal_metrics"),
        (final, "pooled_metrics"),
        (final, "train_global_model"),
        (final, "predict_global_model"),
        (runner, "structural_preflight"),
        (runner, "load_global_model"),
        (runner.pd, "read_parquet"),
        (runner.pd, "read_csv"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    assert runner.main(["--publish-only", *args]) == 0
    assert "no accuracy recalculation" in capsys.readouterr().out
    for name, payload in expected.items():
        assert (output / "reveal" / name).read_bytes() == payload
    assert runner.file_sha256(output / "reveal_started.json") == started_hash
    assert calculate.call_count == 1
    assert (output / "reveal_failure.json").is_file()
    assert json.loads(expected["manifest.json"])["accuracy_calculations"] == 1
    with pytest.raises(ValueError, match="already published"):
        runner.main(["--publish-only", *args])
    with pytest.raises(ValueError, match="already revealed"):
        runner.main(["--reveal", *args])


@pytest.mark.parametrize("filename", [*runner.SCORED_REPORTS, "checksums.json"])
def test_recovery_rejects_modified_scored_result(interrupted_publication, filename):
    (args, output, artifacts, _), calculate = interrupted_publication
    path = artifacts / "scored_result" / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.main(["--publish-only", *args])
    assert not (output / "reveal").exists()
    assert calculate.call_count == 1


def test_recovery_rejects_replaced_score_and_updated_checksums(interrupted_publication):
    (args, output, artifacts, _), calculate = interrupted_publication
    scored = artifacts / "scored_result"
    (scored / "overall.csv").write_text("substituted result")
    checksums = json.loads((scored / "checksums.json").read_text())
    checksums["files"]["overall.csv"] = runner.file_sha256(scored / "overall.csv")
    runner.write_json(scored / "checksums.json", checksums)
    with pytest.raises(ValueError, match="checksum manifest changed"):
        runner.main(["--publish-only", *args])
    assert not (output / "reveal").exists()
    assert calculate.call_count == 1


def test_recovery_can_retry_publication_failure_without_recalculating(
    interrupted_publication, monkeypatch
):
    (args, output, _, _), calculate = interrupted_publication
    seal_hash = runner.file_sha256(output / "score_completed.json")

    def fail(*a, **k):
        raise OSError("second transient publication failure")

    with monkeypatch.context() as failure:
        failure.setattr(Path, "write_bytes", fail)
        with pytest.raises(OSError, match="second transient"):
            runner.main(["--publish-only", *args])
    assert not (output / "reveal").exists()
    assert runner.file_sha256(output / "score_completed.json") == seal_hash
    assert runner.main(["--publish-only", *args]) == 0
    assert calculate.call_count == 1


def test_publication_only_requires_complete_score(sandbox, monkeypatch):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])

    def forbidden(*a, **k):
        pytest.fail("Publication-only command attempted a new accuracy calculation")

    monkeypatch.setattr(runner, "reveal_metrics", forbidden)
    with pytest.raises(ValueError, match="No complete durable scored result"):
        runner.main(["--publish-only", *args])
    assert not (output / "reveal_started.json").exists()
    assert not (artifacts / "scored_result").exists()


def test_failed_score_persistence_cannot_trigger_rescoring(sandbox, monkeypatch):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    calculate = Mock(wraps=runner.reveal_metrics)
    monkeypatch.setattr(runner, "reveal_metrics", calculate)
    original_publish = runner.publish_directory

    def persistence_failure(destination, writer):
        if destination == artifacts / "scored_result":
            raise OSError("synthetic persistence failure")
        return original_publish(destination, writer)

    monkeypatch.setattr(runner, "publish_directory", persistence_failure)
    with pytest.raises(OSError, match="persistence failure"):
        runner.main(["--reveal", *args])
    with pytest.raises(FileExistsError):
        runner.main(["--reveal", *args])
    with pytest.raises(ValueError, match="No complete durable scored result"):
        runner.main(["--publish-only", *args])
    assert calculate.call_count == 1
    assert not (output / "score_completed.json").exists()


def test_exclusive_primary_reservation_prevents_concurrent_different_destinations(
    sandbox, monkeypatch
):
    args, output, artifacts, _ = sandbox
    barrier = Barrier(2)
    reserve = runner.reserve_primary
    fit = Mock(wraps=final.train_global_model)
    monkeypatch.setattr(final, "train_global_model", fit)

    def simultaneous_reservation(*a):
        barrier.wait(timeout=10)
        return reserve(*a)

    monkeypatch.setattr(runner, "reserve_primary", simultaneous_reservation)
    second_output = output.parent / "other-primary"
    second_artifacts = artifacts.parent / "other-primary"
    second_args = [
        str(second_output)
        if arg == str(output)
        else str(second_artifacts)
        if arg == str(artifacts)
        else arg
        for arg in args
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = [
            executor.submit(runner.main, ["--preflight", *argv])
            for argv in (args, second_args)
        ]
        outcomes = []
        for attempt in attempts:
            try:
                outcomes.append(attempt.result(timeout=20))
            except FileExistsError:
                outcomes.append("reserved")
    assert outcomes.count(0) == outcomes.count("reserved") == 1
    assert fit.call_count == 1
    assert sum(path.exists() for path in (output, second_output)) == 1
    reservation = json.loads((output.parent / "primary_reserved.json").read_text())
    assert Path(reservation["reports"]).is_dir()
    assert Path(reservation["artifacts"]).is_dir()


def test_failed_primary_cannot_be_restarted_at_a_different_destination(
    sandbox, monkeypatch
):
    args, output, artifacts, _ = sandbox

    def fail(*a, **k):
        raise OSError("synthetic preflight failure")

    monkeypatch.setattr(runner, "structural_preflight", fail)
    with pytest.raises(OSError):
        runner.main(["--preflight", *args])
    other = [
        str(output.parent / "other")
        if arg == str(output)
        else str(artifacts.parent / "other")
        if arg == str(artifacts)
        else arg
        for arg in args
    ]
    with pytest.raises(FileExistsError):
        runner.main(["--preflight", *other])


def test_blind_preflight_has_no_metric_calls_or_scored_artifacts(
    sandbox, monkeypatch, capsys
):
    from src import cycle2_metrics, evaluation, metrics

    args, output, artifacts, _ = sandbox

    def forbidden(*a, **k):
        pytest.fail("Blind preflight entered accuracy calculation or publication")

    for module, names in (
        (metrics, ["wape", "mae", "rmse"]),
        (evaluation, ["evaluate_forecasts", "wape", "mae", "rmse"]),
        (cycle2_metrics, ["aggregation_metrics"]),
        (final, ["pooled_metrics", "reveal_metrics"]),
        (runner, ["reveal_metrics", "persist_scored_result", "publish_scored_result"]),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    runner.main(["--preflight", *args])
    assert capsys.readouterr().out.strip() == (
        "Blind structural preflight passed. No accuracy metrics calculated or displayed."
    )
    manifest = json.loads((output / "preflight.json").read_text())

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not keys(manifest).intersection(
        {
            "daily_wape",
            "weekly_wape",
            "day28_wape",
            "mae",
            "rmse",
            "residual",
            "total_actual",
        }
    )
    assert not (artifacts / "scored_result").exists()
    assert not (output / "score_completed.json").exists()
    assert not list(output.glob("*.csv"))


def test_reproduction_requires_exact_artifacts_and_only_one_attempt(sandbox):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    second = [
        str(output.parent / "reproduction")
        if value == str(output)
        else str(artifacts.parent / "reproduction")
        if value == str(artifacts)
        else value
        for value in args
    ]
    runner.main(["--preflight", "--verify-against", str(output), *second])
    runner.main(["--reveal", *args])
    runner.main(["--reveal", *second])
    repro = output.parent / "reproduction"
    record = json.loads((repro / "reveal/manifest.json").read_text())
    assert "match exactly" in record["independent_reproduction"]
    third = [value.replace("/reproduction", "/third") for value in second]
    with pytest.raises(FileExistsError):
        runner.main(["--preflight", "--verify-against", str(output), *third])


@pytest.mark.parametrize(
    "relative",
    [
        "reports/first_run",
        "reports/cycle2_calendar/new",
        "reports/ml_final_evaluation/../outside",
        "reports/ml_final_evaluation",
    ],
)
def test_output_namespace_protected(sandbox, relative):
    args, _, _, _ = sandbox
    with pytest.raises(ValueError):
        runner.main(["--preflight", *args, "--output-dir", str(runner.ROOT / relative)])


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        runner.validate_destination(root / "link/run", root)


def test_failed_directory_write_is_atomic(tmp_path):
    def writer(folder):
        (folder / "partial").write_text("incomplete")
        raise OSError("disk full")

    destination = tmp_path / "run"
    with pytest.raises(OSError):
        runner.publish_directory(destination, writer)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_directory_never_overwritten(tmp_path):
    (tmp_path / "sentinel").write_text("preserved")
    with pytest.raises(ValueError):
        runner.publish_directory(tmp_path, lambda folder: None)
    assert (tmp_path / "sentinel").read_text() == "preserved"


@pytest.fixture(params=["primary", "reproduction"])
def registered_run(sandbox, request):
    args, output, artifacts, inputs = sandbox
    runner.main(["--preflight", *args])
    primary = output
    if request.param == "reproduction":
        target_output = output.parent / "reproduction"
        target_artifacts = artifacts.parent / "reproduction"
        args = [
            str(target_output)
            if arg == str(output)
            else str(target_artifacts)
            if arg == str(artifacts)
            else arg
            for arg in args
        ]
        runner.main(["--preflight", "--verify-against", str(primary), *args])
        output, artifacts = target_output, target_artifacts
    return args, output, artifacts, inputs, primary


@pytest.mark.parametrize("rewrite", ["none", "report", "report_and_artifacts"])
def test_copied_manifest_cannot_authorize_scoring(registered_run, monkeypatch, rewrite):
    _, output, artifacts, _, _ = registered_run
    alias = output.parent / f"{output.name}-backup"
    alias.mkdir()
    manifest = json.loads((output / "preflight.json").read_text())
    if rewrite != "none":
        manifest["reports_dir"] = str(alias)
    if rewrite == "report_and_artifacts":
        cloned_artifacts = artifacts.parent / f"{artifacts.name}-backup"
        shutil.copytree(artifacts, cloned_artifacts)
        manifest["artifacts_dir"] = str(cloned_artifacts)
        runner.write_json(cloned_artifacts / "preflight.json", manifest)
    runner.write_json(alias / "preflight.json", manifest)
    boundary = Mock(side_effect=AssertionError("Copied manifest reached scoring"))
    monkeypatch.setattr(runner, "reveal_metrics", boundary)
    with pytest.raises(ValueError, match="registration|Unregistered"):
        runner.main(["--reveal", "--output-dir", str(alias)])
    assert boundary.call_count == 0
    assert not (alias / "reveal_started.json").exists()
    # Copy rejection leaves the registered run valid and unconsumed.
    assert runner.verified_artifacts(output)[0] == output
    assert not (output / "reveal_started.json").exists()


def test_copied_manifest_cannot_authorize_reproduction_or_publication(
    registered_run, monkeypatch
):
    args, output, artifacts, _, _ = registered_run
    alias = output.parent / f"{output.name}-backup"
    alias.mkdir()
    shutil.copyfile(output / "preflight.json", alias / "preflight.json")
    fit = Mock(side_effect=AssertionError("Copied manifest authorized training"))
    publish = Mock(side_effect=AssertionError("Copied manifest authorized publication"))
    monkeypatch.setattr(runner, "structural_preflight", fit)
    monkeypatch.setattr(runner, "publish_scored_result", publish)
    next_args = [
        str(output.parent / "extra")
        if arg == str(output)
        else str(artifacts.parent / "extra")
        if arg == str(artifacts)
        else arg
        for arg in args
    ]
    with pytest.raises(ValueError, match="registration|Unregistered"):
        runner.main(["--preflight", "--verify-against", str(alias), *next_args])
    with pytest.raises(ValueError, match="registration|Unregistered"):
        runner.main(["--publish-only", "--output-dir", str(alias)])
    assert fit.call_count == publish.call_count == 0


@pytest.mark.parametrize("role_change", ["add_reproduction", "change_artifacts"])
def test_modified_registered_identity_is_rejected(sandbox, monkeypatch, role_change):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    manifest = json.loads((output / "preflight.json").read_text())
    if role_change == "add_reproduction":
        manifest["reproduction_of"] = str(output)
    else:
        cloned = artifacts.parent / "different-identity"
        shutil.copytree(artifacts, cloned)
        manifest["artifacts_dir"] = str(cloned)
        runner.write_json(cloned / "preflight.json", manifest)
    runner.write_json(output / "preflight.json", manifest)
    boundary = Mock(side_effect=AssertionError("Modified identity reached scoring"))
    monkeypatch.setattr(runner, "reveal_metrics", boundary)
    with pytest.raises(ValueError, match="registration|Primary run"):
        runner.main(["--reveal", *args])
    assert boundary.call_count == 0
    assert not (output / "reveal_started.json").exists()


def test_concurrent_registered_reveals_reach_scoring_once(sandbox, monkeypatch):
    args, output, _, _ = sandbox
    runner.main(["--preflight", *args])
    barrier = Barrier(2)
    verify = runner.verified_artifacts
    score = Mock(wraps=runner.reveal_metrics)

    def synchronized(*a):
        result = verify(*a)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(runner, "verified_artifacts", synchronized)
    monkeypatch.setattr(runner, "reveal_metrics", score)
    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = [executor.submit(runner.main, ["--reveal", *args]) for _ in range(2)]
        results = []
        for attempt in attempts:
            try:
                results.append(attempt.result(timeout=20))
            except FileExistsError:
                results.append("blocked")
    assert results.count(0) == results.count("blocked") == 1
    assert score.call_count == 1
    assert (output / "reveal/manifest.json").is_file()


@pytest.mark.parametrize("filename", ["predictions.parquet", "lookups.parquet"])
def test_replacement_after_path_verification_is_rejected_before_parsing(
    sandbox, monkeypatch, filename
):
    args, output, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    path = artifacts / filename
    changed = pd.read_parquet(path)
    if filename == "predictions.parquet":
        changed.loc[changed.model == runner.MODEL, "predicted"] = 12345.0
    else:
        changed["demand_group"] = "sparse"
    # Replace at the former gap: after verified_artifacts but before table loading.
    verify = runner.verified_artifacts

    def replace_after_verification(*a):
        result = verify(*a)
        changed.to_parquet(path, index=False)
        return result

    boundary = Mock(side_effect=AssertionError("Modified artifact reached scoring"))
    monkeypatch.setattr(runner, "verified_artifacts", replace_after_verification)
    monkeypatch.setattr(runner, "reveal_metrics", boundary)
    parse = Mock(wraps=runner.pd.read_parquet)
    monkeypatch.setattr(runner.pd, "read_parquet", parse)
    with pytest.raises(ValueError, match="changed before loading"):
        runner.main(["--reveal", *args])
    assert boundary.call_count == 0
    assert parse.call_count == (0 if filename == "predictions.parquet" else 1)
    assert not (output / "score_completed.json").exists()


@pytest.mark.parametrize("filename", ["predictions.parquet", "lookups.parquet"])
def test_replacement_after_snapshot_verification_cannot_change_parsed_bytes(
    sandbox, monkeypatch, filename
):
    args, _, artifacts, _ = sandbox
    runner.main(["--preflight", *args])
    path = artifacts / filename
    expected = pd.read_parquet(path)
    expected_hash = runner.file_sha256(path)
    changed = expected.copy()
    if filename == "predictions.parquet":
        changed.loc[changed.model == runner.MODEL, "predicted"] = 12345.0
    else:
        changed["demand_group"] = "sparse"
    parse = runner.pd.read_parquet
    seen = []

    def replace_source_before_parse(snapshot, *a, **k):
        assert hasattr(snapshot, "read") and hasattr(snapshot, "seek")
        assert snapshot.tell() == 0
        changed.to_parquet(path, index=False)
        seen.append(1)
        return parse(snapshot, *a, **k)

    monkeypatch.setattr(runner.pd, "read_parquet", replace_source_before_parse)
    result = runner.read_verified_parquet(path, expected_hash)
    pd.testing.assert_frame_equal(result, expected)
    assert len(seen) == 1
    assert runner.file_sha256(path) != expected_hash


def test_verified_snapshot_copies_in_bounded_chunks(tmp_path, monkeypatch):
    path = tmp_path / "large-synthetic.parquet"
    content = b"synthetic bytes" * 240000
    path.write_bytes(content)
    expected_hash = runner.file_sha256(path)
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.stream = original_open(path, "rb")
            return self

        def __exit__(self, *exc):
            self.stream.close()

        def read(self, size):
            reads.append(size)
            assert size == 1024 * 1024
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda *a, **k: BoundedReader())

    def inspect_snapshot(snapshot):
        assert snapshot.read() == content
        return "verified"

    monkeypatch.setattr(runner.pd, "read_parquet", inspect_snapshot)
    assert runner.read_verified_parquet(path, expected_hash) == "verified"
    assert len(reads) >= 4
