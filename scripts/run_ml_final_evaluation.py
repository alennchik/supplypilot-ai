"""Issue #28: blind preflight and separate, explicit artifact-only metrics reveal."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from scripts.run_cycle2_calendar import (
    CALENDAR_INPUT_COLUMNS,
    execution_metadata,
    file_sha256,
    validate_sample_metadata,
)
from scripts.train_ml_model import peak_rss_mib
from src.ml_final_evaluation import (
    CUTOFF,
    EXPECTED_SERIES,
    KEYS,
    MODEL,
    ORIGINS,
    align_methods,
    protocol,
    reveal_metrics,
    structural_preflight,
)
from src.ml_model import load_global_model, save_global_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = ("model.txt", "model.json", "predictions.parquet", "lookups.parquet")
METRIC_REPORTS = (
    "overall.csv",
    "by_window.csv",
    "by_category.csv",
    "by_demand_group.csv",
    "by_horizon_week.csv",
)
SCORED_REPORTS = (*METRIC_REPORTS, "manifest.json")


def source_identity():
    files = [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "config.yaml"]
    for directory in ("src", "scripts", "tests"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in files}


def frame_hash(frame, columns):
    canonical = frame[columns].sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


def validate_destination(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Use a fresh run subdirectory under {root}")
    if path.exists() or path.is_symlink():
        raise ValueError(
            "Run destination already exists; outputs are never overwritten"
        )
    return path


def publish_directory(destination, writer):
    """Publish a complete directory only after every write succeeds."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
    try:
        writer(temporary)
        # Flush the entire result before making its directory visible. In
        # particular, scoring must survive a later public-report write failure.
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Destination appeared during writing")
        os.rename(temporary, destination)
        sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def reserve_primary(reports, artifacts):
    """One durable reservation for Issue #28, independent of run destinations.

    Keep the reservation on failure: restarting a process or choosing a different
    destination must not silently start another primary experiment.
    """
    root = ROOT / "reports/ml_final_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    sync_directory(root.parent)
    write_json(
        root / "primary_reserved.json",
        {
            "issue": 28,
            "reports": str(reports),
            "artifacts": str(artifacts),
        },
        exclusive=True,
    )


def run_preflight(args):
    reports = validate_destination(
        args.output_dir, ROOT / "reports/ml_final_evaluation"
    )
    artifacts = validate_destination(
        args.artifacts_dir, ROOT / "data/processed/ml_final_evaluation"
    )
    reference = None
    if not args.verify_against and any(
        (ROOT / "reports/ml_final_evaluation").rglob("preflight.json")
    ):
        raise ValueError(
            "A primary preflight already exists; only one identical --verify-against run is permitted"
        )
    if not args.verify_against:
        reserve_primary(reports, artifacts)
    if args.verify_against:
        primary_report, _, reference = verified_artifacts(args.verify_against)
        if reference.get("reproduction_of"):
            raise ValueError("A reproduction cannot be reproduced again")
        # Exclusive reservation permits at most one independent verification attempt.
        write_json(
            primary_report / "reproduction_reserved.json",
            {
                "issue": 28,
                "reports": str(reports),
                "artifacts": str(artifacts),
                "reproduction_of": str(primary_report),
            },
            exclusive=True,
        )
    paths = {
        "sales": args.sales,
        "metadata": args.metadata,
        "calendar": args.calendar,
        "baselines": args.baselines,
        "sample_manifest": ROOT / "reports/sample_series.csv",
        "phase1_manifest": ROOT / "reports/m5_backtesting_manifest.json",
        "adoption": ROOT / "reports/cycle2_calendar/first_run/protocol_selection.json",
    }
    paths.update({p.name: p for p in (ROOT / "reports").glob("m5_baselines_*.csv")})
    paths = {name: path.resolve() for name, path in paths.items()}
    # Hash inputs before and after execution, detecting concurrent input changes.
    execution = execution_metadata(paths)
    execution["source_files_sha256"] = source_identity()
    started = time.perf_counter()
    metadata = pd.read_parquet(args.metadata)
    validate_sample_metadata(metadata, paths["sample_manifest"])
    adoption = json.loads(paths["adoption"].read_text())
    # The frozen report is an input, never rewritten.
    selected = adoption.get("selected_candidate", {})
    if (
        selected.get("candidate") != MODEL
        or selected.get("best_iteration") != 1743
        or selected.get("n_origins") != 8
        or selected.get("adopted_cycle2") is not True
    ):
        raise ValueError("Frozen adoption record does not contain the adopted model")
    sales = pd.read_parquet(
        args.sales,
        columns=["series_id", "date", "quantity"],
        filters=[("date", "<=", ORIGINS[-1] + pd.Timedelta(days=28))],
    )
    calendar = pd.read_csv(
        args.calendar, usecols=CALENDAR_INPUT_COLUMNS, keep_default_na=False
    )
    baselines = pd.read_parquet(args.baselines)
    model, forecasts, lookups, checks = structural_preflight(
        sales, metadata, calendar, baselines
    )
    if checks["series_count"] != EXPECTED_SERIES:
        raise ValueError("Wrong production cohort")
    after = execution_metadata(paths)
    if (
        execution["inputs"] != after["inputs"]
        or execution["source_files_sha256"] != source_identity()
    ):
        raise ValueError("Sources or inputs changed during preflight")
    execution.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_rss_mib": peak_rss_mib(),
            "rss_note": "Whole-process cumulative high-water mark",
        }
    )
    manifest = {
        "protocol": protocol(),
        "execution": execution,
        "structural_checks": checks,
        "cohort_sha256": frame_hash(lookups, ["series_id"]),
        "aligned_observations_sha256": frame_hash(
            forecasts.loc[forecasts.model == MODEL], [*KEYS, "actual"]
        ),
        "artifacts_dir": str(artifacts),
        "reports_dir": str(reports),
        "exceptions": [],
        "failures": [],
    }

    if reference is not None:
        for key in ("source_commit", "source_files_sha256", "dependencies", "inputs"):
            if execution[key] != reference["execution"][key]:
                raise ValueError(f"Independent reproduction execution differs: {key}")
        for key in ("protocol", "cohort_sha256", "aligned_observations_sha256"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Independent reproduction differs: {key}")
        manifest["reproduction_of"] = str(primary_report)

    def write_artifacts(folder):
        save_global_model(model, folder / "model.txt")
        loaded = load_global_model(folder / "model.txt")
        if (
            loaded.booster.num_trees() != 1743
            or loaded.booster.current_iteration() != 1743
        ):
            raise ValueError("Saved model tree count changed")
        forecasts.to_parquet(folder / "predictions.parquet", index=False)
        lookups.to_parquet(folder / "lookups.parquet", index=False)
        manifest["artifact_sha256"] = {
            name: file_sha256(folder / name) for name in ARTIFACT_NAMES
        }
        if (
            reference is not None
            and manifest["artifact_sha256"] != reference["artifact_sha256"]
        ):
            raise ValueError("Independent reproduction artifacts differ")
        write_json(folder / "preflight.json", manifest)

    publish_directory(artifacts, write_artifacts)
    # This success marker is published last. Orphaned artifacts after a failed report
    # write cannot be revealed; no incomplete report set is presented as a pass.
    publish_directory(
        reports, lambda folder: write_json(folder / "preflight.json", manifest)
    )
    print(
        "Blind structural preflight passed. No accuracy metrics calculated or displayed."
    )


def verify_run_registration(report, manifest):
    """Authorize only the canonical primary or its single registered reproduction.

    A copied preflight manifest is data, not permission to fit or score again.
    The independently created exclusive reservations own both output locations.
    """
    root = ROOT / "reports/ml_final_evaluation"
    try:
        primary = json.loads((root / "primary_reserved.json").read_text())
        if primary.get("issue") != 28:
            raise ValueError("Invalid primary run registration")
        if str(report) == primary["reports"]:
            if manifest.get("reproduction_of") is not None:
                raise ValueError("Primary run cannot be a reproduction")
            registration = primary
            expected = {
                "issue": 28,
                "reports": str(report),
                "artifacts": manifest["artifacts_dir"],
            }
        else:
            if manifest.get("reproduction_of") != primary["reports"]:
                raise ValueError("Unregistered final-evaluation report directory")
            registration = json.loads(
                (Path(primary["reports"]) / "reproduction_reserved.json").read_text()
            )
            expected = {
                "issue": 28,
                "reports": str(report),
                "artifacts": manifest["artifacts_dir"],
                "reproduction_of": primary["reports"],
            }
        if registration != expected or manifest.get("reports_dir") != str(report):
            raise ValueError("Run does not match its canonical registration")
    except FileNotFoundError as exc:
        raise ValueError("Missing final-evaluation run registration") from exc


def verified_artifacts(output_dir):
    report = Path(output_dir).resolve()
    root = (ROOT / "reports/ml_final_evaluation").resolve()
    if not report.is_relative_to(root) or report == root:
        raise ValueError(
            "Preflight reports must be in the isolated final-evaluation namespace"
        )
    manifest = json.loads((report / "preflight.json").read_text())
    verify_run_registration(report, manifest)
    if (
        manifest["protocol"] != protocol()
        or manifest["structural_checks"]["status"]
        != "passed_blind_structural_preflight"
        or manifest["structural_checks"]["accuracy_calculated"] is not False
        or manifest["structural_checks"]["realized_trees"] != 1743
        or manifest["structural_checks"]["training_label_max"] != str(CUTOFF.date())
        or manifest["structural_checks"]["series_count"] != EXPECTED_SERIES
    ):
        raise ValueError("Missing or incompatible frozen preflight")
    if manifest["execution"]["source_files_sha256"] != source_identity():
        raise ValueError("Sources changed since blind preflight; reveal refused")
    for record in manifest["execution"]["inputs"].values():
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Input changed since blind preflight")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if revision != manifest["execution"]["source_commit"]:
        raise ValueError("Code revision changed since blind preflight")
    for name, version in manifest["execution"]["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Dependency changed since blind preflight")
    artifacts = Path(manifest["artifacts_dir"]).resolve()
    if str(artifacts) != manifest["artifacts_dir"]:
        raise ValueError("Artifact directory no longer matches its canonical identity")
    if not artifacts.is_relative_to(
        (ROOT / "data/processed/ml_final_evaluation").resolve()
    ):
        raise ValueError("Artifacts outside isolated data namespace")
    if json.loads((artifacts / "preflight.json").read_text()) != manifest:
        raise ValueError("Preflight manifests differ")
    if set(manifest["artifact_sha256"]) != set(ARTIFACT_NAMES):
        raise ValueError("Incomplete artifact inventory")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(artifacts / name) != digest:
            raise ValueError("Preflight artifact changed")
    return report, artifacts, manifest


def read_verified_parquet(path, expected_sha256):
    """Parse the exact verified snapshot, copying in bounded 1 MiB chunks.

    Hash while copying to an anonymous temporary file, then parse that same
    private handle. Replacing or editing the source path cannot alter the
    verified snapshot. Process artifacts sequentially; never buffer both files.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source, tempfile.TemporaryFile() as snapshot:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            if snapshot.write(block) != len(block):
                raise OSError("Incomplete verified artifact snapshot")
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Preflight artifact changed before loading")
        snapshot.seek(0)
        return pd.read_parquet(snapshot)


def persist_scored_result(report, artifacts, manifest, tables, details):
    """Seal all scored tables in ignored storage before publishing any metrics."""
    if {f"{name}.csv" for name in tables} != set(METRIC_REPORTS):
        raise ValueError("Incomplete scored result")
    preflight_hash = file_sha256(report / "preflight.json")
    prediction_hash = manifest["artifact_sha256"]["predictions.parquet"]
    scored = artifacts / "scored_result"

    def writer(folder):
        for name, table in tables.items():
            table.to_csv(folder / f"{name}.csv", index=False)
        write_json(
            folder / "manifest.json",
            {
                "protocol": protocol(),
                "preflight_sha256": preflight_hash,
                "prediction_sha256": prediction_hash,
                "retrained": False,
                "accuracy_calculations": 1,
                **details,
            },
        )
        write_json(
            folder / "checksums.json",
            {
                "preflight_sha256": preflight_hash,
                "prediction_sha256": prediction_hash,
                "files": {name: file_sha256(folder / name) for name in SCORED_REPORTS},
            },
        )

    publish_directory(scored, writer)
    # A separate durable anchor prevents a damaged result/checksum pair from
    # being silently substituted during publication-only recovery.
    write_json(
        report / "score_completed.json",
        {
            "preflight_sha256": preflight_hash,
            "reveal_started_sha256": file_sha256(report / "reveal_started.json"),
            "checksums_sha256": file_sha256(scored / "checksums.json"),
        },
        exclusive=True,
    )


def verified_scored_result(report, artifacts, manifest):
    """Read and authenticate report bytes only; never load predictions or score."""
    completion_path = report / "score_completed.json"
    if not completion_path.is_file():
        raise ValueError(
            "No complete durable scored result; accuracy will not be recalculated"
        )
    completion = json.loads(completion_path.read_text())
    preflight_hash = file_sha256(report / "preflight.json")
    if completion["preflight_sha256"] != preflight_hash or completion[
        "reveal_started_sha256"
    ] != file_sha256(report / "reveal_started.json"):
        raise ValueError("Scored result is bound to a different reveal")
    scored = artifacts / "scored_result"
    checksum_bytes = (scored / "checksums.json").read_bytes()
    if hashlib.sha256(checksum_bytes).hexdigest() != completion["checksums_sha256"]:
        raise ValueError("Scored-result checksum manifest changed")
    checksums = json.loads(checksum_bytes)
    if (
        checksums["preflight_sha256"] != preflight_hash
        or checksums["prediction_sha256"]
        != manifest["artifact_sha256"]["predictions.parquet"]
        or set(checksums["files"]) != set(SCORED_REPORTS)
    ):
        raise ValueError("Incompatible scored-result inventory")
    payloads = {name: (scored / name).read_bytes() for name in SCORED_REPORTS}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != checksums["files"][name]:
            raise ValueError("Persisted scored result changed")
    details = json.loads(payloads["manifest.json"])
    if (
        details["protocol"] != protocol()
        or details["preflight_sha256"] != preflight_hash
        or details["prediction_sha256"] != checksums["prediction_sha256"]
        or details["accuracy_calculations"] != 1
        or details["retrained"] is not False
    ):
        raise ValueError("Incompatible scored-result metadata")
    return payloads


def publish_scored_result(report, artifacts, manifest):
    destination = report / "reveal"
    if destination.exists() or destination.is_symlink():
        raise ValueError("Metrics were already published; refusing another publication")
    payloads = verified_scored_result(report, artifacts, manifest)

    def writer(folder):
        for name, payload in payloads.items():
            (folder / name).write_bytes(payload)

    publish_directory(destination, writer)


def run_publication_only(args):
    """Recover sealed report bytes without scoring or loading prediction tables."""
    report, artifacts, manifest = verified_artifacts(args.output_dir)
    publish_scored_result(report, artifacts, manifest)
    print(
        f"Published persisted final reports under {report / 'reveal'}; no accuracy recalculation or retraining."
    )


def run_reveal(args):
    report, artifacts, manifest = verified_artifacts(args.output_dir)
    destination = report / "reveal"
    if destination.exists():
        raise ValueError("Metrics were already revealed; refusing another reveal")
    primary_tables = None
    if manifest.get("reproduction_of"):
        primary = Path(manifest["reproduction_of"]) / "reveal"
        if not (primary / "manifest.json").is_file():
            raise ValueError("Reveal the primary run before revealing its reproduction")
        primary_tables = primary
    # A durable exclusive marker prevents a failed metric/report write from being
    # represented as a fresh, untouched reveal. Preserve and disclose any failure.
    write_json(
        report / "reveal_started.json",
        {
            "preflight_sha256": file_sha256(report / "preflight.json"),
        },
        exclusive=True,
    )
    try:
        forecasts = read_verified_parquet(
            artifacts / "predictions.parquet",
            manifest["artifact_sha256"]["predictions.parquet"],
        )
        lookups = read_verified_parquet(
            artifacts / "lookups.parquet",
            manifest["artifact_sha256"]["lookups.parquet"],
        )
        align_methods(forecasts, lookups.series_id)
        if (
            frame_hash(lookups, ["series_id"]) != manifest["cohort_sha256"]
            or frame_hash(forecasts.loc[forecasts.model == MODEL], [*KEYS, "actual"])
            != manifest["aligned_observations_sha256"]
        ):
            raise ValueError("Cohort or actual alignment changed")
        started = time.perf_counter()
        tables = reveal_metrics(forecasts, lookups)
        if primary_tables is not None:
            for name, table in tables.items():
                # Canonical CSV parsing gives identical date types on both sides.
                import io

                current = pd.read_csv(io.StringIO(table.to_csv(index=False)))
                previous = pd.read_csv(primary_tables / f"{name}.csv")
                pd.testing.assert_frame_equal(current, previous, check_exact=True)

        persist_scored_result(
            report,
            artifacts,
            manifest,
            tables,
            {
                "runtime_seconds": time.perf_counter() - started,
                "peak_process_rss_mib": peak_rss_mib(),
                "exceptions": [],
                "independent_reproduction": (
                    "all artifacts and metric tables match exactly"
                    if primary_tables is not None
                    else "not performed"
                ),
            },
        )
        publish_scored_result(report, artifacts, manifest)
    except Exception as exc:
        # A full disk must not mask the original exception or clear the durable
        # start marker. Recovery remains possible only from a complete seal.
        with contextlib.suppress(OSError):
            write_json(
                report / "reveal_failure.json",
                {
                    "exception_type": type(exc).__name__,
                    "disclosure": "Reveal attempted; do not describe a correction as an untouched holdout.",
                },
            )
        raise
    print(
        f"Final metrics saved under {destination}; existing predictions reused, no retraining."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--preflight",
        action="store_true",
        help="Explicitly train and run blind structural checks",
    )
    action.add_argument(
        "--reveal",
        action="store_true",
        help="Explicitly score existing preflight predictions once",
    )
    action.add_argument(
        "--publish-only",
        action="store_true",
        help="Recover report publication from a complete durable score; never recalculate accuracy",
    )
    parser.add_argument(
        "--sales", type=Path, default=Path("data/processed/sales.parquet")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/processed/series.parquet")
    )
    parser.add_argument("--calendar", type=Path, default=Path("data/raw/calendar.csv"))
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("data/processed/m5_backtest_predictions.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/ml_final_evaluation/primary")
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/processed/ml_final_evaluation/primary"),
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="One independent identical preflight against an existing primary run",
    )
    args = parser.parse_args(argv)
    if args.verify_against and not args.preflight:
        parser.error("--verify-against requires --preflight")
    if args.preflight:
        run_preflight(args)
    elif args.reveal:
        run_reveal(args)
    elif args.publish_only:
        run_publication_only(args)
    else:
        print(json.dumps(protocol(), indent=2))
        print(
            "DRY RUN: no data read, fitting, accuracy calculation, or artifact writes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
