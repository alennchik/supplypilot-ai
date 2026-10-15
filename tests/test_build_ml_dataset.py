"""Integration tests for the ML training dataset CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_ml_dataset import main, run_experiment


def make_inputs(tmp_path):
    days = 110
    dates = pd.date_range("2015-01-01", periods=days, freq="D")

    sales = pd.DataFrame(
        {
            "series_id": np.repeat(["one", "two"], days),
            "date": np.tile(dates, 2),
            "quantity": np.concatenate(
                [
                    np.arange(days, dtype=np.int32) % 7,
                    np.zeros(days, dtype=np.int32),
                ]
            ),
        }
    )

    metadata = pd.DataFrame(
        {
            "series_id": ["one", "two"],
            "store_id": ["S1", "S2"],
            "dept_id": ["D1", "D2"],
            "cat_id": ["FOODS", "HOBBIES"],
        }
    )

    sales_path = tmp_path / "sales.parquet"
    metadata_path = tmp_path / "series.parquet"

    sales.to_parquet(sales_path, index=False)
    metadata.to_parquet(metadata_path, index=False)

    return sales_path, metadata_path, dates


def test_real_parquet_round_trip(tmp_path):
    sales_path, metadata_path, dates = make_inputs(tmp_path)
    output = tmp_path / "training.parquet"
    manifest_path = tmp_path / "manifest.json"

    manifest = run_experiment(
        sales_path,
        metadata_path,
        output,
        manifest_path,
        cutoff=dates[-1].date().isoformat(),
        n_origins=2,
    )

    result = pd.read_parquet(output)
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest == on_disk
    assert manifest["rows"] == 2 * 2 * 28
    assert manifest["unique_series"] == 2
    assert len(manifest["origins"]) == 2
    assert len(result) == 112
    assert result["date"].max() == dates[-1]
    assert not result["target"].isna().any()
    assert output.stat().st_size > 0


def test_failure_does_not_replace_existing_files(tmp_path):
    sales_path, metadata_path, dates = make_inputs(tmp_path)

    broken = pd.read_parquet(sales_path)
    broken = broken.loc[
        ~((broken["series_id"] == "one") & (broken["date"] == dates[-2]))
    ]
    broken.to_parquet(sales_path, index=False)

    output = tmp_path / "training.parquet"
    manifest_path = tmp_path / "manifest.json"

    output.write_bytes(b"existing-training-file")
    manifest_path.write_bytes(b"existing-manifest-file")

    with pytest.raises(ValueError, match="Missing target"):
        run_experiment(
            sales_path,
            metadata_path,
            output,
            manifest_path,
            cutoff=dates[-1].date().isoformat(),
            n_origins=2,
        )

    assert output.read_bytes() == b"existing-training-file"
    assert manifest_path.read_bytes() == b"existing-manifest-file"


def test_cli_custom_paths(tmp_path, capsys):
    sales_path, metadata_path, dates = make_inputs(tmp_path)

    output = tmp_path / "custom.parquet"
    manifest_path = tmp_path / "custom.json"

    main(
        [
            "--sales",
            str(sales_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--cutoff",
            dates[-1].date().isoformat(),
            "--origins",
            "1",
        ]
    )

    assert output.exists()
    assert manifest_path.exists()
    assert "Training rows: 56" in capsys.readouterr().out
