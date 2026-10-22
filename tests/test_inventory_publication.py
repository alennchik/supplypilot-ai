"""Tests for the one-time Phase 3 run registration and publication."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.inventory_publication import (
    CHECKSUMS,
    PUBLIC_REPORT,
    RESERVATION,
    RESULT_STORE,
    publish_sealed_results,
    read_sealed_results,
    reserve_primary_run,
    seal_results,
)

CONTENTS = {"curves.csv": b"z,fill\n0.0,0.9\n", "summary.json": b'{"verdict": "x"}\n'}


def test_a_second_reservation_is_refused(tmp_path):
    reserve_primary_run(tmp_path, {"attempt": 1})

    with pytest.raises(ValueError, match="already registered"):
        reserve_primary_run(tmp_path, {"attempt": 2})

    assert json.loads((tmp_path / RESERVATION).read_text()) == {"attempt": 1}


def test_concurrent_reservations_let_exactly_one_run_proceed(tmp_path):
    barrier = Barrier(8)

    def attempt(index):
        barrier.wait()
        try:
            reserve_primary_run(tmp_path, {"attempt": index})
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))

    assert outcomes.count(True) == 1


def test_sealed_results_are_published_unchanged_and_only_once(tmp_path):
    private, public = tmp_path / "private", tmp_path / "reports"
    reserve_primary_run(private, {"attempt": 1})
    seal_results(private, CONTENTS)

    destination = publish_sealed_results(private, public)

    assert destination == public / PUBLIC_REPORT
    for name, data in CONTENTS.items():
        assert (destination / name).read_bytes() == data
    assert (destination / CHECKSUMS).is_file()

    with pytest.raises(ValueError, match="never overwritten"):
        publish_sealed_results(private, public)


def test_results_cannot_be_sealed_twice(tmp_path):
    seal_results(tmp_path, CONTENTS)

    with pytest.raises(ValueError, match="never overwritten"):
        seal_results(tmp_path, {"curves.csv": b"different"})


def test_a_changed_sealed_file_is_never_published(tmp_path):
    private = tmp_path / "private"
    reserve_primary_run(private, {"attempt": 1})
    seal_results(private, CONTENTS)
    (private / RESULT_STORE / "curves.csv").write_bytes(b"edited")

    with pytest.raises(ValueError, match="Sealed result changed: curves.csv"):
        publish_sealed_results(private, tmp_path / "reports")

    assert not (tmp_path / "reports" / PUBLIC_REPORT).exists()


def test_an_extra_sealed_file_is_refused(tmp_path):
    seal_results(tmp_path, CONTENTS)
    (tmp_path / RESULT_STORE / "extra.csv").write_bytes(b"x")

    with pytest.raises(ValueError, match="incomplete or have extra"):
        read_sealed_results(tmp_path)


def test_nothing_is_published_without_a_registered_run(tmp_path):
    seal_results(tmp_path / "private", CONTENTS)

    with pytest.raises(ValueError, match="No registered primary run"):
        publish_sealed_results(tmp_path / "private", tmp_path / "reports")


def test_the_checksum_file_name_is_reserved(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        seal_results(tmp_path, {CHECKSUMS: b"{}"})
