"""One-time registration and atomic publication for the Phase 3 run.

The real experiment runs once. A reservation is created exclusively after
all checks pass and before the first simulation; it is shared by every output
location, is kept if anything later fails, and is never overwritten. Results
are first sealed in a private store with checksums, then copied to the public
report folder. If publication fails, the stored result is published later
without recomputing anything.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

RESERVATION = "primary_reserved.json"
RESULT_STORE = "primary_result"
PUBLIC_REPORT = "primary"
CHECKSUMS = "checksums.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def reserve_primary_run(private_root, record: dict) -> Path:
    """Atomically register the one real run; refuse if one is registered."""
    root = Path(private_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / RESERVATION

    try:
        with path.open("xb") as stream:
            stream.write(json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError(
            f"A primary Phase 3 run is already registered ({path}). The experiment "
            "runs once; a further attempt after a failure must be disclosed, never "
            "silent."
        ) from exc

    _sync_directory(root)
    return path


def _publish_files(destination: Path, contents: dict[str, bytes]) -> Path:
    """Write every file to a hidden folder, then rename it into place."""
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"{destination} already exists and is never overwritten")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))

    try:
        for name, data in contents.items():
            with (temporary / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

        _sync_directory(temporary)

        if destination.exists() or destination.is_symlink():
            raise ValueError(f"{destination} appeared during writing")

        os.rename(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return destination


def seal_results(private_root, contents: dict[str, bytes]) -> Path:
    """Store results privately with a checksum file, exactly once."""
    if CHECKSUMS in contents:
        raise ValueError(f"{CHECKSUMS} is reserved")

    checksums = {name: _sha256_bytes(data) for name, data in contents.items()}
    sealed = {**contents, CHECKSUMS: json_bytes(checksums)}

    return _publish_files(Path(private_root) / RESULT_STORE, sealed)


def read_sealed_results(private_root) -> dict[str, bytes]:
    """Read the sealed results back, refusing any file that changed."""
    store = Path(private_root) / RESULT_STORE

    if not store.is_dir():
        raise ValueError(f"No sealed results at {store}")

    checksums = json.loads((store / CHECKSUMS).read_text(encoding="utf-8"))
    present = {path.name for path in store.iterdir()} - {CHECKSUMS}

    if present != set(checksums):
        raise ValueError("Sealed results are incomplete or have extra files")

    contents = {}

    for name, expected in checksums.items():
        data = (store / name).read_bytes()

        if _sha256_bytes(data) != expected:
            raise ValueError(f"Sealed result changed: {name}")

        contents[name] = data

    return {**contents, CHECKSUMS: (store / CHECKSUMS).read_bytes()}


def publish_sealed_results(private_root, report_root) -> Path:
    """Copy verified sealed results to the public report folder, once."""
    if not (Path(private_root) / RESERVATION).is_file():
        raise ValueError("No registered primary run; nothing may be published")

    contents = read_sealed_results(private_root)
    return _publish_files(Path(report_root) / PUBLIC_REPORT, contents)
