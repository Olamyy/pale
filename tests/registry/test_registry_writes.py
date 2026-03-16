import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pale.errors import CheckpointAlreadyExistsError
from pale.manifest import CheckpointManifest, ManifestWriter
from pale.models import TensorArrayRecord, ChunkRef
from pale.registry.registry import Registry
from pale.registry.schema import create_tables


def _setup() -> Registry:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    return Registry(conn)


def _insert_blob(
    reg: Registry, blob_hash: str, size: int = 0, age_hours: int = 0
) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    reg._conn.execute(
        "INSERT OR IGNORE INTO blobs (blob_hash, size_bytes, created_at) VALUES (?, ?, ?)",
        (blob_hash, size, ts),
    )
    reg._conn.commit()


def _write_manifest(path: Path, blob_hashes: list[str]) -> None:
    """Write a minimal manifest referencing the given blob hashes."""
    tensors = {}
    for i, h in enumerate(blob_hashes):
        tensors[f"t{i}"] = TensorArrayRecord(
            name=f"t{i}",
            shape=[1],
            dtype="float32",
            byte_length=4,
            full_hash=h,
            chunks=[ChunkRef(hash=h, size=4)],
        )
    manifest = CheckpointManifest(
        chunk_size=262144,
        run_id="run_a",
        step=1,
        created_at=datetime.now(timezone.utc),
        tensors=tensors,
    )
    ManifestWriter.write(manifest, path)


def test_register_creates_run_if_missing():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1", "h2"])
    assert "run_a" in reg.list_runs()


def test_register_checkpoint_is_queryable():
    reg = _setup()
    reg.register_checkpoint("run_a", 10, Path("/m/10.json"), ["h1"])
    assert reg.checkpoint_exists("run_a", 10)
    assert reg.get_manifest_path("run_a", 10) == Path("/m/10.json")


def test_register_blob_dedup_across_checkpoints():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), [], new_blob_hashes=["shared", "h1"])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), [], new_blob_hashes=["shared", "h2"])
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash = 'shared'"
    ).fetchone()
    assert row[0] == 1


def test_register_parent_step_stored():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), [])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), [], parent_step=1)
    row = reg._conn.execute(
        "SELECT parent_step FROM checkpoints WHERE run_id='run_a' AND step=2"
    ).fetchone()
    assert row[0] == 1


def test_register_duplicate_step_raises():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["existing_blob"])
    with pytest.raises(CheckpointAlreadyExistsError):
        reg.register_checkpoint("run_a", 1, Path("/m/dupe.json"), ["new_blob"])


def test_register_duplicate_step_no_side_effects():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), [], new_blob_hashes=["existing_blob"])
    with pytest.raises(CheckpointAlreadyExistsError):
        reg.register_checkpoint("run_a", 1, Path("/m/dupe.json"), [], new_blob_hashes=["new_blob"])
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash='new_blob'"
    ).fetchone()
    assert row[0] == 0


def test_delete_removes_checkpoint():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1"])
    reg.delete_checkpoint("run_a", 1)
    assert not reg.checkpoint_exists("run_a", 1)


def test_delete_nonexistent_is_noop():
    reg = _setup()
    reg.delete_checkpoint("run_a", 999)


def test_delete_leaves_blobs():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), [], new_blob_hashes=["exclusive"])
    reg.delete_checkpoint("run_a", 1)
    assert (
        reg._conn.execute(
            "SELECT COUNT(*) FROM blobs WHERE blob_hash='exclusive'"
        ).fetchone()[0]
        == 1
    )


def test_gc_sweeps_old_orphaned_blob():
    reg = _setup()
    _insert_blob(reg, "old_orphan", size=1024, age_hours=48)
    report = reg.gc(grace_period_hours=24)
    assert report.deleted_blobs == 1
    assert report.freed_bytes == 1024
    assert "old_orphan" in report.swept_hashes
    assert (
        reg._conn.execute(
            "SELECT COUNT(*) FROM blobs WHERE blob_hash='old_orphan'"
        ).fetchone()[0]
        == 0
    )


def test_gc_does_not_sweep_fresh_orphan():
    reg = _setup()
    _insert_blob(reg, "fresh_orphan", size=512, age_hours=0)
    report = reg.gc(grace_period_hours=24)
    assert report.deleted_blobs == 0


def test_gc_does_not_sweep_live_blob():
    reg = _setup()
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "run_a" / "step_000001.json"
        _write_manifest(manifest_path, ["live_blob"])
        reg.register_checkpoint(
            "run_a", 1, manifest_path, [], new_blob_hashes=["live_blob"],
            blob_sizes={"live_blob": 256},
        )
        reg._conn.execute(
            "UPDATE blobs SET created_at='2000-01-01 00:00:00' WHERE blob_hash='live_blob'"
        )
        reg._conn.commit()
        report = reg.gc(grace_period_hours=24)
    assert report.deleted_blobs == 0


def test_gc_frees_orphans_after_delete_checkpoint():
    reg = _setup()
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "run_a" / "step_000001.json"
        _write_manifest(manifest_path, ["h1"])
        reg.register_checkpoint(
            "run_a", 1, manifest_path, [], new_blob_hashes=["h1"],
            blob_sizes={"h1": 512},
        )
        reg.delete_checkpoint("run_a", 1)
        reg._conn.execute(
            "UPDATE blobs SET created_at='2000-01-01 00:00:00' WHERE blob_hash='h1'"
        )
        reg._conn.commit()
        report = reg.gc(grace_period_hours=24)
    assert report.deleted_blobs == 1
    assert "h1" in report.swept_hashes
