
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pale.errors import CheckpointAlreadyExistsError
from pale.registry.registry import GCReport, Registry
from pale.registry.schema import create_tables


def _setup() -> Registry:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    return Registry(conn)


def _insert_blob(
    reg: Registry, blob_hash: str, size: int = 0, age_hours: int = 0
) -> None:
    """Insert a blob row with a created_at offset by age_hours into the past."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    reg._conn.execute(
        "INSERT OR IGNORE INTO blobs (blob_hash, size_bytes, created_at) VALUES (?, ?, ?)",
        (blob_hash, size, ts),
    )
    reg._conn.commit()


# ---------------------------------------------------------------------------
# register_checkpoint
# ---------------------------------------------------------------------------


def test_register_creates_run_if_missing():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1", "h2"])
    assert "run_a" in reg.list_runs()


def test_register_idempotent_run():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1"])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), ["h2"])
    assert reg.list_runs().count("run_a") == 1


def test_register_checkpoint_is_queryable():
    reg = _setup()
    reg.register_checkpoint("run_a", 10, Path("/m/10.json"), ["h1"])
    assert reg.checkpoint_exists("run_a", 10)
    assert reg.get_manifest_path("run_a", 10) == Path("/m/10.json")


def test_register_stores_blobs():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["ha", "hb", "hc"])
    rows = reg._conn.execute(
        "SELECT blob_hash FROM blobs ORDER BY blob_hash"
    ).fetchall()
    assert {r[0] for r in rows} == {"ha", "hb", "hc"}


def test_register_blob_dedup_across_checkpoints():
    """Shared blob hash registered in two checkpoints should appear once in blobs."""
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["shared", "h1"])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), ["shared", "h2"])
    rows = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash = 'shared'"
    ).fetchone()
    assert rows[0] == 1


def test_register_refs_recorded():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1", "h2"])
    cp_id = reg._conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE run_id='run_a' AND step=1"
    ).fetchone()[0]
    refs = reg._conn.execute(
        "SELECT blob_hash FROM refs WHERE checkpoint_id=? ORDER BY blob_hash", (cp_id,)
    ).fetchall()
    assert {r[0] for r in refs} == {"h1", "h2"}


def test_register_blob_sizes_stored():
    reg = _setup()
    reg.register_checkpoint(
        "run_a",
        1,
        Path("/m/1.json"),
        ["h1", "h2"],
        blob_sizes={"h1": 512, "h2": 1024},
    )
    rows = reg._conn.execute(
        "SELECT blob_hash, size_bytes FROM blobs ORDER BY blob_hash"
    ).fetchall()
    assert dict(rows) == {"h1": 512, "h2": 1024}


def test_register_parent_step_stored():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), [])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), [], parent_step=1)
    row = reg._conn.execute(
        "SELECT parent_step FROM checkpoints WHERE run_id='run_a' AND step=2"
    ).fetchone()
    assert row[0] == 1


def test_register_duplicate_step_raises():
    """Registering the same (run_id, step) twice must raise CheckpointNotFoundError."""
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["existing_blob"])

    with pytest.raises(CheckpointAlreadyExistsError):
        reg.register_checkpoint("run_a", 1, Path("/m/dupe.json"), ["new_blob"])


def test_register_duplicate_step_no_side_effects():
    """After the duplicate raises, the new blob must not have been committed."""
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["existing_blob"])

    with pytest.raises(CheckpointAlreadyExistsError):
        reg.register_checkpoint("run_a", 1, Path("/m/dupe.json"), ["new_blob"])

    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash='new_blob'"
    ).fetchone()
    assert row[0] == 0


# ---------------------------------------------------------------------------
# delete_checkpoint
# ---------------------------------------------------------------------------


def test_delete_removes_checkpoint():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1"])
    reg.delete_checkpoint("run_a", 1)
    assert not reg.checkpoint_exists("run_a", 1)


def test_delete_nonexistent_is_noop():
    reg = _setup()
    reg.delete_checkpoint("run_a", 999)  # should not raise


def test_delete_removes_refs_but_leaves_blobs():
    """delete_checkpoint removes refs but leaves blob rows for GC to sweep."""
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["exclusive"])
    reg.delete_checkpoint("run_a", 1)

    # Blob row must still exist (GC's responsibility to remove it)
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash='exclusive'"
    ).fetchone()
    assert row[0] == 1

    # Refs must be gone
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM refs WHERE blob_hash='exclusive'"
    ).fetchone()
    assert row[0] == 0


def test_delete_does_not_affect_shared_blob_refs():
    """Deleting one checkpoint must not remove refs belonging to another."""
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["shared", "only_in_1"])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), ["shared", "only_in_2"])

    reg.delete_checkpoint("run_a", 1)

    # shared blob still referenced by step 2
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM refs WHERE blob_hash='shared'"
    ).fetchone()
    assert row[0] == 1

    # both blobs still in blobs table (GC handles cleanup)
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash IN ('shared', 'only_in_1')"
    ).fetchone()
    assert row[0] == 2


def test_delete_leaves_other_checkpoints_intact():
    reg = _setup()
    reg.register_checkpoint("run_a", 1, Path("/m/1.json"), ["h1"])
    reg.register_checkpoint("run_a", 2, Path("/m/2.json"), ["h2"])
    reg.delete_checkpoint("run_a", 1)
    assert reg.checkpoint_exists("run_a", 2)
    assert reg.list_checkpoints("run_a") == [2]


# ---------------------------------------------------------------------------
# gc
# ---------------------------------------------------------------------------


def test_gc_no_candidates_returns_empty_report():
    reg = _setup()
    report = reg.gc(grace_period_hours=24)
    assert report.deleted_blobs == 0
    assert report.freed_bytes == 0
    assert report.swept_hashes == []


def test_gc_sweeps_old_orphaned_blob():
    reg = _setup()
    # Insert an orphaned blob older than grace period
    _insert_blob(reg, "old_orphan", size=1024, age_hours=48)

    report = reg.gc(grace_period_hours=24)

    assert report.deleted_blobs == 1
    assert report.freed_bytes == 1024
    assert "old_orphan" in report.swept_hashes

    # Row must be gone
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash='old_orphan'"
    ).fetchone()
    assert row[0] == 0


def test_gc_does_not_sweep_fresh_orphan():
    reg = _setup()
    # Blob created just now — within grace period
    _insert_blob(reg, "fresh_orphan", size=512, age_hours=0)

    report = reg.gc(grace_period_hours=24)

    assert report.deleted_blobs == 0
    row = reg._conn.execute(
        "SELECT COUNT(*) FROM blobs WHERE blob_hash='fresh_orphan'"
    ).fetchone()
    assert row[0] == 1


def test_gc_does_not_sweep_live_blob():
    """A blob with a live ref must not be swept regardless of age."""
    reg = _setup()
    reg.register_checkpoint(
        "run_a", 1, Path("/m/1.json"), ["live_blob"], blob_sizes={"live_blob": 256}
    )
    # Manually backdate the blob's created_at to look ancient
    reg._conn.execute(
        "UPDATE blobs SET created_at='2000-01-01 00:00:00' WHERE blob_hash='live_blob'"
    )
    reg._conn.commit()

    report = reg.gc(grace_period_hours=24)

    assert report.deleted_blobs == 0
    assert "live_blob" not in report.swept_hashes


def test_gc_returns_swept_hashes():
    """Swept hashes are returned so StorageEngine can delete .chunk files."""
    reg = _setup()
    _insert_blob(reg, "a", size=100, age_hours=48)
    _insert_blob(reg, "b", size=200, age_hours=48)

    report = reg.gc(grace_period_hours=24)

    assert set(report.swept_hashes) == {"a", "b"}


def test_gc_frees_orphans_after_delete_checkpoint():
    """Full flow: register → delete → gc sweeps the now-orphaned blobs."""
    reg = _setup()
    reg.register_checkpoint(
        "run_a", 1, Path("/m/1.json"), ["h1"], blob_sizes={"h1": 512}
    )
    reg.delete_checkpoint("run_a", 1)

    # Backdate blob so it falls outside grace period
    reg._conn.execute(
        "UPDATE blobs SET created_at='2000-01-01 00:00:00' WHERE blob_hash='h1'"
    )
    reg._conn.commit()

    report = reg.gc(grace_period_hours=24)

    assert report.deleted_blobs == 1
    assert "h1" in report.swept_hashes
