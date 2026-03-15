import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from pale.errors import CheckpointAlreadyExistsError


class GCReport(NamedTuple):
    deleted_blobs: int
    freed_bytes: int
    swept_hashes: List[str]


def _now_utc() -> str:
    """Return current UTC time as SQLite-compatible string (YYYY-MM-DD HH:MM:SS)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Registry:
    """Checkpoint and run metadata backed by SQLite.

    All writes are transactional. The connection is assumed to be opened with
    isolation_level=None (autocommit off) or managed by the caller — use
    Registry.from_path() for the standard setup.

    Blob rows are never deleted by delete_checkpoint(). Orphaned blobs
    (no live refs) are left in the blobs table for gc() to sweep after
    the grace period. This ensures .chunk files are never silently leaked.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def from_path(cls, db_path: Path) -> "Registry":
        """Open (or create) a SQLite registry at db_path and return a Registry."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        registry = cls(conn)
        registry._create_tables()
        return registry

    def _create_tables(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id   TEXT PRIMARY KEY,
                    run_id          TEXT NOT NULL REFERENCES runs(run_id),
                    step            INTEGER NOT NULL,
                    parent_step     INTEGER,
                    manifest_path   TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    UNIQUE(run_id, step)
                );

                CREATE TABLE IF NOT EXISTS blobs (
                    blob_hash   TEXT PRIMARY KEY,
                    size_bytes  INTEGER NOT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS refs (
                    checkpoint_id   TEXT NOT NULL REFERENCES checkpoints(checkpoint_id),
                    blob_hash       TEXT NOT NULL REFERENCES blobs(blob_hash),
                    PRIMARY KEY (checkpoint_id, blob_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step
                    ON checkpoints(run_id, step);

                CREATE INDEX IF NOT EXISTS idx_refs_blob_hash
                    ON refs(blob_hash);
            """)

    def list_runs(self) -> List[str]:
        """Return all run_ids in ascending order."""
        rows = self._conn.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
        return [r[0] for r in rows]

    def list_checkpoints(self, run_id: str) -> List[int]:
        """Return all steps for run_id in ascending order.

        Returns an empty list if run_id is unknown — does not raise.
        """
        rows = self._conn.execute(
            "SELECT step FROM checkpoints WHERE run_id = ? ORDER BY step",
            (run_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_manifest_path(self, run_id: str, step: int) -> Optional[Path]:
        """Return manifest path for (run_id, step), or None if not found."""
        row = self._conn.execute(
            "SELECT manifest_path FROM checkpoints WHERE run_id = ? AND step = ?",
            (run_id, step),
        ).fetchone()
        return Path(row[0]) if row else None

    def checkpoint_exists(self, run_id: str, step: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM checkpoints WHERE run_id = ? AND step = ?",
            (run_id, step),
        ).fetchone()
        return row is not None

    def register_checkpoint(
        self,
        run_id: str,
        step: int,
        manifest_path: Path,
        blob_hashes: List[str],
        blob_sizes: Optional[Dict[str, int]] = None,
        parent_step: Optional[int] = None,
    ) -> None:
        """Register a checkpoint and its blob references transactionally.

        Creates the run row if it doesn't exist. Inserts blob rows for any
        hashes not already in the blobs table (idempotent on blob_hash).
        Inserts the checkpoint and all ref rows atomically.

        Raises:
            CheckpointAlreadyExistsError: if (run_id, step) already exists.
                Callers must explicitly delete before re-registering.

        Args:
            run_id: Identifier for the training run.
            step: Training step / epoch number for this checkpoint.
            manifest_path: Absolute path to the manifest JSON file.
            blob_hashes: All chunk hashes referenced by this checkpoint.
            blob_sizes: Optional hash → size_bytes mapping. Blobs with no
                entry are stored with size_bytes=0 (GC uses size for
                reporting only — correctness is not affected).
            parent_step: Step of the preceding checkpoint, if any.
                Used for lineage traversal. None for the first checkpoint.
        """
        if self.checkpoint_exists(run_id, step):
            raise CheckpointAlreadyExistsError(
                f"Checkpoint already exists for run_id={run_id!r}, step={step}. "
                "Call delete_checkpoint() before re-registering."
            )

        now = _now_utc()
        checkpoint_id = str(uuid.uuid4())
        sizes = blob_sizes or {}

        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, created_at) VALUES (?, ?)",
                (run_id, now),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO blobs (blob_hash, size_bytes, created_at) "
                "VALUES (?, ?, ?)",
                [(h, sizes.get(h, 0), now) for h in blob_hashes],
            )
            self._conn.execute(
                "INSERT INTO checkpoints "
                "(checkpoint_id, run_id, step, parent_step, manifest_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (checkpoint_id, run_id, step, parent_step, str(manifest_path), now),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO refs (checkpoint_id, blob_hash) VALUES (?, ?)",
                [(checkpoint_id, h) for h in blob_hashes],
            )

    def delete_checkpoint(self, run_id: str, step: int) -> None:
        """Delete a checkpoint row and its ref rows.

        Blob rows are intentionally left intact. Orphaned blobs (those with
        no remaining refs) will be swept by gc() after the grace period.
        This ensures .chunk files are never leaked — gc() is the sole owner
        of blob row deletion and physical file removal.

        Does nothing if the checkpoint does not exist.
        """
        with self._conn:
            row = self._conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE run_id = ? AND step = ?",
                (run_id, step),
            ).fetchone()
            if row is None:
                return

            checkpoint_id = row[0]
            self._conn.execute(
                "DELETE FROM refs WHERE checkpoint_id = ?", (checkpoint_id,)
            )
            self._conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            )

    def gc(self, grace_period_hours: int = 24) -> GCReport:
        """Mark-and-sweep GC over orphaned blobs.

        Mark: blobs with no live refs whose created_at is older than
        grace_period_hours. The grace period protects blobs written by
        in-flight saves that haven't been registered yet.

        Sweep: deletes matching blob rows from the DB and returns their
        hashes in GCReport.swept_hashes. The caller (StorageEngine) is
        responsible for deleting the corresponding .chunk files —
        Registry does not know filesystem paths for chunk files.
        """
        from datetime import timedelta

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=grace_period_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with self._conn:
            candidates = self._conn.execute(
                """
                SELECT blob_hash, size_bytes FROM blobs
                WHERE created_at < ?
                  AND blob_hash NOT IN (SELECT blob_hash FROM refs)
                """,
                (cutoff,),
            ).fetchall()

            if not candidates:
                return GCReport(deleted_blobs=0, freed_bytes=0, swept_hashes=[])

            hashes = [r[0] for r in candidates]
            freed = sum(r[1] for r in candidates)

            self._conn.executemany(
                "DELETE FROM blobs WHERE blob_hash = ?",
                [(h,) for h in hashes],
            )

        return GCReport(
            deleted_blobs=len(hashes), freed_bytes=freed, swept_hashes=hashes
        )
