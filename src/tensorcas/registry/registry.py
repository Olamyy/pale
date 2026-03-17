import sqlite3
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from tensorcas.errors import CheckpointAlreadyExistsError, CorruptManifestError


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
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        registry = cls(conn)
        registry._create_tables()
        return registry

    def _create_tables(self) -> None:
        """Create tables if they don't exist. Idempotent.

        The refs table has been removed. Blob liveness is now determined at
        GC time by scanning manifest files, not by a per-chunk join table.
        This makes register_checkpoint O(new blobs) instead of O(total blobs).
        """
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

                DROP TABLE IF EXISTS refs;

                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step
                    ON checkpoints(run_id, step);
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
        new_blob_hashes: Optional[List[str]] = None,
        blob_sizes: Optional[Dict[str, int]] = None,
        parent_step: Optional[int] = None,
    ) -> None:
        """Register a checkpoint and record any newly written blobs.

        Creates the run row if it doesn't exist. Inserts blob rows only for
        newly written chunks (new_blob_hashes). Blob liveness for GC is
        determined at gc() time by scanning manifest files — there is no
        per-chunk refs table.

        Raises:
            CheckpointAlreadyExistsError: if (run_id, step) already exists.
                Callers must explicitly delete before re-registering.

        Args:
            run_id: Identifier for the training run.
            step: Training step / epoch number for this checkpoint.
            manifest_path: Absolute path to the manifest JSON file.
            blob_hashes: Ignored (kept for call-site compatibility).
            new_blob_hashes: Blobs written this step. Only these are inserted
                into the blobs table. If None, falls back to blob_hashes.
            blob_sizes: hash → size_bytes mapping for new blobs.
            parent_step: Step of the preceding checkpoint, if any.
        """
        if self.checkpoint_exists(run_id, step):
            raise CheckpointAlreadyExistsError(
                f"Checkpoint already exists for run_id={run_id!r}, step={step}. "
                "Call delete_checkpoint() before re-registering."
            )

        now = _now_utc()
        checkpoint_id = str(uuid.uuid4())
        sizes = blob_sizes or {}
        blobs_to_insert = new_blob_hashes if new_blob_hashes is not None else blob_hashes

        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, created_at) VALUES (?, ?)",
                (run_id, now),
            )
            if blobs_to_insert:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO blobs (blob_hash, size_bytes, created_at) "
                    "VALUES (?, ?, ?)",
                    [(h, sizes.get(h, 0), now) for h in blobs_to_insert],
                )
            self._conn.execute(
                "INSERT INTO checkpoints "
                "(checkpoint_id, run_id, step, parent_step, manifest_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (checkpoint_id, run_id, step, parent_step, str(manifest_path), now),
            )

    def delete_checkpoint(self, run_id: str, step: int) -> None:
        """Delete a checkpoint row.

        Blob rows are intentionally left intact. Orphaned blobs are swept by
        gc(), which determines liveness by scanning manifest files.

        Does nothing if the checkpoint does not exist.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM checkpoints WHERE run_id = ? AND step = ?",
                (run_id, step),
            )

    def gc(self, grace_period_hours: int = 24) -> GCReport:
        """Mark-and-sweep GC over orphaned blobs.

        Mark: blobs in the blobs table whose created_at is older than
        grace_period_hours AND whose hash does not appear in any live
        manifest file. The grace period protects blobs written by in-flight
        saves not yet registered.

        Liveness is determined by scanning all manifest files registered in
        the checkpoints table — there is no refs join table. Corrupt or
        missing manifest files are skipped with a warning (their blobs are
        treated as live to avoid accidental deletion).

        Sweep: deletes matching blob rows from the DB and returns their
        hashes in GCReport.swept_hashes. The caller is responsible for
        deleting the corresponding .chunk files.
        """
        from tensorcas.manifest import ManifestReader

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=grace_period_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Collect all candidate blobs (old enough to consider sweeping).
        candidates = self._conn.execute(
            "SELECT blob_hash, size_bytes FROM blobs WHERE created_at < ?",
            (cutoff,),
        ).fetchall()

        if not candidates:
            return GCReport(deleted_blobs=0, freed_bytes=0, swept_hashes=[])

        candidate_set = {r[0]: r[1] for r in candidates}

        # Build live set from all registered manifest files.
        live: set[str] = set()
        manifest_paths = self._conn.execute(
            "SELECT manifest_path FROM checkpoints"
        ).fetchall()
        for (path_str,) in manifest_paths:
            path = Path(path_str)
            try:
                manifest = ManifestReader.read(path)
                for record in manifest.tensors.values():
                    for ref in record.chunks:
                        live.add(ref.hash)
            except CorruptManifestError:
                warnings.warn(
                    f"GC: skipping corrupt/missing manifest {path} — "
                    "its blobs are treated as live.",
                    stacklevel=2,
                )

        orphans = [h for h in candidate_set if h not in live]
        if not orphans:
            return GCReport(deleted_blobs=0, freed_bytes=0, swept_hashes=[])

        freed = sum(candidate_set[h] for h in orphans)
        with self._conn:
            self._conn.executemany(
                "DELETE FROM blobs WHERE blob_hash = ?",
                [(h,) for h in orphans],
            )

        return GCReport(
            deleted_blobs=len(orphans), freed_bytes=freed, swept_hashes=orphans
        )
