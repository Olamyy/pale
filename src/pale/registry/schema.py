import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id   TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    step            INTEGER NOT NULL,
    parent_step     INTEGER,
    manifest_path   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (run_id, step)
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
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables. Idempotent — safe to call multiple times."""
    conn.executescript(DDL)
    conn.commit()


def open_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the registry SQLite database at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    return conn
