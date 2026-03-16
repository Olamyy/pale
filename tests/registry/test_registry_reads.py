import sqlite3
from datetime import datetime, timezone

from pale.registry.registry import Registry
from pale.registry.schema import create_tables


def _setup() -> Registry:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    return Registry(conn)


def _insert_checkpoint(reg, run_id, step, manifest_path="/tmp/m.json"):
    now = datetime.now(timezone.utc).isoformat()
    reg._conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, created_at) VALUES (?, ?)", (run_id, now)
    )
    reg._conn.execute(
        "INSERT INTO checkpoints (checkpoint_id, run_id, step, manifest_path, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"{run_id}:{step}", run_id, step, manifest_path, now),
    )
    reg._conn.commit()


def test_list_checkpoints_sorted():
    reg = _setup()
    for step in [30, 10, 20]:
        _insert_checkpoint(reg, "run_x", step)
    assert reg.list_checkpoints("run_x") == [10, 20, 30]
