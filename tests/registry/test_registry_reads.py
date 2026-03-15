
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pale.registry.schema import create_tables
from pale.registry.registry import Registry


def _setup() -> Registry:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    return Registry(conn)


def _insert_checkpoint(
    reg: Registry, run_id: str, step: int, manifest_path: str = "/tmp/m.json"
) -> None:
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


def test_list_runs_empty():
    reg = _setup()
    assert reg.list_runs() == []


def test_list_runs_populated():
    reg = _setup()
    _insert_checkpoint(reg, "run_a", 1)
    _insert_checkpoint(reg, "run_b", 1)
    assert reg.list_runs() == ["run_a", "run_b"]


def test_list_checkpoints_empty():
    reg = _setup()
    assert reg.list_checkpoints("unknown_run") == []


def test_list_checkpoints_populated():
    reg = _setup()
    for step in [30, 10, 20]:
        _insert_checkpoint(reg, "run_x", step)
    assert reg.list_checkpoints("run_x") == [10, 20, 30]


def test_list_checkpoints_unknown_run_returns_empty():
    reg = _setup()
    _insert_checkpoint(reg, "run_x", 1)
    assert reg.list_checkpoints("run_y") == []


def test_get_manifest_path_exists():
    reg = _setup()
    _insert_checkpoint(reg, "run_x", 5, "/some/path/step_5.json")
    result = reg.get_manifest_path("run_x", 5)
    assert result == Path("/some/path/step_5.json")


def test_get_manifest_path_missing():
    reg = _setup()
    assert reg.get_manifest_path("run_x", 999) is None


def test_checkpoint_exists_true():
    reg = _setup()
    _insert_checkpoint(reg, "run_x", 10)
    assert reg.checkpoint_exists("run_x", 10) is True


def test_checkpoint_exists_false():
    reg = _setup()
    assert reg.checkpoint_exists("run_x", 10) is False
