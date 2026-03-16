import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pale.cas.engine import CASEngine
from pale.errors import CheckpointAlreadyExistsError, CheckpointNotFoundError, StorageError
from pale.registry.registry import Registry
from pale.registry.schema import create_tables
from pale.storage import StorageEngine


class DictBackend:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def has(self, h: str) -> bool:
        return h in self._store

    def put(self, h: str, data: bytes) -> None:
        self._store[h] = data

    def get(self, h: str) -> bytes:
        return self._store[h]

    def batch_has(self, hashes):
        return {h: h in self._store for h in hashes}

    def delete(self, h: str) -> None:
        del self._store[h]


def make_engine(tmp_path: Path):
    backend = DictBackend()
    cas = CASEngine(backend, chunk_size=256)
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    reg = Registry(conn)
    storage = StorageEngine(cas, reg, tmp_path / "manifests")
    return storage, cas, backend, reg


def test_save_writes_manifest_file(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})
    assert (tmp_path / "manifests" / "run_a" / "step_000001.json").exists()


def test_save_cas_failure_raises_storage_error(tmp_path):
    storage, cas, _, _ = make_engine(tmp_path)
    with patch.object(cas, "store_tensor", side_effect=RuntimeError("backend unavailable")):
        with pytest.raises(StorageError, match="CAS store failed"):
            storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})


def test_save_registry_failure_removes_manifest(tmp_path):
    storage, _, _, reg = make_engine(tmp_path)
    with patch.object(reg, "register_checkpoint", side_effect=Exception("db down")):
        with pytest.raises(StorageError, match="Registry registration failed"):
            storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})
    assert not (tmp_path / "manifests" / "run_a" / "step_000001.json").exists()


def test_save_duplicate_step_raises(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})
    with pytest.raises(CheckpointAlreadyExistsError):
        storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})


def test_load_nonexistent_step_raises(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    with pytest.raises(CheckpointNotFoundError):
        storage.load("run_a", 999)


def test_load_after_noop_save_returns_equal_arrays(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    arr = np.ones(20, dtype=np.float32)
    storage.save("run_a", 1, {"w": arr})
    storage.save("run_a", 2, {"w": arr})
    assert np.array_equal(arr, storage.load("run_a", 2)["w"])
