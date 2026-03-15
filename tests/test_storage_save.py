import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pale.cas.engine import CASEngine
from pale.errors import (
    CheckpointAlreadyExistsError,
    CheckpointNotFoundError,
    CorruptChunkError,
    StorageError,
)
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
    manifest_dir = tmp_path / "manifests"
    storage = StorageEngine(cas, reg, manifest_dir)
    return storage, cas, backend, reg


def test_save_returns_manifest(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    tensors = {"w": np.ones(10, dtype=np.float32)}
    manifest = storage.save("run_a", 1, tensors)
    assert manifest.run_id == "run_a"
    assert manifest.step == 1
    assert "w" in manifest.tensors


def test_save_writes_manifest_file(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})
    manifest_path = tmp_path / "manifests" / "run_a" / "step_000001.json"
    assert manifest_path.exists()


def test_save_registers_checkpoint(tmp_path):
    storage, _, _, reg = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.zeros(4, dtype=np.float64)})
    assert reg.checkpoint_exists("run_a", 1)


def test_save_stores_blobs_in_registry(tmp_path):
    storage, _, _, reg = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.arange(8, dtype=np.float32)})
    blobs = reg._conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert blobs > 0


def test_save_writes_chunks_to_backend(tmp_path):
    storage, _, backend, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(100, dtype=np.float32)})
    assert len(backend._store) > 0


def test_save_multiple_tensors(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    tensors = {
        "a": np.ones(10, dtype=np.float32),
        "b": np.zeros(10, dtype=np.float64),
        "c": np.arange(10, dtype=np.int32),
    }
    manifest = storage.save("run_a", 1, tensors)
    assert set(manifest.tensors.keys()) == {"a", "b", "c"}


def test_second_save_unchanged_writes_zero_chunks(tmp_path):
    storage, _, backend, _ = make_engine(tmp_path)
    tensors = {"w": np.ones(50, dtype=np.float32)}

    storage.save("run_a", 1, tensors)
    count_after_first = len(backend._store)

    storage.save("run_a", 2, tensors)
    count_after_second = len(backend._store)

    assert count_after_first == count_after_second


def test_second_save_unchanged_does_not_call_store_tensor(tmp_path):
    storage, cas, _, _ = make_engine(tmp_path)
    tensors = {"w": np.ones(50, dtype=np.float32)}
    storage.save("run_a", 1, tensors)

    with patch.object(cas, "store_tensor", wraps=cas.store_tensor) as mock:
        storage.save("run_a", 2, tensors)

    assert mock.call_count == 0


def test_second_save_changed_tensor_calls_store_tensor(tmp_path):
    storage, cas, _, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(10, dtype=np.float32)})

    with patch.object(cas, "store_tensor", wraps=cas.store_tensor) as mock:
        storage.save("run_a", 2, {"w": np.zeros(10, dtype=np.float32)})

    assert mock.call_count == 1


def test_save_parent_step_stored(tmp_path):
    storage, _, _, reg = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})
    storage.save("run_a", 2, {"w": np.ones(4, dtype=np.float32)}, parent_step=1)
    row = reg._conn.execute(
        "SELECT parent_step FROM checkpoints WHERE run_id='run_a' AND step=2"
    ).fetchone()
    assert row[0] == 1


def test_save_cas_failure_raises_storage_error(tmp_path):
    storage, cas, _, _ = make_engine(tmp_path)

    def boom(name, arr):
        raise RuntimeError("backend unavailable")

    with patch.object(cas, "store_tensor", side_effect=boom):
        with pytest.raises(StorageError, match="CAS store failed"):
            storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})


def test_save_registry_failure_removes_manifest(tmp_path):
    storage, _, _, reg = make_engine(tmp_path)

    with patch.object(reg, "register_checkpoint", side_effect=Exception("db down")):
        with pytest.raises(StorageError, match="Registry registration failed"):
            storage.save("run_a", 1, {"w": np.ones(5, dtype=np.float32)})

    manifest_path = tmp_path / "manifests" / "run_a" / "step_000001.json"
    assert not manifest_path.exists()


def test_save_duplicate_step_raises_checkpoint_already_exists(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})
    with pytest.raises(CheckpointAlreadyExistsError):
        storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})


def test_save_duplicate_step_registry_still_has_original(tmp_path):
    """After a duplicate-step failure, the registry still points to step=1."""
    storage, _, _, reg = make_engine(tmp_path)
    storage.save("run_a", 1, {"w": np.ones(4, dtype=np.float32)})
    with pytest.raises(CheckpointAlreadyExistsError):
        storage.save("run_a", 1, {"w": np.zeros(4, dtype=np.float32)})
    assert reg.checkpoint_exists("run_a", 1)


def test_load_after_save_returns_equal_arrays(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    tensors = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    storage.save("run_a", 1, tensors)
    loaded = storage.load("run_a", 1)
    assert np.array_equal(tensors["w"], loaded["w"])


def test_load_multiple_tensors(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    tensors = {
        "a": np.ones(10, dtype=np.float32),
        "b": np.arange(10, dtype=np.int32),
        "c": np.zeros((3, 4), dtype=np.float64),
    }
    storage.save("run_a", 1, tensors)
    loaded = storage.load("run_a", 1)
    assert set(loaded.keys()) == {"a", "b", "c"}
    for name in tensors:
        assert np.array_equal(tensors[name], loaded[name]), name


def test_load_preserves_dtype_and_shape(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    storage.save("run_a", 1, {"m": arr})
    loaded = storage.load("run_a", 1)
    assert loaded["m"].dtype == arr.dtype
    assert loaded["m"].shape == arr.shape


def test_load_nonexistent_step_raises(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    with pytest.raises(CheckpointNotFoundError):
        storage.load("run_a", 999)


def test_load_nonexistent_run_raises(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    with pytest.raises(CheckpointNotFoundError):
        storage.load("no_such_run", 1)


def test_load_corrupted_chunk_raises(tmp_path):
    storage, _, backend, _ = make_engine(tmp_path)
    arr = np.ones(10, dtype=np.float32)
    manifest = storage.save("run_a", 1, {"w": arr})

    # Corrupt a chunk in the backend
    first_hash = manifest.tensors["w"].chunks[0].hash
    backend._store[first_hash] = b"\x00" * manifest.tensors["w"].chunks[0].size

    with pytest.raises(CorruptChunkError):
        storage.load("run_a", 1)


def test_load_after_noop_save_returns_equal_arrays(tmp_path):
    storage, _, _, _ = make_engine(tmp_path)
    arr = np.ones(20, dtype=np.float32)
    storage.save("run_a", 1, {"w": arr})
    storage.save("run_a", 2, {"w": arr})
    loaded = storage.load("run_a", 2)
    assert np.array_equal(arr, loaded["w"])
