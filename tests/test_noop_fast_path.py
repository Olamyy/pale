import sqlite3
from datetime import datetime, timezone

import numpy as np
from unittest.mock import patch

from pale.cas.engine import CASEngine
from pale.manifest import CheckpointManifest
from pale.models import ChunkRef, TensorArrayRecord
from pale.registry.registry import Registry
from pale.registry.schema import create_tables
from pale.storage import StorageEngine
from pale.hashing import hash_chunk
from pale.serialization import tensor_to_bytes


class DictBackend:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def has(self, h):
        return h in self._store

    def put(self, h, d):
        self._store[h] = d

    def get(self, h):
        return self._store[h]

    def batch_has(self, hs):
        return {h: h in self._store for h in hs}

    def delete(self, h):
        del self._store[h]


def make_storage(tmp_path):
    backend = DictBackend()
    cas = CASEngine(backend, chunk_size=256)
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    reg = Registry(conn)
    return StorageEngine(cas, reg, tmp_path / "manifests")


def _make_record(name: str, arr: np.ndarray) -> TensorArrayRecord:

    raw, dtype_str, shape = tensor_to_bytes(arr)
    full_hash = hash_chunk(raw)
    return TensorArrayRecord(
        name=name,
        dtype=dtype_str,
        shape=shape,
        byte_length=len(raw),
        full_hash=full_hash,
        chunks=[ChunkRef(hash=full_hash, size=len(raw))],
    )


def _make_manifest(tensors: dict) -> CheckpointManifest:
    return CheckpointManifest(
        chunk_size=256,
        run_id="r",
        step=1,
        created_at=datetime.now(timezone.utc),
        tensors=tensors,
    )


def test_no_prev_manifest_returns_false(tmp_path):
    s = make_storage(tmp_path)
    arr = np.ones(10, dtype=np.float32)
    assert s._is_unchanged("w", arr, None) is False


def test_unchanged_tensor_returns_true(tmp_path):
    s = make_storage(tmp_path)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    manifest = _make_manifest({"w": _make_record("w", arr)})
    assert s._is_unchanged("w", arr, manifest) is True


def test_changed_tensor_returns_false(tmp_path):
    s = make_storage(tmp_path)
    arr_v1 = np.ones(10, dtype=np.float32)
    arr_v2 = arr_v1 + 0.001
    manifest = _make_manifest({"w": _make_record("w", arr_v1)})
    assert s._is_unchanged("w", arr_v2, manifest) is False


def test_new_tensor_not_in_manifest_returns_false(tmp_path):
    s = make_storage(tmp_path)
    manifest = _make_manifest({})
    assert s._is_unchanged("new", np.zeros(5, dtype=np.float32), manifest) is False


def test_second_save_unchanged_zero_new_writes(tmp_path):
    s = make_storage(tmp_path)
    arr = np.ones(50, dtype=np.float32)
    s.save("run_a", 1, {"w": arr})

    with patch.object(s._cas, "store_tensor", wraps=s._cas.store_tensor) as mock:
        s.save("run_a", 2, {"w": arr})

    assert mock.call_count == 0


def test_second_save_changed_tensor_calls_store_tensor(tmp_path):
    s = make_storage(tmp_path)
    s.save("run_a", 1, {"w": np.ones(10, dtype=np.float32)})

    with patch.object(s._cas, "store_tensor", wraps=s._cas.store_tensor) as mock:
        s.save("run_a", 2, {"w": np.zeros(10, dtype=np.float32)})

    assert mock.call_count == 1


def test_noop_refs_are_identical_objects(tmp_path):
    """After a no-op save, the loaded record's chunks must be identical to the first save."""
    s = make_storage(tmp_path)
    arr = np.arange(50, dtype=np.float64)
    m1 = s.save("run_a", 1, {"w": arr})
    m2 = s.save("run_a", 2, {"w": arr})
    assert m1.tensors["w"].chunks == m2.tensors["w"].chunks
