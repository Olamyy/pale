import sqlite3
from unittest.mock import patch

import numpy as np

from pale.cas.engine import CASEngine
from pale.registry.registry import Registry
from pale.registry.schema import create_tables
from pale.storage import StorageEngine


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
    return StorageEngine(cas, Registry(conn), tmp_path / "manifests")


def test_noop_refs_are_identical_objects(tmp_path):
    s = make_storage(tmp_path)
    arr = np.arange(50, dtype=np.float64)
    m1 = s.save("run_a", 1, {"w": arr})
    m2 = s.save("run_a", 2, {"w": arr})
    assert m1.tensors["w"].chunks == m2.tensors["w"].chunks
