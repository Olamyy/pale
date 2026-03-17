import numpy as np
import pytest

from tensorcas.cas.engine import CASEngine
from tensorcas.errors import CorruptChunkError


class DictBackend:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def has(self, hash: str) -> bool:
        return hash in self._store

    def put(self, hash: str, data: bytes) -> None:
        self._store[hash] = data

    def get(self, hash: str) -> bytes:
        return self._store[hash]

    def batch_has(self, hashes: list[str]) -> dict[str, bool]:
        return {h: h in self._store for h in hashes}

    def delete(self, hash: str) -> None:
        del self._store[hash]


def make_engine(chunk_size: int = 256 * 1024) -> tuple[CASEngine, DictBackend]:
    backend = DictBackend()
    return CASEngine(backend, chunk_size), backend


def test_load_multi_chunk_round_trip():
    engine, _ = make_engine(chunk_size=8)
    arr = np.arange(32, dtype=np.float32)
    record = engine.store_tensor("big", arr)
    assert len(record.chunks) == 16
    assert np.array_equal(arr, engine.load_tensor(record))


def test_load_corrupted_chunk_raises():
    engine, backend = make_engine()
    arr = np.ones(10, dtype=np.float32)
    record = engine.store_tensor("w", arr)
    backend._store[record.chunks[0].hash] = b"\x00" * record.chunks[0].size
    with pytest.raises(CorruptChunkError):
        engine.load_tensor(record)


def test_load_out_of_order_chunk_completion():
    chunk_size = 8
    arr = np.arange(32, dtype=np.float32)
    engine, backend = make_engine(chunk_size=chunk_size)
    record = engine.store_tensor("arr", arr)

    reversed_backend = DictBackend()
    for h, data in reversed(list(backend._store.items())):
        reversed_backend._store[h] = data

    assert np.array_equal(
        arr, CASEngine(reversed_backend, chunk_size).load_tensor(record)
    )
