import numpy as np
import pytest

from pale.cas.engine import CASEngine
from pale.errors import CorruptChunkError


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


def test_store_returns_correct_record():
    engine, _ = make_engine()
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    record = engine.store_tensor("w", arr)
    assert record.name == "w"
    assert record.dtype == "float32"
    assert record.shape == [3]
    assert record.byte_length == arr.nbytes
    assert len(record.chunks) >= 1


def test_store_chunk_refs_valid():
    engine, _ = make_engine(chunk_size=8)
    arr = np.arange(8, dtype=np.float32)
    record = engine.store_tensor("x", arr)
    record.validate_chunk_refs()


def test_store_writes_chunks_to_backend():
    engine, backend = make_engine(chunk_size=8)
    arr = np.arange(4, dtype=np.float64)
    record = engine.store_tensor("w", arr)
    for ref in record.chunks:
        assert backend.has(ref.hash)


def test_second_store_writes_zero_new_chunks():
    engine, backend = make_engine()
    arr = np.random.default_rng(0).standard_normal(100).astype(np.float32)

    engine.store_tensor("w", arr)
    count_after_first = len(backend._store)

    engine.store_tensor("w", arr)
    count_after_second = len(backend._store)

    assert count_after_first == count_after_second


def test_second_store_calls_put_zero_times(monkeypatch):
    engine, backend = make_engine()
    arr = np.ones(50, dtype=np.float64)
    engine.store_tensor("w", arr)

    put_calls = []
    original_put = backend.put
    monkeypatch.setattr(
        backend, "put", lambda h, d: put_calls.append(h) or original_put(h, d)
    )

    engine.store_tensor("w", arr)
    assert put_calls == []


def test_load_returns_array_equal():
    engine, _ = make_engine()
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    record = engine.store_tensor("m", arr)
    loaded = engine.load_tensor(record)
    assert np.array_equal(arr, loaded)


def test_load_preserves_dtype_and_shape():
    engine, _ = make_engine()
    for dtype in (np.float32, np.float64, np.int32, np.int64):
        arr = np.arange(12, dtype=dtype).reshape(3, 4)
        record = engine.store_tensor("a", arr)
        loaded = engine.load_tensor(record)
        assert loaded.dtype == arr.dtype, f"dtype mismatch for {dtype}"
        assert loaded.shape == arr.shape, f"shape mismatch for {dtype}"
        assert np.array_equal(arr, loaded)


def test_load_multi_chunk_round_trip():
    engine, _ = make_engine(chunk_size=8)
    arr = np.arange(32, dtype=np.float32)
    record = engine.store_tensor("big", arr)
    assert len(record.chunks) == 16
    loaded = engine.load_tensor(record)
    assert np.array_equal(arr, loaded)


def test_load_corrupted_chunk_raises():
    engine, backend = make_engine()
    arr = np.ones(10, dtype=np.float32)
    record = engine.store_tensor("w", arr)

    first_hash = record.chunks[0].hash
    backend._store[first_hash] = b"\x00" * record.chunks[0].size

    with pytest.raises(CorruptChunkError):
        engine.load_tensor(record)


def test_load_out_of_order_chunk_completion():
    """Reassembly must be correct regardless of which chunk completes first.

    Stores chunks then re-inserts them in reverse order into a fresh backend,
    verifying that load_tensor uses positional indexing (not completion order).
    """
    chunk_size = 8
    arr = np.arange(32, dtype=np.float32)

    engine, backend = make_engine(chunk_size=chunk_size)
    record = engine.store_tensor("arr", arr)

    reversed_backend = DictBackend()
    for h, data in reversed(list(backend._store.items())):
        reversed_backend._store[h] = data

    engine2 = CASEngine(reversed_backend, chunk_size)
    loaded = engine2.load_tensor(record)
    assert np.array_equal(arr, loaded), (
        "Positional reassembly failed with reversed chunk storage"
    )


def test_full_hash_matches_raw_bytes():
    from pale.hashing import hash_chunk
    from pale.serialization import tensor_to_bytes

    engine, _ = make_engine()
    arr = np.random.default_rng(1).standard_normal(50).astype(np.float64)
    record = engine.store_tensor("w", arr)

    raw, _, _ = tensor_to_bytes(arr)
    assert record.full_hash == hash_chunk(raw)


def test_context_manager_works():
    backend = DictBackend()
    with CASEngine(backend, chunk_size=256) as engine:
        arr = np.array([1.0, 2.0], dtype=np.float32)
        record = engine.store_tensor("w", arr)
        loaded = engine.load_tensor(record)
        assert np.array_equal(arr, loaded)


def test_max_workers_respected():
    backend = DictBackend()
    engine = CASEngine(backend, chunk_size=8, max_workers=2)
    arr = np.arange(16, dtype=np.float32)
    record = engine.store_tensor("w", arr)
    loaded = engine.load_tensor(record)
    assert np.array_equal(arr, loaded)
