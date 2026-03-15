from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from pale.errors import (
    CheckpointAlreadyExistsError,
    CheckpointNotFoundError,
    CorruptChunkError,
)
from pale.registry.registry import GCReport
from pale.store import PaleStore


class DictAdapter:
    def extract(self, model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in model.items()}

    def reconstruct(
        self, tensors: Dict[str, np.ndarray], original: Any
    ) -> Dict[str, np.ndarray]:
        return tensors


def _model(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"weights": rng.standard_normal(64).astype(np.float32)}


@pytest.fixture()
def store(tmp_path: Path) -> PaleStore:
    return PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())


def test_save_and_load_round_trip(store):
    model = _model(0)
    store.save(model, step=1)
    restored = store.load(step=1)
    np.testing.assert_array_equal(model["weights"], restored["weights"])


def test_load_passes_original_to_adapter(tmp_path):
    received = {}

    class CapturingAdapter:
        def extract(self, model):
            return {"w": model}

        def reconstruct(self, tensors, original):
            received["original"] = original
            return tensors

    store = PaleStore(root=tmp_path, run_id="r", adapter=CapturingAdapter())
    store.save(np.zeros(4, dtype=np.float32), step=1)
    sentinel = object()
    store.load(step=1, original=sentinel)
    assert received["original"] is sentinel


def test_load_tensors_returns_arrays(store):
    model = _model(0)
    store.save(model, step=1)
    tensors = store.load_tensors(step=1)
    np.testing.assert_array_equal(model["weights"], tensors["weights"])


def test_load_missing_step_raises(store):
    with pytest.raises(CheckpointNotFoundError):
        store.load(step=99)


def test_save_duplicate_step_raises(store):
    store.save(_model(0), step=1)
    with pytest.raises(CheckpointAlreadyExistsError):
        store.save(_model(1), step=1)


def test_list_checkpoints_sorted(store):
    store.save(_model(0), step=3)
    store.save(_model(1), step=1)
    store.save(_model(2), step=2)
    assert store.list_checkpoints() == [1, 2, 3]


def test_delete_checkpoint_removes_from_list(store):
    store.save(_model(0), step=1)
    store.save(_model(1), step=2)
    store.delete_checkpoint(step=1)
    assert store.list_checkpoints() == [2]


def test_deleted_checkpoint_not_loadable(store):
    store.save(_model(0), step=1)
    store.delete_checkpoint(step=1)
    with pytest.raises(CheckpointNotFoundError):
        store.load(step=1)


def test_gc_deletes_chunk_files_for_orphaned_blobs(tmp_path):
    store = PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())
    store.save(_model(0), step=1)
    store.delete_checkpoint(step=1)

    store._registry._conn.execute("UPDATE blobs SET created_at = '2000-01-01 00:00:00'")
    store._registry._conn.commit()

    report = store.gc(grace_period_hours=0)
    assert report.deleted_blobs > 0
    assert list((tmp_path / "objects").rglob("*.chunk")) == []


def test_gc_does_not_delete_live_blobs(tmp_path):
    store = PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())
    store.save(_model(0), step=1)
    report = store.gc(grace_period_hours=0)
    assert report.deleted_blobs == 0
    assert len(list((tmp_path / "objects").rglob("*.chunk"))) > 0


def test_stats_dedup_ratio_for_identical_checkpoints(store):
    model = _model(0)
    store.save(model, step=1)
    store.save(model, step=2)
    s = store.stats()
    assert s["checkpoints"] == 2
    assert s["unique_chunks"] < s["total_chunks"]
    assert s["dedup_ratio"] < 1.0


def test_stats_empty_run(store):
    s = store.stats()
    assert s["checkpoints"] == 0
    assert s["total_chunks"] == 0
    assert s["dedup_ratio"] == 1.0


def test_context_manager_shuts_down_executor(tmp_path):
    with PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter()) as store:
        store.save(_model(0), step=1)
        restored = store.load(step=1)
    np.testing.assert_array_equal(_model(0)["weights"], restored["weights"])


def test_identical_consecutive_checkpoints_skip_cas_writes(tmp_path):
    store = PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())
    model = _model(0)
    store.save(model, step=1)
    with patch.object(store._cas._backend, "put") as mock_put:
        store.save(model, step=2)
        mock_put.assert_not_called()


def test_corrupt_chunk_raises(tmp_path):
    store = PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())
    store.save(_model(0), step=1)
    chunk_file = next((tmp_path / "objects").rglob("*.chunk"))
    chunk_file.write_bytes(b"corrupted")
    with pytest.raises(CorruptChunkError):
        store.load(step=1)


def test_store_stats_cross_run_dedup(tmp_path):
    model = _model(0)
    with PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter()) as s:
        s.save(model, step=1)
    with PaleStore(root=tmp_path, run_id="run2", adapter=DictAdapter()) as s:
        s.save(model, step=1)
    stats = PaleStore.store_stats(tmp_path)
    assert stats["runs"] == 2
    assert stats["checkpoints"] == 2
    assert stats["unique_chunks"] < stats["total_chunks"]
    assert stats["dedup_ratio"] < 1.0
