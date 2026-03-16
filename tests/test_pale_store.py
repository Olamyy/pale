from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pale.errors import CheckpointNotFoundError
from pale.store import PaleStore


class DictAdapter:
    def extract(self, model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in model.items()}

    def reconstruct(
        self, tensors: Dict[str, np.ndarray], original: Any
    ) -> Dict[str, np.ndarray]:
        return tensors


def _model(seed: int = 0) -> Dict[str, np.ndarray]:
    return {
        "weights": np.random.default_rng(seed).standard_normal(64).astype(np.float32)
    }


@pytest.fixture()
def store(tmp_path: Path) -> PaleStore:
    return PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())


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


def test_load_missing_step_raises(store):
    with pytest.raises(CheckpointNotFoundError):
        store.load(step=99)


def test_delete_checkpoint_removes_from_list(store):
    store.save(_model(0), step=1)
    store.save(_model(1), step=2)
    store.delete_checkpoint(step=1)
    assert store.list_checkpoints() == [2]


def test_gc_deletes_chunk_files_for_orphaned_blobs(tmp_path):
    store = PaleStore(root=tmp_path, run_id="run1", adapter=DictAdapter())
    store.save(_model(0), step=1)
    store.delete_checkpoint(step=1)
    store._registry._conn.execute("UPDATE blobs SET created_at = '2000-01-01 00:00:00'")
    store._registry._conn.commit()
    report = store.gc(grace_period_hours=0)
    assert report.deleted_blobs > 0
    assert list((tmp_path / "objects").rglob("*.chunk")) == []
