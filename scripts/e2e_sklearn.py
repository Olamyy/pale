import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from pale.adapters.sklearn import SklearnAdapter
from pale.errors import CorruptChunkError
from pale.store import PaleStore

CHUNK_SIZE = 256 * 1024

_RNG = np.random.default_rng(42)
_X = _RNG.standard_normal((200, 8))
_Y = (_X[:, 0] + _X[:, 1] > 0).astype(int)


def train_gbm(n_estimators: int) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, warm_start=True, random_state=0
    )
    model.fit(_X, _Y)
    return model


def check_evolving_checkpoints(root: Path) -> None:
    print(
        "\n--- Check 0: Evolving warm-start checkpoints — dedup fires on frozen trees ---"
    )
    model = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=0)
    model.fit(_X, _Y)

    cas_root = root / "c0"
    with PaleStore(
        root=cas_root, run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        tensors_step1 = store.stats()["total_chunks"]

        model.set_params(n_estimators=20)
        model.fit(_X, _Y)
        store.save(model, step=2)

        s = store.stats()

    unique = s["unique_chunks"]
    total = s["total_chunks"]
    assert unique < total, f"Expected dedup: unique={unique} < total={total}"

    reused = total - unique
    print(
        f"  step1={tensors_step1} tensors  step2={total - tensors_step1} tensors  total={total}  unique={unique}  reused={reused}"
    )
    print(f"  PASS: {reused} frozen tree tensors reused from step 1 (no-op fast path)")


def check_round_trip(root: Path) -> None:
    print("\n--- Check 1: Save + load round-trip + prediction equality ---")
    model = train_gbm(20)
    with PaleStore(
        root=root / "c1", run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        restored = store.load(step=1, original=model)
    assert np.array_equal(model.predict(_X), restored.predict(_X)), (
        "Reconstructed model predictions differ from original"
    )
    print("  PASS: predictions match after save/load")


def check_cross_run_cas_dedup(root: Path) -> None:
    print(
        "\n--- Check 2: Cross-run CAS dedup (two runs sharing the same model weights) ---"
    )
    model = train_gbm(20)
    cas_root = root / "c2"

    with PaleStore(
        root=cas_root, run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)

    with PaleStore(
        root=cas_root, run_id="run2", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        with patch.object(store._cas._backend, "put") as mock_put:
            store.save(model, step=1)
            new_writes = mock_put.call_count

    print(
        f"  run2 save: {new_writes} new chunk writes (should be 0 — all chunks already in CAS)"
    )
    assert new_writes == 0, f"Expected 0 new writes, got {new_writes}"
    print("  PASS: cross-run CAS dedup fires, zero new chunks written")


def check_noop_fast_path(root: Path) -> None:
    print(
        "\n--- Check 3: No-op fast path (unchanged tensors skip CASEngine entirely) ---"
    )
    model = train_gbm(20)
    with PaleStore(
        root=root / "c3", run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        with patch.object(
            store._cas, "store_tensor", wraps=store._cas.store_tensor
        ) as mock_store:
            store.save(model, step=2)
            call_count = mock_store.call_count
    print(f"  store_tensor called {call_count} time(s) on second save (should be 0)")
    assert call_count == 0, (
        f"Expected store_tensor never called, called {call_count} time(s)"
    )
    print("  PASS: no-op fast path fires for all unchanged tensors")


def check_stats_dedup(root: Path) -> None:
    print("\n--- Check 4: stats() reports dedup savings ---")
    model = train_gbm(20)
    with PaleStore(
        root=root / "c4", run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        store.save(model, step=2)
        s = store.stats()
    print(
        f"  total_chunks={s['total_chunks']}  unique_chunks={s['unique_chunks']}  dedup_ratio={s['dedup_ratio']}  total_bytes={s['total_bytes']}"
    )
    assert s["total_chunks"] > s["unique_chunks"], (
        "Expected dedup across identical checkpoints"
    )
    assert s["dedup_ratio"] < 1.0, (
        f"dedup_ratio should be < 1.0, got {s['dedup_ratio']}"
    )
    assert s["total_bytes"] > 0, "total_bytes should be positive"
    print("  PASS: dedup savings reported correctly")


def check_store_wide_stats(root: Path) -> None:
    print("\n--- Check 5: store_stats() shows cross-run dedup ---")
    model = train_gbm(20)
    cas_root = root / "c5s"

    with PaleStore(
        root=cas_root, run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
    with PaleStore(
        root=cas_root, run_id="run2", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
    with PaleStore(
        root=cas_root, run_id="run3", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)

    s = PaleStore.store_stats(cas_root)
    print(
        f"  runs={s['runs']}  checkpoints={s['checkpoints']}  total_chunks={s['total_chunks']}  unique_chunks={s['unique_chunks']}  dedup_ratio={s['dedup_ratio']}"
    )
    assert s["runs"] == 3
    assert s["checkpoints"] == 3
    assert s["unique_chunks"] < s["total_chunks"], "Expected cross-run dedup"
    assert s["dedup_ratio"] < 1.0
    print("  PASS: store_stats() shows cross-run dedup across 3 independent runs")


def check_corrupt_chunk_detection(root: Path) -> None:
    print("\n--- Check 6: Corrupt chunk raises CorruptChunkError ---")
    model = train_gbm(20)
    with PaleStore(
        root=root / "c6", run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        chunk_file = next((root / "c6" / "objects").rglob("*.chunk"))
        chunk_file.write_bytes(b"corrupted")
        try:
            store.load(step=1, original=model)
            raise AssertionError("Expected CorruptChunkError was not raised")
        except CorruptChunkError:
            pass
    print("  PASS: corrupt chunk raises CorruptChunkError")


def check_delete_and_gc(root: Path) -> None:
    print("\n--- Check 7: delete_checkpoint + GC removes chunk files ---")
    model = train_gbm(20)
    cas_root = root / "c7"
    with PaleStore(
        root=cas_root, run_id="run1", adapter=SklearnAdapter(), chunk_size=CHUNK_SIZE
    ) as store:
        store.save(model, step=1)
        assert store.list_checkpoints() == [1]

        store.delete_checkpoint(step=1)
        assert store.list_checkpoints() == [], (
            "Checkpoint should be removed from list after delete"
        )

        chunk_files_before = list((cas_root / "objects").rglob("*.chunk"))
        assert len(chunk_files_before) > 0, "Chunk files should still exist before GC"

        store._registry._conn.execute(
            "UPDATE blobs SET created_at = datetime('now', '-2 hours')"
        )
        store._registry._conn.commit()

        report = store.gc(grace_period_hours=1)
        assert report.deleted_blobs > 0, "GC should have swept orphaned blobs"

        chunk_files_after = list((cas_root / "objects").rglob("*.chunk"))
        assert len(chunk_files_after) == 0, "All chunk files should be deleted after GC"
    print(f"  Deleted {report.deleted_blobs} blobs, freed {report.freed_bytes} bytes")
    print("  PASS: delete + GC round-trip cleans up chunk files")


if __name__ == "__main__":
    KEEP_ON_FAILURE = os.environ.get("PALE_E2E_KEEP", "0") == "1"
    root = Path(tempfile.mkdtemp(prefix="pale_e2e_"))
    print(f"Store dir: {root}")

    try:
        check_evolving_checkpoints(root)
        check_round_trip(root)
        check_cross_run_cas_dedup(root)
        check_noop_fast_path(root)
        check_stats_dedup(root)
        check_store_wide_stats(root)
        check_corrupt_chunk_detection(root)
        check_delete_and_gc(root)
        print("\n=== All checks passed ===")
        shutil.rmtree(root)
    except Exception:
        if KEEP_ON_FAILURE:
            print(f"\nStore dir preserved for inspection: {root}")
        else:
            shutil.rmtree(root)
        raise
