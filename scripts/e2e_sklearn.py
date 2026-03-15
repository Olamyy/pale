"""
End-to-end smoke test for the pale sklearn pipeline.

Wires together: SklearnAdapter → CASEngine (FilesystemBackend) →
ManifestWriter/Reader + apply_noop_fast_path.

Validates:
  1. Tensor round-trip + prediction equality + full_hash correctness
  2. CAS dedup: saving identical model twice writes zero new chunks
  3. No-op fast path: unchanged tensors skip CASEngine entirely
  4. Manifest coherence: all hashes present, tensors reload correctly
  5. Evolving checkpoints: warm-start frozen trees dedup via per-tree storage

Run:
    python scripts/e2e_sklearn.py

Set PALE_E2E_KEEP=1 to preserve the store dir on failure for inspection.
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from pale.adapters.sklearn import SklearnAdapter
from pale.cas.engine import CASEngine
from pale.cas.filesystem import FilesystemBackend
from pale.hashing import hash_chunk
from pale.manifest import CheckpointManifest, ManifestReader, ManifestWriter
from pale.serialization import tensor_to_bytes
from pale.storage import apply_noop_fast_path

CHUNK_SIZE = 256 * 1024  # 256KB

# Shared dataset — generated once, used by all checks
_RNG = np.random.default_rng(42)
_X = _RNG.standard_normal((200, 8))
_Y = (_X[:, 0] + _X[:, 1] > 0).astype(int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def train_gbm(
    n_estimators: int, warm_start: bool = False
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, warm_start=warm_start, random_state=0
    )
    model.fit(_X, _Y)
    return model


def count_chunk_files(cas_root: Path) -> int:
    objects_dir = cas_root / "objects"
    if not objects_dir.exists():
        return 0
    return sum(1 for _ in objects_dir.rglob("*.chunk"))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_round_trip(store_dir: Path) -> None:
    print("\n--- Check 1: Tensor round-trip + prediction equality + full_hash ---")
    backend = FilesystemBackend(store_dir / "cas")
    engine = CASEngine(backend, CHUNK_SIZE)
    adapter = SklearnAdapter()
    model = train_gbm(20)

    tensors_original = adapter.extract(model)
    records = {
        name: engine.store_tensor(name, arr) for name, arr in tensors_original.items()
    }
    print(f"  Stored {len(records)} tensors")

    for name, record in records.items():
        raw, _, _ = tensor_to_bytes(tensors_original[name])
        assert record.full_hash == hash_chunk(raw), f"full_hash mismatch for '{name}'"
    print("  full_hash correct for all tensors")

    tensors_loaded = {
        name: engine.load_tensor(record) for name, record in records.items()
    }
    for name in tensors_original:
        orig = tensors_original[name]
        loaded = tensors_loaded[name]
        ok = np.array_equal(orig, loaded)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name}: shape={orig.shape} dtype={orig.dtype}"
        )
        assert ok, f"Round-trip failed for tensor '{name}'"

    reconstructed = adapter.reconstruct(tensors_loaded, model)
    assert np.array_equal(model.predict(_X), reconstructed.predict(_X)), (
        "Reconstructed model predictions differ from original"
    )
    print("  PASS: tensors round-trip correctly, predictions match")


def check_cas_dedup(store_dir: Path) -> None:
    print("\n--- Check 2: CAS dedup (identical model saved twice) ---")
    cas_root = store_dir / "cas_dedup"
    backend = FilesystemBackend(cas_root)
    engine = CASEngine(backend, CHUNK_SIZE)
    tensors = SklearnAdapter().extract(train_gbm(20))

    for name, arr in tensors.items():
        engine.store_tensor(name, arr)
    count_after_first = count_chunk_files(cas_root)

    for name, arr in tensors.items():
        engine.store_tensor(name, arr)
    count_after_second = count_chunk_files(cas_root)

    print(f"  First save:  {count_after_first} total chunk files")
    print(
        f"  Second save: {count_after_second} total chunk files (should be unchanged)"
    )

    assert count_after_first == count_after_second, (
        f"Chunk file count changed on second save: {count_after_first} → {count_after_second}"
    )
    print("  PASS: second save writes zero new chunks")


def check_noop_fast_path(store_dir: Path) -> None:
    print("\n--- Check 3: No-op fast path (unchanged tensors skip CASEngine) ---")
    backend = FilesystemBackend(store_dir / "cas_noop")
    engine = CASEngine(backend, CHUNK_SIZE)
    adapter = SklearnAdapter()
    tensors = adapter.extract(train_gbm(20))

    # First save — build prev_manifest
    prev_records = {
        name: engine.store_tensor(name, arr) for name, arr in tensors.items()
    }
    prev_manifest = CheckpointManifest(
        chunk_size=CHUNK_SIZE,
        run_id="noop_run",
        step=1,
        created_at=datetime.now(timezone.utc),
        tensors=prev_records,
    )

    # Second save — all tensors unchanged, fast path should fire for all
    with patch.object(engine, "store_tensor", wraps=engine.store_tensor) as mock_store:
        records = {}
        for name, arr in tensors.items():
            fast = apply_noop_fast_path(name, arr, prev_manifest)
            if fast is not None:
                records[name] = fast
            else:
                records[name] = engine.store_tensor(name, arr)

    print(f"  store_tensor called {mock_store.call_count} time(s) on second save")
    assert mock_store.call_count == 0, (
        f"Expected store_tensor never called (no-op fast path), called {mock_store.call_count} time(s)"
    )
    assert set(records.keys()) == set(tensors.keys()), (
        "Missing tensor keys after fast path"
    )
    print(
        "  PASS: no-op fast path fires — store_tensor not called for unchanged tensors"
    )


def check_manifest_coherence(store_dir: Path) -> None:
    print("\n--- Check 4: Manifest coherence ---")
    cas_root = store_dir / "cas_manifest"
    backend = FilesystemBackend(cas_root)
    engine = CASEngine(backend, CHUNK_SIZE)
    adapter = SklearnAdapter()
    manifest_path = store_dir / "manifests" / "step_000020.json"

    tensors = adapter.extract(train_gbm(20))
    records = {name: engine.store_tensor(name, arr) for name, arr in tensors.items()}

    manifest = CheckpointManifest(
        chunk_size=CHUNK_SIZE,
        run_id="smoke_run",
        step=20,
        created_at=datetime.now(timezone.utc),
        tensors=records,
    )
    ManifestWriter.write(manifest, manifest_path)
    print(f"  Manifest written to {manifest_path}")

    loaded_manifest = ManifestReader.read(manifest_path)
    assert loaded_manifest.run_id == "smoke_run"
    assert loaded_manifest.step == 20
    assert set(loaded_manifest.tensors.keys()) == set(records.keys())

    ManifestReader.validate(loaded_manifest, backend)
    print(
        f"  All {sum(len(r.chunks) for r in records.values())} chunk hashes present in CAS"
    )

    for record in loaded_manifest.tensors.values():
        record.validate_chunk_refs()

    tensors_reloaded = {
        name: engine.load_tensor(record)
        for name, record in loaded_manifest.tensors.items()
    }
    for name in tensors:
        assert np.array_equal(tensors[name], tensors_reloaded[name]), (
            f"Manifest round-trip failed for '{name}'"
        )
    print("  PASS: manifest coherent, tensors reload correctly")


def check_evolving_checkpoints(store_dir: Path) -> None:
    print("\n--- Check 5: Evolving checkpoints (warm-start frozen tree dedup) ---")
    cas_root = store_dir / "cas_evolving"
    backend = FilesystemBackend(cas_root)
    engine = CASEngine(backend, CHUNK_SIZE)
    adapter = SklearnAdapter()

    # Step 1: train 10 trees
    model = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=0)
    model.fit(_X, _Y)
    tensors_10 = adapter.extract(model)

    with patch.object(backend, "put", wraps=backend.put) as mock_put:
        for name, arr in tensors_10.items():
            engine.store_tensor(name, arr)
    chunks_step1 = mock_put.call_count

    # Step 2: continue on the same model — first 10 trees are frozen
    model.set_params(n_estimators=20)
    model.fit(_X, _Y)
    tensors_20 = adapter.extract(model)

    with patch.object(backend, "put", wraps=backend.put) as mock_put:
        for name, arr in tensors_20.items():
            engine.store_tensor(name, arr)
    chunks_step2 = mock_put.call_count

    total_files = count_chunk_files(cas_root)

    print(f"  Step 1 (10 trees): {chunks_step1} new chunks written")
    print(f"  Step 2 (20 trees): {chunks_step2} new chunks written")
    print(f"  Total chunk files on disk: {total_files}")

    assert chunks_step1 > 0 and chunks_step2 > 0, "Both steps must write chunks"
    assert total_files == chunks_step1 + chunks_step2, (
        f"Unexpected file count: {total_files} != {chunks_step1} + {chunks_step2}"
    )
    assert chunks_step2 < chunks_step1, (
        f"Expected step 2 (20 trees, 10 frozen) to write fewer chunks than step 1 "
        f"(10 new trees): {chunks_step2} >= {chunks_step1}"
    )
    print("  PASS: frozen tree chunks deduplicated across warm-start checkpoints")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    KEEP_ON_FAILURE = os.environ.get("PALE_E2E_KEEP", "0") == "1"
    store_dir = Path(tempfile.mkdtemp(prefix="pale_e2e_"))
    print(f"Store dir: {store_dir}")

    try:
        check_round_trip(store_dir)
        check_cas_dedup(store_dir)
        check_noop_fast_path(store_dir)
        check_manifest_coherence(store_dir)
        check_evolving_checkpoints(store_dir)
        print("\n=== All checks passed ===")
        shutil.rmtree(store_dir)
    except Exception:
        if KEEP_ON_FAILURE:
            print(f"\nStore dir preserved for inspection: {store_dir}")
        else:
            shutil.rmtree(store_dir)
        raise
