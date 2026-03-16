import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from pale.cas.engine import CASEngine
from pale.errors import (
    CheckpointAlreadyExistsError,
    CheckpointNotFoundError,
    CorruptManifestError,
    StorageError,
)
from pale.hashing import hash_chunk
from pale.manifest import CheckpointManifest, ManifestReader, ManifestWriter
from pale.models import TensorArrayRecord
from pale.registry.registry import Registry
from pale.serialization import tensor_to_bytes


class StorageEngine:
    """Orchestrates save and load of checkpoints.

    Owns the coordination between:
    - CASEngine: chunk-level dedup storage
    - Per-tensor identity check against previous manifest (no-op fast path)
    - ManifestWriter/Reader: atomic manifest persistence
    - Registry: transactional checkpoint registration

    Does not own the CASEngine executor lifecycle — the engine is passed in
    and should be shut down by the caller.
    """

    def __init__(
        self,
        cas_engine: CASEngine,
        registry: Registry,
        manifest_dir: Path,
    ) -> None:
        self._cas = cas_engine
        self._registry = registry
        self._manifest_dir = manifest_dir
        self._hash_cache: Dict[str, str] = {}  # tensor name → full_hash
        self._prev_manifest: Optional[CheckpointManifest] = None  # last written manifest

    def save(
        self,
        run_id: str,
        step: int,
        tensors: Dict[str, np.ndarray],
        parent_step: Optional[int] = None,
    ) -> CheckpointManifest:
        """Save a checkpoint transactionally.

        For each tensor:
          1. Check if content is unchanged vs. the previous manifest.
          2. If unchanged — reuse previous ChunkRefs, zero CAS writes.
          3. If new or changed — store via CASEngine.

        Then:
          4. Write manifest atomically to disk.
          5. Register checkpoint + blob refs in the Registry.

        Raises:
            StorageError: if CAS store or manifest write fails.
            CheckpointAlreadyExistsError: if (run_id, step) is already registered.
                Propagated unwrapped so callers can distinguish it from infra errors.

        On failure after CAS writes, orphaned chunks are left for GC to sweep.

        Returns the written CheckpointManifest.
        """
        prev_manifest = self._get_prev_manifest(run_id, step)

        # Serial no-op partition: O(1) dict lookups with cache, no allocation.
        unchanged: Dict[str, TensorArrayRecord] = {}
        changed: Dict[str, np.ndarray] = {}
        for name, arr in tensors.items():
            if self._is_unchanged(name, arr, prev_manifest):
                unchanged[name] = prev_manifest.tensors[name]  # type: ignore[index]
            else:
                changed[name] = arr

        # Parallel CAS writes only for tensors that actually changed.
        records: Dict[str, TensorArrayRecord] = dict(unchanged)
        errors: list[tuple[str, Exception]] = []
        if changed:
            n_workers = min(len(changed), 8)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {
                    pool.submit(self._cas.store_tensor, nm, ar): nm
                    for nm, ar in changed.items()
                }
                for fut in as_completed(futs):
                    nm = futs[fut]
                    try:
                        record = fut.result()
                        self._hash_cache[nm] = record.full_hash
                        records[nm] = record
                    except Exception as exc:
                        errors.append((nm, exc))
        if errors:
            detail = "; ".join(f"'{nm}': {exc}" for nm, exc in errors)
            raise StorageError(
                f"CAS store failed for {len(errors)} tensor(s) "
                f"(run={run_id}, step={step}): {detail}"
            )

        manifest_path = self._manifest_path(run_id, step)
        manifest = CheckpointManifest(
            chunk_size=self._cas.chunk_size,
            run_id=run_id,
            step=step,
            created_at=datetime.now(timezone.utc),
            tensors=records,
        )
        try:
            ManifestWriter.write(manifest, manifest_path)
        except Exception as exc:
            raise StorageError(
                f"Manifest write failed (run={run_id}, step={step}): {exc}"
            ) from exc
        self._prev_manifest = manifest

        blob_hashes = [ref.hash for rec in records.values() for ref in rec.chunks]
        blob_sizes = {
            ref.hash: ref.size for rec in records.values() for ref in rec.chunks
        }
        try:
            self._registry.register_checkpoint(
                run_id=run_id,
                step=step,
                manifest_path=manifest_path,
                blob_hashes=blob_hashes,
                blob_sizes=blob_sizes,
                parent_step=parent_step,
            )
        except CheckpointAlreadyExistsError:
            try:
                os.unlink(manifest_path)
            except OSError:
                pass
            raise
        except Exception as exc:
            try:
                os.unlink(manifest_path)
            except OSError:
                pass
            raise StorageError(
                f"Registry registration failed (run={run_id}, step={step}): {exc}"
            ) from exc

        return manifest

    def load(self, run_id: str, step: int) -> Dict[str, np.ndarray]:
        """Load a checkpoint and return its tensors.

        Raises:
            CheckpointNotFoundError: if (run_id, step) is not in the registry.
            CorruptManifestError: if the manifest is missing or invalid.
            CorruptChunkError: if any chunk fails hash verification.
        """
        manifest_path = self._registry.get_manifest_path(run_id, step)
        if manifest_path is None:
            raise CheckpointNotFoundError(
                f"No checkpoint found for run={run_id!r}, step={step}"
            )

        manifest = ManifestReader.read(manifest_path)
        ManifestReader.validate(manifest, self._cas.backend)

        n_workers = min(len(manifest.tensors), 8)
        results: Dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(self._cas.load_tensor, record): name
                for name, record in manifest.tensors.items()
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results

    def _manifest_path(self, run_id: str, step: int) -> Path:
        return self._manifest_dir / run_id / f"step_{step:06d}.json"

    def _is_unchanged(
        self,
        name: str,
        arr: np.ndarray,
        prev_manifest: Optional[CheckpointManifest],
    ) -> bool:
        """Return True if arr is byte-identical to the previous manifest's record."""
        if prev_manifest is None:
            return False
        prev_record = prev_manifest.tensors.get(name)
        if prev_record is None:
            return False
        if list(arr.shape) != prev_record.shape or str(arr.dtype) != prev_record.dtype:
            self._hash_cache.pop(name, None)
            return False
        cached = self._hash_cache.get(name)
        if cached is not None:
            return cached == prev_record.full_hash
        raw, _, _ = tensor_to_bytes(arr)
        h = hash_chunk(raw)
        self._hash_cache[name] = h
        return h == prev_record.full_hash

    def _get_prev_manifest(
        self, run_id: str, step: int
    ) -> Optional[CheckpointManifest]:
        """Return the manifest for (run_id, step-1).

        Uses the in-memory cached manifest when it matches (run_id, step-1) to
        avoid re-reading and deserializing the JSON on every sequential save.
        Falls back to disk for the first call or after a gap in steps.
        """
        prev_step = step - 1
        if (
            self._prev_manifest is not None
            and self._prev_manifest.run_id == run_id
            and self._prev_manifest.step == prev_step
        ):
            return self._prev_manifest
        prev_path = self._registry.get_manifest_path(run_id, prev_step)
        if prev_path is None or not prev_path.exists():
            return None
        try:
            return ManifestReader.read(prev_path)
        except CorruptManifestError:
            warnings.warn(
                f"Previous manifest for run={run_id!r} step={prev_step} is corrupt — "
                "no-op fast path disabled for this checkpoint.",
                stacklevel=3,
            )
            return None
        except Exception as exc:
            raise StorageError(
                f"Failed to load previous manifest (run={run_id!r}, step={prev_step}): {exc}"
            ) from exc
