import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tensorcas.adapters.base import ModelAdapter
from tensorcas.cas.engine import CASEngine
from tensorcas.cas.filesystem import FilesystemBackend
from tensorcas.manifest import ManifestReader
from tensorcas.registry.registry import GCReport, Registry
from tensorcas.storage import StorageEngine

_DEFAULT_CHUNK_SIZE = 1_048_576


class tensorcasStore:
    """High-level API for checkpoint storage and retrieval.

    Wires together FilesystemBackend, CASEngine, Registry, and StorageEngine
    under a single root directory.

    Layout under root:
        {root}/objects/       — CAS chunk files
        {root}/manifests/     — Per-run manifest JSON files
        {root}/registry.db    — SQLite registry

    Args:
        root:       Root directory for all storage. Created if absent.
        run_id:     Identifier for this training run.
        adapter:    ModelAdapter instance for the model framework being used.
        chunk_size: Chunk size in bytes for CAS splitting. Default 1 MiB.
    """

    def __init__(
        self,
        root: Path,
        run_id: str,
        adapter: ModelAdapter,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_workers: int = 8,
    ) -> None:
        root = Path(root)
        self._run_id = run_id
        self._adapter = adapter

        backend = FilesystemBackend(root)
        cas = CASEngine(backend, chunk_size, max_workers=max_workers)
        registry = Registry.from_path(root / "registry.db")
        self._storage = StorageEngine(
            cas_engine=cas,
            registry=registry,
            manifest_dir=root / "manifests",
        )
        self._registry = registry
        self._cas = cas

    def save(self, model: Any, step: int, parent_step: Optional[int] = None) -> None:
        """Extract tensors from model and save as checkpoint (run_id, step).

        Raises:
            AdapterError:                 if tensor extraction fails.
            CheckpointAlreadyExistsError: if (run_id, step) is already saved.
            StorageError:                 if CAS or manifest write fails.
        """
        tensors = self._adapter.extract(model)
        self._storage.save(
            run_id=self._run_id,
            step=step,
            tensors=tensors,
            parent_step=parent_step,
        )

    def load(self, step: int, original: Any = None) -> Any:
        """Load checkpoint (run_id, step) and reconstruct the model.

        For adapters that require a template (e.g. SklearnAdapter), pass the
        original fitted model via original. Adapters that encode full state
        (e.g. XGBoostAdapter) ignore it.

        Raises:
            CheckpointNotFoundError: if (run_id, step) does not exist.
            CorruptManifestError:    if the manifest is missing or invalid.
            CorruptChunkError:       if any chunk fails hash verification.
            AdapterError:            if model reconstruction fails.
        """
        tensors = self._storage.load(run_id=self._run_id, step=step)
        return self._adapter.reconstruct(tensors, original=original)

    def load_tensors(self, step: int) -> Dict[str, np.ndarray]:
        """Load raw tensors for (run_id, step) without reconstruction."""
        return self._storage.load(run_id=self._run_id, step=step)

    def list_checkpoints(self) -> List[int]:
        """Return all saved steps for this run in ascending order."""
        return self._registry.list_checkpoints(self._run_id)

    def delete_checkpoint(self, step: int) -> None:
        """Delete checkpoint (run_id, step) from the registry.

        Blob files are not deleted immediately — orphaned blobs are swept by gc().
        Does nothing if the checkpoint does not exist.
        """
        self._registry.delete_checkpoint(self._run_id, step)

    def gc(self, grace_period_hours: int = 24) -> GCReport:
        """Sweep orphaned blobs older than grace_period_hours.

        Deletes blob rows from the registry and removes the corresponding
        .chunk files from the filesystem.
        """
        report = self._registry.gc(grace_period_hours=grace_period_hours)
        for h in report.swept_hashes:
            self._cas.delete_chunk(h)
        return report

    def stats(self) -> Dict[str, Any]:
        """Return storage statistics for this run.

        Walks all manifests for the run to compute chunk-level dedup metrics.
        stats is run-scoped — it reports only on this run_id. For store-wide
        dedup across all runs, unique chunks from all runs would need to be
        aggregated separately.
        """
        steps = self._registry.list_checkpoints(self._run_id)
        total_chunks = 0
        unique_chunks: set = set()
        total_bytes = 0

        for step in steps:
            path = self._registry.get_manifest_path(self._run_id, step)
            if path and path.exists():
                manifest = ManifestReader.read(path)
                for record in manifest.tensors.values():
                    total_bytes += record.byte_length
                    for ref in record.chunks:
                        total_chunks += 1
                        unique_chunks.add(ref.hash)

        dedup_ratio = len(unique_chunks) / total_chunks if total_chunks else 1.0
        return {
            "run_id": self._run_id,
            "checkpoints": len(steps),
            "steps": steps,
            "total_chunks": total_chunks,
            "unique_chunks": len(unique_chunks),
            "dedup_ratio": round(dedup_ratio, 4),
            "total_bytes": total_bytes,
        }

    @classmethod
    def store_stats(cls, root: Path) -> Dict[str, Any]:
        """Return store-wide statistics across all runs under root.

        Walks every manifest in the registry to compute unique chunk counts
        across all runs. This is the number that shows cross-run dedup value:
        how many logical chunks exist vs how many are actually stored on disk.
        """
        root = Path(root)
        registry = Registry.from_path(root / "registry.db")

        runs = registry.list_runs()
        total_chunks = 0
        unique_chunks: set = set()
        total_bytes = 0
        total_checkpoints = 0

        for run_id in runs:
            for step in registry.list_checkpoints(run_id):
                path = registry.get_manifest_path(run_id, step)
                if path and path.exists():
                    manifest = ManifestReader.read(path)
                    total_checkpoints += 1
                    for record in manifest.tensors.values():
                        total_bytes += record.byte_length
                        for ref in record.chunks:
                            total_chunks += 1
                            unique_chunks.add(ref.hash)

        dedup_ratio = len(unique_chunks) / total_chunks if total_chunks else 1.0
        return {
            "runs": len(runs),
            "checkpoints": total_checkpoints,
            "total_chunks": total_chunks,
            "unique_chunks": len(unique_chunks),
            "dedup_ratio": round(dedup_ratio, 4),
            "total_bytes": total_bytes,
        }

    def __enter__(self) -> "tensorcasStore":
        return self

    def __exit__(self, *_) -> None:
        self._shutdown()

    def __del__(self) -> None:
        self._shutdown()

    def _shutdown(self) -> None:
        try:
            self._cas.shutdown(wait=False)
        except Exception as exc:
            warnings.warn(f"CASEngine shutdown failed: {exc}", stacklevel=2)
