from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np

from pale.cas.backend import CASBackend
from pale.chunking import chunk_bytes, reassemble_bytes
from pale.errors import CorruptChunkError
from pale.hashing import hash_chunk
from pale.models import ChunkRef, TensorArrayRecord
from pale.serialization import bytes_to_tensor, tensor_to_bytes

_DEFAULT_MAX_WORKERS = 8


class CASEngine:
    """Chunk-level storage and retrieval for numpy arrays.

    Responsibilities:
    - Split arrays into fixed-size chunks
    - Deduplicate via batch_has before writing
    - Parallel put / get via a shared thread pool
    - Hash-verify every chunk on load

    Does NOT know about manifests, checkpoints, or previous steps.
    The no-op fast path (full_hash comparison) lives in StorageEngine.
    """

    def __init__(
        self,
        backend: CASBackend,
        chunk_size: int,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self._backend = backend
        self._chunk_size = chunk_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def backend(self) -> CASBackend:
        return self._backend

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the shared thread pool. Call when the engine is no longer needed."""
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "CASEngine":
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def store_tensor(self, name: str, arr: np.ndarray) -> TensorArrayRecord:
        """Chunk, deduplicate, and store a numpy array.

        Returns a TensorArrayRecord describing the stored chunks.
        Chunks already present in the backend are not re-written.

        NOTE: if any put() raises, partial chunks may be written to the backend
        with no referencing manifest. These are orphaned blobs and will be
        collected by GC's grace-period sweep. This is intentional.
        """
        raw, dtype_str, shape = tensor_to_bytes(arr)
        full_hash = hash_chunk(raw)
        chunks: List[bytes] = chunk_bytes(raw, self._chunk_size)
        hashes: List[str] = [hash_chunk(c) for c in chunks]

        presence = self._backend.batch_has(hashes)

        missing = [(h, c) for h, c in zip(hashes, chunks) if not presence[h]]
        if missing:
            futures = {
                self._executor.submit(self._backend.put, h, c): h for h, c in missing
            }
            for future in as_completed(futures):
                future.result()

        refs = [ChunkRef(hash=h, size=len(c)) for h, c in zip(hashes, chunks)]
        record = TensorArrayRecord(
            name=name,
            dtype=dtype_str,
            shape=shape,
            byte_length=len(raw),
            full_hash=full_hash,
            chunks=refs,
        )
        # Validate before returning — raises if byte_length / chunk sizes are inconsistent.
        # Chunks are already written at this point; a failure here means a logic bug
        # in chunking or serialization, not a storage failure.
        record.validate_chunk_refs()
        return record

    def load_tensor(self, record: TensorArrayRecord) -> np.ndarray:
        """Fetch, verify, and reassemble a numpy array from its TensorArrayRecord.

        Raises CorruptChunkError if any chunk's content doesn't match its hash.
        Chunks are fetched in parallel; reassembly uses positional indexing so
        out-of-order completion is handled correctly.
        """
        n = len(record.chunks)
        raw_chunks: List[bytes] = [b""] * n

        futures = {
            self._executor.submit(self._fetch, i, ref, record): i
            for i, ref in enumerate(record.chunks)
        }
        for future in as_completed(futures):
            idx, data = future.result()
            raw_chunks[idx] = data

        raw = reassemble_bytes(raw_chunks)
        return bytes_to_tensor(raw, record.dtype, record.shape)

    def _fetch(
        self, idx: int, ref: ChunkRef, record: TensorArrayRecord
    ) -> tuple[int, bytes]:
        data = self._backend.get(ref.hash)
        actual = hash_chunk(data)
        if actual != ref.hash:
            raise CorruptChunkError(
                f"Chunk hash mismatch for tensor '{record.name}' "
                f"chunk {idx}: expected {ref.hash}, got {actual}"
            )
        return idx, data
