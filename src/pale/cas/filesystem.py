import os
import tempfile
from pathlib import Path
from typing import Dict, List

import zstandard as zstd

from pale.cas.backend import CASBackend
from pale.errors import CorruptChunkError, StorageError

_ZSTD_LEVEL = 3
_cctx = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
_dctx = zstd.ZstdDecompressor()


class FilesystemBackend(CASBackend):
    """CAS backend that stores chunks on the local filesystem.

    Layout: {root}/objects/{hash[:2]}/{hash[2:4]}/{hash[4:]}.chunk
    Three-level sharding handles millions of chunks without directory pressure.

    put  — compresses with zstd level 3, writes atomically (temp + rename)
    get  — reads and decompresses; callers always see raw uncompressed bytes
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _chunk_path(self, hash: str) -> Path:
        return self._root / "objects" / hash[:2] / hash[2:4] / (hash[4:] + ".chunk")

    def has(self, hash: str) -> bool:
        return self._chunk_path(hash).exists()

    def put(self, hash: str, data: bytes) -> None:
        """Compress data and write atomically. No-op if hash already exists."""
        path = self._chunk_path(hash)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        compressed = _cctx.compress(data)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(compressed)
            os.replace(tmp, path)
        except Exception as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise StorageError(f"Failed to write chunk {hash}: {e}") from e

    def get(self, hash: str) -> bytes:
        """Return raw (decompressed) bytes for hash."""
        path = self._chunk_path(hash)
        try:
            compressed = path.read_bytes()
        except FileNotFoundError:
            raise StorageError(f"Chunk not found: {hash}")
        except OSError as e:
            raise StorageError(f"Failed to read chunk {hash}: {e}") from e
        try:
            return _dctx.decompress(compressed)
        except Exception as e:
            raise CorruptChunkError(
                f"Chunk decompression failed for {hash}: {e}"
            ) from e

    def batch_has(self, hashes: List[str]) -> Dict[str, bool]:
        """Check existence of multiple hashes."""
        return {h: self._chunk_path(h).exists() for h in hashes}

    def delete(self, hash: str) -> None:
        """Delete chunk file. No-op if not present."""
        path = self._chunk_path(hash)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            raise StorageError(f"Failed to delete chunk {hash}: {e}") from e
