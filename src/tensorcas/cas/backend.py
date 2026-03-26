from typing import Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CASBackend(Protocol):
    def has(self, hash: str) -> bool:
        """Return True if the chunk identified by hash exists in the store."""
        ...

    def put(self, hash: str, data: bytes) -> None:
        """Store data under hash. No-op if hash already exists."""
        ...

    def get(self, hash: str) -> bytes:
        """Return raw (uncompressed) bytes for hash. Raises StorageError if missing."""
        ...

    def batch_has(self, hashes: List[str]) -> Dict[str, bool]:
        """Return presence map for a list of hashes."""
        ...

    def delete(self, hash: str) -> None:
        """Delete the chunk identified by hash. No-op if not present."""
        ...
