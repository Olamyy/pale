from typing import List

import msgspec

from pale.errors import CorruptManifestError


class ChunkRef(msgspec.Struct, frozen=True):
    hash: str
    size: int


class TensorArrayRecord(msgspec.Struct, frozen=True):
    name: str
    dtype: str
    shape: List[int]
    byte_length: int
    full_hash: str
    chunks: List[ChunkRef]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise CorruptManifestError(f"TensorArrayRecord '{self.name}' has no chunks")

    def validate_chunk_refs(self) -> None:
        """Assert sum(ref.size) == byte_length. Raises CorruptManifestError if not."""
        total = sum(ref.size for ref in self.chunks)
        if total != self.byte_length:
            raise CorruptManifestError(
                f"TensorArrayRecord '{self.name}': sum of chunk sizes {total} "
                f"!= byte_length {self.byte_length}"
            )
