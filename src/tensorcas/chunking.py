from typing import List


def chunk_bytes(data: bytes, chunk_size: int) -> List[bytes]:
    """Split data into fixed-size chunks.

    The final chunk may be smaller than chunk_size (ragged trailing chunk).
    Guaranteed: reassemble_bytes(chunk_bytes(data, n)) == data for any n > 0.
    If data is empty, returns an empty list.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not data:
        return []
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def reassemble_bytes(chunks: List[bytes]) -> bytes:
    """Concatenate chunks back into the original byte sequence."""
    return b"".join(chunks)
