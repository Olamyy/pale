import blake3


def hash_chunk(raw_bytes: bytes) -> str:
    """Return the BLAKE3 hex digest of raw_bytes.

    Always hashes raw uncompressed bytes — the hash is the chunk's identity
    independent of how it is stored (compression level, version, backend).
    """
    return blake3.blake3(raw_bytes).hexdigest()
