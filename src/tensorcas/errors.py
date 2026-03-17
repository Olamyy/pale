class tensorcasError(Exception):
    """Base class for all tensorcas exceptions."""


class StorageError(tensorcasError):
    """Raised when a CAS read or write operation fails."""


class CorruptChunkError(tensorcasError):
    """Raised when a fetched chunk's BLAKE3 hash does not match the recorded hash."""


class CorruptManifestError(tensorcasError):
    """Raised when a manifest is missing, incomplete, or internally inconsistent."""


class CheckpointNotFoundError(tensorcasError):
    """Raised when a requested (run_id, step) does not exist in the registry."""


class AdapterError(tensorcasError):
    """Raised when a ModelAdapter cannot extract or reconstruct a model."""


class CheckpointAlreadyExistsError(tensorcasError):
    """Raised when attempting to register a (run_id, step) that already exists."""
