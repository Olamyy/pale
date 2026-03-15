class PaleError(Exception):
    """Base class for all pale exceptions."""


class StorageError(PaleError):
    """Raised when a CAS read or write operation fails."""


class CorruptChunkError(PaleError):
    """Raised when a fetched chunk's BLAKE3 hash does not match the recorded hash."""


class CorruptManifestError(PaleError):
    """Raised when a manifest is missing, incomplete, or internally inconsistent."""


class CheckpointNotFoundError(PaleError):
    """Raised when a requested (run_id, step) does not exist in the registry."""


class AdapterError(PaleError):
    """Raised when a ModelAdapter cannot extract or reconstruct a model."""


class CheckpointAlreadyExistsError(PaleError):
    """Raised when attempting to register a (run_id, step) that already exists."""
