import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import msgspec

from tensorcas._version import __version__
from tensorcas.errors import CorruptManifestError
from tensorcas.models import TensorArrayRecord

if TYPE_CHECKING:
    from tensorcas.cas.backend import CASBackend


class CheckpointManifest(msgspec.Struct, frozen=True):
    chunk_size: int
    run_id: str
    step: int
    created_at: datetime
    tensors: Dict[str, TensorArrayRecord]
    format_version: int = 1
    tensorcas_version: str = __version__
    metrics: Dict[str, float] = {}

    def __repr__(self) -> str:
        return (
            f"CheckpointManifest(run_id={self.run_id!r}, step={self.step}, "
            f"tensors={list(self.tensors)}, created_at={self.created_at.isoformat()})"
        )


_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(CheckpointManifest)


class ManifestWriter:
    @staticmethod
    def write(manifest: CheckpointManifest, path: Path) -> None:
        """Write manifest to path atomically (temp file + rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _encoder.encode(manifest)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class ManifestReader:
    @staticmethod
    def read(path: Path) -> CheckpointManifest:
        """Read and parse a manifest file. Raises CorruptManifestError on failure."""
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise CorruptManifestError(f"Manifest not found: {path}")
        except OSError as e:
            raise CorruptManifestError(f"Could not read manifest {path}: {e}")

        if not raw:
            raise CorruptManifestError(f"Manifest is empty: {path}")

        try:
            return _decoder.decode(raw)
        except msgspec.DecodeError as e:
            raise CorruptManifestError(
                f"Manifest schema validation failed ({path}): {e}"
            )

    @staticmethod
    def validate(manifest: CheckpointManifest, cas_backend: "CASBackend") -> None:
        """Verify all ChunkRef hashes exist in the CAS backend.

        Raises CorruptManifestError listing any missing hashes.
        """
        all_hashes = [
            ref.hash for record in manifest.tensors.values() for ref in record.chunks
        ]
        if not all_hashes:
            return

        presence = cas_backend.batch_has(all_hashes)
        missing = [h for h, exists in presence.items() if not exists]
        if missing:
            suffix = f" ... and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise CorruptManifestError(
                f"Manifest for run={manifest.run_id} step={manifest.step} "
                f"references {len(missing)} missing chunk(s): {missing[:5]}{suffix}"
            )
