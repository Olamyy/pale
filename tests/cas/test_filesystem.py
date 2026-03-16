import os

import pytest

from pale.cas.filesystem import FilesystemBackend
from pale.errors import StorageError


def test_put_is_idempotent(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.put("h1", b"first")
    b.put("h1", b"second")
    assert b.get("h1") == b"first"


def test_three_level_sharding(tmp_path):
    b = FilesystemBackend(tmp_path)
    h = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
    b.put(h, b"data")
    assert (tmp_path / "objects" / h[:2] / h[2:4] / (h[4:] + ".chunk")).exists()


def test_get_missing_raises_storage_error(tmp_path):
    b = FilesystemBackend(tmp_path)
    with pytest.raises(StorageError):
        b.get("nonexistent")


def test_atomicity_no_partial_on_failure(tmp_path, monkeypatch):
    b = FilesystemBackend(tmp_path)
    h = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(StorageError):
        b.put(h, b"data")
    assert not (tmp_path / "objects" / h[:2] / h[2:4] / (h[4:] + ".chunk")).exists()


def test_compression_transparent(tmp_path):
    b = FilesystemBackend(tmp_path)
    data = bytes(range(256)) * 100
    b.put("datahash", data)
    assert b.get("datahash") == data
