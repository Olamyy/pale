import os
from pathlib import Path

import pytest

from pale.cas.filesystem import FilesystemBackend
from pale.errors import StorageError


def test_store_retrieve_round_trip(tmp_path):
    b = FilesystemBackend(tmp_path)
    data = b"hello pale"
    b.put("abc123", data)
    assert b.get("abc123") == data


def test_has_returns_false_before_put(tmp_path):
    b = FilesystemBackend(tmp_path)
    assert not b.has("deadbeef")


def test_has_returns_true_after_put(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.put("aabbcc", b"data")
    assert b.has("aabbcc")


def test_put_is_idempotent(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.put("h1", b"first")
    b.put("h1", b"second")
    assert b.get("h1") == b"first"


def test_three_level_sharding(tmp_path):
    b = FilesystemBackend(tmp_path)
    h = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
    b.put(h, b"data")
    expected = tmp_path / "objects" / h[:2] / h[2:4] / (h[4:] + ".chunk")
    assert expected.exists()


def test_get_missing_raises_storage_error(tmp_path):
    b = FilesystemBackend(tmp_path)
    with pytest.raises(StorageError):
        b.get("nonexistent")


def test_atomicity_no_partial_on_failure(tmp_path, monkeypatch):
    b = FilesystemBackend(tmp_path)
    h = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"

    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(StorageError):
        b.put(h, b"data")

    path = tmp_path / "objects" / h[:2] / h[2:4] / (h[4:] + ".chunk")
    assert not path.exists()


def test_batch_has_correctness(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.put("h1", b"a")
    b.put("h2", b"b")
    result = b.batch_has(["h1", "h2", "h3"])
    assert result == {"h1": True, "h2": True, "h3": False}


def test_batch_has_empty(tmp_path):
    b = FilesystemBackend(tmp_path)
    assert b.batch_has([]) == {}


def test_delete_removes_file(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.put("h1", b"data")
    b.delete("h1")
    assert not b.has("h1")


def test_delete_missing_is_noop(tmp_path):
    b = FilesystemBackend(tmp_path)
    b.delete("nonexistent")


def test_compression_transparent(tmp_path):
    """Raw bytes returned by get() must equal what was passed to put()."""
    b = FilesystemBackend(tmp_path)
    data = bytes(range(256)) * 100
    b.put("datahash", data)
    assert b.get("datahash") == data
