from typing import Dict, List

from pale.cas.backend import CASBackend
from pale.cas.filesystem import FilesystemBackend


class DictBackend:
    """In-memory CAS backend for testing."""

    def __init__(self):
        self._store: Dict[str, bytes] = {}

    def has(self, hash: str) -> bool:
        return hash in self._store

    def put(self, hash: str, data: bytes) -> None:
        self._store[hash] = data

    def get(self, hash: str) -> bytes:
        return self._store[hash]

    def batch_has(self, hashes: List[str]) -> Dict[str, bool]:
        return {h: h in self._store for h in hashes}

    def delete(self, hash: str) -> None:
        self._store.pop(hash, None)


def test_dict_backend_satisfies_protocol():
    assert isinstance(DictBackend(), CASBackend)


def test_filesystem_backend_satisfies_protocol(tmp_path):
    assert isinstance(FilesystemBackend(tmp_path), CASBackend)


def test_dict_backend_put_get():
    b = DictBackend()
    b.put("abc", b"hello")
    assert b.get("abc") == b"hello"


def test_dict_backend_has():
    b = DictBackend()
    assert not b.has("xyz")
    b.put("xyz", b"data")
    assert b.has("xyz")


def test_dict_backend_batch_has():
    b = DictBackend()
    b.put("h1", b"a")
    result = b.batch_has(["h1", "h2"])
    assert result == {"h1": True, "h2": False}


def test_dict_backend_delete():
    b = DictBackend()
    b.put("h1", b"a")
    b.delete("h1")
    assert not b.has("h1")
    b.delete("h1")
