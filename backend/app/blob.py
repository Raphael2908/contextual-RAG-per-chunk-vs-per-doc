"""Original-file storage, behind a Protocol with a filesystem impl + fake.

The filesystem impl writes into a path backed by a docker volume — no S3/Supabase
in this slice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...

    def get(self, key: str) -> bytes: ...


class FilesystemBlobStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("..", "_")
        return self.root / safe

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()


class InMemoryBlobStore:
    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        self._data[key] = data
        return f"mem://{key}"

    def get(self, key: str) -> bytes:
        return self._data[key]
