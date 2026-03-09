"""A tiny file-based cache for repeated retrieval and answer work."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from raglab.storage.json_store import ensure_dir, read_json, write_json


class FileCache:
    """Very small file-based cache keyed by stable request attributes."""

    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def _key_path(self, namespace: str, payload: str) -> Path:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, payload: str) -> dict[str, Any] | None:
        path = self._key_path(namespace, payload)
        if not path.exists():
            return None
        return read_json(path)

    def set(self, namespace: str, payload: str, value: dict[str, Any]) -> None:
        path = self._key_path(namespace, payload)
        write_json(path, value)
