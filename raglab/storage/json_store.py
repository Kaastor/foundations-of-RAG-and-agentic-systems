"""JSON and JSONL helpers used throughout the repository."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Any


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.exists():
        if default is not None:
            return default
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(target.parent), encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(target)


def copy_tree(source: str | Path, destination: str | Path) -> None:
    src = Path(source)
    dst = Path(destination)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def remove_path(path: str | Path) -> None:
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
