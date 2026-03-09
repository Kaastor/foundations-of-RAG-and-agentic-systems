"""Workspace layout helpers for staging, snapshots, and publishing."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from raglab.storage.json_store import ensure_dir, read_json, write_json


def workspace_dirs(workspace: Path) -> dict[str, Path]:
    """Return the canonical workspace directories."""
    workspace = workspace.resolve()
    return {
        "root": workspace,
        "staged": workspace / "staged",
        "snapshots": workspace / "snapshots",
        "traces": workspace / "traces",
        "cache": workspace / "cache",
        "sessions": workspace / "sessions",
        "live_pointer": workspace / "live.json",
    }


def init_workspace(workspace: Path) -> dict[str, Path]:
    """Create the expected directory structure and return it."""
    dirs = workspace_dirs(workspace)
    for key, path in dirs.items():
        if key != "live_pointer":
            ensure_dir(path)
    return dirs


def live_snapshot_path(workspace: Path) -> Path:
    """Resolve the currently published live snapshot."""
    dirs = init_workspace(workspace)
    pointer = dirs["live_pointer"]
    if not pointer.exists():
        raise FileNotFoundError("No live snapshot has been published yet.")
    payload = read_json(pointer)
    return dirs["snapshots"] / payload["snapshot_id"]


def staged_snapshot_path(workspace: Path) -> Path:
    """Return the mutable staged snapshot directory."""
    return init_workspace(workspace)["staged"]


def publish_staged_snapshot(workspace: Path, note: str = "") -> Path:
    """Promote the staged directory into an immutable snapshot and point live to it."""
    dirs = init_workspace(workspace)
    staged = dirs["staged"]
    if not staged.exists():
        raise FileNotFoundError("Staged workspace does not exist.")
    manifest = staged / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Staged manifest.json is missing; run ingest and index first.")
    snapshot_id = f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}"
    destination = dirs["snapshots"] / snapshot_id
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(staged, destination)
    write_json(dirs["live_pointer"], {"snapshot_id": snapshot_id, "note": note})
    return destination


def list_snapshots(workspace: Path) -> list[dict[str, Any]]:
    """List published snapshots and identify the live one."""
    dirs = init_workspace(workspace)
    live_id = None
    if dirs["live_pointer"].exists():
        live_id = read_json(dirs["live_pointer"]).get("snapshot_id")
    snapshots: list[dict[str, Any]] = []
    for path in sorted(dirs["snapshots"].iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            manifest = {"snapshot_id": path.name}
        snapshots.append(
            {
                "snapshot_id": path.name,
                "live": path.name == live_id,
                "created_at": manifest.get("created_at", ""),
                "document_count": manifest.get("document_count", 0),
                "chunk_count": manifest.get("chunk_count", 0),
            }
        )
    return snapshots
