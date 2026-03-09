"""Build helpers that sit between ingestion and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.config import AppConfig
from raglab.domain.models import ChunkRecord
from raglab.ops.publish import staged_snapshot_path
from raglab.retrieval.indexes import build_all_indexes
from raglab.storage.json_store import read_json, read_jsonl, write_json


def load_staged_manifest(workspace: Path) -> dict[str, Any]:
    """Load the mutable staged manifest."""
    return read_json(staged_snapshot_path(workspace) / "manifest.json")


def load_staged_chunks(workspace: Path) -> list[ChunkRecord]:
    """Load staged chunks as dataclasses."""
    rows = read_jsonl(staged_snapshot_path(workspace) / "chunks.jsonl")
    return [ChunkRecord.from_dict(row) for row in rows]


def index_staged_snapshot(workspace: Path, config: AppConfig | None = None) -> dict[str, Any]:
    """Build indexes inside the staged workspace."""
    staged = staged_snapshot_path(workspace)
    manifest = read_json(staged / "manifest.json")
    config = config or AppConfig(**manifest["config"])
    chunks = load_staged_chunks(workspace)
    summary = build_all_indexes(chunks, config=config, output_dir=staged / "indexes")
    manifest["index_summary"] = summary
    write_json(staged / "manifest.json", manifest)
    return summary
