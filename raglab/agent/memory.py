"""Session memory helpers for the controller."""

from __future__ import annotations

from pathlib import Path

from raglab.domain.models import AgentState, MemoryEntry
from raglab.storage.json_store import ensure_dir, read_json, write_json


def load_session(session_dir: Path, session_id: str) -> dict[str, MemoryEntry]:
    """Load a session memory file if it exists."""
    ensure_dir(session_dir)
    path = session_dir / f"{session_id}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {key: MemoryEntry.from_dict(value) for key, value in payload.get("memory", {}).items()}


def save_session(session_dir: Path, session_id: str, state: AgentState) -> Path:
    """Persist current working memory and selected state fields."""
    ensure_dir(session_dir)
    path = session_dir / f"{session_id}.json"
    write_json(
        path,
        {
            "query": state.query,
            "intent": state.intent.to_dict(),
            "memory": {key: value.to_dict() for key, value in state.memory.items()},
            "history": state.history,
            "notes": state.notes,
        },
    )
    return path


def write_memory_entry(state: AgentState, key: str, value: str, source: str, scope: str, confidence: float, expires_with_task: bool = True) -> None:
    """Write a scoped memory entry."""
    state.memory[key] = MemoryEntry(
        key=key,
        value=value,
        source=source,
        scope=scope,
        confidence=confidence,
        expires_with_task=expires_with_task,
    )


def prune_memory_for_new_scope(state: AgentState, keep_scopes: set[str]) -> None:
    """Drop entries whose scopes no longer apply."""
    state.memory = {key: entry for key, entry in state.memory.items() if entry.scope in keep_scopes}
