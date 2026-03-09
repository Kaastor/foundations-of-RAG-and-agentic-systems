"""Application-wide defaults and small helpers for CLI configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppConfig:
    """Configurable defaults for the pedagogical system.

    The values are intentionally conservative and easy to inspect. They are not
    meant to be optimal beyond the bundled demo corpus.
    """

    chunk_tokens: int = 120
    chunk_overlap: int = 24
    sparse_k1: float = 1.4
    sparse_b: float = 0.75
    vector_dims: int = 192
    ann_tables: int = 4
    ann_bits_per_table: int = 10
    ann_probe_tables: int = 2
    default_top_k: int = 8
    default_rerank_top_m: int = 5
    default_context_budget: int = 420
    max_agent_steps: int = 6
    max_retries: int = 2
    low_quality_threshold: float = 0.45
    low_trust_threshold: float = 0.40
    cache_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return asdict(self)


DEFAULT_CONFIG = AppConfig()


def workspace_path(value: str | Path) -> Path:
    """Resolve a workspace path without creating it yet."""
    return Path(value).expanduser().resolve()
