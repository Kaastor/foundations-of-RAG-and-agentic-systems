"""Heuristic trust and injection checks used during ingestion and retrieval."""

from __future__ import annotations

import re
from typing import Iterable


_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all|any|the)? ?previous instructions", re.IGNORECASE),
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"\breveal\b.*\binternal\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b", re.IGNORECASE),
]


def instruction_like_language(text: str) -> list[str]:
    """Return matched suspicious fragments from untrusted content."""
    matches: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        hit = pattern.search(text)
        if hit:
            matches.append(hit.group(0))
    return matches


def trust_score(text: str, metadata_title: str = "") -> float:
    """Score content trust using a few obvious signals.

    This is intentionally lightweight. The repository demonstrates the control
    surface, not a full security product.
    """
    suspicious = len(instruction_like_language(text))
    uppercase_title_bonus = 0.05 if metadata_title.isupper() and len(metadata_title) > 10 else 0.0
    score = 1.0 - suspicious * 0.35 - uppercase_title_bonus
    return max(0.0, min(1.0, score))


def quality_score(text: str, title: str = "", ocr_confidence: float | None = None) -> float:
    """Estimate a coarse document quality score."""
    score = 1.0
    if not title.strip():
        score -= 0.15
    if len(text.strip()) < 80:
        score -= 0.20
    weird_ratio = sum(1 for char in text if char in {"�", "□", "¤"}) / max(1, len(text))
    score -= min(0.30, weird_ratio * 5.0)
    if ocr_confidence is not None and ocr_confidence < 0.75:
        score -= 0.35
    return max(0.0, min(1.0, score))


def should_quarantine(score: float, trust: float, reasons: Iterable[str]) -> bool:
    """Decide whether a document should stay out of the live index."""
    return score < 0.45 or trust < 0.40 or any(reason.startswith("suspicious") for reason in reasons)
