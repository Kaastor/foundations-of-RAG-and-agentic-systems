"""Small text utilities shared across ingestion, retrieval, and generation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_IDENTIFIER_RE = re.compile(r"\b(?:[A-Z]{1,5}-\d{1,4}|[A-Z]\d{1,3}|V\d{1,3}|X\d{1,3}|IR-\d{1,4}|HMX-\d{1,4}|FW-\d\.\d)\b")


def normalize_whitespace(text: str) -> str:
    """Collapse most layout noise while preserving paragraph breaks."""
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_query(text: str) -> str:
    """Normalize a user query while preserving identifiers."""
    return normalize_whitespace(text)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms while preserving identifier-like strings."""
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def identifiers(text: str) -> list[str]:
    """Extract likely exact identifiers such as bulletin IDs and model names."""
    return [match.group(0) for match in _IDENTIFIER_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Split prose into rough sentence units.

    The implementation is intentionally simple and deterministic.
    """
    parts = _SENTENCE_RE.split(normalize_whitespace(text))
    return [part.strip() for part in parts if part.strip()]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Return character n-grams for vector hashing."""
    lowered = f"  {normalize_whitespace(text).lower()}  "
    return [lowered[index : index + n] for index in range(max(0, len(lowered) - n + 1))]


def term_frequency(tokens: Iterable[str]) -> Counter[str]:
    """Count terms from an iterable."""
    return Counter(tokens)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between dense vectors."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def value_density(score: float, token_cost: int) -> float:
    """A tiny helper for budget-aware packing."""
    return score / max(1, token_cost)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity for duplicate detection."""
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))
