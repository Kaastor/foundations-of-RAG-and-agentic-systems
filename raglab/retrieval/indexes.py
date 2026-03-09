"""Sparse, dense, and approximate vector indexes."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from raglab.config import AppConfig
from raglab.domain.models import ChunkRecord
from raglab.storage.json_store import read_json, write_json
from raglab.text import char_ngrams, cosine_similarity, term_frequency, tokenize


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


@dataclass(slots=True)
class SparseSearchResult:
    chunk_id: str
    score: float


class SparseBM25Index:
    """A tiny BM25 index with enough structure to be teachable."""

    def __init__(self, postings: dict[str, dict[str, int]], doc_lengths: dict[str, int], avgdl: float, chunk_titles: dict[str, str]) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.avgdl = avgdl
        self.chunk_titles = chunk_titles
        self.doc_freq = {term: len(postings_for_term) for term, postings_for_term in postings.items()}

    @classmethod
    def build(cls, chunks: Iterable[ChunkRecord]) -> "SparseBM25Index":
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        doc_lengths: dict[str, int] = {}
        titles: dict[str, str] = {}
        for chunk in chunks:
            tokens = tokenize(f"{chunk.title} {chunk.text}")
            counts = term_frequency(tokens)
            doc_lengths[chunk.chunk_id] = len(tokens)
            titles[chunk.chunk_id] = chunk.title
            for term, count in counts.items():
                postings[term][chunk.chunk_id] = count
        avgdl = sum(doc_lengths.values()) / max(1, len(doc_lengths))
        return cls(dict(postings), doc_lengths, avgdl, titles)

    def search(self, query: str, top_k: int, k1: float, b: float, eligible_ids: set[str] | None = None) -> list[SparseSearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores: Counter[str] = Counter()
        total_docs = max(1, len(self.doc_lengths))
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (total_docs - self.doc_freq.get(term, 0) + 0.5) / (self.doc_freq.get(term, 0) + 0.5))
            for chunk_id, tf in postings.items():
                if eligible_ids is not None and chunk_id not in eligible_ids:
                    continue
                dl = self.doc_lengths.get(chunk_id, 0)
                denom = tf + k1 * (1 - b + b * dl / max(1.0, self.avgdl))
                scores[chunk_id] += idf * (tf * (k1 + 1)) / max(1e-9, denom)
        results = [SparseSearchResult(chunk_id=chunk_id, score=score) for chunk_id, score in scores.most_common(top_k)]
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "chunk_titles": self.chunk_titles,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseBM25Index":
        return cls(
            postings={term: {chunk_id: int(count) for chunk_id, count in postings.items()} for term, postings in payload["postings"].items()},
            doc_lengths={chunk_id: int(length) for chunk_id, length in payload["doc_lengths"].items()},
            avgdl=float(payload["avgdl"]),
            chunk_titles=payload.get("chunk_titles", {}),
        )


class HashingEmbedder:
    """A deterministic hashed vectorizer used as an educational stand-in for neural embeddings.

    The book's chapters on embeddings and dense retrieval are mapped onto this
    class because the repository intentionally avoids third-party ML runtime
    dependencies. The implementation is honest about being a pedagogical
    substitute.
    """

    def __init__(self, dims: int = 192) -> None:
        self.dims = dims

    def encode(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dims)]
        features = tokenize(text) + char_ngrams(text, n=3)
        if not features:
            return vector
        counts = term_frequency(features)
        for feature, count in counts.items():
            hashed = _stable_hash(feature)
            index = hashed % self.dims
            sign = -1.0 if (hashed >> 8) & 1 else 1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class DenseVectorIndex:
    """Exact cosine search over hashed vectors."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: Iterable[ChunkRecord], embedder: HashingEmbedder) -> "DenseVectorIndex":
        return cls(vectors={chunk.chunk_id: embedder.encode(f"{chunk.title} {chunk.text}") for chunk in chunks})

    def search(self, query: str, embedder: HashingEmbedder, top_k: int, eligible_ids: set[str] | None = None) -> list[tuple[str, float]]:
        query_vec = embedder.encode(query)
        scored = []
        for chunk_id, vector in self.vectors.items():
            if eligible_ids is not None and chunk_id not in eligible_ids:
                continue
            scored.append((chunk_id, cosine_similarity(query_vec, vector)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {"vectors": self.vectors}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseVectorIndex":
        return cls(vectors={chunk_id: [float(value) for value in vector] for chunk_id, vector in payload["vectors"].items()})


class AnnLSHIndex:
    """A small approximate nearest-neighbor index using random-hyperplane LSH."""

    def __init__(self, dims: int, tables: int, bits_per_table: int, hyperplanes: list[list[list[float]]], buckets: list[dict[str, list[str]]]) -> None:
        self.dims = dims
        self.tables = tables
        self.bits_per_table = bits_per_table
        self.hyperplanes = hyperplanes
        self.buckets = buckets

    @classmethod
    def build(cls, vectors: dict[str, list[float]], dims: int, tables: int, bits_per_table: int, seed: int = 7) -> "AnnLSHIndex":
        rng = random.Random(seed)
        hyperplanes: list[list[list[float]]] = []
        for _ in range(tables):
            table_planes: list[list[float]] = []
            for _ in range(bits_per_table):
                plane = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
                table_planes.append(plane)
            hyperplanes.append(table_planes)

        buckets: list[dict[str, list[str]]] = [defaultdict(list) for _ in range(tables)]
        index = cls(dims=dims, tables=tables, bits_per_table=bits_per_table, hyperplanes=hyperplanes, buckets=[{} for _ in range(tables)])
        for chunk_id, vector in vectors.items():
            for table_index in range(tables):
                key = index.signature(vector, table_index)
                bucket = buckets[table_index]
                bucket[key].append(chunk_id)

        finalized = [dict(table) for table in buckets]
        return cls(dims=dims, tables=tables, bits_per_table=bits_per_table, hyperplanes=hyperplanes, buckets=finalized)

    def signature(self, vector: list[float], table_index: int) -> str:
        bits: list[str] = []
        for plane in self.hyperplanes[table_index]:
            dot = sum(a * b for a, b in zip(vector, plane))
            bits.append("1" if dot >= 0 else "0")
        return "".join(bits)

    def candidates(self, vector: list[float], probe_tables: int) -> set[str]:
        candidate_ids: set[str] = set()
        for table_index in range(min(self.tables, probe_tables)):
            key = self.signature(vector, table_index)
            candidate_ids.update(self.buckets[table_index].get(key, []))
        return candidate_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "dims": self.dims,
            "tables": self.tables,
            "bits_per_table": self.bits_per_table,
            "hyperplanes": self.hyperplanes,
            "buckets": self.buckets,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnnLSHIndex":
        return cls(
            dims=int(payload["dims"]),
            tables=int(payload["tables"]),
            bits_per_table=int(payload["bits_per_table"]),
            hyperplanes=[[[float(value) for value in plane] for plane in table] for table in payload["hyperplanes"]],
            buckets=[{key: list(values) for key, values in table.items()} for table in payload["buckets"]],
        )


def build_all_indexes(chunks: list[ChunkRecord], config: AppConfig, output_dir: Path) -> dict[str, Any]:
    """Build sparse, dense, and approximate indexes into one directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse = SparseBM25Index.build(chunks)
    embedder = HashingEmbedder(config.vector_dims)
    dense = DenseVectorIndex.build(chunks, embedder)
    ann = AnnLSHIndex.build(dense.vectors, dims=config.vector_dims, tables=config.ann_tables, bits_per_table=config.ann_bits_per_table)
    write_json(output_dir / "sparse.json", sparse.to_dict())
    write_json(output_dir / "dense.json", dense.to_dict())
    write_json(output_dir / "ann.json", ann.to_dict())
    write_json(output_dir / "vectorizer.json", {"dims": config.vector_dims})
    return {
        "sparse_terms": len(sparse.postings),
        "dense_vectors": len(dense.vectors),
        "ann_tables": config.ann_tables,
    }


def load_indexes(index_dir: Path) -> tuple[SparseBM25Index, DenseVectorIndex, AnnLSHIndex, HashingEmbedder]:
    """Load all built indexes from disk."""
    sparse = SparseBM25Index.from_dict(read_json(index_dir / "sparse.json"))
    dense = DenseVectorIndex.from_dict(read_json(index_dir / "dense.json"))
    ann = AnnLSHIndex.from_dict(read_json(index_dir / "ann.json"))
    vectorizer_payload = read_json(index_dir / "vectorizer.json")
    embedder = HashingEmbedder(int(vectorizer_payload["dims"]))
    return sparse, dense, ann, embedder
