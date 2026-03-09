"""Evaluation metrics used by the bundled benchmark harness."""

from __future__ import annotations

import math
from typing import Iterable

from raglab.domain.models import AnswerResult, SearchHit


def recall_at_k(relevant_doc_ids: set[str], hits: Iterable[SearchHit], k: int) -> float:
    top = [hit.chunk.doc_id for hit in list(hits)[:k]]
    if not relevant_doc_ids:
        return 0.0
    return len(relevant_doc_ids & set(top)) / len(relevant_doc_ids)


def precision_at_k(relevant_doc_ids: set[str], hits: Iterable[SearchHit], k: int) -> float:
    top = [hit.chunk.doc_id for hit in list(hits)[:k]]
    if not top:
        return 0.0
    return len(relevant_doc_ids & set(top)) / len(top)


def mrr(relevant_doc_ids: set[str], hits: Iterable[SearchHit]) -> float:
    for index, hit in enumerate(hits, start=1):
        if hit.chunk.doc_id in relevant_doc_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance_by_doc: dict[str, int], hits: Iterable[SearchHit], k: int) -> float:
    dcg = 0.0
    ranked = list(hits)[:k]
    seen_docs: set[str] = set()
    for index, hit in enumerate(ranked, start=1):
        if hit.chunk.doc_id in seen_docs:
            rel = 0
        else:
            rel = relevance_by_doc.get(hit.chunk.doc_id, 0)
            seen_docs.add(hit.chunk.doc_id)
        dcg += (2**rel - 1) / math.log2(index + 1)
    ideal_rels = sorted(relevance_by_doc.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(index + 1) for index, rel in enumerate(ideal_rels, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def groundedness(result: AnswerResult) -> float:
    if not result.claims:
        return 0.0
    return sum(1 for claim in result.claims if claim.supported) / len(result.claims)


def clarification_precision(results: list[dict[str, bool]]) -> float:
    asked = [row for row in results if row["asked"]]
    if not asked:
        return 0.0
    return sum(1 for row in asked if row["needed"]) / len(asked)


def clarification_recall(results: list[dict[str, bool]]) -> float:
    needed = [row for row in results if row["needed"]]
    if not needed:
        return 0.0
    return sum(1 for row in needed if row["asked"]) / len(needed)


def retry_gain(before_scores: list[float], after_scores: list[float]) -> float:
    if not before_scores or not after_scores:
        return 0.0
    paired = zip(before_scores, after_scores)
    deltas = [after - before for before, after in paired]
    return sum(deltas) / max(1, len(deltas))
