"""Grounded answer synthesis from packed evidence."""

from __future__ import annotations

import re
from typing import Iterable

from raglab.domain.models import AnswerResult, Claim, QueryIntent, SearchHit
from raglab.generation.verify import verify_answer
from raglab.retrieval.engine import PackedContext
from raglab.text import identifiers, split_sentences, tokenize


def _best_evidence_sentences(query: str, evidence: Iterable[SearchHit], limit: int = 3) -> list[str]:
    query_tokens = set(tokenize(query))
    scored: list[tuple[float, str]] = []
    for hit in evidence:
        for sentence in split_sentences(hit.chunk.text):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_tokens & sentence_tokens)
            identifier_bonus = sum(0.5 for value in identifiers(query) if value.lower() in sentence.lower())
            scored.append((overlap + identifier_bonus, sentence.strip()))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    for _score, sentence in scored:
        if sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) >= limit:
            break
    return chosen


def _comparison_claims(query: str, packed: PackedContext) -> list[Claim]:
    hits = packed.hits
    if not hits:
        return [Claim(text="The system did not retrieve evidence strong enough for a comparison.", supported=False)]
    newest = max(hits, key=lambda hit: hit.chunk.effective_date or "")
    oldest = min(hits, key=lambda hit: hit.chunk.effective_date or "")
    claims = [
        Claim(text=f"Newest evidence: {newest.chunk.title}. {split_sentences(newest.chunk.text)[0]}")
    ]
    if oldest.chunk.doc_id != newest.chunk.doc_id:
        claims.append(Claim(text=f"Older baseline: {oldest.chunk.title}. {split_sentences(oldest.chunk.text)[0]}"))
    for conflict in packed.conflicts:
        claims.append(Claim(text=f"Conflict note: {conflict}", supported=False))
    return claims


def _procedural_claims(query: str, packed: PackedContext) -> list[Claim]:
    hits = packed.hits
    if not hits:
        return [Claim(text="No procedural evidence was retrieved.", supported=False)]
    claims = []
    for sentence in _best_evidence_sentences(query, hits, limit=2):
        claims.append(Claim(text=sentence))
    titles = ", ".join(dict.fromkeys(hit.chunk.title for hit in hits[:2]))
    claims.append(Claim(text=f"Primary sources: {titles}."))
    return claims


def _default_claims(query: str, packed: PackedContext) -> list[Claim]:
    evidence_sentences = _best_evidence_sentences(query, packed.hits, limit=3)
    if not evidence_sentences:
        return [Claim(text="I could not find supporting evidence in the current snapshot.", supported=False)]
    claims = [Claim(text=sentence) for sentence in evidence_sentences[:2]]
    if packed.conflicts:
        claims.append(Claim(text=f"Conflict note: {packed.conflicts[0]}", supported=False))
    return claims


def _shape_answer(claims: list[Claim], query: str, intent: QueryIntent, packed: PackedContext) -> str:
    question = query.strip().rstrip("?")
    if not claims:
        return "No answer was produced."
    if intent.task_type == "procedural":
        lead = f"Grounded answer for: {question}."
    elif intent.task_type == "comparison":
        lead = f"Grounded comparison for: {question}."
    else:
        lead = f"Grounded answer for: {question}."
    body = " ".join(claim.text for claim in claims)
    if packed.conflicts:
        body += " The evidence set includes version tension that the answer keeps visible."
    return f"{lead} {body}".strip()


def synthesize_answer(query: str, packed: PackedContext, intent: QueryIntent, mode: str = "grounded", careful: bool = True) -> AnswerResult:
    """Generate an answer purely from packed evidence.

    The synthesizer is intentionally extractive and conservative. It demonstrates
    grounded answering without pretending to be a full language model.
    """
    if not packed.hits:
        answer = AnswerResult(
            mode=mode,
            query=query,
            answer_text="I could not find supporting material in the current snapshot.",
            claims=[Claim(text="No supporting material found.", supported=False)],
            evidence=[],
            abstained=True,
        )
        return verify_answer(answer)

    if intent.missing_scope:
        clarification = f"More scope is required before answering reliably: {', '.join(intent.missing_scope)}."
        answer = AnswerResult(
            mode=mode,
            query=query,
            answer_text=clarification,
            claims=[Claim(text=clarification, supported=False)],
            evidence=packed.hits,
            abstained=True,
            clarification_request=clarification,
            diagnostics={"conflicts": packed.conflicts},
        )
        return verify_answer(answer)

    if intent.task_type == "comparison":
        claims = _comparison_claims(query, packed)
    elif intent.task_type == "procedural":
        claims = _procedural_claims(query, packed)
    else:
        claims = _default_claims(query, packed)

    if careful and len(packed.conflicts) > 1:
        claims.append(Claim(text="Multiple conflicting sources remain unresolved; a human review may be appropriate.", supported=False))

    answer_text = _shape_answer(claims, query, intent, packed)
    result = AnswerResult(
        mode=mode,
        query=query,
        answer_text=answer_text,
        claims=claims,
        evidence=packed.hits,
        diagnostics={"packed_token_count": packed.token_count, "conflicts": packed.conflicts},
    )
    return verify_answer(result)
