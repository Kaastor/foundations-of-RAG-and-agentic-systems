"""Support checks for claim-level groundedness."""

from __future__ import annotations

import re
from typing import Iterable

from raglab.domain.models import AnswerResult, Citation, Claim, SearchHit
from raglab.text import split_sentences, tokenize


_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def support_score(claim_text: str, supporting_text: str) -> float:
    """Estimate support with overlap plus exact-number preservation."""
    claim_tokens = set(tokenize(claim_text))
    support_tokens = set(tokenize(supporting_text))
    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & support_tokens) / max(1, len(claim_tokens))
    claim_numbers = set(_NUMBER_RE.findall(claim_text))
    support_numbers = set(_NUMBER_RE.findall(supporting_text))
    number_ratio = 1.0 if not claim_numbers else len(claim_numbers & support_numbers) / len(claim_numbers)
    return 0.7 * overlap + 0.3 * number_ratio


def choose_citation_for_claim(claim_text: str, evidence: Iterable[SearchHit]) -> Citation | None:
    """Pick the best citation span for a claim."""
    best: tuple[float, Citation | None] = (0.0, None)
    for hit in evidence:
        for sentence in split_sentences(hit.chunk.text):
            score = support_score(claim_text, sentence)
            if score > best[0]:
                citation = Citation(
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    title=hit.chunk.title,
                    section=hit.chunk.section,
                    support_span=sentence.strip(),
                )
                best = (score, citation)
    return best[1]


def verify_answer(result: AnswerResult, threshold: float = 0.55) -> AnswerResult:
    """Attach citations and flag unsupported claims."""
    verified_claims: list[Claim] = []
    supported_count = 0
    for claim in result.claims:
        citation = choose_citation_for_claim(claim.text, result.evidence)
        citations = [citation] if citation is not None else []
        supported = False
        if citation is not None:
            supported = support_score(claim.text, citation.support_span) >= threshold
        notes = list(claim.notes)
        if not supported:
            notes.append("support below threshold")
        else:
            supported_count += 1
        verified_claims.append(Claim(text=claim.text, citations=citations, supported=supported, notes=notes))
    result.claims = verified_claims
    result.diagnostics["groundedness"] = supported_count / max(1, len(verified_claims))
    return result
