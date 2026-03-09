"""A tiny closed-book baseline used only for pedagogical contrast."""

from __future__ import annotations

from raglab.domain.models import AnswerResult, Claim


COMMON_FACTS = {
    "what does gpu stand for": "GPU stands for graphics processing unit.",
    "what is rag": "RAG stands for retrieval-augmented generation.",
}


def closed_book_answer(query: str) -> AnswerResult:
    """Return a deliberately limited baseline answer.

    The point is not to be clever. The point is to contrast parameter-only
    behavior with evidence-grounded behavior on source-sensitive questions.
    """
    lowered = query.strip().lower().rstrip("?")
    if lowered in COMMON_FACTS:
        text = COMMON_FACTS[lowered]
        return AnswerResult(mode="closed_book", query=query, answer_text=text, claims=[Claim(text=text)], evidence=[])
    text = (
        "Closed-book mode did not inspect the knowledge base. "
        "For source-sensitive, current, or local questions this repository intentionally "
        "treats that as insufficient."
    )
    return AnswerResult(mode="closed_book", query=query, answer_text=text, claims=[Claim(text=text, supported=False)], evidence=[], abstained=True)
