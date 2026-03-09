"""Benchmark runner over the bundled query set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.agent.controller import run_agent
from raglab.config import AppConfig
from raglab.domain.models import UserContext
from raglab.evaluation.metrics import clarification_precision, clarification_recall, groundedness, mrr, ndcg_at_k, precision_at_k, recall_at_k
from raglab.generation.synthesizer import synthesize_answer
from raglab.ops.governance import load_named_user, user_from_role
from raglab.retrieval.engine import KnowledgeBase
from raglab.storage.json_store import read_json, read_jsonl


def _query_user(examples_root: Path, row: dict[str, Any]) -> UserContext:
    users_path = examples_root / "users.json"
    if row.get("user_id"):
        return load_named_user(users_path, row["user_id"])
    return user_from_role(row.get("user_role", "field_support"), region=row.get("region", "global"), high_stakes=row.get("high_stakes", False))


def run_benchmark(snapshot_path: Path, dataset_path: Path, examples_root: Path, mode: str = "answer") -> dict[str, Any]:
    """Run the bundled benchmark and return a structured report."""
    kb = KnowledgeBase.load(snapshot_path)
    rows = read_jsonl(dataset_path)
    retrieval_rows: list[dict[str, Any]] = []
    clarification_rows: list[dict[str, bool]] = []
    answer_rows: list[dict[str, Any]] = []

    for row in rows:
        user = _query_user(examples_root, row)
        query = row["query"]
        relevant = set(row.get("relevant_doc_ids", []))
        graded = {doc_id: 2 for doc_id in relevant}

        hits, packed, intent, diagnostics = kb.retrieve(query, user=user, route=row.get("route", "hybrid"), top_k=8, rerank_top_m=5)
        retrieval_rows.append(
            {
                "query_id": row["query_id"],
                "recall@5": recall_at_k(relevant, hits, 5),
                "precision@5": precision_at_k(relevant, hits, 5),
                "mrr": mrr(relevant, hits),
                "ndcg@5": ndcg_at_k(graded, hits, 5),
            }
        )

        if row["task"] == "clarification":
            asked = bool(intent.missing_scope or intent.ambiguous_options)
            clarification_rows.append({"needed": True, "asked": asked})
            continue

        if mode == "agent":
            assumptions = row.get("assumptions", {})
            answer, _state = run_agent(query, kb, user=user, workspace=snapshot_path.parent.parent, assumptions=assumptions)
        else:
            answer = synthesize_answer(query, packed, intent=intent, mode=mode, careful=True)

        expected_fragments = [fragment.lower() for fragment in row.get("expected_fragments", [])]
        answer_text = answer.answer_text.lower()
        fragment_hits = sum(1 for fragment in expected_fragments if fragment in answer_text)
        answer_rows.append(
            {
                "query_id": row["query_id"],
                "groundedness": groundedness(answer),
                "fragment_hit_rate": fragment_hits / max(1, len(expected_fragments)) if expected_fragments else 0.0,
                "abstained": answer.abstained,
                "clarification_requested": bool(answer.clarification_request),
                "escalated": bool(answer.escalation_request),
            }
        )
        if row.get("needs_clarification", False):
            clarification_rows.append({"needed": True, "asked": bool(answer.clarification_request)})
        else:
            clarification_rows.append({"needed": False, "asked": bool(answer.clarification_request)})

    def _mean(items: list[float]) -> float:
        if not items:
            return 0.0
        return round(sum(items) / len(items), 4)

    report = {
        "dataset": str(dataset_path),
        "retrieval": {
            "recall@5": _mean([row["recall@5"] for row in retrieval_rows]),
            "precision@5": _mean([row["precision@5"] for row in retrieval_rows]),
            "mrr": _mean([row["mrr"] for row in retrieval_rows]),
            "ndcg@5": _mean([row["ndcg@5"] for row in retrieval_rows]),
        },
        "answers": {
            "groundedness": _mean([row["groundedness"] for row in answer_rows]),
            "fragment_hit_rate": _mean([row["fragment_hit_rate"] for row in answer_rows]),
            "abstention_rate": _mean([1.0 if row["abstained"] else 0.0 for row in answer_rows]),
            "clarification_rate": _mean([1.0 if row["clarification_requested"] else 0.0 for row in answer_rows]),
            "escalation_rate": _mean([1.0 if row["escalated"] else 0.0 for row in answer_rows]),
        },
        "agent_process": {
            "clarification_precision": round(clarification_precision(clarification_rows), 4),
            "clarification_recall": round(clarification_recall(clarification_rows), 4),
        },
        "per_query": {
            "retrieval": retrieval_rows,
            "answers": answer_rows,
        },
    }
    return report
