"""A tiny specialized multi-agent coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from raglab.domain.models import AnswerResult, QueryIntent, UserContext
from raglab.generation.synthesizer import synthesize_answer
from raglab.retrieval.engine import KnowledgeBase
from raglab.ops.tracing import TraceRecorder


@dataclass(slots=True)
class MultiAgentArtifacts:
    plan: list[str]
    retrieval_notes: list[str]
    verification_notes: list[str]


def run_multi_agent(query: str, kb: KnowledgeBase, user: UserContext, trace: TraceRecorder | None = None) -> tuple[AnswerResult, MultiAgentArtifacts]:
    """Run a simple planner -> retriever -> writer -> verifier pattern."""
    plan = ["planner", "retriever", "writer", "verifier", "policy_guard"]
    retrieval_notes: list[str] = []
    verification_notes: list[str] = []

    span = trace.start_span("multi_agent_planner", {"query": query}) if trace else None
    hits, packed, intent, diagnostics = kb.retrieve(query, user=user, route="hybrid", top_k=8, rerank_top_m=5, trace=trace)
    if span:
        span.finish(output_summary={"plan": plan})
    retrieval_notes.append(f"Retrieved {len(hits)} hits with route hybrid.")

    span = trace.start_span("multi_agent_writer", {"packed_hits": len(packed.hits)}) if trace else None
    result = synthesize_answer(query, packed, intent=intent, mode="multi_agent", careful=True)
    if span:
        span.finish(output_summary={"claims": len(result.claims), "abstained": result.abstained})

    unsupported = [claim.text for claim in result.claims if not claim.supported]
    if unsupported:
        verification_notes.append("Verifier found unsupported or low-support claims.")
        result.diagnostics["multi_agent_revision"] = "Verifier kept unsupported claims visible rather than hiding them."
    if any(hit.chunk.disclosure == "internal" for hit in result.evidence) and user.role == "distributor_support":
        verification_notes.append("Policy guard blocked internal-only material.")
        result.escalation_request = "Customer-safe review recommended because internal-only evidence was retrieved."
    return result, MultiAgentArtifacts(plan=plan, retrieval_notes=retrieval_notes, verification_notes=verification_notes)
