"""Routing policies for retrievers, tools, and synthesis profiles."""

from __future__ import annotations

from raglab.domain.models import AgentState


def choose_route(state: AgentState) -> str:
    """Select a retrieval route for the current state."""
    query = state.intent.normalized_query.lower()
    if state.intent.task_type == "structured":
        return "structured_then_policy"
    if state.intent.identifiers:
        return "sparse_first"
    if state.user.high_stakes or state.intent.task_type in {"comparison", "multi_hop"}:
        return "hybrid_plus_careful"
    if "policy" in query or "warranty" in query:
        return "hybrid_plus_careful"
    return "cheap_default"


def synthesis_profile(route: str) -> str:
    """Map a route to a synthesis profile name."""
    if route in {"hybrid_plus_careful", "structured_then_policy"}:
        return "careful"
    return "fast"
