"""Simple planning and task decomposition."""

from __future__ import annotations

from raglab.domain.models import AgentState


def initial_plan(state: AgentState) -> list[str]:
    """Return a coarse ordered plan.

    The plan is intentionally small. The controller may still revise it based on
    observations, which is the pedagogical point.
    """
    if state.intent.missing_scope:
        return ["clarify", "retrieve", "synthesize"]
    if state.intent.task_type == "structured":
        return ["structured_lookup", "retrieve", "synthesize"]
    if state.intent.task_type == "multi_hop":
        return ["retrieve", "follow_reference", "verify", "synthesize"]
    if state.intent.task_type == "procedural":
        return ["retrieve", "tool_lookup", "verify", "synthesize"]
    if state.intent.task_type == "comparison":
        return ["retrieve", "verify", "synthesize"]
    return ["retrieve", "synthesize"]
