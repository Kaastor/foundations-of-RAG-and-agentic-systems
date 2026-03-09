"""Stateful controller for the pedagogical agentic RAG loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.agent.memory import load_session, prune_memory_for_new_scope, save_session, write_memory_entry
from raglab.agent.multi_agent import run_multi_agent
from raglab.agent.planner import initial_plan
from raglab.agent.router import choose_route, synthesis_profile
from raglab.agent.tools import ToolRuntime
from raglab.config import AppConfig
from raglab.domain.models import AgentState, AnswerResult, Claim, QueryIntent, SearchHit, UserContext
from raglab.generation.synthesizer import synthesize_answer
from raglab.ops.publish import init_workspace
from raglab.ops.tracing import TraceRecorder
from raglab.retrieval.engine import KnowledgeBase, infer_intent


def _apply_user_assumptions(intent: QueryIntent, assumptions: dict[str, str]) -> QueryIntent:
    for key, value in assumptions.items():
        if key in {"region", "product", "disclosure"} and value:
            intent.filters[key] = value
        if key in intent.missing_scope:
            intent.missing_scope = [item for item in intent.missing_scope if item != key]
    return intent


def _apply_session_memory(state: AgentState) -> None:
    query_lower = state.intent.normalized_query.lower()
    for key, entry in list(state.memory.items()):
        if key == "product" and "product" not in state.intent.filters:
            state.intent.filters["product"] = entry.value
            state.notes.append(f"Applied session memory for product={entry.value}.")
        if key == "region" and "region" not in state.intent.filters and entry.scope in {"task", "session"}:
            state.intent.filters["region"] = entry.value
            state.notes.append(f"Applied session memory for region={entry.value}.")
        if key == "product" and state.intent.filters.get("product") and state.intent.filters["product"].lower() not in query_lower and any(token in query_lower for token in ["r-12", "x12", "v14"]):
            state.notes.append("Current query conflicts with stored product memory; dropping task-scoped memory.")
            prune_memory_for_new_scope(state, keep_scopes={"profile"})


def _clarification_prompt(intent: QueryIntent) -> str:
    if intent.ambiguous_options:
        options = "; ".join(intent.ambiguous_options)
        return f"I need one missing scope decision before continuing. Please choose among: {options}."
    if intent.missing_scope:
        return f"I need one missing scope value before continuing: {', '.join(intent.missing_scope)}."
    return "I need more context before continuing."


def _structured_region(state: AgentState) -> str:
    return state.intent.filters.get("region") or (state.user.region if state.user.region != "global" else "EU")


def _structured_model(state: AgentState) -> str:
    return state.intent.filters.get("product") or "V14"



def _search_query(query: str, state: AgentState, assumptions: dict[str, str]) -> str:
    augmented = query
    for key in ("region", "product", "disclosure"):
        value = assumptions.get(key) or state.intent.filters.get(key)
        if value and value.lower() not in augmented.lower():
            augmented += f" {value}"
    return augmented


def _retry_route(state: AgentState) -> str:
    if state.route == "sparse_first":
        return "hybrid_plus_careful"
    if state.route == "cheap_default":
        return "hybrid_plus_careful"
    return state.route


def run_agent(
    query: str,
    knowledge_base: KnowledgeBase,
    user: UserContext,
    workspace: Path,
    config: AppConfig | None = None,
    assumptions: dict[str, str] | None = None,
    session_id: str | None = None,
    interactive: bool = False,
    multi_agent: bool = False,
) -> tuple[AnswerResult, AgentState]:
    """Execute a bounded agentic control loop."""
    config = config or knowledge_base.config
    init_workspace(workspace)
    trace = TraceRecorder(workspace / "traces", request_kind="agent", query=query)
    intent = infer_intent(query, user=user)
    assumptions = assumptions or {}
    intent = _apply_user_assumptions(intent, assumptions)

    state = AgentState(
        query=query,
        intent=intent,
        user=user,
        budget_remaining=config.max_agent_steps,
        open_questions=list(intent.missing_scope),
    )
    if session_id:
        state.memory = load_session(workspace / "sessions", session_id)
        _apply_session_memory(state)

    if multi_agent:
        multi_agent_query = _search_query(query, state, assumptions)
        result, artifacts = run_multi_agent(multi_agent_query, knowledge_base, user=user, trace=trace)
        state.history.append({"action": "multi_agent", "artifacts": {"plan": artifacts.plan, "retrieval_notes": artifacts.retrieval_notes, "verification_notes": artifacts.verification_notes}})
        if session_id:
            save_session(workspace / "sessions", session_id, state)
        result.trace_path = str(trace.save())
        return result, state

    tool_runtime = ToolRuntime(knowledge_base)
    state.route = choose_route(state)
    plan = initial_plan(state)
    trace.add_event("plan_created", {"plan": plan, "route": state.route})

    while state.budget_remaining > 0:
        state.step += 1
        state.budget_remaining -= 1
        state.history.append({"step": state.step, "route": state.route, "intent": state.intent.to_dict()})

        if state.intent.missing_scope and not assumptions:
            if interactive:
                response = input(_clarification_prompt(state.intent) + " ")
                if response.strip():
                    if state.intent.missing_scope:
                        assumptions[state.intent.missing_scope[0]] = response.strip()
                    state.intent = _apply_user_assumptions(state.intent, assumptions)
                    continue
            clarification = _clarification_prompt(state.intent)
            result = AnswerResult(
                mode="agent",
                query=query,
                answer_text=clarification,
                claims=[Claim(text=clarification, supported=False)],
                evidence=state.evidence,
                abstained=True,
                clarification_request=clarification,
                diagnostics={"route": state.route, "plan": plan, "notes": state.notes},
            )
            result.trace_path = str(trace.save())
            if session_id:
                save_session(workspace / "sessions", session_id, state)
            return result, state

        if state.route == "structured_then_policy" and state.intent.task_type == "structured":
            span = trace.start_span("tool_lookup_service_centers", {"region": _structured_region(state), "model": _structured_model(state)})
            tool_result = tool_runtime.execute(
                "lookup_service_centers",
                {"region": _structured_region(state), "model": _structured_model(state)},
                user=user,
                governance_state={"user_role": user.role, "requested_disclosure": "public"},
            )
            span.finish(output_summary={"status": tool_result.status, "rows": len(tool_result.payload.get("rows", []))})
            state.history.append({"action": "lookup_service_centers", "result": tool_result.to_dict()})
            if tool_result.status == "ok" and tool_result.payload.get("rows"):
                centers = ", ".join(row["name"] for row in tool_result.payload["rows"])
                program = tool_result.payload["rows"][0]["warranty_programs"][0]
                write_memory_entry(state, "warranty_program", program, source="structured_tool", scope="task", confidence=0.95)
                state.intent.filters.setdefault("region", _structured_region(state))
                search_query = f"{query} warranty program {program}"
                hits, packed, _intent, diagnostics = knowledge_base.retrieve(search_query, user=user, route="sparse", top_k=5, trace=trace)
                state.evidence = hits
                answer = synthesize_answer(search_query, packed, intent=state.intent, mode="agent", careful=True)
                answer.answer_text = f"Structured lookup found service centers: {centers}. {answer.answer_text}"
                answer.diagnostics["structured_rows"] = tool_result.payload["rows"]
                answer.trace_path = str(trace.save())
                if session_id:
                    save_session(workspace / "sessions", session_id, state)
                return answer, state

        retrieval_route = "hybrid"
        if state.route == "sparse_first":
            retrieval_route = "sparse"
        elif state.route == "cheap_default":
            retrieval_route = "sparse"
        elif state.route in {"hybrid_plus_careful", "structured_then_policy"}:
            retrieval_route = "hybrid"

        search_query = _search_query(query, state, assumptions)
        hits, packed, current_intent, diagnostics = knowledge_base.retrieve(search_query, user=user, route=retrieval_route, top_k=8, rerank_top_m=5, trace=trace)
        state.intent = _apply_user_assumptions(current_intent, assumptions)
        _apply_session_memory(state)
        state.evidence = hits
        state.history.append({"action": "retrieve", "route": retrieval_route, "hits": [hit.chunk.chunk_id for hit in hits], "diagnostics": diagnostics})

        if state.intent.missing_scope and not assumptions:
            clarification = _clarification_prompt(state.intent)
            result = AnswerResult(
                mode="agent",
                query=query,
                answer_text=clarification,
                claims=[Claim(text=clarification, supported=False)],
                evidence=hits,
                abstained=True,
                clarification_request=clarification,
                diagnostics=diagnostics,
            )
            result.trace_path = str(trace.save())
            if session_id:
                save_session(workspace / "sessions", session_id, state)
            return result, state

        followup_done = any(item.get("action") == "follow_reference" for item in state.history if isinstance(item, dict))
        if state.intent.task_type in {"procedural", "multi_hop"} and not followup_done:
            referenced_chunks = knowledge_base.follow_references(hits)
            if referenced_chunks:
                state.history.append({"action": "follow_reference", "doc_ids": sorted({chunk.doc_id for chunk in referenced_chunks})})
                if state.intent.task_type == "procedural":
                    environment = "staging" if "staging" in query.lower() else "sandbox"
                    span = trace.start_span("tool_get_runbook_steps", {"environment": environment})
                    tool_result = tool_runtime.execute(
                        "get_runbook_steps",
                        {"environment": environment, "procedure_name": "key rotation rollback"},
                        user=user,
                        governance_state={"user_role": user.role, "requested_disclosure": "internal"},
                    )
                    span.finish(output_summary={"status": tool_result.status})
                    state.history.append({"action": "get_runbook_steps", "result": tool_result.to_dict()})
                    if tool_result.status == "ok":
                        lines = "; ".join(tool_result.payload["steps"][:3])
                        write_memory_entry(state, "runbook_steps", lines, source="tool:get_runbook_steps", scope="task", confidence=0.95)
                        result = synthesize_answer(query, packed, intent=state.intent, mode="agent", careful=True)
                        result.answer_text += f" Tool lookup found current {environment} steps: {lines}."
                        result.diagnostics["tool_result"] = tool_result.payload
                        result.trace_path = str(trace.save())
                        if session_id:
                            save_session(workspace / "sessions", session_id, state)
                        return result, state
                else:
                    extra_hits = [
                        SearchHit(chunk=chunk, scores={"follow_reference": 1.0}, final_score=1.0, rank=index + 1, reasons=["reference follow-up"])
                        for index, chunk in enumerate(referenced_chunks[:3])
                    ]
                    combined_hits = hits + extra_hits
                    packed = knowledge_base.pack_context(query, combined_hits[:6], budget_tokens=knowledge_base.config.default_context_budget)
                    result = synthesize_answer(query, packed, intent=state.intent, mode="agent", careful=True)
                    result.trace_path = str(trace.save())
                    if session_id:
                        save_session(workspace / "sessions", session_id, state)
                    return result, state

        if not hits:
            state.notes.append("No hits retrieved.")
            if state.budget_remaining > 0:
                state.route = _retry_route(state)
                continue
            result = AnswerResult(
                mode="agent",
                query=query,
                answer_text="I could not retrieve evidence. A human review or a more specific query is needed.",
                claims=[Claim(text="No evidence retrieved.", supported=False)],
                evidence=[],
                abstained=True,
                escalation_request="No evidence after bounded search.",
                diagnostics={"route": state.route, "notes": state.notes},
            )
            result.trace_path = str(trace.save())
            if session_id:
                save_session(workspace / "sessions", session_id, state)
            return result, state

        top_support = max((hit.final_score for hit in hits), default=0.0)
        if diagnostics.get("conflicts") and user.high_stakes:
            result = synthesize_answer(query, packed, intent=state.intent, mode="agent", careful=True)
            result.escalation_request = "Conflicting high-stakes evidence remains after bounded search."
            result.trace_path = str(trace.save())
            if session_id:
                save_session(workspace / "sessions", session_id, state)
            return result, state

        if top_support < 0.02 and state.budget_remaining > 0:
            state.route = _retry_route(state)
            state.notes.append("Low support triggered route escalation.")
            continue

        result = synthesize_answer(query, packed, intent=state.intent, mode="agent", careful=synthesis_profile(state.route) == "careful")
        if state.intent.filters.get("product"):
            write_memory_entry(state, "product", state.intent.filters["product"], source="query", scope="task", confidence=0.9)
        if state.intent.filters.get("region"):
            write_memory_entry(state, "region", state.intent.filters["region"], source="query", scope="task", confidence=0.9)
        result.diagnostics["route"] = state.route
        result.diagnostics["plan"] = plan
        result.diagnostics["notes"] = state.notes
        result.trace_path = str(trace.save())
        if session_id:
            save_session(workspace / "sessions", session_id, state)
        return result, state

    result = AnswerResult(
        mode="agent",
        query=query,
        answer_text="The controller exhausted its search budget before reaching a reliable answer.",
        claims=[Claim(text="Search budget exhausted.", supported=False)],
        evidence=state.evidence,
        abstained=True,
        escalation_request="Budget exhausted",
        diagnostics={"history": state.history},
    )
    result.trace_path = str(trace.save())
    if session_id:
        save_session(workspace / "sessions", session_id, state)
    return result, state
