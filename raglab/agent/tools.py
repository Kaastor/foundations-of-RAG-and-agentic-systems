"""Tool schemas and runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from raglab.domain.models import UserContext
from raglab.ops.governance import allow_action
from raglab.retrieval.engine import KnowledgeBase


@dataclass(slots=True)
class ToolSchema:
    """Metadata describing one tool contract."""

    name: str
    purpose: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...] = ()
    enum_args: dict[str, tuple[str, ...]] = field(default_factory=dict)
    read_only: bool = True
    description: str = ""


@dataclass(slots=True)
class ToolResult:
    """Normalized tool output."""

    status: str
    payload: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "payload": self.payload, "error": self.error}


class ToolRuntime:
    """Validate tool calls before delegating to Python functions."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.schemas: dict[str, ToolSchema] = {}
        self.handlers: dict[str, Callable[[dict[str, Any], UserContext], ToolResult]] = {}
        self._register_builtin_tools()

    def _register(self, schema: ToolSchema, handler: Callable[[dict[str, Any], UserContext], ToolResult]) -> None:
        self.schemas[schema.name] = schema
        self.handlers[schema.name] = handler

    def _register_builtin_tools(self) -> None:
        self._register(
            ToolSchema(
                name="search_documents",
                purpose="retrieve chunks from the knowledge base",
                required_args=("query",),
                optional_args=("route", "top_k"),
                enum_args={"route": ("sparse", "dense", "hybrid")},
                read_only=True,
                description="Read-only retrieval tool returning chunk IDs, titles, scores, and diagnostics.",
            ),
            self._search_documents,
        )
        self._register(
            ToolSchema(
                name="get_runbook_steps",
                purpose="read current rollback or rotation steps for an environment",
                required_args=("environment", "procedure_name"),
                enum_args={"environment": ("sandbox", "staging", "production")},
                read_only=True,
                description="Read-only runbook lookup by environment and procedure name.",
            ),
            self._get_runbook_steps,
        )
        self._register(
            ToolSchema(
                name="lookup_service_centers",
                purpose="query structured service-center records",
                required_args=("region", "model"),
                optional_args=("warranty_program",),
                read_only=True,
                description="Structured lookup returning service-center rows.",
            ),
            self._lookup_service_centers,
        )
        self._register(
            ToolSchema(
                name="get_document_metadata",
                purpose="read metadata for one source document",
                required_args=("doc_id",),
                read_only=True,
                description="Read-only metadata lookup for a document ID.",
            ),
            self._get_document_metadata,
        )
        self._register(
            ToolSchema(
                name="submit_ticket_update",
                purpose="simulate a side-effecting operational tool",
                required_args=("ticket_id", "note"),
                read_only=False,
                description="Toy side-effecting tool intentionally blocked unless governance approves it.",
            ),
            self._submit_ticket_update,
        )

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        if tool_name not in self.schemas:
            return False, f"Unknown tool: {tool_name}"
        schema = self.schemas[tool_name]
        for required in schema.required_args:
            if required not in arguments or arguments[required] in {"", None}:
                return False, f"Missing required argument: {required}"
        for field_name, allowed in schema.enum_args.items():
            if field_name in arguments and arguments[field_name] not in allowed:
                allowed_text = ", ".join(allowed)
                return False, f"Invalid value for {field_name}. Allowed: {allowed_text}"
        return True, None

    def execute(self, tool_name: str, arguments: dict[str, Any], user: UserContext, governance_state: dict[str, Any] | None = None) -> ToolResult:
        ok, error = self.validate(tool_name, arguments)
        if not ok:
            return ToolResult(status="validation_error", payload={}, error=error)
        schema = self.schemas[tool_name]
        state = dict(governance_state or {})
        state.setdefault("user_role", user.role)
        state.setdefault("requested_disclosure", arguments.get("disclosure", "public"))
        action_name = f"{'read' if schema.read_only else 'side_effect'}:{tool_name}"
        if not allow_action(action_name, state):
            return ToolResult(status="permission_denied", payload={}, error="Governance policy blocked the tool call.")
        return self.handlers[tool_name](arguments, user)

    def _search_documents(self, arguments: dict[str, Any], user: UserContext) -> ToolResult:
        query = str(arguments["query"])
        route = str(arguments.get("route", "hybrid"))
        top_k = int(arguments.get("top_k", 5))
        hits, _packed, intent, diagnostics = self.knowledge_base.retrieve(query, user=user, route=route, top_k=top_k)
        payload = {
            "intent": intent.to_dict(),
            "hits": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_id,
                    "title": hit.chunk.title,
                    "score": round(hit.final_score, 5),
                }
                for hit in hits
            ],
            "diagnostics": diagnostics,
        }
        return ToolResult(status="ok", payload=payload)

    def _get_runbook_steps(self, arguments: dict[str, Any], user: UserContext) -> ToolResult:
        runbooks = self.knowledge_base.structured.get("runbooks", [])
        environment = str(arguments["environment"])
        procedure_name = str(arguments["procedure_name"]).lower()
        for row in runbooks:
            if row["environment"] == environment and procedure_name in row["procedure_name"].lower():
                return ToolResult(
                    status="ok",
                    payload={
                        "steps": list(row["steps"]),
                        "source_id": row["source_id"],
                        "environment": environment,
                        "procedure_name": row["procedure_name"],
                    },
                )
        return ToolResult(status="not_found", payload={}, error="No matching runbook procedure was found.")

    def _lookup_service_centers(self, arguments: dict[str, Any], user: UserContext) -> ToolResult:
        centers = self.knowledge_base.structured.get("service_centers", [])
        region = str(arguments["region"])
        model = str(arguments["model"])
        warranty_program = arguments.get("warranty_program")
        results = []
        for row in centers:
            if row["region"] != region:
                continue
            if model not in row["models"]:
                continue
            if warranty_program and warranty_program not in row["warranty_programs"]:
                continue
            results.append(row)
        return ToolResult(status="ok", payload={"rows": results})

    def _get_document_metadata(self, arguments: dict[str, Any], user: UserContext) -> ToolResult:
        doc_id = str(arguments["doc_id"])
        document = self.knowledge_base.documents.get(doc_id)
        if document is None:
            return ToolResult(status="not_found", payload={}, error="Unknown document ID.")
        return ToolResult(
            status="ok",
            payload={
                "doc_id": document.doc_id,
                "title": document.title,
                "doc_type": document.doc_type,
                "effective_date": document.effective_date,
                "authority": document.authority,
                "status": document.status,
                "disclosure": document.disclosure,
                "references": list(document.references),
            },
        )

    def _submit_ticket_update(self, arguments: dict[str, Any], user: UserContext) -> ToolResult:
        return ToolResult(
            status="blocked",
            payload={
                "ticket_id": arguments["ticket_id"],
                "note": arguments["note"],
            },
            error="This pedagogical repository exposes the schema but intentionally blocks side effects by default.",
        )
