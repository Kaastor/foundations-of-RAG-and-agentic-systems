"""Dataclasses shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(slots=True)
class UserContext:
    """Represents a requesting user, their role, and retrieval constraints."""

    user_id: str
    role: str
    region: str = "global"
    language: str = "en"
    allowed_disclosures: tuple[str, ...] = ("public",)
    source_preference: str = "authority"
    high_stakes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentRecord:
    """A normalized source document ready for chunking."""

    doc_id: str
    title: str
    doc_type: str
    text: str
    source_path: str
    product: str = ""
    region: str = "global"
    effective_date: str = ""
    version: str = ""
    authority: str = "reference"
    status: str = "active"
    disclosure: str = "internal"
    allowed_roles: tuple[str, ...] = ("field_support",)
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    trust_label: str = "trusted"
    quality_score: float = 1.0
    trust_score: float = 1.0
    duplicate_of: str | None = None
    quarantine_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRecord":
        return cls(**payload)


@dataclass(slots=True)
class ChunkRecord:
    """A retrievable chunk derived from one source document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str
    order: int
    token_count: int
    product: str = ""
    region: str = "global"
    effective_date: str = ""
    version: str = ""
    authority: str = "reference"
    status: str = "active"
    disclosure: str = "internal"
    allowed_roles: tuple[str, ...] = ("field_support",)
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    trust_score: float = 1.0
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkRecord":
        return cls(**payload)


@dataclass(slots=True)
class QueryIntent:
    """Normalized query plus routing and clarification hints."""

    raw_query: str
    normalized_query: str
    tokens: list[str]
    identifiers: list[str]
    task_type: str
    filters: dict[str, str] = field(default_factory=dict)
    expansions: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    ambiguous_options: list[str] = field(default_factory=list)
    missing_scope: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    """One retrieved chunk plus scoring diagnostics."""

    chunk: ChunkRecord
    scores: dict[str, float]
    final_score: float
    rank: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunk"] = self.chunk.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchHit":
        return cls(
            chunk=ChunkRecord.from_dict(payload["chunk"]),
            scores=payload["scores"],
            final_score=payload["final_score"],
            rank=payload["rank"],
            reasons=payload.get("reasons", []),
        )


@dataclass(slots=True)
class Citation:
    """A support pointer back to evidence."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    support_span: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Citation":
        return cls(**payload)


@dataclass(slots=True)
class Claim:
    """One answer claim and its citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    supported: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Claim":
        return cls(
            text=payload["text"],
            citations=[Citation.from_dict(item) for item in payload.get("citations", [])],
            supported=payload.get("supported", True),
            notes=payload.get("notes", []),
        )


@dataclass(slots=True)
class AnswerResult:
    """The final system response plus supporting diagnostics."""

    mode: str
    query: str
    answer_text: str
    claims: list[Claim]
    evidence: list[SearchHit]
    abstained: bool = False
    clarification_request: str | None = None
    escalation_request: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query": self.query,
            "answer_text": self.answer_text,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [hit.to_dict() for hit in self.evidence],
            "abstained": self.abstained,
            "clarification_request": self.clarification_request,
            "escalation_request": self.escalation_request,
            "diagnostics": self.diagnostics,
            "trace_path": self.trace_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerResult":
        return cls(
            mode=payload["mode"],
            query=payload["query"],
            answer_text=payload["answer_text"],
            claims=[Claim.from_dict(item) for item in payload.get("claims", [])],
            evidence=[SearchHit.from_dict(item) for item in payload.get("evidence", [])],
            abstained=payload.get("abstained", False),
            clarification_request=payload.get("clarification_request"),
            escalation_request=payload.get("escalation_request"),
            diagnostics=payload.get("diagnostics", {}),
            trace_path=payload.get("trace_path"),
        )


@dataclass(slots=True)
class MemoryEntry:
    """One scoped working-memory entry."""

    key: str
    value: str
    source: str
    scope: str
    confidence: float
    expires_with_task: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        return cls(**payload)


@dataclass(slots=True)
class AgentState:
    """State for the simple controller."""

    query: str
    intent: QueryIntent
    user: UserContext
    budget_remaining: int
    route: str = "hybrid"
    step: int = 0
    open_questions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    evidence: list[SearchHit] = field(default_factory=list)
    memory: dict[str, MemoryEntry] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    waiting_for_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.to_dict(),
            "user": self.user.to_dict(),
            "budget_remaining": self.budget_remaining,
            "route": self.route,
            "step": self.step,
            "open_questions": self.open_questions,
            "blockers": self.blockers,
            "evidence": [hit.to_dict() for hit in self.evidence],
            "memory": {key: entry.to_dict() for key, entry in self.memory.items()},
            "history": self.history,
            "notes": self.notes,
            "waiting_for_user": self.waiting_for_user,
        }
