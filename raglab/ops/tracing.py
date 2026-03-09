"""Minimal structured tracing for educational debugging."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from raglab.storage.json_store import ensure_dir, write_json


@dataclass(slots=True)
class Span:
    """One timed span in a request trace."""

    name: str
    started_at: float
    ended_at: float | None = None
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def finish(self, output_summary: dict[str, Any] | None = None, diagnostics: dict[str, Any] | None = None) -> None:
        self.ended_at = time.time()
        if output_summary:
            self.output_summary = output_summary
        if diagnostics:
            self.diagnostics = diagnostics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ms"] = None if self.ended_at is None else round((self.ended_at - self.started_at) * 1000, 3)
        return payload


class TraceRecorder:
    """Collect span data and persist it as JSON.

    The implementation is deliberately simple and file-based so learners can
    inspect traces without extra infrastructure.
    """

    def __init__(self, trace_dir: Path, request_kind: str, query: str) -> None:
        ensure_dir(trace_dir)
        self.trace_dir = trace_dir
        self.trace_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.request_kind = request_kind
        self.query = query
        self.spans: list[Span] = []
        self.events: list[dict[str, Any]] = []

    def start_span(self, name: str, input_summary: dict[str, Any] | None = None) -> Span:
        span = Span(name=name, started_at=time.time(), input_summary=input_summary or {})
        self.spans.append(span)
        return span

    def add_event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "time": time.time(),
                "payload": payload or {},
            }
        )

    def save(self) -> Path:
        payload = {
            "trace_id": self.trace_id,
            "request_kind": self.request_kind,
            "query": self.query,
            "spans": [span.to_dict() for span in self.spans],
            "events": self.events,
        }
        target = self.trace_dir / f"{self.trace_id}.json"
        write_json(target, payload)
        return target


def load_trace(path: str | Path) -> dict[str, Any]:
    """Load a persisted trace for the CLI inspector."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trace_summary(trace: dict[str, Any]) -> str:
    """Render a compact plain-text summary."""
    lines = [f"Trace {trace['trace_id']} ({trace['request_kind']})", f"Query: {trace['query']}"]
    for span in trace.get("spans", []):
        lines.append(
            f"- {span['name']}: {span.get('duration_ms', 'n/a')} ms | input={span.get('input_summary', {})} | output={span.get('output_summary', {})}"
        )
    return "\n".join(lines)
