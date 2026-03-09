"""Command-line interface for the pedagogical RAG systems lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from raglab.agent.controller import run_agent
from raglab.build import index_staged_snapshot
from raglab.config import DEFAULT_CONFIG, AppConfig, workspace_path
from raglab.demos import chapter_demo_text, ensure_demo_workspace, examples_root, run_chapter_demo
from raglab.domain.models import AnswerResult, SearchHit
from raglab.evaluation.runner import run_benchmark
from raglab.generation.closed_book import closed_book_answer
from raglab.generation.synthesizer import synthesize_answer
from raglab.ingest.pipeline import ingest_corpus
from raglab.ops.governance import load_named_user, user_from_role
from raglab.ops.publish import init_workspace, list_snapshots, live_snapshot_path, publish_staged_snapshot, staged_snapshot_path
from raglab.ops.tracing import load_trace, trace_summary
from raglab.retrieval.engine import KnowledgeBase


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_workspace() -> Path:
    return (_repo_root() / ".workspace" / "demo").resolve()


def _resolve_snapshot(workspace: Path, snapshot_selector: str) -> Path:
    if snapshot_selector == "live":
        return live_snapshot_path(workspace)
    if snapshot_selector == "staged":
        return staged_snapshot_path(workspace)
    candidate = Path(snapshot_selector)
    if candidate.exists():
        return candidate.resolve()
    snapshots = workspace / "snapshots"
    named = snapshots / snapshot_selector
    if named.exists():
        return named.resolve()
    raise FileNotFoundError(f"Could not resolve snapshot selector: {snapshot_selector}")


def _resolve_user(args: argparse.Namespace) -> Any:
    examples = examples_root()
    if getattr(args, "user_id", None):
        return load_named_user(examples / "users.json", args.user_id)
    role = getattr(args, "user_role", "field_support")
    region = getattr(args, "region", "global")
    high_stakes = getattr(args, "high_stakes", False)
    return user_from_role(role, region=region, high_stakes=high_stakes)


def _parse_assumptions(items: list[str] | None) -> dict[str, str]:
    assumptions: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Assumption must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        assumptions[key.strip()] = value.strip()
    return assumptions


def _print_hits(hits: list[SearchHit], diagnostics: dict[str, Any]) -> str:
    lines = [f"Route: {diagnostics.get('route', 'n/a')}"]
    if diagnostics.get("missing_scope"):
        lines.append(f"Missing scope: {', '.join(diagnostics['missing_scope'])}")
    if diagnostics.get("ambiguous_options"):
        lines.append(f"Ambiguity options: {', '.join(diagnostics['ambiguous_options'])}")
    for hit in hits:
        lines.append(
            f"- #{hit.rank} {hit.chunk.doc_id} | {hit.chunk.title} | section={hit.chunk.section} | score={hit.final_score:.4f}"
        )
        if hit.reasons:
            lines.append(f"    reasons: {', '.join(hit.reasons)}")
    if diagnostics.get("conflicts"):
        lines.append("Conflicts:")
        for conflict in diagnostics["conflicts"]:
            lines.append(f"  - {conflict}")
    return "\n".join(lines)


def _print_answer(result: AnswerResult) -> str:
    lines = [result.answer_text]
    if result.clarification_request:
        lines.append(f"Clarification request: {result.clarification_request}")
    if result.escalation_request:
        lines.append(f"Escalation request: {result.escalation_request}")
    if result.claims:
        lines.append("Claims and citations:")
        for claim in result.claims:
            citations = ", ".join(f"{citation.doc_id}/{citation.section}" for citation in claim.citations) or "none"
            status = "supported" if claim.supported else "unsupported"
            lines.append(f"- ({status}) {claim.text}")
            lines.append(f"    citations: {citations}")
    if result.trace_path:
        lines.append(f"Trace: {result.trace_path}")
    return "\n".join(lines)


def _build_default_sources(include_updates: bool = True) -> list[Path]:
    root = examples_root() / "corpus"
    sources = [root / "base"]
    if include_updates:
        sources.append(root / "update")
    return sources


def _load_kb_for_command(workspace: Path, snapshot_selector: str) -> KnowledgeBase:
    snapshot = _resolve_snapshot(workspace, snapshot_selector)
    return KnowledgeBase.load(snapshot)


def cmd_ingest(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    sources = [Path(item).resolve() for item in args.source]
    manifest = ingest_corpus(sources, workspace, DEFAULT_CONFIG)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    summary = index_staged_snapshot(workspace, DEFAULT_CONFIG)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    snapshot = publish_staged_snapshot(workspace, note=args.note or "")
    print(str(snapshot))
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    print(json.dumps(list_snapshots(workspace), indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    kb = _load_kb_for_command(workspace, args.snapshot)
    user = _resolve_user(args)
    hits, _packed, _intent, diagnostics = kb.retrieve(args.query, user=user, route=args.route, top_k=args.top_k, rerank_top_m=args.rerank_top_m, use_ann=not args.no_ann)
    if args.json:
        print(json.dumps({"hits": [hit.to_dict() for hit in hits], "diagnostics": diagnostics}, indent=2))
    else:
        print(_print_hits(hits, diagnostics))
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    if args.mode == "closed_book":
        result = closed_book_answer(args.query)
        print(_print_answer(result))
        return 0
    kb = _load_kb_for_command(workspace, args.snapshot)
    user = _resolve_user(args)
    hits, packed, intent, diagnostics = kb.retrieve(args.query, user=user, route=args.route, top_k=args.top_k, rerank_top_m=args.rerank_top_m, use_ann=not args.no_ann)
    result = synthesize_answer(args.query, packed, intent=intent, mode="grounded", careful=not args.fast)
    result.diagnostics.update(diagnostics)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_print_answer(result))
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    kb = _load_kb_for_command(workspace, args.snapshot)
    user = _resolve_user(args)
    assumptions = _parse_assumptions(args.assume)
    result, _state = run_agent(
        args.query,
        kb,
        user=user,
        workspace=workspace,
        assumptions=assumptions,
        session_id=args.session_id,
        interactive=args.interactive,
        multi_agent=args.multi_agent,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_print_answer(result))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    snapshot = _resolve_snapshot(workspace, args.snapshot)
    report = run_benchmark(snapshot, Path(args.dataset).resolve(), examples_root(), mode=args.mode)
    print(json.dumps(report, indent=2))
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    candidate = Path(args.trace_ref)
    if not candidate.exists():
        candidate = workspace / "traces" / f"{args.trace_ref}.json"
    trace = load_trace(candidate)
    print(trace_summary(trace))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    workspace = workspace_path(args.workspace)
    if args.subdemo == "chapter":
        chapter = int(args.chapter)
        if args.run:
            print(run_chapter_demo(chapter, workspace))
        else:
            print(chapter_demo_text(chapter))
        return 0
    if args.subdemo == "prepare":
        snapshot = ensure_demo_workspace(workspace, include_updates=not args.base_only)
        print(str(snapshot))
        return 0
    return 1


def cmd_coverage(args: argparse.Namespace) -> int:
    docs_path = _repo_root() / "docs" / "concept_coverage.md"
    print(str(docs_path))
    print()
    print(docs_path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="Pedagogical reference implementation for a textbook-sized RAG and agentic-RAG codebase.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse, normalize, deduplicate, chunk, and quarantine source files into a staged workspace.")
    ingest.add_argument("--source", action="append", required=True, help="Source directory to ingest. Repeat for multiple roots.")
    ingest.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    ingest.set_defaults(func=cmd_ingest)

    index = subparsers.add_parser("index", help="Build sparse, dense, and approximate indexes inside the staged workspace.")
    index.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    index.set_defaults(func=cmd_index)

    publish = subparsers.add_parser("publish", help="Promote the staged workspace into an immutable live snapshot.")
    publish.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    publish.add_argument("--note", default="", help="Optional release note stored beside the live pointer.")
    publish.set_defaults(func=cmd_publish)

    snapshots = subparsers.add_parser("snapshots", help="List published snapshots and indicate which one is live.")
    snapshots.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    snapshots.set_defaults(func=cmd_snapshots)

    retrieve = subparsers.add_parser("retrieve", help="Run retrieval only and inspect first-pass plus reranked evidence.")
    retrieve.add_argument("query", help="User query.")
    retrieve.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    retrieve.add_argument("--snapshot", default="live", help="Snapshot selector: live, staged, a snapshot id, or a path.")
    retrieve.add_argument("--route", choices=("sparse", "dense", "hybrid"), default="hybrid", help="Retrieval route.")
    retrieve.add_argument("--top-k", type=int, default=8, help="First-pass depth.")
    retrieve.add_argument("--rerank-top-m", type=int, default=5, help="How many reranked hits to keep.")
    retrieve.add_argument("--no-ann", action="store_true", help="Use exact dense search instead of the approximate LSH index.")
    retrieve.add_argument("--user-id", help="Named user from examples/users.json.")
    retrieve.add_argument("--user-role", default="field_support", help="Fallback role if --user-id is omitted.")
    retrieve.add_argument("--region", default="global", help="Fallback user region.")
    retrieve.add_argument("--high-stakes", action="store_true", help="Flag the request as high-stakes.")
    retrieve.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    retrieve.set_defaults(func=cmd_retrieve)

    answer = subparsers.add_parser("answer", help="Run the fixed workflow: retrieve, rerank, pack context, synthesize, verify.")
    answer.add_argument("query", help="User query.")
    answer.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    answer.add_argument("--snapshot", default="live", help="Snapshot selector: live, staged, a snapshot id, or a path.")
    answer.add_argument("--route", choices=("sparse", "dense", "hybrid"), default="hybrid", help="Retrieval route.")
    answer.add_argument("--top-k", type=int, default=8, help="First-pass depth.")
    answer.add_argument("--rerank-top-m", type=int, default=5, help="How many reranked hits to keep.")
    answer.add_argument("--mode", choices=("grounded", "closed_book"), default="grounded", help="Answering mode.")
    answer.add_argument("--fast", action="store_true", help="Use a lighter synthesis profile.")
    answer.add_argument("--no-ann", action="store_true", help="Use exact dense search instead of the approximate LSH index.")
    answer.add_argument("--user-id", help="Named user from examples/users.json.")
    answer.add_argument("--user-role", default="field_support", help="Fallback role if --user-id is omitted.")
    answer.add_argument("--region", default="global", help="Fallback user region.")
    answer.add_argument("--high-stakes", action="store_true", help="Flag the request as high-stakes.")
    answer.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    answer.set_defaults(func=cmd_answer)

    agent = subparsers.add_parser("agent", help="Run the bounded agentic controller with routing, tool use, retries, and clarification.")
    agent.add_argument("query", help="User query.")
    agent.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    agent.add_argument("--snapshot", default="live", help="Snapshot selector: live, staged, a snapshot id, or a path.")
    agent.add_argument("--user-id", help="Named user from examples/users.json.")
    agent.add_argument("--user-role", default="field_support", help="Fallback role if --user-id is omitted.")
    agent.add_argument("--region", default="global", help="Fallback user region.")
    agent.add_argument("--high-stakes", action="store_true", help="Flag the request as high-stakes.")
    agent.add_argument("--assume", action="append", help="Provide a clarification answer or scope assumption as KEY=VALUE.")
    agent.add_argument("--session-id", help="Persist and reuse working memory for this session id.")
    agent.add_argument("--interactive", action="store_true", help="Prompt for missing clarification interactively.")
    agent.add_argument("--multi-agent", action="store_true", help="Use the specialized planner/retriever/writer/verifier path.")
    agent.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    agent.set_defaults(func=cmd_agent)

    evaluate = subparsers.add_parser("evaluate", help="Run the bundled benchmark for retrieval, groundedness, and agent-process behavior.")
    evaluate.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    evaluate.add_argument("--snapshot", default="live", help="Snapshot selector: live, staged, a snapshot id, or a path.")
    evaluate.add_argument("--dataset", default=str(examples_root() / "eval" / "queries.jsonl"), help="JSONL dataset path.")
    evaluate.add_argument("--mode", choices=("answer", "agent"), default="answer", help="Evaluate the fixed workflow or the agentic controller.")
    evaluate.set_defaults(func=cmd_evaluate)

    trace = subparsers.add_parser("trace", help="Inspect a stored JSON trace by file path or trace id.")
    trace.add_argument("trace_ref", help="Trace file path or bare trace id.")
    trace.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    trace.set_defaults(func=cmd_trace)

    demo = subparsers.add_parser("demo", help="Prepare the bundled demo workspace or show chapter-specific scenarios.")
    demo_subparsers = demo.add_subparsers(dest="subdemo", required=True)

    demo_prepare = demo_subparsers.add_parser("prepare", help="Build and publish the bundled demo snapshot if it does not already exist.")
    demo_prepare.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    demo_prepare.add_argument("--base-only", action="store_true", help="Use only the base corpus, which intentionally leaves the live snapshot stale.")
    demo_prepare.set_defaults(func=cmd_demo)

    demo_chapter = demo_subparsers.add_parser("chapter", help="Show or run the scenario mapped to a textbook chapter.")
    demo_chapter.add_argument("chapter", help="Chapter number from the textbook.")
    demo_chapter.add_argument("--workspace", default=str(_default_workspace()), help="Workspace directory.")
    demo_chapter.add_argument("--run", action="store_true", help="Execute the scenario instead of just describing it.")
    demo_chapter.set_defaults(func=cmd_demo)

    coverage = subparsers.add_parser("coverage", help="Print the generated concept coverage report.")
    coverage.set_defaults(func=cmd_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
