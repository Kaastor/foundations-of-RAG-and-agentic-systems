"""Named and chapter-based demos."""

from __future__ import annotations

from pathlib import Path

from raglab.agent.controller import run_agent
from raglab.build import index_staged_snapshot
from raglab.config import DEFAULT_CONFIG
from raglab.generation.closed_book import closed_book_answer
from raglab.generation.synthesizer import synthesize_answer
from raglab.ingest.pipeline import ingest_corpus
from raglab.ops.governance import load_named_user
from raglab.ops.publish import init_workspace, live_snapshot_path, publish_staged_snapshot
from raglab.retrieval.engine import KnowledgeBase


CHAPTER_SCENARIOS: dict[int, str] = {
    1: "foundations",
    2: "foundations",
    3: "basic_retrieval",
    4: "basic_retrieval",
    5: "basic_rag",
    6: "corpus_quality",
    7: "dense_vs_sparse",
    8: "dense_vs_sparse",
    9: "query_understanding",
    10: "basic_rag",
    11: "basic_rag",
    12: "basic_rag",
    13: "evaluation",
    14: "dense_vs_sparse",
    15: "structured_retrieval",
    16: "query_understanding",
    17: "multi_hop",
    18: "basic_rag",
    19: "freshness",
    20: "session_memory",
    21: "agentic",
    22: "agentic",
    23: "agentic",
    24: "agentic",
    25: "session_memory",
    26: "agentic",
    27: "agentic",
    28: "multi_agent",
    29: "agentic",
    30: "citations",
    31: "failure_modes",
    32: "security",
    33: "evaluation",
    34: "ops",
    35: "ops",
    36: "governance",
    37: "domain_agents",
    38: "freshness",
    39: "future_only",
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "foundations": "Compare a closed-book baseline with grounded retrieval on a source-sensitive query.",
    "basic_retrieval": "Run sparse retrieval and inspect BM25-style evidence selection.",
    "basic_rag": "Retrieve, rerank, pack context, and synthesize a grounded answer with citations.",
    "corpus_quality": "Inspect quarantine behavior, duplicate hints, and chunking decisions.",
    "dense_vs_sparse": "Compare sparse, dense, ANN-backed dense, and hybrid retrieval on exact and paraphrased queries.",
    "query_understanding": "Trigger ambiguity detection, rewrites, and clarification requests.",
    "structured_retrieval": "Combine structured lookup with text retrieval.",
    "multi_hop": "Follow references across multiple documents before answering.",
    "freshness": "Build a stale base snapshot, then publish an updated snapshot and compare results.",
    "session_memory": "Persist and reuse scoped session memory across turns.",
    "agentic": "Run the bounded controller with retrieval retries, tool use, and clarification.",
    "multi_agent": "Run the planner/retriever/writer/verifier specialization path.",
    "citations": "Show claim-level citation attachment and support verification.",
    "failure_modes": "Contrast stale, low-quality, and conflicting evidence handling.",
    "security": "Inspect quarantined malicious documents and trust-aware retrieval.",
    "evaluation": "Run the bundled benchmark and inspect per-query metrics.",
    "ops": "Inspect traces, caches, and snapshot publishing behavior.",
    "governance": "Contrast retrieval behavior under different user roles and disclosure policies.",
    "domain_agents": "Compare support and distributor-focused routes over the same corpus.",
    "future_only": "Read documentation only; this chapter is intentionally not executable in code.",
}


def examples_root() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


def demo_sources(include_updates: bool = True) -> list[Path]:
    root = examples_root() / "corpus"
    sources = [root / "base"]
    if include_updates:
        sources.append(root / "update")
    return sources


def ensure_demo_workspace(workspace: Path, include_updates: bool = True) -> Path:
    """Build and publish a demo snapshot if no live snapshot exists yet."""
    init_workspace(workspace)
    try:
        return live_snapshot_path(workspace)
    except FileNotFoundError:
        sources = demo_sources(include_updates=include_updates)
        ingest_corpus(sources, workspace, DEFAULT_CONFIG)
        index_staged_snapshot(workspace, DEFAULT_CONFIG)
        publish_staged_snapshot(workspace, note="demo")
        return live_snapshot_path(workspace)


def chapter_demo_text(chapter: int) -> str:
    scenario = CHAPTER_SCENARIOS.get(chapter, "basic_rag")
    description = SCENARIO_DESCRIPTIONS.get(scenario, "")
    return f"Chapter {chapter} -> scenario '{scenario}': {description}"


def run_chapter_demo(chapter: int, workspace: Path) -> str:
    scenario = CHAPTER_SCENARIOS.get(chapter, "basic_rag")
    root = examples_root()
    if scenario == "future_only":
        return chapter_demo_text(chapter) + "\nThis chapter is documented only in docs/chapters."

    include_updates = scenario != "freshness"
    snapshot = ensure_demo_workspace(workspace, include_updates=include_updates)
    kb = KnowledgeBase.load(snapshot)
    user_path = root / "users.json"

    if scenario == "foundations":
        query = "Does firmware 3.2 change the V14 installation torque, and where is that stated?"
        user = load_named_user(user_path, "field-eu")
        hits, packed, intent, _ = kb.retrieve(query, user=user, route="hybrid")
        grounded = synthesize_answer(query, packed, intent)
        closed = closed_book_answer(query)
        return (
            chapter_demo_text(chapter)
            + "\n\nClosed-book:\n"
            + closed.answer_text
            + "\n\nGrounded:\n"
            + grounded.answer_text
        )

    if scenario == "dense_vs_sparse":
        query = "What changed in tightening requirements after the 3.2 update for V14?"
        user = load_named_user(user_path, "field-eu")
        sparse_hits, _, _, _ = kb.retrieve(query, user=user, route="sparse")
        dense_hits, _, _, _ = kb.retrieve(query, user=user, route="dense")
        hybrid_hits, _, _, _ = kb.retrieve(query, user=user, route="hybrid")
        return (
            chapter_demo_text(chapter)
            + "\n\nSparse top hit: "
            + (sparse_hits[0].chunk.title if sparse_hits else "none")
            + "\nDense top hit: "
            + (dense_hits[0].chunk.title if dense_hits else "none")
            + "\nHybrid top hit: "
            + (hybrid_hits[0].chunk.title if hybrid_hits else "none")
        )

    if scenario == "structured_retrieval":
        query = "Which EU service centers support V14 sensor replacement under the current warranty program?"
        user = load_named_user(user_path, "distributor-eu")
        answer, _state = run_agent(query, kb, user=user, workspace=workspace, assumptions={"region": "EU", "product": "V14"})
        return chapter_demo_text(chapter) + "\n\n" + answer.answer_text

    if scenario == "multi_hop":
        query = "Which supplier's revised seal specification is referenced by bulletin SB-82, and what failure mode motivated it?"
        user = load_named_user(user_path, "compliance-analyst")
        answer, _state = run_agent(query, kb, user=user, workspace=workspace)
        return chapter_demo_text(chapter) + "\n\n" + answer.answer_text

    if scenario == "agentic":
        query = "Where is the rollback procedure for X12 staging key rotation documented?"
        user = load_named_user(user_path, "field-eu")
        answer, _state = run_agent(query, kb, user=user, workspace=workspace)
        return chapter_demo_text(chapter) + "\n\n" + answer.answer_text

    if scenario == "multi_agent":
        query = "Write a short answer for a distributor explaining the latest warranty exclusions for V14."
        user = load_named_user(user_path, "distributor-eu")
        answer, _state = run_agent(query, kb, user=user, workspace=workspace, assumptions={"region": "EU"}, multi_agent=True)
        return chapter_demo_text(chapter) + "\n\n" + answer.answer_text

    if scenario == "security":
        quarantine = (workspace / "staged" / "quarantine.jsonl")
        if not quarantine.exists():
            ingest_corpus(demo_sources(include_updates=True), workspace, DEFAULT_CONFIG)
        return chapter_demo_text(chapter) + f"\n\nInspect quarantine file: {quarantine}"

    return chapter_demo_text(chapter) + "\n\nUse the CLI commands documented in docs/chapters for the full walkthrough."
