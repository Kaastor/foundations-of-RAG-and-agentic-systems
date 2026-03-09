"""Query understanding, retrieval, reranking, and context packing."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from raglab.config import AppConfig
from raglab.domain.models import ChunkRecord, DocumentRecord, QueryIntent, SearchHit, UserContext
from raglab.ops.governance import can_view
from raglab.ops.tracing import TraceRecorder
from raglab.retrieval.indexes import HashingEmbedder, SparseBM25Index, DenseVectorIndex, AnnLSHIndex, load_indexes
from raglab.storage.json_store import read_json, read_jsonl
from raglab.text import identifiers, normalize_query, split_sentences, tokenize, value_density


KNOWN_REGIONS = {
    "eu": "EU",
    "europe": "EU",
    "poland": "Poland",
    "north america": "North America",
    "na": "North America",
    "global": "global",
}
KNOWN_PRODUCTS = {"v14": "V14", "x12": "X12", "phoenix": "Phoenix"}

ALIASES = {
    "tightening requirements": ["installation torque"],
    "rollback": ["credential recovery", "rollback steps", "runbook"],
    "seal bulletin": ["SB-82", "elastomer revision"],
    "new seal bulletin": ["SB-82", "coastal maintenance bulletin"],
    "coastal units": ["coastal maintenance", "saline deployments"],
    "latest": ["current", "effective now"],
}

AMBIGUOUS_TERMS = {
    "phoenix": [
        "Phoenix product line",
        "Phoenix migration project",
        "Phoenix field office",
    ],
}

EXACT_IDENTIFIER_RE = re.compile(r"\b(?:SB-\d{1,4}|IR-\d{1,4}|HMX-\d{1,4}|V\d{1,3}|X\d{1,3}|FW-\d\.\d)\b", re.IGNORECASE)
AFTER_DATE_RE = re.compile(r"after (january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})", re.IGNORECASE)


@dataclass(slots=True)
class PackedContext:
    """The evidence actually passed to the synthesizer."""

    hits: list[SearchHit]
    rendered_blocks: list[str]
    token_count: int
    conflicts: list[str]

    def as_text(self) -> str:
        return "\n\n".join(self.rendered_blocks)


def _parse_loose_date(value: str) -> tuple[int, int, int]:
    if not value:
        return (0, 0, 0)
    parts = value.replace("/", "-").split("-")
    parts = [int(part) for part in parts if part.isdigit()]
    if len(parts) == 1:
        return (parts[0], 1, 1)
    if len(parts) == 2:
        return (parts[0], parts[1], 1)
    return (parts[0], parts[1], parts[2])


def _after_date_from_query(query: str) -> str | None:
    match = AFTER_DATE_RE.search(query)
    if not match:
        return None
    month_name, year_text = match.groups()
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = months[month_name.lower()]
    return f"{int(year_text):04d}-{month:02d}-01"


def infer_intent(query: str, user: UserContext | None = None) -> QueryIntent:
    """Infer task hints, filters, clarifications, and rewrites."""
    normalized = normalize_query(query)
    tokens = tokenize(normalized)
    ids = identifiers(normalized)
    task_type = "lookup"
    lowered = normalized.lower()

    if any(term in lowered for term in ["compare", "difference", "what changed", "changed after"]):
        task_type = "comparison"
    if any(term in lowered for term in ["where is", "documented", "procedure", "steps", "rollback"]):
        task_type = "procedural"
    if any(term in lowered for term in ["service center", "service centres", "structured", "which eu service centers"]):
        task_type = "structured"
    if any(term in lowered for term in ["which supplier", "failure mode", "why "]):
        task_type = "multi_hop"

    filters: dict[str, str] = {}
    expansions: list[str] = []
    subqueries: list[str] = []
    missing_scope: list[str] = []
    ambiguous_options: list[str] = []
    notes: list[str] = []

    for phrase, rewrites in ALIASES.items():
        if phrase in lowered:
            expansions.extend(rewrites)

    for term, options in AMBIGUOUS_TERMS.items():
        if term in lowered:
            ambiguous_options.extend(options)
            notes.append(f"'{term}' has multiple plausible corpus meanings.")

    for key, canonical in KNOWN_REGIONS.items():
        if key in lowered:
            filters["region"] = canonical
            break
    if user and user.region != "global" and "region" not in filters:
        notes.append("User profile region is available as a soft preference.")
    if any(key in lowered for key in ["warranty", "policy", "travel approval", "travel threshold"]) and "region" not in filters:
        if "poland" not in lowered and "north america" not in lowered and "eu" not in lowered:
            missing_scope.append("region")

    for key, canonical in KNOWN_PRODUCTS.items():
        if key in lowered:
            filters["product"] = canonical
            break

    if "customer-shareable" in lowered or "customer shareable" in lowered or "public-only" in lowered:
        filters["disclosure"] = "partner_shareable"
    elif "internal" in lowered or "incident report" in lowered:
        filters["disclosure"] = "internal"

    after_date = _after_date_from_query(normalized)
    if after_date:
        filters["after_date"] = after_date

    if "this quarter" in lowered and "travel" in lowered and "poland" in lowered:
        filters["after_date"] = "2025-01-01"

    if "and" in lowered and task_type in {"comparison", "multi_hop"}:
        parts = [part.strip() for part in re.split(r"\band\b", normalized, flags=re.IGNORECASE) if part.strip()]
        if len(parts) > 1:
            subqueries.extend(parts[:3])

    if "latest distributor warranty terms" in lowered and "region" not in filters:
        ambiguous_options.extend(["EU distributor policy", "North American distributor policy", "internal reseller terms"])
        missing_scope.append("policy family")

    return QueryIntent(
        raw_query=query,
        normalized_query=normalized,
        tokens=tokens,
        identifiers=ids,
        task_type=task_type,
        filters=filters,
        expansions=expansions,
        subqueries=subqueries,
        ambiguous_options=ambiguous_options,
        missing_scope=missing_scope,
        notes=notes,
    )


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], constant: int = 60) -> dict[str, float]:
    """Combine multiple rank lists without assuming score calibration."""
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for index, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (constant + index)
    return dict(fused)


def score_sentence(query_tokens: set[str], sentence: str) -> float:
    tokens = set(tokenize(sentence))
    overlap = len(query_tokens & tokens)
    identifier_bonus = sum(1 for term in query_tokens if term.isupper() and term.lower() in tokens)
    return overlap + identifier_bonus * 0.5


class KnowledgeBase:
    """Loaded view over a built snapshot."""

    def __init__(
        self,
        snapshot_path: Path,
        config: AppConfig,
        documents: dict[str, DocumentRecord],
        chunks: dict[str, ChunkRecord],
        sparse_index: SparseBM25Index,
        dense_index: DenseVectorIndex,
        ann_index: AnnLSHIndex,
        embedder: HashingEmbedder,
        structured: dict[str, Any],
    ) -> None:
        self.snapshot_path = snapshot_path
        self.config = config
        self.documents = documents
        self.chunks = chunks
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.ann_index = ann_index
        self.embedder = embedder
        self.structured = structured

    @classmethod
    def load(cls, snapshot_path: Path) -> "KnowledgeBase":
        manifest = read_json(snapshot_path / "manifest.json")
        config = AppConfig(**manifest["config"])
        documents = {row["doc_id"]: DocumentRecord.from_dict(row) for row in read_jsonl(snapshot_path / "docs.jsonl")}
        chunks = {row["chunk_id"]: ChunkRecord.from_dict(row) for row in read_jsonl(snapshot_path / "chunks.jsonl")}
        sparse_index, dense_index, ann_index, embedder = load_indexes(snapshot_path / "indexes")
        structured: dict[str, Any] = {}
        structured_dir = snapshot_path / "structured"
        if structured_dir.exists():
            for path in structured_dir.glob("*.json"):
                structured[path.stem] = read_json(path)
        return cls(
            snapshot_path=snapshot_path,
            config=config,
            documents=documents,
            chunks=chunks,
            sparse_index=sparse_index,
            dense_index=dense_index,
            ann_index=ann_index,
            embedder=embedder,
            structured=structured,
        )

    def user_visible_chunks(self, user: UserContext, intent: QueryIntent) -> set[str]:
        eligible: set[str] = set()
        for chunk in self.chunks.values():
            if not can_view(chunk.disclosure, user):
                continue
            if chunk.trust_score < self.config.low_trust_threshold:
                continue
            if intent.filters.get("product") and chunk.product and intent.filters["product"] != chunk.product:
                continue
            if intent.filters.get("region"):
                region = intent.filters["region"]
                if chunk.region not in {region, "global"}:
                    continue
            if intent.filters.get("disclosure"):
                disclosure = intent.filters["disclosure"]
                if disclosure == "partner_shareable" and chunk.disclosure not in {"public", "partner_shareable"}:
                    continue
                if disclosure == "internal" and chunk.disclosure == "public":
                    # keep internal and partner_shareable; drop purely public if user explicitly asked for internal detail
                    continue
            if intent.filters.get("after_date") and chunk.effective_date:
                if _parse_loose_date(chunk.effective_date) < _parse_loose_date(intent.filters["after_date"]):
                    continue
            eligible.add(chunk.chunk_id)
        return eligible

    def first_pass_search(
        self,
        query: str,
        user: UserContext,
        route: str,
        top_k: int,
        use_ann: bool = True,
        trace: TraceRecorder | None = None,
    ) -> tuple[list[SearchHit], dict[str, Any], QueryIntent]:
        """Run query understanding and first-pass search."""
        intent = infer_intent(query, user=user)
        span = trace.start_span("query_understanding", {"query": query}) if trace else None
        if span:
            span.finish(output_summary={"task_type": intent.task_type, "filters": intent.filters, "missing_scope": intent.missing_scope})
        eligible_ids = self.user_visible_chunks(user, intent)
        search_query = " ".join([intent.normalized_query] + intent.expansions[:2])

        if route == "sparse":
            ranking = [(item.chunk_id, item.score) for item in self.sparse_index.search(search_query, top_k=top_k * 2, k1=self.config.sparse_k1, b=self.config.sparse_b, eligible_ids=eligible_ids)]
            rankings = [ranking]
            diagnostics = {"route": "sparse"}
        elif route == "dense":
            if use_ann:
                query_vec = self.embedder.encode(search_query)
                candidates = self.ann_index.candidates(query_vec, probe_tables=self.config.ann_probe_tables)
                if eligible_ids:
                    candidates &= eligible_ids
                ranking = self.dense_index.search(search_query, self.embedder, top_k=top_k * 2, eligible_ids=candidates or eligible_ids)
                diagnostics = {"route": "dense_ann", "ann_candidates": len(candidates)}
            else:
                ranking = self.dense_index.search(search_query, self.embedder, top_k=top_k * 2, eligible_ids=eligible_ids)
                diagnostics = {"route": "dense_exact"}
            rankings = [ranking]
        else:
            sparse_ranking = [(item.chunk_id, item.score) for item in self.sparse_index.search(search_query, top_k=top_k * 2, k1=self.config.sparse_k1, b=self.config.sparse_b, eligible_ids=eligible_ids)]
            if use_ann:
                query_vec = self.embedder.encode(search_query)
                candidates = self.ann_index.candidates(query_vec, probe_tables=self.config.ann_probe_tables)
                if eligible_ids:
                    candidates &= eligible_ids
                dense_ranking = self.dense_index.search(search_query, self.embedder, top_k=top_k * 2, eligible_ids=candidates or eligible_ids)
            else:
                dense_ranking = self.dense_index.search(search_query, self.embedder, top_k=top_k * 2, eligible_ids=eligible_ids)
            fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])
            ranking = sorted(fused.items(), key=lambda item: item[1], reverse=True)
            rankings = [sparse_ranking, dense_ranking]
            diagnostics = {"route": "hybrid", "sparse_candidates": len(sparse_ranking), "dense_candidates": len(dense_ranking)}

        span = trace.start_span("first_pass_retrieval", {"route": route, "eligible_ids": len(eligible_ids)}) if trace else None
        hits = self._as_hits(ranking[: top_k * 2], rankings=rankings)
        if span:
            span.finish(
                output_summary={"hits": [hit.chunk.chunk_id for hit in hits[:top_k]]},
                diagnostics=diagnostics,
            )
        return hits[: top_k * 2], diagnostics, intent

    def _as_hits(self, ranking: list[tuple[str, float]], rankings: list[list[tuple[str, float]]]) -> list[SearchHit]:
        rank_lookup: list[dict[str, int]] = []
        score_lookup: list[dict[str, float]] = []
        for ranking_list in rankings:
            rank_lookup.append({chunk_id: index for index, (chunk_id, _score) in enumerate(ranking_list, start=1)})
            score_lookup.append({chunk_id: score for chunk_id, score in ranking_list})
        hits: list[SearchHit] = []
        for index, (chunk_id, score) in enumerate(ranking, start=1):
            chunk = self.chunks[chunk_id]
            scores = {
                f"signal_{signal_index + 1}": score_map.get(chunk_id, 0.0)
                for signal_index, score_map in enumerate(score_lookup)
            }
            reasons = []
            for identifier in identifiers(chunk.text + " " + chunk.title):
                if identifier.lower() in chunk.text.lower():
                    reasons.append(f"contains identifier {identifier}")
            hits.append(SearchHit(chunk=chunk, scores=scores, final_score=score, rank=index, reasons=reasons))
        return hits

    def rerank(self, query: str, hits: list[SearchHit], intent: QueryIntent, top_m: int) -> list[SearchHit]:
        """Refine first-pass candidates with simple metadata-aware heuristics."""
        query_tokens = set(tokenize(query))
        exact_ids = {value.lower() for value in intent.identifiers}
        reranked: list[SearchHit] = []
        for hit in hits:
            chunk = hit.chunk
            score = hit.final_score
            reasons = list(hit.reasons)
            chunk_tokens = set(tokenize(chunk.text))
            if exact_ids and exact_ids & chunk_tokens:
                score += 1.2
                reasons.append("exact identifier match")
            if intent.filters.get("product") and chunk.product == intent.filters["product"]:
                score += 0.4
                reasons.append("product filter match")
            if intent.filters.get("region") and chunk.region in {intent.filters["region"], "global"}:
                score += 0.3
                reasons.append("region filter match")
            if chunk.authority == "approved":
                score += 0.25
                reasons.append("approved authority")
            if chunk.status == "superseded":
                score -= 0.35
                reasons.append("superseded document penalty")
            if chunk.quality_score < 0.7:
                score -= 0.20
                reasons.append("low quality penalty")
            if chunk.trust_score < 0.7:
                score -= 0.20
                reasons.append("low trust penalty")
            overlap = len(query_tokens & chunk_tokens)
            score += overlap * 0.03
            reranked.append(SearchHit(chunk=chunk, scores=hit.scores, final_score=score, rank=hit.rank, reasons=reasons))

        reranked.sort(key=lambda item: item.final_score, reverse=True)

        selected: list[SearchHit] = []
        seen_docs: set[str] = set()
        for hit in reranked:
            redundancy_penalty = 0.0
            if hit.chunk.doc_id in seen_docs:
                redundancy_penalty += 0.15
            adjusted_score = hit.final_score - redundancy_penalty
            if adjusted_score < 0:
                continue
            selected.append(SearchHit(chunk=hit.chunk, scores=hit.scores, final_score=adjusted_score, rank=len(selected) + 1, reasons=hit.reasons + (["diversity penalty"] if redundancy_penalty else [])))
            seen_docs.add(hit.chunk.doc_id)
            if len(selected) >= top_m:
                break
        return selected

    def detect_conflicts(self, hits: Iterable[SearchHit]) -> list[str]:
        """Detect obvious stale/new or conflicting-version situations."""
        grouped: defaultdict[tuple[str, str], list[SearchHit]] = defaultdict(list)
        for hit in hits:
            key = (hit.chunk.product, hit.chunk.region)
            grouped[key].append(hit)
        conflicts: list[str] = []
        for (_product, _region), group_hits in grouped.items():
            active_titles = {hit.chunk.title for hit in group_hits if hit.chunk.status == "active"}
            superseded_titles = {hit.chunk.title for hit in group_hits if hit.chunk.status == "superseded"}
            if active_titles and superseded_titles:
                conflicts.append(
                    f"Active and superseded sources appear together: active={sorted(active_titles)}, superseded={sorted(superseded_titles)}"
                )
        return conflicts

    def pack_context(self, query: str, hits: list[SearchHit], budget_tokens: int) -> PackedContext:
        """Build a token-budgeted context from reranked evidence."""
        query_tokens = set(tokenize(query))
        selected_hits: list[SearchHit] = []
        rendered_blocks: list[str] = []
        token_count = 0
        for hit in hits:
            sentences = split_sentences(hit.chunk.text)
            if not sentences:
                continue
            scored_sentences = sorted(sentences, key=lambda sentence: score_sentence(query_tokens, sentence), reverse=True)
            best_sentences = scored_sentences[:2]
            block = f"[{hit.chunk.doc_id} | {hit.chunk.section}] " + " ".join(best_sentences)
            block_tokens = len(tokenize(block))
            if token_count + block_tokens > budget_tokens:
                continue
            selected_hits.append(hit)
            rendered_blocks.append(block)
            token_count += block_tokens
        conflicts = self.detect_conflicts(selected_hits)
        return PackedContext(hits=selected_hits, rendered_blocks=rendered_blocks, token_count=token_count, conflicts=conflicts)

    def retrieve(
        self,
        query: str,
        user: UserContext,
        route: str = "hybrid",
        top_k: int | None = None,
        rerank_top_m: int | None = None,
        budget_tokens: int | None = None,
        use_ann: bool = True,
        trace: TraceRecorder | None = None,
    ) -> tuple[list[SearchHit], PackedContext, QueryIntent, dict[str, Any]]:
        """Retrieve, rerank, and pack context."""
        top_k = top_k or self.config.default_top_k
        rerank_top_m = rerank_top_m or self.config.default_rerank_top_m
        budget_tokens = budget_tokens or self.config.default_context_budget
        first_pass_hits, diagnostics, intent = self.first_pass_search(query, user=user, route=route, top_k=top_k, use_ann=use_ann, trace=trace)
        span = trace.start_span("rerank", {"input_hits": len(first_pass_hits)}) if trace else None
        reranked = self.rerank(query, first_pass_hits, intent=intent, top_m=rerank_top_m)
        if span:
            span.finish(output_summary={"top_hits": [hit.chunk.chunk_id for hit in reranked]})
        span = trace.start_span("context_pack", {"budget_tokens": budget_tokens}) if trace else None
        packed = self.pack_context(query, reranked, budget_tokens=budget_tokens)
        if span:
            span.finish(output_summary={"packed_hits": [hit.chunk.chunk_id for hit in packed.hits], "token_count": packed.token_count}, diagnostics={"conflicts": packed.conflicts})
        diagnostics.update(
            {
                "missing_scope": intent.missing_scope,
                "ambiguous_options": intent.ambiguous_options,
                "packed_token_count": packed.token_count,
                "conflicts": packed.conflicts,
                "exact_identifiers": intent.identifiers,
            }
        )
        return reranked, packed, intent, diagnostics

    def follow_references(self, hits: Iterable[SearchHit]) -> list[ChunkRecord]:
        """Return chunks whose doc IDs were explicitly referenced by retrieved evidence."""
        referenced_doc_ids: set[str] = set()
        for hit in hits:
            referenced_doc_ids.update(hit.chunk.references)
            text_refs = set(EXACT_IDENTIFIER_RE.findall(hit.chunk.text))
            referenced_doc_ids.update(text_refs)
        followed: list[ChunkRecord] = []
        for chunk in self.chunks.values():
            if chunk.doc_id in referenced_doc_ids:
                followed.append(chunk)
        return followed
