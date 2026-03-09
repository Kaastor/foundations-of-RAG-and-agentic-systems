"""Offline ingestion pipeline for the sample corpus."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from raglab.config import AppConfig
from raglab.domain.models import ChunkRecord, DocumentRecord
from raglab.ops.publish import init_workspace, staged_snapshot_path
from raglab.ops.security import instruction_like_language, quality_score, should_quarantine, trust_score
from raglab.storage.json_store import ensure_dir, write_json, write_jsonl
from raglab.text import jaccard_similarity, normalize_whitespace, tokenize


_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw_meta, body = match.groups()
    metadata: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body


def _parse_text_file(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(text)
    return metadata, body


def _parse_csv_file(path: Path) -> tuple[dict[str, str], str, list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(text)
    rows = list(csv.DictReader(body.splitlines()))
    return metadata, body, rows


def _build_document(path: Path, text: str, metadata: dict[str, str]) -> DocumentRecord:
    title = metadata.get("title") or path.stem.replace("_", " ").title()
    reasons = []
    suspicious_hits = instruction_like_language(text)
    if suspicious_hits:
        reasons.append(f"suspicious:{'; '.join(suspicious_hits)}")
    ocr_confidence = float(metadata.get("ocr_confidence", "1.0"))
    if ocr_confidence < 0.75:
        reasons.append("low_ocr_confidence")
    quality = quality_score(text, title=title, ocr_confidence=ocr_confidence)
    trust = trust_score(text, metadata_title=title)
    allowed_roles = tuple(item.strip() for item in metadata.get("allowed_roles", "field_support").split(",") if item.strip())
    tags = tuple(item.strip() for item in metadata.get("tags", "").split(",") if item.strip())
    references = tuple(item.strip() for item in metadata.get("references", "").split(",") if item.strip())
    doc = DocumentRecord(
        doc_id=metadata.get("doc_id", path.stem),
        title=title,
        doc_type=metadata.get("doc_type", path.suffix.lstrip(".") or "text"),
        text=normalize_whitespace(text),
        source_path=str(path),
        product=metadata.get("product", ""),
        region=metadata.get("region", "global"),
        effective_date=metadata.get("effective_date", ""),
        version=metadata.get("version", ""),
        authority=metadata.get("authority", "reference"),
        status=metadata.get("status", "active"),
        disclosure=metadata.get("disclosure", "internal"),
        allowed_roles=allowed_roles or ("field_support",),
        tags=tags,
        references=references,
        trust_label=metadata.get("trust", "trusted"),
        quality_score=quality,
        trust_score=trust,
        quarantine_reason="; ".join(reasons) if should_quarantine(quality, trust, reasons) or any(reason == "low_ocr_confidence" for reason in reasons) else None,
        metadata={key: value for key, value in metadata.items() if key not in {"allowed_roles", "tags", "references"}},
    )
    return doc


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Body"
    current_lines: list[str] = []
    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = heading_match.group(2).strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, body) for title, body in sections if body.strip()]


def _table_chunks(section_text: str, section_title: str) -> list[tuple[str, str]]:
    lines = [line for line in section_text.splitlines() if line.strip()]
    chunks: list[tuple[str, str]] = []
    current_table: list[str] = []
    for line in lines:
        if "|" in line and line.count("|") >= 2:
            current_table.append(line)
        else:
            if len(current_table) >= 2:
                chunks.extend(_materialize_markdown_table(current_table, section_title))
            current_table = []
    if len(current_table) >= 2:
        chunks.extend(_materialize_markdown_table(current_table, section_title))
    return chunks


def _materialize_markdown_table(lines: list[str], section_title: str) -> list[tuple[str, str]]:
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    body_lines = [line for line in lines[2:] if set(line.strip()) != {"|", "-"}]
    results: list[tuple[str, str]] = []
    for row_line in body_lines:
        cells = [cell.strip() for cell in row_line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rendered = "; ".join(f"{column}: {value}" for column, value in zip(header, cells))
        results.append((f"{section_title} table row", rendered))
    return results


def _sliding_windows(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    tokens = tokenize(text)
    if len(tokens) <= chunk_tokens:
        return [text.strip()]
    windows: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_tokens)
        token_window = set(tokens[start:end])
        sentence_candidates = [line.strip() for line in text.splitlines() if line.strip()]
        selected = [line for line in sentence_candidates if token_window & set(tokenize(line))]
        if not selected:
            selected = sentence_candidates
        rendered = " ".join(selected[: max(1, chunk_tokens // 24)])
        windows.append(rendered.strip())
        if end == len(tokens):
            break
        start = max(end - overlap, start + 1)
    return windows


def chunk_document(document: DocumentRecord, config: AppConfig) -> list[ChunkRecord]:
    """Chunk one document with a mix of section-aware and row-aware logic."""
    chunks: list[ChunkRecord] = []
    if document.doc_type == "csv":
        metadata, _, rows = _parse_csv_file(Path(document.source_path))
        for index, row in enumerate(rows):
            rendered = "; ".join(f"{key}: {value}" for key, value in row.items())
            chunk_text = f"{document.title}. Row {index + 1}. {rendered}"
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}:row-{index + 1:02d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=chunk_text,
                    section="row",
                    order=index,
                    token_count=len(tokenize(chunk_text)),
                    product=document.product,
                    region=document.region,
                    effective_date=document.effective_date,
                    version=document.version,
                    authority=document.authority,
                    status=document.status,
                    disclosure=document.disclosure,
                    allowed_roles=document.allowed_roles,
                    tags=document.tags,
                    references=document.references,
                    trust_score=document.trust_score,
                    quality_score=document.quality_score,
                    metadata={"row": row, **document.metadata},
                )
            )
        return chunks

    sections = _markdown_sections(document.text)
    if not sections:
        sections = [("Body", document.text)]

    order = 0
    for section_title, section_text in sections:
        table_rows = _table_chunks(section_text, section_title)
        for table_title, table_text in table_rows:
            chunk_text = f"{document.title}. {table_title}. {table_text}"
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}:chunk-{order:02d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=chunk_text,
                    section=table_title,
                    order=order,
                    token_count=len(tokenize(chunk_text)),
                    product=document.product,
                    region=document.region,
                    effective_date=document.effective_date,
                    version=document.version,
                    authority=document.authority,
                    status=document.status,
                    disclosure=document.disclosure,
                    allowed_roles=document.allowed_roles,
                    tags=document.tags,
                    references=document.references,
                    trust_score=document.trust_score,
                    quality_score=document.quality_score,
                    metadata=document.metadata,
                )
            )
            order += 1

        windows = _sliding_windows(section_text, config.chunk_tokens, config.chunk_overlap)
        for window_text in windows:
            chunk_text = f"{document.title}. {section_title}. {window_text}"
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}:chunk-{order:02d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=chunk_text,
                    section=section_title,
                    order=order,
                    token_count=len(tokenize(chunk_text)),
                    product=document.product,
                    region=document.region,
                    effective_date=document.effective_date,
                    version=document.version,
                    authority=document.authority,
                    status=document.status,
                    disclosure=document.disclosure,
                    allowed_roles=document.allowed_roles,
                    tags=document.tags,
                    references=document.references,
                    trust_score=document.trust_score,
                    quality_score=document.quality_score,
                    metadata=document.metadata,
                )
            )
            order += 1
    return chunks


def _duplicate_key(document: DocumentRecord) -> tuple[str, str, str]:
    hash_digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    return (document.title.lower(), document.product.lower(), hash_digest)


def _near_duplicate_signature(document: DocumentRecord) -> set[str]:
    return set(tokenize(document.title + " " + document.text)) - {"the", "and", "for", "with", "this", "that"}


def ingest_corpus(source_roots: Iterable[str | Path], workspace: Path, config: AppConfig) -> dict[str, Any]:
    """Ingest source files into the staged workspace.

    The function writes:
    - staged/docs.jsonl
    - staged/chunks.jsonl
    - staged/quarantine.jsonl
    - staged/structured/*
    - staged/manifest.json
    """
    init_workspace(workspace)
    staged = staged_snapshot_path(workspace)
    if staged.exists():
        shutil.rmtree(staged)
    ensure_dir(staged)
    structured_out = ensure_dir(staged / "structured")

    documents: list[DocumentRecord] = []
    quarantined: list[DocumentRecord] = []
    exact_seen: dict[tuple[str, str, str], str] = {}
    near_seen: list[tuple[str, set[str]]] = []
    structured_files: list[str] = []

    for root in [Path(item).resolve() for item in source_roots]:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if "structured" in path.parts and path.suffix == ".json":
                destination = structured_out / path.name
                shutil.copy2(path, destination)
                structured_files.append(str(destination))
                continue
            if path.suffix not in {".md", ".txt", ".csv"}:
                continue

            if path.suffix == ".csv":
                metadata, body, _ = _parse_csv_file(path)
                document = _build_document(path, body, metadata)
            else:
                metadata, body = _parse_text_file(path)
                document = _build_document(path, body, metadata)

            exact_key = _duplicate_key(document)
            if exact_key in exact_seen:
                document.duplicate_of = exact_seen[exact_key]
            else:
                exact_seen[exact_key] = document.doc_id

            signature = _near_duplicate_signature(document)
            for other_doc_id, other_signature in near_seen:
                similarity = jaccard_similarity(signature, other_signature)
                if similarity >= 0.92 and other_doc_id != document.doc_id:
                    document.metadata["near_duplicate_of"] = other_doc_id
                    document.metadata["near_duplicate_similarity"] = round(similarity, 3)
                    break
            near_seen.append((document.doc_id, signature))

            if document.quarantine_reason:
                quarantined.append(document)
            else:
                documents.append(document)

    chunks: list[ChunkRecord] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))

    manifest = {
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "source_roots": [str(Path(item).resolve()) for item in source_roots],
        "document_count": len(documents),
        "quarantine_count": len(quarantined),
        "chunk_count": len(chunks),
        "structured_files": structured_files,
        "config": config.to_dict(),
        "notes": [
            "Structured files are copied alongside staged documents for tool access.",
            "Near-duplicate hints are recorded in document metadata, not removed aggressively.",
        ],
    }

    write_jsonl(staged / "docs.jsonl", [document.to_dict() for document in documents])
    write_jsonl(staged / "chunks.jsonl", [chunk.to_dict() for chunk in chunks])
    write_jsonl(staged / "quarantine.jsonl", [document.to_dict() for document in quarantined])
    write_json(staged / "manifest.json", manifest)

    return manifest
