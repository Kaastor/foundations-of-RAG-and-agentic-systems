# Assumptions and deliberate simplifications

This repository follows the textbook closely in architecture and control flow, but it makes several small, explicit assumptions so the code can stay runnable with only the Python standard library.

## Core assumptions

1. **Dense retrieval uses deterministic hashed vectors instead of neural embeddings.**  
   This keeps the repository dependency-free while still demonstrating vector representations, cosine similarity, and approximate nearest-neighbor search.

2. **Approximate nearest-neighbor search uses LSH instead of HNSW or IVF/PQ.**  
   The textbook discusses several ANN families. The repository implements a simpler locality-sensitive hashing index so learners can read the full algorithm in one sitting.

3. **Answer synthesis is extractive and rule-based rather than LLM-driven.**  
   The code demonstrates grounded answering, claim support, citations, abstention, and partial answers without requiring external model APIs or heavyweight local ML stacks.

4. **Messy data is simulated with noisy text, markdown tables, duplicates, and quarantined files.**  
   The sample corpus includes low-OCR text and a malicious note, but it does not parse real PDFs.

5. **Structured tools use local JSON files.**  
   This stands in for SQL queries, workflow APIs, and structured operational systems.

6. **Governance, privacy, and security are policy hooks, not enterprise control planes.**  
   The repository includes role-based disclosure controls, tool guards, trace retention tags, and trust scoring, but it does not implement real IAM, SIEM, or regulatory workflows.

## Why these assumptions are acceptable

The textbook is primarily about systems structure: clean boundaries, evidence flow, routing, correction, observability, and operational trade-offs. Those lessons remain visible even when the retrieval and generation primitives are simplified.

## What to do next if you want a less toy-like build

- Replace `HashingEmbedder` with a real embedding model.
- Swap `AnnLSHIndex` for an HNSW or IVF/PQ backend.
- Replace the extractive synthesizer with an LLM-backed generator and keep the same claim-verification interface.
- Back structured tools with SQL or service APIs instead of JSON.
- Move traces and caches to a distributed telemetry stack.
