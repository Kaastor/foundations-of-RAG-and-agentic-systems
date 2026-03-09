# Foundations of RAG and Agentic Systems

A complete, runnable, standard-library-only teaching repository that maps the attached *Agentic RAG* textbook into code. It implements a miniature but realistic system with offline ingestion, sparse/dense/hybrid retrieval, ANN search, reranking, context packing, grounded answer synthesis, an agentic controller, tool schemas, session memory, evaluation, tracing, publishing, and governance hooks.

The attached textbook moves from foundational RAG ideas through advanced retrieval, agentic control, reliability, safety, and production operations. This repository mirrors that arc with a small codebase you can actually read end to end. 

## What this repository is

This is **not** a production search stack and it is **not** a benchmark-chasing demo. It is a pedagogical reference implementation whose job is to make the textbook's system boundaries visible in code:

- offline preparation vs online serving
- corpus quality and quarantine
- sparse, dense, ANN, and hybrid retrieval
- reranking and context packing
- grounded answer synthesis with citations
- agentic retrieval, tool use, clarification, and escalation
- evaluation, tracing, caching hooks, governance, and publishable snapshots

## How it maps to the textbook

The textbook covers:

- foundations of parametric vs explicit knowledge
- basic RAG construction
- advanced retrieval design
- agentic RAG control loops
- reliability, safety, observability, and operations
- enterprise deployment patterns and future directions

This repository maps those chapters into four code strata:

1. **Knowledge preparation**  
   `raglab/ingest/*`, `raglab/build.py`, `raglab/ops/publish.py`

2. **Retrieval and grounded answering**  
   `raglab/retrieval/*`, `raglab/generation/*`

3. **Agentic control and tools**  
   `raglab/agent/*`

4. **Evaluation and operations**  
   `raglab/evaluation/*`, `raglab/ops/*`

Chapter-by-chapter documentation lives in `docs/chapters/`, and the concept inventory plus coverage status lives in `docs/concept_coverage.md`.

## Installation with Poetry

The project is configured as a Poetry package.

```bash
poetry install
```

Then run the CLI with either form:

```bash
poetry run raglab --help
poetry run python -m raglab --help
```

## Quick start

Prepare the bundled demo workspace:

```bash
poetry run raglab demo prepare --workspace .workspace/demo
```

Retrieve evidence only:

```bash
poetry run raglab retrieve "Which bulletin changed the torque for V14 and mentions SB-118?"   --workspace .workspace/demo   --user-id field-eu   --route sparse
```

Run the fixed non-agentic workflow:

```bash
poetry run raglab answer "Does firmware 3.2 change the V14 installation torque, and where is that stated?"   --workspace .workspace/demo   --user-id field-eu
```

Run the agentic controller:

```bash
poetry run raglab agent "Where is the rollback procedure for X12 staging key rotation documented?"   --workspace .workspace/demo   --user-id field-eu
```

Run a chapter demo:

```bash
poetry run raglab demo chapter 24 --workspace .workspace/demo --run
```

Run the benchmark:

```bash
poetry run raglab evaluate --workspace .workspace/demo --mode answer
```

Inspect the most recent trace:

```bash
TRACE_ID=$(poetry run python - <<'PY'
from pathlib import Path

trace = max(Path(".workspace/demo/traces").glob("*.json"), key=lambda path: path.stat().st_mtime)
print(trace.stem)
PY
)
poetry run raglab trace "$TRACE_ID" --workspace .workspace/demo
```

## How to rebuild the sample knowledge base from source

The bundled examples are split into a base corpus and an update corpus so you can demonstrate freshness and staged publishing.

```bash
poetry run raglab ingest   --source examples/corpus/base   --source examples/corpus/update   --workspace .workspace/demo

poetry run raglab index --workspace .workspace/demo
poetry run raglab publish --workspace .workspace/demo --note "initial demo publish"
```

## How to run tests

```bash
poetry run python -m unittest -v
```

## Where to start reading the code

1. `raglab/cli.py`  
   The CLI is the best high-level map of the system.

2. `raglab/ingest/pipeline.py`  
   Shows how source documents become normalized documents, quarantined failures, and chunks.

3. `raglab/retrieval/engine.py`  
   Handles query understanding, retrieval, reranking, and context packing.

4. `raglab/generation/synthesizer.py` and `raglab/generation/verify.py`  
   Shows grounded answer construction and citation verification.

5. `raglab/agent/controller.py`  
   Shows the stateful control loop, clarification, retries, route changes, tool use, and escalation.

## Documentation map

- `docs/architecture.md` - system structure and data flow
- `docs/assumptions.md` - explicit simplifications
- `docs/chapters/README.md` - per-chapter docs index
- `docs/concept_coverage.md` - concept inventory and honest coverage report

## Scope and simplifications

This repository intentionally stays dependency-light and readable.

- Dense retrieval uses deterministic hashed vectors instead of neural embeddings.
- Approximate search uses LSH instead of HNSW or IVF/PQ.
- Answer synthesis is extractive and rule-based instead of LLM-backed.
- Structured tools are local JSON readers instead of SQL or external APIs.
- Governance, security, and compliance are represented by explicit policy hooks rather than enterprise infrastructure.

Those simplifications are documented, not hidden. The goal is to preserve the textbook's systems lessons without burying them under vendor SDKs or opaque frameworks.
