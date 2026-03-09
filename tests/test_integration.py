from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from raglab.agent.controller import run_agent
from raglab.build import load_staged_chunks
from raglab.storage.json_store import read_jsonl
from raglab.config import DEFAULT_CONFIG
from raglab.evaluation.runner import run_benchmark
from raglab.generation.synthesizer import synthesize_answer
from raglab.ingest.pipeline import ingest_corpus
from raglab.ops.governance import load_named_user
from raglab.ops.publish import live_snapshot_path, publish_staged_snapshot, staged_snapshot_path
from raglab.retrieval.engine import KnowledgeBase
from raglab.build import index_staged_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"


class DemoWorkspaceMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="raglab-tests-"))
        cls.workspace = cls.temp_root / "workspace"
        sources = [EXAMPLES_ROOT / "corpus" / "base", EXAMPLES_ROOT / "corpus" / "update"]
        ingest_corpus(sources, cls.workspace, DEFAULT_CONFIG)
        index_staged_snapshot(cls.workspace, DEFAULT_CONFIG)
        publish_staged_snapshot(cls.workspace, note="tests")
        cls.kb = KnowledgeBase.load(live_snapshot_path(cls.workspace))
        cls.field_user = load_named_user(EXAMPLES_ROOT / "users.json", "field-eu")
        cls.distributor_user = load_named_user(EXAMPLES_ROOT / "users.json", "distributor-eu")
        cls.compliance_user = load_named_user(EXAMPLES_ROOT / "users.json", "compliance-analyst")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root)


class IngestionTests(DemoWorkspaceMixin):
    def test_quarantine_contains_malicious_and_low_quality_docs(self) -> None:
        quarantine_rows = read_jsonl(staged_snapshot_path(self.workspace) / "quarantine.jsonl")
        quarantined_ids = {row["doc_id"] for row in quarantine_rows}
        self.assertIn("UNTRUSTED-X12-NOTE", quarantined_ids)
        self.assertIn("OCR-SCAN-TORQUE", quarantined_ids)

    def test_table_row_chunk_exists_for_sb118(self) -> None:
        chunks = load_staged_chunks(self.workspace)
        table_rows = [chunk for chunk in chunks if chunk.doc_id == "SB-118" and "table row" in chunk.section.lower()]
        self.assertTrue(table_rows, "expected a row-level chunk from the markdown table")


class RetrievalTests(DemoWorkspaceMixin):
    def test_sparse_retrieval_finds_exact_identifier(self) -> None:
        hits, _packed, _intent, _diagnostics = self.kb.retrieve(
            "Which bulletin changed the torque for V14 and mentions SB-118?",
            user=self.field_user,
            route="sparse",
            top_k=5,
            rerank_top_m=3,
        )
        self.assertEqual(hits[0].chunk.doc_id, "SB-118")

    def test_dense_retrieval_handles_paraphrase(self) -> None:
        hits, _packed, _intent, _diagnostics = self.kb.retrieve(
            "What changed in tightening requirements after the 3.2 update for V14?",
            user=self.field_user,
            route="dense",
            top_k=5,
            rerank_top_m=3,
        )
        doc_ids = {hit.chunk.doc_id for hit in hits[:3]}
        self.assertIn("SB-118", doc_ids)


class AnswerAndAgentTests(DemoWorkspaceMixin):
    def test_grounded_answer_includes_supported_claim(self) -> None:
        hits, packed, intent, _diagnostics = self.kb.retrieve(
            "Does firmware 3.2 change the V14 installation torque, and where is that stated?",
            user=self.field_user,
            route="hybrid",
        )
        result = synthesize_answer(
            "Does firmware 3.2 change the V14 installation torque, and where is that stated?",
            packed,
            intent=intent,
        )
        self.assertFalse(result.abstained)
        self.assertTrue(any("0.48" in claim.text for claim in result.claims))
        self.assertTrue(all(claim.citations for claim in result.claims if claim.supported))

    def test_agent_requests_clarification_for_ambiguous_policy_query(self) -> None:
        result, _state = run_agent(
            "What changed in the latest distributor warranty terms for V14?",
            self.kb,
            user=self.distributor_user,
            workspace=self.workspace,
        )
        self.assertTrue(result.abstained)
        self.assertIsNotNone(result.clarification_request)

    def test_agent_can_use_structured_tool(self) -> None:
        result, _state = run_agent(
            "Which EU service centers support V14 sensor replacement under the current warranty program?",
            self.kb,
            user=self.distributor_user,
            workspace=self.workspace,
            assumptions={"region": "EU", "product": "V14"},
        )
        self.assertIn("Warsaw", result.answer_text)
        self.assertIn("Gdansk", result.answer_text)


class OpsAndCliTests(DemoWorkspaceMixin):
    def test_trace_file_created_for_agent_run(self) -> None:
        result, _state = run_agent(
            "Where is the rollback procedure for X12 staging key rotation documented?",
            self.kb,
            user=self.field_user,
            workspace=self.workspace,
        )
        self.assertTrue(result.trace_path)
        self.assertTrue(Path(result.trace_path).exists())

    def test_benchmark_runner_returns_metrics(self) -> None:
        report = run_benchmark(
            live_snapshot_path(self.workspace),
            EXAMPLES_ROOT / "eval" / "queries.jsonl",
            EXAMPLES_ROOT,
            mode="answer",
        )
        self.assertIn("retrieval", report)
        self.assertIn("answers", report)
        self.assertIn("recall@5", report["retrieval"])

    def test_python_module_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "raglab", "--help"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIn("Pedagogical reference implementation", completed.stdout)


if __name__ == "__main__":
    unittest.main()
