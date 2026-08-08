from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.close_x1_5_c1_gate import (
    assess_gate_closure,
    build_adjudication_queue,
    build_source_verification_status,
    write_closure_outputs,
)


class C1GateClosureTests(unittest.TestCase):
    def fixture(self):
        novelty = [{
            "record_id": "r-direct",
            "title": "GPU negative sampling",
            "study_family_id": "f1",
            "c1_relevance": "DIRECT",
            "overlap_class": "DIRECT-FUNCTIONAL",
            "peer_review_status": "VERIFIED",
            "full_text_status": "REMOTE_LOCATED",
            "evidence_locator_count": "1",
            "blocking_conditions": "",
        }]
        human = [{
            "record_id": "r-direct", "issue_type": "POTENTIAL_C1", "priority": "HIGH",
            "status": "PENDING", "decision": "", "adjudicator_note": "verify",
        }]
        pages = [{"index": "DBLP", "status": "FAILED", "query": "q"}]
        decision = {"claim_id": "C1", "verdict": "UNRESOLVED", "blockers": ["human_adjudication_pending", "retrieval_failed"]}
        return novelty, human, pages, decision

    def test_queue_orders_high_priority_and_preserves_pending_issue(self) -> None:
        novelty, human, _, _ = self.fixture()
        queue = build_adjudication_queue(human)
        self.assertEqual(queue[0]["priority"], "HIGH")
        self.assertEqual(queue[0]["record_id"], "r-direct")
        self.assertEqual(queue[0]["status"], "PENDING")

    def test_source_status_propagates_human_and_retrieval_blockers(self) -> None:
        novelty, human, pages, _ = self.fixture()
        status = build_source_verification_status(novelty, human, pages)
        self.assertIn("human_adjudication_pending", status[0]["blocking_conditions"])
        self.assertIn("retrieval_failed", status[0]["blocking_conditions"])

    def test_source_status_does_not_relabel_verified_candidate_from_global_matrix_blockers(self) -> None:
        novelty, human, pages, _ = self.fixture()
        novelty[0]["blocking_conditions"] = "human_adjudication_pending;peer_review_status_unverified;retrieval_failed"
        human[0]["status"] = "RESOLVED"
        pages = [{"index": "DBLP", "status": "OK", "query": "q"}]
        status = build_source_verification_status(novelty, human, pages)
        self.assertEqual(status[0]["blocking_conditions"], "")

    def test_gate_is_fail_closed_until_queue_and_retrieval_are_resolved(self) -> None:
        novelty, human, pages, decision = self.fixture()
        queue = build_adjudication_queue(human)
        status = build_source_verification_status(novelty, human, pages)
        result = assess_gate_closure(decision, queue, status, pages, {"status": "OPEN"})
        self.assertEqual(result["closure_status"], "UNRESOLVED")
        self.assertIn("human_adjudication_pending", result["blockers"])
        self.assertIn("retrieval_failed", result["blockers"])

    def test_gate_can_be_ready_without_issuing_a_substantive_verdict(self) -> None:
        novelty, human, _, decision = self.fixture()
        human[0]["status"] = "RESOLVED"
        pages = [{"index": "DBLP", "status": "OK", "query": "q"}]
        queue = build_adjudication_queue(human)
        status = build_source_verification_status(novelty, human, pages)
        result = assess_gate_closure({"verdict": "UNRESOLVED", "blockers": []}, queue, status, pages, {"status": "CLOSED"})
        self.assertEqual(result["closure_status"], "READY_FOR_HUMAN_DECISION")
        self.assertEqual(result["c1_verdict"], "UNRESOLVED")

    def test_writer_is_deterministic_and_serializes_csv_json(self) -> None:
        novelty, human, pages, decision = self.fixture()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            paths_a = write_closure_outputs(Path(first), Path(first), novelty=novelty, human=human, pages=pages, novelty_decision=decision, cutoff={"status": "OPEN"})
            paths_b = write_closure_outputs(Path(second), Path(second), novelty=novelty, human=human, pages=pages, novelty_decision=decision, cutoff={"status": "OPEN"})
            self.assertEqual({k: v.read_bytes() for k, v in paths_a.items()}, {k: v.read_bytes() for k, v in paths_b.items()})
            with paths_a["c1_adjudication_queue.csv"].open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            json.loads(paths_a["c1_gate_closure.json"].read_text(encoding="utf-8"))

    def test_qualified_fallback_is_not_a_c1_retrieval_blocker(self) -> None:
        novelty, human, pages, decision = self.fixture()
        human[0]["status"] = "RESOLVED"
        closure = {
            "status": "CLOSED_WITH_FALLBACK",
            "blocking_conditions": [],
            "advisories": ["dblp_missing_page_with_qualified_fallback"],
        }
        status = build_source_verification_status(novelty, human, pages, closure)
        self.assertEqual(status[0]["retrieval_status"], "CLOSED_WITH_FALLBACK")
        self.assertNotIn("retrieval_failed", status[0]["blocking_conditions"])
        result = assess_gate_closure(
            {"verdict": "UNRESOLVED", "blockers": ["retrieval_failed"]},
            [], status, pages, {"status": "CLOSED"}, closure,
        )
        self.assertEqual(result["closure_status"], "READY_FOR_HUMAN_DECISION")
        self.assertNotIn("retrieval_failed", result["blockers"])
        self.assertEqual(result["retrieval_status"], "CLOSED_WITH_FALLBACK")

    def test_uncovered_fallback_remains_a_hard_gate_blocker(self) -> None:
        novelty, human, pages, decision = self.fixture()
        human[0]["status"] = "RESOLVED"
        closure = {"status": "CLOSED_BLOCKED", "blocking_conditions": ["retrieval_gap_uncovered"], "advisories": []}
        status = build_source_verification_status(novelty, human, pages, closure)
        self.assertIn("retrieval_gap_uncovered", status[0]["blocking_conditions"])
        result = assess_gate_closure(
            {"verdict": "UNRESOLVED", "blockers": []}, [], status, pages, {"status": "CLOSED"}, closure,
        )
        self.assertEqual(result["closure_status"], "UNRESOLVED")
        self.assertIn("retrieval_gap_uncovered", result["blockers"])

    def test_writer_derives_closed_with_fallback_without_stale_retrieval_blocker(self) -> None:
        novelty, human, pages, decision = self.fixture()
        human[0]["status"] = "RESOLVED"
        pages = [{"index": "DBLP", "status": "FAILED", "query": "q", "retrieval_stage": "G0_SEEDS"}]
        fallback = [{
            "failed_index": "DBLP", "query": "q", "retrieval_stage": "G0_SEEDS",
            "qualification": "QUALIFIED", "fallback_status": "AVAILABLE",
            "fallback_indexes": "OpenAlex", "matched_record_ids": "doi:q",
            "raw_evidence": "OpenAlex:" + "a" * 64, "disposition": "ADVISORY_ONLY",
        }]
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            input_dir.joinpath("fallback_coverage.csv").write_text(
                "failed_index,query,retrieval_stage,fallback_status,fallback_indexes,qualification,matched_record_ids,raw_evidence,disposition\n"
                "DBLP,q,G0_SEEDS,AVAILABLE,OpenAlex,QUALIFIED,doi:q,OpenAlex:" + "a" * 64 + ",ADVISORY_ONLY\n",
                encoding="utf-8",
            )
            paths = write_closure_outputs(
                input_dir, input_dir, novelty=novelty, human=human, pages=pages,
                novelty_decision=decision, cutoff={"status": "CLOSED"},
            )
            closure = json.loads(paths["c1_gate_closure.json"].read_text(encoding="utf-8"))
            self.assertEqual(closure["retrieval_status"], "CLOSED_WITH_FALLBACK")
            self.assertNotIn("retrieval_failed", closure["blockers"])
