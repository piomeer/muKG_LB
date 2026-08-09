from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.synthesize_x1_5_mapping import (
    build_literature_mapping,
    build_novelty_evidence_matrix,
    summarize_mapping,
    write_mapping_outputs,
)


class X15MappingTests(unittest.TestCase):
    def fixture(self) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        records = [
            {
                "record_id": "r-direct",
                "title": "GPU negative sampling runtime",
                "year": "2022",
                "doi": "10.1000/direct",
                "peer_reviewed": "true",
                "peer_review_status": "VERIFIED",
                "topic_relevance": "IN_SCOPE",
                "full_text_status": "REMOTE_LOCATED",
                "c1_relevance": "DIRECT",
                "overlap_class": "DIRECT-FUNCTIONAL",
                "mechanism_match": "true",
                "integration_match": "true",
                "evidence_match": "false",
                "study_family_id": "family-a",
            },
            {
                "record_id": "r-component",
                "title": "GPU graph embedding systems",
                "year": "2021",
                "doi": "10.1000/component",
                "peer_reviewed": "true",
                "peer_review_status": "VERIFIED",
                "topic_relevance": "IN_SCOPE",
                "full_text_status": "LOCATED",
                "c1_relevance": "STRONG-COMPONENT",
                "overlap_class": "STRONG-COMPONENT",
                "mechanism_match": "true",
                "integration_match": "true",
                "evidence_match": "false",
                "study_family_id": "family-b",
            },
        ]
        screening = [
            {"record_id": "r-direct", "channel": "NEUTRAL_ELIGIBILITY", "decision": "INCLUDE"},
            {"record_id": "r-component", "channel": "NEUTRAL_ELIGIBILITY", "decision": "INCLUDE"},
        ]
        evidence = [
            {"record_id": "r-direct", "facet": "runtime_integration", "value": "GPU", "locator": "https://example.org/direct", "source_type": "WEB"}
        ]
        return records, screening, evidence

    def test_mapping_assigns_mq_facets_and_preserves_provenance(self) -> None:
        records, screening, evidence = self.fixture()
        rows = build_literature_mapping(records, screening, evidence)

        self.assertEqual([row["record_id"] for row in rows], ["r-component", "r-direct"])
        direct = next(row for row in rows if row["record_id"] == "r-direct")
        self.assertEqual(direct["mq1_gpu_negative_sampling"], "true")
        self.assertEqual(direct["mq2_runtime_integration"], "true")
        self.assertEqual(direct["evidence_locator_count"], "1")
        self.assertEqual(direct["study_family_id"], "family-a")

    def test_novelty_matrix_propagates_missing_evidence_and_gate_blockers(self) -> None:
        records, screening, evidence = self.fixture()
        decision = {"verdict": "UNRESOLVED", "blockers": ["human_adjudication_pending"]}
        rows = build_novelty_evidence_matrix(records, evidence, decision)

        direct = next(row for row in rows if row["record_id"] == "r-direct")
        self.assertEqual(direct["evidence_locator_count"], "1")
        self.assertEqual(direct["gate_verdict"], "UNRESOLVED")
        self.assertIn("human_adjudication_pending", direct["blocking_conditions"])

    def test_summary_counts_study_families_and_facets(self) -> None:
        records, screening, evidence = self.fixture()
        mapping = build_literature_mapping(records, screening, evidence)
        novelty = build_novelty_evidence_matrix(records, evidence, {"verdict": "UNRESOLVED", "blockers": ["x"]})
        summary = summarize_mapping(mapping, novelty, {"verdict": "UNRESOLVED", "blockers": ["x"]})

        self.assertEqual(summary["mapping_record_count"], 2)
        self.assertEqual(summary["study_family_count"], 2)
        self.assertEqual(summary["mq_counts"]["mq1_gpu_negative_sampling"], 1)
        self.assertEqual(summary["c1_gate_verdict"], "UNRESOLVED")

    def test_output_writer_is_byte_deterministic(self) -> None:
        records, screening, evidence = self.fixture()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (Path(first), Path(second)):
                (directory / "records.csv").write_text("", encoding="utf-8")
            first_paths = write_mapping_outputs(Path(first), Path(first), records=records, screening=screening, evidence=evidence, novelty_decision={"verdict": "UNRESOLVED", "blockers": ["x"]})
            second_paths = write_mapping_outputs(Path(second), Path(second), records=records, screening=screening, evidence=evidence, novelty_decision={"verdict": "UNRESOLVED", "blockers": ["x"]})
            first_bytes = {name: path.read_bytes() for name, path in sorted(first_paths.items())}
            second_bytes = {name: path.read_bytes() for name, path in sorted(second_paths.items())}
            self.assertEqual(first_bytes, second_bytes)
            with (Path(first) / "literature_mapping.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            json.loads((Path(first) / "mapping_summary.json").read_text(encoding="utf-8"))

    def test_output_writer_applies_manual_overlay_fields_and_passes_coverage(self) -> None:
        records, screening, evidence = self.fixture()
        records[0].pop("mechanism_match")
        records[0].pop("integration_match")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manual_adjudications.json").write_text(
                json.dumps([{"record_id": "r-direct", "mechanism_match": True, "integration_match": True, "evidence_match": False}]),
                encoding="utf-8",
            )
            paths = write_mapping_outputs(root, root, records=records, screening=screening, evidence=evidence, novelty_decision={"verdict": "UNRESOLVED", "blockers": []})
            checks = json.loads(paths["coverage_checks.json"].read_text(encoding="utf-8"))
            with paths["novelty_evidence_matrix.csv"].open(encoding="utf-8", newline="") as handle:
                novelty_rows = list(csv.DictReader(handle))
            direct = next(row for row in novelty_rows if row["record_id"] == "r-direct")
            self.assertEqual(direct["mechanism_match"], "true")
            self.assertEqual(direct["integration_match"], "true")
            self.assertEqual(checks["status"], "PASS")

    def test_missing_input_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                write_mapping_outputs(Path(directory), Path(directory))
