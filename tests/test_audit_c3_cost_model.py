from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_c3_cost_model import (
    build_audit,
    run_cpu_fixtures,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class AuditC3CostModelTests(unittest.TestCase):
    def test_cpu_fixtures_cover_metric_and_filter_invariants(self) -> None:
        fixtures = run_cpu_fixtures(REPO_ROOT)
        self.assertAlmostEqual(fixtures["pearson_r"], 0.8, places=12)
        self.assertAlmostEqual(fixtures["r_squared"], 0.64, places=12)
        self.assertAlmostEqual(fixtures["ols_r_squared"], 0.64, places=12)
        self.assertEqual(fixtures["full_rows"], 4)
        self.assertEqual(fixtures["complete_rows"], 3)
        self.assertEqual(fixtures["rounded_rows"], 2)
        self.assertAlmostEqual(fixtures["crossover_linear_n"], 225000.0, places=6)

    def test_repository_audit_has_one_grade_per_c3_claim_and_expected_lineage_flags(self) -> None:
        audit = build_audit(REPO_ROOT)
        grades = {row["claim_id"]: row["grade"] for row in audit["claim_verdicts"]}
        self.assertEqual(
            grades,
            {
                "C3.1-L": "D",
                "C3.1-R1": "C",
                "C3.2": "C",
                "C3.3": "A",
                "C3.4": "C",
                "C3.5": "D",
                "C3.6": "A",
            },
        )
        checks = audit["audit_checks"]
        self.assertTrue(checks["synthetic_candidate_construction"])
        self.assertTrue(checks["phase10_target_is_deterministic_cost_table"])
        self.assertTrue(checks["candidate_size_ignores_relation_type"])
        self.assertTrue(checks["feature_cache_lacks_provenance_metadata"])
        self.assertEqual(checks["weight_validation_numeric_rows"], 400)
        self.assertEqual(checks["runtime_attribution_rows"], 546)
        self.assertEqual(checks["gpu_microbenchmark_rows"], 7)

    def test_output_is_byte_deterministic_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = write_outputs(REPO_ROOT, Path(first))
            second_paths = write_outputs(REPO_ROOT, Path(second))
            self.assertEqual(tree_hashes(Path(first)), tree_hashes(Path(second)))
            self.assertEqual(
                set(first_paths),
                {
                    "audit_checks",
                    "claim_verdicts",
                    "recomputed_metrics",
                    "source_manifest",
                    "variable_lineage",
                },
            )
            for key in ("audit_checks", "source_manifest"):
                json.loads(first_paths[key].read_text(encoding="utf-8"))
            with first_paths["claim_verdicts"].open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 7)
            for key in ("recomputed_metrics", "variable_lineage"):
                with first_paths[key].open(newline="", encoding="utf-8") as handle:
                    self.assertGreater(len(list(csv.DictReader(handle))), 0)
            manifest = json.loads(first_paths["source_manifest"].read_text(encoding="utf-8"))
            for source in manifest["sources"]:
                path = REPO_ROOT / source["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_x1_5_freeze_manifest_matches_preserved_snapshot(self) -> None:
        manifest_path = REPO_ROOT / "output/results/evidence_audit_x1_5/x1_5_freeze_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["freeze_kind"], "X1.5 governance snapshot")
        for entry in manifest["files"]:
            path = REPO_ROOT / entry["path"]
            self.assertTrue(path.is_file(), entry["path"])
            self.assertEqual(path.stat().st_size, entry["bytes"], entry["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
