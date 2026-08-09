from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_c2_framework import (
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


class AuditC2FrameworkTests(unittest.TestCase):
    def test_cpu_fixtures_cover_frozen_architecture_properties(self) -> None:
        fixtures = run_cpu_fixtures(REPO_ROOT)

        self.assertTrue(fixtures["cost_model_deterministic"])
        self.assertTrue(fixtures["cost_model_float32"])
        self.assertEqual(fixtures["cost_model_shape"], [6])
        self.assertEqual(
            fixtures["scheduler_combinations"],
            [
                "CostSorter+ChunkPacker",
                "CostSorter+FFDPacker",
                "RandomSorter+ChunkPacker",
                "RandomSorter+FFDPacker",
            ],
        )
        self.assertTrue(fixtures["batch_provider_full_coverage"])
        self.assertTrue(fixtures["rank_partitions_disjoint"])
        self.assertTrue(fixtures["rank_partitions_cover_all_batches"])
        self.assertTrue(fixtures["ffd_equals_chunk_on_frozen_fixture"])

    def test_repository_audit_freezes_claim_grades_and_driver_facts(self) -> None:
        audit = build_audit(REPO_ROOT)
        grades = {
            claim["claim_id"]: claim["grade"]
            for claim in audit["audit_checks"]["claims"]
        }

        self.assertEqual(
            grades,
            {
                "C2.1-R1": "A",
                "C2.2": "B",
                "C2.3": "A",
                "C2.4": "A",
                "C2.5": "A",
                "C2.6": "D",
            },
        )
        facts = audit["audit_checks"]["facts"]
        self.assertEqual(facts["phase9_config_labels"], ["BL", "CBP", "CBP+GPU", "GPU"])
        self.assertEqual(facts["phase9_per_config_write_suffix"], ".md")
        self.assertEqual(facts["phase9_per_config_read_suffix"], ".csv")
        self.assertFalse(facts["runtime_policy_implemented"])
        self.assertFalse(facts["gpu_execution_implemented"])
        self.assertTrue(facts["training_loop_selects_backend"])
        self.assertEqual(
            facts["factory_scheduler_combinations"],
            ["CostSorter+FFDPacker", "RandomSorter+ChunkPacker"],
        )
        self.assertEqual(
            facts["phase9_scheduler_combinations"],
            ["CostSorter+FFDPacker", "RandomSorter+ChunkPacker"],
        )
        self.assertFalse(facts["factory_supports_all_four_combinations"])
        self.assertFalse(facts["experiment_validates_all_four_scheduler_combinations"])
        self.assertTrue(facts["scheduling_occurs_per_iterate_call"])
        self.assertTrue(facts["runtime_cost_access_uses_array_subscript"])
        self.assertIn("pack_batches", facts["actual_interface_signatures"])
        self.assertIn("iterate", facts["actual_interface_signatures"])

    def test_metrics_include_phase6_and_c1_r1_overheads(self) -> None:
        audit = build_audit(REPO_ROOT)
        metrics = {
            row["metric_id"]: float(row["value"])
            for row in audit["recomputed_metrics"]
        }

        self.assertAlmostEqual(metrics["phase6_bl_scheduler_overhead_ms"], 64.757, places=6)
        self.assertAlmostEqual(metrics["phase6_cbp_scheduler_overhead_ms"], 1165.0, places=6)
        self.assertAlmostEqual(metrics["c1_r1_bl_scheduler_mean_ms"], 73.0879968667, places=9)
        self.assertAlmostEqual(metrics["c1_r1_gpu_scheduler_mean_ms"], 66.8442073333, places=9)
        self.assertAlmostEqual(metrics["c1_r1_bl_scheduler_epoch_pct"], 0.279904, places=6)
        self.assertAlmostEqual(metrics["c1_r1_gpu_scheduler_epoch_pct"], 1.529412, places=6)

    def test_output_is_byte_deterministic_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = write_outputs(REPO_ROOT, Path(first))
            second_paths = write_outputs(REPO_ROOT, Path(second))

            self.assertEqual(tree_hashes(Path(first)), tree_hashes(Path(second)))
            self.assertEqual(
                set(first_paths),
                {
                    "architecture_mapping",
                    "audit_checks",
                    "recomputed_metrics",
                    "source_manifest",
                },
            )
            for key in ("audit_checks", "source_manifest"):
                json.loads(first_paths[key].read_text(encoding="utf-8"))
            for key in ("architecture_mapping", "recomputed_metrics"):
                with first_paths[key].open(newline="", encoding="utf-8") as handle:
                    self.assertGreater(len(list(csv.DictReader(handle))), 0)

            manifest = json.loads(
                first_paths["source_manifest"].read_text(encoding="utf-8")
            )
            for source in manifest["sources"]:
                path = REPO_ROOT / source["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )


if __name__ == "__main__":
    unittest.main()
