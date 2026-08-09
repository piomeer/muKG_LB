from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.check_x0_5_quarantine import build_checks, write_output


ROOT = Path(__file__).resolve().parents[1]
LEGACY_FILES = (
    "paper/draft/method.md",
    "docs/paper_outline.md",
    "docs/paper_story_freeze.md",
    "docs/runtime_framework_spec.md",
    "docs/phase8_architecture_freeze.md",
)
MARKER = "<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->"


class X05QuarantineCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo = Path(self.tempdir.name)
        for relative in (
            "docs/phase_x_x0_research_freeze.md",
            "docs/evidence_audit_part1_claim_inventory.md",
            "docs/evidence_audit_part2_c1_gpu_runtime.md",
            "docs/evidence_audit_part3_c2_framework.md",
            "docs/evidence_audit_part4_c3_cost_model.md",
            "docs/evidence_audit_part5_c4_cbp.md",
            "docs/unified_runtime_architecture_freeze.md",
            "docs/baseline_freeze.md",
            "docs/validation_plan.md",
            "docs/evidence_matrix.md",
        ):
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("placeholder\n", encoding="utf-8")
        shutil.copy2(
            ROOT / "docs/evidence_audit_part1_claim_inventory.md",
            self.repo / "docs/evidence_audit_part1_claim_inventory.md",
        )
        for relative in LEGACY_FILES:
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# title\n{MARKER}\n", encoding="utf-8")
        for relative in (
            "output/results/c1_r1_combined_rerun",
            "output/results/evidence_audit_part2",
            "output/results/evidence_audit_part3",
            "output/results/evidence_audit_part4",
            "output/results/evidence_audit_part5",
            "output/results/unified_runtime",
            "output/results/phase9_step2",
            "output/results/phase9_step3",
            "output/results/runtime_attribution",
            "output/results/phase9_step4_5",
            "output/results/integration_validation",
        ):
            (self.repo / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "docs/phase_x_x0_5_legacy_narrative_quarantine.md", self.repo / "docs/phase_x_x0_5_legacy_narrative_quarantine.md")
        register = self.repo / "docs/phase_x_x0_5_legacy_narrative_quarantine.md"
        text = register.read_text(encoding="utf-8")
        text = text.replace("output/results/evidence_audit_part2/", "output/results/evidence_audit_part2/")
        register.write_text(text, encoding="utf-8")

    def test_real_repository_passes_x05_gate(self) -> None:
        result = build_checks(ROOT)
        self.assertEqual(result["overall_status"], "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in result["checks"]))

    def test_missing_header_fails_closed(self) -> None:
        target = self.repo / LEGACY_FILES[0]
        target.write_text("# title\n", encoding="utf-8")
        self.assertEqual(build_checks(self.repo)["overall_status"], "FAIL")

    def test_duplicate_header_fails_closed(self) -> None:
        target = self.repo / LEGACY_FILES[0]
        target.write_text(f"# title\n{MARKER}\n{MARKER}\n", encoding="utf-8")
        self.assertEqual(build_checks(self.repo)["overall_status"], "FAIL")

    def test_missing_source_fails_closed(self) -> None:
        (self.repo / "output/results/evidence_audit_part5").rmdir()
        self.assertEqual(build_checks(self.repo)["overall_status"], "FAIL")

    def test_output_is_byte_deterministic(self) -> None:
        result = build_checks(ROOT)
        first = self.repo / "first.json"
        second = self.repo / "second.json"
        write_output(result, first)
        write_output(result, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(json.loads(first.read_text()), json.loads(second.read_text()))


if __name__ == "__main__":
    unittest.main()
