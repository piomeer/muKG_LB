import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.triage_phase_x_contributions import (
    APPENDIX,
    PRIMARY,
    SUPPORTING,
    REPLACEMENTS,
    decision_artifact,
    decision_for,
    rows_for,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


class X55ContributionTriageTests(unittest.TestCase):
    def test_exact_inventory_and_replacement_coverage(self):
        rows = rows_for(ROOT)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r["claim"] for r in rows}), 36)
        self.assertEqual({r["claim"] for r in rows if r["lineage_status"] == "REPLACEMENT"}, set(REPLACEMENTS))
        self.assertEqual(validate(ROOT, rows), [])

    def test_frozen_contribution_set(self):
        rows = rows_for(ROOT)
        self.assertEqual({r["claim"] for r in rows if r["decision"] == "RETAIN_PRIMARY"}, PRIMARY)
        self.assertEqual({r["claim"] for r in rows if r["decision"] == "RETAIN_SUPPORTING"}, SUPPORTING)
        self.assertEqual({r["claim"] for r in rows if r["decision"] == "APPENDIX"}, APPENDIX)
        self.assertFalse(any(r["decision"] in {"RETAIN_PRIMARY", "RETAIN_SUPPORTING", "EXPLORATORY"} for r in rows if r["claim"].startswith(("C3", "C4"))))

    def test_waiver_schema_and_scope(self):
        artifact = decision_artifact(rows_for(ROOT))
        self.assertEqual(artifact["status"], "FINAL")
        self.assertEqual(artifact["x6_5_status"], "WAIVED")
        self.assertEqual(artifact["branches"]["C3"]["decision"], "WAIVED")
        self.assertIsNone(artifact["branches"]["C3"]["primary_promotion_estimand"])
        self.assertIn("X8 clean-room reproduction", artifact["waiver_scope"]["not_covered"])
        self.assertNotIn("C1 sensitivity", artifact["waiver_scope"]["covered"])

    def test_replacement_parent_must_be_removed(self):
        rows = rows_for(ROOT)
        parent = next(r for r in rows if r["claim"] == "C1.2")
        parent["decision"] = "APPENDIX"
        self.assertTrue(validate(ROOT, rows))

    def test_self_test_cli_contract_is_deterministic(self):
        self.assertEqual(decision_for("C1.2-R1"), "RETAIN_PRIMARY")
        self.assertEqual(decision_for("C3.1"), "REMOVE")
        self.assertEqual(decision_for("C4.7-R1"), "APPENDIX")


if __name__ == "__main__":
    unittest.main()
