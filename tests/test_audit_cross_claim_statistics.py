import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cross_claim_statistics import (
    claim_ids,
    contract_status,
    geometric_mean,
    log_t_ci,
    metrics_from_pairs,
    registry_rows,
    self_test,
)


ROOT = Path(__file__).resolve().parents[1]


class CrossClaimStatisticsTests(unittest.TestCase):
    def test_c1_pair_recomputation_and_simultaneous_interval(self):
        with (ROOT / "output/results/c1_r1_combined_rerun/analysis/paired_metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        out = metrics_from_pairs(rows)
        e1 = {r["statistic"]: float(r["value"]) for r in out if r["metric"] == "E1" and r["statistic"] != "direction_count"}
        e2 = {r["statistic"]: float(r["value"]) for r in out if r["metric"] == "E2" and r["statistic"] != "direction_count"}
        self.assertAlmostEqual(e1["geometric_mean"], 6.013389739959145, places=12)
        self.assertAlmostEqual(e2["geometric_mean"], 87.8771, places=3)
        self.assertGreater(e1["simultaneous_ci97_5_lower"], 1)
        self.assertGreater(e2["simultaneous_ci97_5_lower"], 1)
        self.assertEqual(next(r["value"] for r in out if r["metric"] == "E1" and r["statistic"] == "direction_count"), "6")

    def test_fixture_math_and_leave_one_out(self):
        self.assertEqual(geometric_mean([1, 4, 16]), 4.0)
        lo, hi = log_t_ci([2, 3, 4, 5, 6, 7])
        self.assertLess(lo, 4.0)
        self.assertGreater(hi, 4.0)

    def test_all_part1_claims_and_replacements(self):
        ids = claim_ids(ROOT)
        self.assertEqual(len(ids), 28)
        rows = registry_rows(ids, ROOT)
        self.assertEqual(len([r for r in rows if r["replacement_of"]]), 8)
        self.assertEqual({r["claim_id"] for r in rows if r["replacement_of"]}, {"C1.2-R1", "C1.3-R1", "C1.7-R1", "C2.1-R1", "C3.1-R1", "C4.1-R1", "C4.3-R1", "C4.7-R1"})

    def test_missing_x5_5_fails_closed(self):
        status, reasons = contract_status(ROOT)
        self.assertEqual(status, "BLOCKED_X5_5_INPUT")
        self.assertTrue(reasons)

    def test_self_test(self):
        self_test()


if __name__ == "__main__":
    unittest.main()
