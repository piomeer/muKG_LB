import unittest
from pathlib import Path

from scripts import audit_x8_c1_r1_clean_room as audit
from scripts import run_x8_c1_r1_clean_room as executor


ROOT = Path(__file__).resolve().parents[1]


class X8C1CleanRoomContractTests(unittest.TestCase):
    def test_executor_loads_a_frozen_allowlisted_execution_contract(self):
        """Catches an executor that silently accepts unfrozen or broad inputs."""
        contract = executor.load_contract(ROOT)

        self.assertEqual(contract["contract_id"], "X8-C1-R1-clean-room-v1")
        self.assertEqual(contract["status"], "FROZEN")
        self.assertEqual(
            contract["source_hashes"]
            ["src/py/experiments/c1_r1_combined_rerun.py"],
            "2556df6aa6e50d20ae2c188fe987a7694dff1743473aaf0dd62a4e96615710ab",
        )
        self.assertEqual(contract["execution_matrix"]["primary_job_count"], 24)
        self.assertEqual(contract["execution_matrix"]["diagnostic_job_count"], 6)
        self.assertTrue(contract["analysis_controls"]["pooling_forbidden"])
        self.assertNotIn(
            "output/results/c1_r1_combined_rerun",
            contract["capsule"]["allowlisted_paths"],
        )
        self.assertTrue(contract["environment"]["network_forbidden"])

    def test_audit_loads_the_same_blind_statistical_contract(self):
        """Catches analysis that changes the primary family, filters, or verdicts."""
        contract = audit.load_contract(ROOT)

        self.assertEqual(contract["analysis"]["primary_family"], ["E1", "E2"])
        self.assertTrue(contract["analysis_controls"]["pooling_forbidden"])
        self.assertEqual(
            contract["analysis"]["simultaneous_interval"],
            {"method": "Bonferroni", "confidence_level": 0.975, "lower_bound": 1.0},
        )
        self.assertTrue(
            contract["analysis"]["primary_gate"]["direction_consistency_required"]
        )
        self.assertEqual(
            contract["analysis"]["filters"]["E2"],
            {
                "is_partial": False,
                "batch_size_actual": 5000,
                "aggregation": "mean_of_five_epoch_population_sds",
                "ddof": 0,
            },
        )
        self.assertEqual(
            contract["analysis"]["numerical_fidelity_ratios"],
            {"E1": [0.9, 1.1], "E2": [0.75, 1.25], "E3": [0.9, 1.1]},
        )
        self.assertEqual(
            contract["analysis"]["verdict_states"],
            [
                "VERIFIED",
                "SUPPORTED_WITH_NUMERICAL_DRIFT",
                "NOT_REPRODUCED",
                "INCOMPLETE",
                "BLOCKED_ENVIRONMENT",
            ],
        )
        self.assertTrue(audit.load_contract(ROOT)["analysis_controls"]["pooling_forbidden"])


if __name__ == "__main__":
    unittest.main()
