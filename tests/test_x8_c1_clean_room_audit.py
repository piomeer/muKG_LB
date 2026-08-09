import csv
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import audit_x8_c1_r1_clean_room as audit


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_split() -> dict[str, object]:
    return {
        "declared_triples": 17115,
        "raw_triples": 17115,
        "held_out_size": 5000,
        "training_set_size": 12115,
        "split_seed": 42,
        "split_algorithm": "fixture frozen shuffle; first 5000 held out",
        "source_path": "src/py/data/FB15K237/train2id.txt",
        "source_sha256": "f" * 64,
        "raw_order_sha256": "a" * 64,
        "file_order_sha256": "b" * 64,
        "held_out_order_sha256": "c" * 64,
        "training_order_sha256": "d" * 64,
    }


def fixture_contract() -> dict[str, object]:
    epoch_fields = [
        "protocol_id", "pass_name", "config", "seed", "epoch",
        "epoch_time_ns", "num_steps", "full_batch_count", "partial_batch_count",
        "partial_batch_size", "training_examples", "loss_finite",
    ]
    step_fields = [
        "protocol_id", "pass_name", "config", "seed", "epoch", "step",
        "batch_size_actual", "is_partial", "is_first_measured_step",
        "neg_time_ns", "component_sum_ns", "total_step_ns", "timing_residual_ns",
    ]
    telemetry_fields = [
        "protocol_id", "config", "seed", "pass_name", "event", "time_ns",
        "thermal_slowdown", "other_compute_processes", "raw_gpu_query",
        "raw_process_query", "query_error",
    ]
    return {
        "contract_id": "X8-C1-R1-clean-room-v1",
        "status": "FROZEN",
        "source_hashes": {
            "src/py/data/FB15K237/train2id.txt": "f" * 64,
        },
        "protocol": {
            "protocol_id": "C1-R1-v1.1",
            "batch_size": 5000,
            "epochs_per_job": 5,
            "training_examples": 12115,
            "paired_order": {
                str(seed): (["BL", "GPU"] if seed % 2 == 0 else ["GPU", "BL"])
                for seed in range(42, 48)
            },
        },
        "execution_matrix": {
            "primary": {
                "seeds": list(range(42, 48)),
                "passes": ["throughput", "trace"],
                "configs": ["BL", "GPU"],
            },
            "diagnostic": {"seeds": []},
        },
        "analysis": {
            "primary_family": ["E1", "E2"],
            "filters": {
                "E1": {"pass_name": "throughput"},
                "E2": {
                    "is_partial": False,
                    "batch_size_actual": 5000,
                    "ddof": 0,
                },
                "E3": {"is_partial": False, "batch_size_actual": 5000},
            },
            "primary_gate": {
                "complete_pairs_required": 6,
                "simultaneous_lower_bound_strictly_above": 1.0,
            },
            "t_critical_values": {
                "df5_ci95": 2.570581835636314,
                "df5_ci97_5": 3.163381449748624,
            },
            "numerical_fidelity_ratios": {
                "E1": [0.9, 1.1], "E2": [0.75, 1.25], "E3": [0.9, 1.1],
            },
            "verdict_states": [
                "VERIFIED", "SUPPORTED_WITH_NUMERICAL_DRIFT", "NOT_REPRODUCED",
                "INCOMPLETE", "BLOCKED_ENVIRONMENT",
            ],
        },
        "analysis_controls": {"pooling_forbidden": True},
        "retry_policy": {
            "maximum_retries_per_pass_seed_pair": 1,
            "paired_configs": ["BL", "GPU"],
            "eligible_failure_reasons": ["thermal_slowdown", "telemetry_failure"],
        },
        "raw_artifact_schemas": {
            "environment.json": {
                "required_fields": [
                    "protocol_id", "cuda_available", "gpu", "python", "pytorch",
                    "torch_cuda_runtime", "raw_command_captures",
                ]
            },
            "preflight/result.json": {
                "required_fields": ["protocol_id", "all_passed", "checks", "split"]
            },
            "status.json": {
                "required_fields": [
                    "protocol_id", "pass_name", "config", "seed", "split",
                    "row_counts", "valid", "invalid_reasons", "warnings",
                ]
            },
            "per_epoch.csv": {"required_columns": epoch_fields},
            "per_step.csv": {"required_columns": step_fields},
            "gpu_telemetry.csv": {"required_columns": telemetry_fields},
            "material_passport.json": {
                "required_fields": [
                    "contract_id", "contract_sha256", "capsule_manifest_sha256",
                    "environment_manifest_sha256", "raw_artifact_manifest_sha256",
                    "stage", "sealed_at_ns",
                ]
            },
        },
        "material_passport": {
            "required_fields": [
                "contract_id", "contract_sha256", "capsule_manifest_sha256",
                "environment_manifest_sha256", "raw_artifact_manifest_sha256",
                "stage", "sealed_at_ns",
            ]
        },
    }


def telemetry_row(pass_name: str, config: str, seed: int) -> dict[str, object]:
    return {
        "protocol_id": "C1-R1-v1.1",
        "config": config,
        "seed": seed,
        "pass_name": pass_name,
        "event": "after_job",
        "time_ns": 1,
        "thermal_slowdown": "False",
        "other_compute_processes": "",
        "raw_gpu_query": "fixture",
        "raw_process_query": "",
        "query_error": "",
    }


def create_sealed_raw_fixture(root: Path) -> None:
    contract = fixture_contract()
    write_json(root / "frozen_contract.json", contract)
    write_json(root / "capsule_manifest.json", {"contract_id": contract["contract_id"], "files": []})
    write_json(root / "environment_manifest.json", {"contract_id": contract["contract_id"]})
    raw = root / "raw"
    write_json(
        raw / "environment.json",
        {
            "protocol_id": "C1-R1-v1.1",
            "cuda_available": True,
            "gpu": {"name": "fixture"},
            "python": "3.11",
            "pytorch": "fixture",
            "torch_cuda_runtime": "fixture",
            "raw_command_captures": {},
        },
    )
    preflight_root = raw / "attempts/preflight_attempt0/preflight"
    write_json(
        preflight_root / "result.json",
        {
            "protocol_id": "C1-R1-v1.1",
            "all_passed": True,
            "checks": [],
            "split": fixture_split(),
        },
    )
    telemetry_fields = contract["raw_artifact_schemas"]["gpu_telemetry.csv"]["required_columns"]
    write_csv(
        preflight_root / "gpu_telemetry.csv",
        telemetry_fields,
        [telemetry_row("preflight", "GPU", -1)],
    )

    jobs: list[dict[str, object]] = [
        {"id": "preflight", "kind": "preflight", "state": "COMPLETED", "attempt": 0}
    ]
    epoch_fields = contract["raw_artifact_schemas"]["per_epoch.csv"]["required_columns"]
    step_fields = contract["raw_artifact_schemas"]["per_step.csv"]["required_columns"]
    for seed in range(42, 48):
        for pass_name in ("throughput", "trace"):
            for config in contract["protocol"]["paired_order"][str(seed)]:
                job = {
                    "id": f"{pass_name}_{config}_seed{seed}",
                    "kind": "job",
                    "pass_name": pass_name,
                    "config": config,
                    "seed": seed,
                    "state": "COMPLETED",
                    "attempt": 0,
                }
                jobs.append(job)
                job_dir = (
                    raw / "attempts" / f"{pass_name}_seed{seed}_attempt0" / "jobs"
                    / f"{pass_name}_{config}_seed{seed}"
                )
                epoch_time = 20 if config == "BL" else 10
                epoch_rows = [
                    {
                        "protocol_id": "C1-R1-v1.1", "pass_name": pass_name,
                        "config": config, "seed": seed, "epoch": epoch,
                        "epoch_time_ns": epoch_time, "num_steps": 3,
                        "full_batch_count": 2, "partial_batch_count": 1,
                        "partial_batch_size": 2115, "training_examples": 12115,
                        "loss_finite": "True",
                    }
                    for epoch in range(5)
                ]
                write_csv(job_dir / "per_epoch.csv", epoch_fields, epoch_rows)
                step_count = 0
                if pass_name == "trace":
                    step_rows = []
                    full_values = [10, 30] if config == "BL" else [10, 20]
                    for epoch in range(5):
                        for step, neg_time in enumerate(full_values + [999_999_999]):
                            partial = step == 2
                            step_rows.append(
                                {
                                    "protocol_id": "C1-R1-v1.1", "pass_name": "trace",
                                    "config": config, "seed": seed, "epoch": epoch,
                                    "step": step,
                                    "batch_size_actual": 2115 if partial else 5000,
                                    "is_partial": str(partial),
                                    "is_first_measured_step": str(epoch == 0 and step == 0),
                                    "neg_time_ns": neg_time, "component_sum_ns": 100,
                                    "total_step_ns": 110, "timing_residual_ns": 10,
                                }
                            )
                    write_csv(job_dir / "per_step.csv", step_fields, step_rows)
                    step_count = 15
                write_json(
                    job_dir / "status.json",
                    {
                        "protocol_id": "C1-R1-v1.1", "pass_name": pass_name,
                        "config": config, "seed": seed, "split": fixture_split(),
                        "row_counts": {"epochs": 5, "steps": step_count},
                        "valid": True, "invalid_reasons": [], "warnings": [],
                    },
                )
                write_csv(
                    job_dir / "gpu_telemetry.csv",
                    telemetry_fields,
                    [telemetry_row(pass_name, config, seed)],
                )

    execution_manifest = {
        "contract_id": contract["contract_id"],
        "protocol_id": "C1-R1-v1.1",
        "state": "RAW_COMPLETE",
        "contract_sha256": sha256_file(root / "frozen_contract.json"),
        "capsule_manifest_sha256": sha256_file(root / "capsule_manifest.json"),
        "environment_manifest_sha256": sha256_file(root / "environment_manifest.json"),
        "jobs": jobs,
        "remediations": [],
    }
    write_json(root / "execution_manifest.json", execution_manifest)
    artifacts = []
    raw_paths = sorted(
        [path for path in raw.rglob("*") if path.is_file()] + [root / "execution_manifest.json"],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in raw_paths:
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        root / "raw_artifact_manifest.json",
        {"contract_id": contract["contract_id"], "stage": "raw", "artifacts": artifacts},
    )
    write_json(
        root / "material_passport.json",
        {
            "contract_id": contract["contract_id"],
            "contract_sha256": execution_manifest["contract_sha256"],
            "capsule_manifest_sha256": execution_manifest["capsule_manifest_sha256"],
            "environment_manifest_sha256": execution_manifest["environment_manifest_sha256"],
            "raw_artifact_manifest_sha256": sha256_file(root / "raw_artifact_manifest.json"),
            "stage": "raw",
            "sealed_at_ns": 123456789,
        },
    )


def reseal_raw_fixture(root: Path) -> None:
    contract = json.loads((root / "frozen_contract.json").read_text(encoding="utf-8"))
    raw_paths = sorted(
        [path for path in (root / "raw").rglob("*") if path.is_file()]
        + [root / "execution_manifest.json"],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    write_json(
        root / "raw_artifact_manifest.json",
        {
            "contract_id": contract["contract_id"],
            "stage": "raw",
            "artifacts": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in raw_paths
            ],
        },
    )
    passport = json.loads((root / "material_passport.json").read_text(encoding="utf-8"))
    passport["raw_artifact_manifest_sha256"] = sha256_file(
        root / "raw_artifact_manifest.json"
    )
    write_json(root / "material_passport.json", passport)


def reseal_independent_fixture(root: Path) -> None:
    paths = [
        root / "derived/independent/summary.json",
        root / "derived/independent/checks.json",
        root / "derived/independent/seed_level_metrics.csv",
        root / "derived/independent/leave_one_seed_out.csv",
    ]
    raw_passport = json.loads(
        (root / "material_passport.json").read_text(encoding="utf-8")
    )
    contract = json.loads((root / "frozen_contract.json").read_text(encoding="utf-8"))
    write_json(
        root / "independent_artifact_manifest.json",
        {
            "contract_id": contract["contract_id"],
            "stage": "independent",
            "raw_artifact_manifest_sha256": raw_passport[
                "raw_artifact_manifest_sha256"
            ],
            "artifacts": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(paths)
            ],
        },
    )
    passport_path = root / "independent_material_passport.json"
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    passport["independent_artifact_manifest_sha256"] = sha256_file(
        root / "independent_artifact_manifest.json"
    )
    write_json(passport_path, passport)


def install_paired_retry(root: Path, *, pass_name: str, seed: int) -> None:
    source = root / f"raw/attempts/{pass_name}_seed{seed}_attempt0"
    destination = root / f"raw/attempts/{pass_name}_seed{seed}_attempt1"
    shutil.copytree(source, destination)
    manifest_path = root / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = fixture_contract()["protocol"]["paired_order"][str(seed)]
    pair = [
        job
        for job in manifest["jobs"]
        if job.get("kind") == "job"
        and job.get("pass_name") == pass_name
        and job.get("seed") == seed
    ]
    for job in pair:
        job["analysis_eligible"] = False
        job["superseded_by"] = 1
        job["state"] = "INVALID" if job["config"] == order[0] else "SKIPPED_INVALID_PAIR"
        if job["state"] == "INVALID":
            job["invalid_reason"] = "thermal_slowdown"
    retry_jobs = [
        {
            "id": f"{pass_name}_{config}_seed{seed}_retry1",
            "kind": "job",
            "pass_name": pass_name,
            "config": config,
            "seed": seed,
            "attempt": 1,
            "dispatch": 1,
            "state": "COMPLETED",
            "analysis_eligible": True,
        }
        for config in order
    ]
    manifest["remediations"].append(
        {
            "pass_name": pass_name,
            "seed": seed,
            "attempt": 1,
            "dispatch": 1,
            "state": "COMPLETED",
            "analysis_eligible": True,
            "trigger_reasons": ["thermal_slowdown"],
            "jobs": retry_jobs,
        }
    )
    write_json(manifest_path, manifest)
    reseal_raw_fixture(root)


class X8C1CleanRoomAuditStatisticsTests(unittest.TestCase):
    def test_hand_derived_geometric_and_arithmetic_t_intervals(self):
        """Catches arithmetic pooling of ratios or the wrong t critical value."""
        geometric = audit.geometric_summary(
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            t95=2.570581835636314,
            t97_5=3.163381449748624,
        )
        arithmetic = audit.arithmetic_summary(
            [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            t95=2.570581835636314,
        )

        self.assertAlmostEqual(geometric["estimate"], 5.656854249492381)
        self.assertAlmostEqual(geometric["ci95"]["low"], 1.4506361363703022)
        self.assertAlmostEqual(geometric["ci95"]["high"], 22.05928778257831)
        self.assertAlmostEqual(geometric["ci97_5_bonferroni"]["low"], 1.0598995254217827)
        self.assertAlmostEqual(geometric["ci97_5_bonferroni"]["high"], 30.191541021084735)
        self.assertAlmostEqual(arithmetic["estimate"], 15.0)
        self.assertAlmostEqual(arithmetic["sample_sd"], 3.7416573867739413)
        self.assertAlmostEqual(arithmetic["ci95"]["low"], 11.073371386039353)
        self.assertAlmostEqual(arithmetic["ci95"]["high"], 18.92662861396065)


class X8C1CleanRoomIndependentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name) / "clean-room"
        create_sealed_raw_fixture(self.root)

    def test_independent_analysis_recomputes_filtered_estimands_and_is_deterministic(self):
        """Catches partial-batch leakage, pooled effects, or timestamped derived bytes."""
        first = audit.run_independent(self.root)
        paths = [
            self.root / "derived/independent/summary.json",
            self.root / "derived/independent/checks.json",
            self.root / "derived/independent/seed_level_metrics.csv",
            self.root / "derived/independent/leave_one_seed_out.csv",
            self.root / "independent_artifact_manifest.json",
            self.root / "independent_material_passport.json",
        ]
        first_bytes = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in paths}

        second = audit.run_independent(self.root)
        second_bytes = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in paths}

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["status"], "ANALYZED")
        self.assertEqual(first["estimands"]["E1"]["estimate"], 2.0)
        self.assertEqual(first["estimands"]["E1"]["ci95"], {"low": 2.0, "high": 2.0})
        self.assertEqual(first["estimands"]["E2"]["estimate"], 2.0)
        self.assertEqual(first["estimands"]["E2"]["ci97_5_bonferroni"]["low"], 2.0)
        self.assertEqual(first["estimands"]["E3"]["estimate_ns"], 15.0)
        self.assertEqual(first["estimands"]["E3"]["sample_sd_ns"], 0.0)
        self.assertEqual(first["direction_consistency"]["E1"], {"count": 6, "required": 6, "passed": True})
        self.assertEqual(first["direction_consistency"]["E2"], {"count": 6, "required": 6, "passed": True})
        self.assertTrue(first["primary_gate_passed"])
        self.assertTrue(audit.validate_independent_seal(self.root))
        with (self.root / "derived/independent/seed_level_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            seed_rows = list(csv.DictReader(handle))
        with (self.root / "derived/independent/leave_one_seed_out.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            leave_one_rows = list(csv.DictReader(handle))
        self.assertEqual(len(seed_rows), 6)
        self.assertTrue(all(row["E1_ratio"] == "2.0" for row in seed_rows))
        self.assertTrue(all(row["E2_ratio"] == "2.0" for row in seed_rows))
        self.assertTrue(all(row["E3_gpu_full_neg_mean_ns"] == "15.0" for row in seed_rows))
        self.assertEqual(len(leave_one_rows), 6)
        self.assertTrue(all(row["E1_estimate"] == "2.0" for row in leave_one_rows))
        self.assertFalse(any(path.is_relative_to(self.root / "raw") for path in paths))

    def test_raw_seal_rejects_a_malformed_passport_timestamp(self):
        """Catches accepting a materially tampered passport field as valid lineage."""
        passport_path = self.root / "material_passport.json"
        passport = json.loads(passport_path.read_text(encoding="utf-8"))
        passport["sealed_at_ns"] = "not-an-integer"
        write_json(passport_path, passport)

        with self.assertRaisesRegex(RuntimeError, "sealed_at_ns"):
            audit.validate_raw_seal(self.root)

    def test_raw_and_independent_seals_detect_post_seal_byte_tampering(self):
        """Catches acceptance of altered raw or derived bytes after sealing."""
        raw_path = next((self.root / "raw").rglob("per_epoch.csv"))
        raw_path.write_text(raw_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "raw artifact.*mismatch"):
            audit.run_independent(self.root)

        create_sealed_raw_fixture(self.root)
        audit.run_independent(self.root)
        summary_path = self.root / "derived/independent/summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "independent artifact.*mismatch"):
            audit.validate_independent_seal(self.root)

    def test_self_consistent_seal_still_rejects_a_raw_schema_violation(self):
        """Catches trusting hashes while skipping the frozen CSV schema."""
        path = (
            self.root
            / "raw/attempts/trace_seed42_attempt0/jobs/trace_BL_seed42/per_step.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fields = [field for field in rows[0] if field != "is_partial"]
        write_csv(
            path,
            fields,
            [{field: row[field] for field in fields} for row in rows],
        )
        reseal_raw_fixture(self.root)

        with self.assertRaisesRegex(RuntimeError, "schema missing columns.*is_partial"):
            audit.run_independent(self.root)

    def test_only_the_complete_predeclared_paired_retry_is_selected(self):
        """Catches reusing an excluded attempt or accepting a single-config retry."""
        install_paired_retry(self.root, pass_name="throughput", seed=42)
        retry_bl = (
            self.root
            / "raw/attempts/throughput_seed42_attempt1/jobs/throughput_BL_seed42/per_epoch.csv"
        )
        with retry_bl.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["epoch_time_ns"] = "30"
        write_csv(retry_bl, list(rows[0]), rows)
        reseal_raw_fixture(self.root)

        summary = audit.run_independent(self.root)

        self.assertEqual(summary["estimands"]["E1"]["seed_level_ratios"][0], 3.0)
        with (self.root / "derived/independent/seed_level_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            seed_rows = list(csv.DictReader(handle))
        self.assertEqual(seed_rows[0]["throughput_attempt"], "1")

        manifest_path = self.root / "execution_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["remediations"][0]["jobs"].pop()
        write_json(manifest_path, manifest)
        reseal_raw_fixture(self.root)
        with self.assertRaisesRegex(RuntimeError, "pair/order"):
            audit.run_independent(self.root)

    def test_retry_trigger_reasons_must_equal_actual_eligible_invalid_reasons(self):
        """Catches missing, outcome-driven, or relabeled retry lineage."""
        cases = (
            ("missing", None, ["thermal_slowdown"]),
            ("noneligible", "outcome_regression", ["thermal_slowdown"]),
            ("mismatch", "thermal_slowdown", ["telemetry_failure"]),
        )
        for name, actual_reason, declared_reasons in cases:
            with self.subTest(name=name):
                root = Path(self.tempdir.name) / f"retry-{name}"
                create_sealed_raw_fixture(root)
                install_paired_retry(root, pass_name="throughput", seed=42)
                manifest_path = root / "execution_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                invalid = next(
                    job
                    for job in manifest["jobs"]
                    if job.get("state") == "INVALID"
                    and job.get("pass_name") == "throughput"
                    and job.get("seed") == 42
                )
                invalid.pop("invalid_reason", None)
                if actual_reason is not None:
                    invalid["invalid_reason"] = actual_reason
                manifest["remediations"][0]["trigger_reasons"] = declared_reasons
                write_json(manifest_path, manifest)
                reseal_raw_fixture(root)

                with self.assertRaisesRegex(RuntimeError, "invalid reason|trigger reason"):
                    audit.run_independent(root)

    def test_paired_effects_require_exact_preflight_and_frozen_split_lineage(self):
        """Catches pairing jobs from different splits or a non-frozen split size."""
        status_path = (
            self.root
            / "raw/attempts/throughput_seed42_attempt0/jobs"
            / "throughput_GPU_seed42/status.json"
        )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["split"]["training_order_sha256"] = "e" * 64
        write_json(status_path, status)
        reseal_raw_fixture(self.root)

        with self.assertRaisesRegex(RuntimeError, "split"):
            audit.run_independent(self.root)
        self.assertFalse(
            (self.root / "derived/independent/summary.json").exists()
        )

        root = Path(self.tempdir.name) / "frozen-split-drift"
        create_sealed_raw_fixture(root)
        preflight_path = root / "raw/attempts/preflight_attempt0/preflight/result.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["split"]["training_set_size"] = 12114
        write_json(preflight_path, preflight)
        for path in (root / "raw/attempts").rglob("status.json"):
            job_status = json.loads(path.read_text(encoding="utf-8"))
            job_status["split"] = preflight["split"]
            write_json(path, job_status)
        reseal_raw_fixture(root)

        with self.assertRaisesRegex(RuntimeError, "split"):
            audit.run_independent(root)

    def test_independent_seal_allows_only_four_regular_declared_outputs(self):
        """Catches sealed extras or writes through attacker-controlled symlinks."""
        extra_root = Path(self.tempdir.name) / "preexisting-extra"
        create_sealed_raw_fixture(extra_root)
        extra_path = extra_root / "derived/independent/extra.json"
        write_json(extra_path, {"unexpected": True})
        with self.assertRaisesRegex(RuntimeError, "unexpected independent output"):
            audit.run_independent(extra_root)

        symlink_root = Path(self.tempdir.name) / "preexisting-symlink"
        create_sealed_raw_fixture(symlink_root)
        outside = Path(self.tempdir.name) / "outside.json"
        outside.write_text("outside stays unchanged\n", encoding="utf-8")
        symlink = symlink_root / "derived/independent/summary.json"
        symlink.parent.mkdir(parents=True)
        symlink.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            audit.run_independent(symlink_root)
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside stays unchanged\n")

        forged_root = Path(self.tempdir.name) / "forged-extra"
        create_sealed_raw_fixture(forged_root)
        audit.run_independent(forged_root)
        forged_extra = forged_root / "derived/independent/extra.json"
        write_json(forged_extra, {"unexpected": True})
        paths = sorted(
            path
            for path in (forged_root / "derived/independent").iterdir()
            if path.is_file()
        )
        manifest_path = forged_root / "independent_artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = [
            {
                "path": path.relative_to(forged_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
        write_json(manifest_path, manifest)
        passport_path = forged_root / "independent_material_passport.json"
        passport = json.loads(passport_path.read_text(encoding="utf-8"))
        passport["independent_artifact_manifest_sha256"] = sha256_file(manifest_path)
        write_json(passport_path, passport)
        with self.assertRaisesRegex(RuntimeError, "unexpected independent output"):
            audit.validate_independent_seal(forged_root)


class X8C1CleanRoomComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.root = self.base / "clean-room"
        self.original = self.base / "original"
        create_sealed_raw_fixture(self.root)

    @staticmethod
    def original_estimates() -> dict[str, float]:
        return {
            "E1": 6.013389739959145,
            "E2": 87.87705683218147,
            "E3": 3_002_619.603144654,
        }

    def write_frozen_original(self) -> None:
        estimates = self.original_estimates()
        write_json(
            self.original / "analysis/summary.json",
            {
                "C1.2-R1": {
                    "geometric_mean_speedup": estimates["E1"],
                    "A_gate_passed": True,
                },
                "C1.3-R1": {
                    "geometric_mean_sd_compression": estimates["E2"],
                    "A_gate_passed": True,
                },
                "C1.7-R1": {
                    "six_run_mean_ms": estimates["E3"] / 1_000_000,
                    "A_gate_passed": True,
                },
            },
        )

    def test_verdict_branches_and_inclusive_fidelity_boundaries(self):
        """Catches exclusive tolerances, drift-as-failure, or fabricated blocked values."""
        contract = fixture_contract()
        original = self.original_estimates()
        supported = {
            "status": "ANALYZED",
            "complete_seed_pairs": 6,
            "primary_gate_passed": True,
            "direction_consistency": {
                "E1": {"count": 6, "required": 6, "passed": True},
                "E2": {"count": 6, "required": 6, "passed": True},
            },
            "estimands": {
                "E1": {"estimate": original["E1"]},
                "E2": {"estimate": original["E2"]},
                "E3": {"estimate_ns": original["E3"]},
            },
            "pooling_performed": False,
        }

        verified = audit.compare_estimates(supported, original, contract)
        boundary = json.loads(json.dumps(supported))
        boundary["estimands"]["E1"]["estimate"] = original["E1"] * 0.90
        boundary["estimands"]["E2"]["estimate"] = original["E2"] * 1.25
        boundary["estimands"]["E3"]["estimate_ns"] = original["E3"] * 1.10
        boundary_result = audit.compare_estimates(boundary, original, contract)
        just_outside = json.loads(json.dumps(supported))
        just_outside["estimands"]["E1"]["estimate"] = (
            original["E1"] * 1.100000000001
        )
        drift = json.loads(json.dumps(supported))
        drift["estimands"]["E2"]["estimate"] = original["E2"] * 1.30
        failed = json.loads(json.dumps(supported))
        failed["primary_gate_passed"] = False

        self.assertEqual(verified["verdict"], "VERIFIED")
        self.assertEqual(boundary_result["verdict"], "VERIFIED")
        self.assertEqual(
            audit.compare_estimates(just_outside, original, contract)["verdict"],
            "SUPPORTED_WITH_NUMERICAL_DRIFT",
        )
        self.assertEqual(
            audit.compare_estimates(drift, original, contract)["verdict"],
            "SUPPORTED_WITH_NUMERICAL_DRIFT",
        )
        self.assertEqual(
            audit.compare_estimates(failed, original, contract)["verdict"],
            "NOT_REPRODUCED",
        )
        for status in ("INCOMPLETE", "BLOCKED_ENVIRONMENT"):
            result = audit.compare_estimates({"status": status}, original, contract)
            self.assertEqual(result, {"verdict": status, "estimands": {}})

    def test_compare_validates_independent_seal_before_reading_original(self):
        """Catches any original-result read before the independent passport gate."""
        with self.assertRaisesRegex(RuntimeError, "independent"):
            audit.run_compare(self.root, self.original / "does-not-exist")

    def test_run_compare_preserves_status_only_verdicts_without_estimands(self):
        """Catches the fallacy gate overwriting terminal no-estimate statuses."""
        self.write_frozen_original()
        for status in ("INCOMPLETE", "BLOCKED_ENVIRONMENT"):
            with self.subTest(status=status):
                audit.run_independent(self.root)
                summary_path = self.root / "derived/independent/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["status"] = status
                summary["estimands"] = {}
                write_json(summary_path, summary)
                reseal_independent_fixture(self.root)

                result = audit.run_compare(self.root, self.original)

                self.assertEqual(result["verdict"], status)
                self.assertEqual(result["estimands"], {})

    def test_compare_emits_one_deterministic_verdict_and_all_fallacy_items(self):
        """Catches pooling, omitted fallacy checks, or multiple conflicting verdicts."""
        audit.run_independent(self.root)
        self.write_frozen_original()

        first = audit.run_compare(self.root, self.original)
        comparison_path = self.root / "derived/comparison/comparison.json"
        fallacy_path = self.root / "derived/comparison/statistical_fallacy_scan.csv"
        first_bytes = (comparison_path.read_bytes(), fallacy_path.read_bytes())
        second = audit.run_compare(self.root, self.original)
        second_bytes = (comparison_path.read_bytes(), fallacy_path.read_bytes())

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["verdict"], "SUPPORTED_WITH_NUMERICAL_DRIFT")
        self.assertFalse(first["pooling_performed"])
        self.assertEqual(sum(key == "verdict" for key in first), 1)
        with fallacy_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["check_id"] for row in rows], [f"F{index:02d}" for index in range(1, 12)])
        self.assertTrue(all(row["passed"] == "True" for row in rows))
        summary = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertNotIn("pooled", summary["estimands"])

    def test_compare_rejects_drift_in_the_frozen_original_estimate(self):
        """Catches silently changing the historical reference during comparison."""
        audit.run_independent(self.root)
        self.write_frozen_original()
        path = self.original / "analysis/summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["C1.2-R1"]["geometric_mean_speedup"] += 0.01
        write_json(path, summary)

        with self.assertRaisesRegex(RuntimeError, "frozen original estimate"):
            audit.run_compare(self.root, self.original)


class X8C1CleanRoomAuditCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.root = self.base / "clean-room"
        self.original = self.base / "original"
        create_sealed_raw_fixture(self.root)

    def test_independent_cli_rejects_original_root_before_analysis(self):
        """Catches an independent command surface that can cross the blind boundary."""
        with self.assertRaisesRegex(ValueError, "forbidden.*independent"):
            audit.main(
                [
                    "--stage", "independent", "--root", str(self.root),
                    "--original-root", str(self.original / "must-not-be-read"),
                ]
            )

    def test_cli_stages_and_self_test_emit_machine_readable_results(self):
        """Catches missing stage dispatch or a self-test that requires experiment data."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(audit.main(["--self-test"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "PASS")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                audit.main(["--stage", "independent", "--root", str(self.root)]),
                0,
            )
        self.assertEqual(json.loads(output.getvalue())["status"], "ANALYZED")

        estimates = X8C1CleanRoomComparisonTests.original_estimates()
        write_json(
            self.original / "analysis/summary.json",
            {
                "C1.2-R1": {"geometric_mean_speedup": estimates["E1"], "A_gate_passed": True},
                "C1.3-R1": {"geometric_mean_sd_compression": estimates["E2"], "A_gate_passed": True},
                "C1.7-R1": {"six_run_mean_ms": estimates["E3"] / 1_000_000, "A_gate_passed": True},
            },
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                audit.main(
                    [
                        "--stage", "compare", "--root", str(self.root),
                        "--original-root", str(self.original),
                    ]
                ),
                0,
            )
        self.assertEqual(
            json.loads(output.getvalue())["verdict"],
            "SUPPORTED_WITH_NUMERICAL_DRIFT",
        )


if __name__ == "__main__":
    unittest.main()
