import contextlib
import hashlib
import csv
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import run_x8_c1_r1_clean_room as executor


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture_contract(source_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "contract_id": "X8-C1-R1-clean-room-v1",
        "contract_version": 1,
        "status": "FROZEN",
        "capsule": {
            "allowlisted_paths": list(source_hashes),
            "excluded_path_prefixes": ["output/results/c1_r1_combined_rerun"],
            "source_mutation_forbidden": True,
        },
        "source_hashes": source_hashes,
        "environment": {
            "clone_active_conda_environment_locally": True,
            "network_forbidden": True,
            "cuda_available_required": True,
            "gpu_contention_forbidden": True,
            "expected_gpu": {
                "name": "NVIDIA GeForce RTX 3070",
                "total_memory_mib": 8192,
                "compute_capability": "8.6",
            },
            "expected_runtime": {
                "pytorch": "2.7.1+cu118",
                "torch_cuda_runtime": "11.8",
            },
        },
        "protocol": {
            "protocol_id": "C1-R1-v1.1",
            "paired_order": {"42": ["BL", "GPU"]},
        },
        "execution_matrix": {
            "primary": {
                "seeds": [42],
                "passes": ["throughput", "trace"],
                "configs": ["BL", "GPU"],
            },
            "diagnostic": {"seeds": [42]},
        },
        "retry_policy": {
            "eligible_failure_reasons": [
                "thermal_slowdown",
                "other_compute_processes",
                "telemetry_failure",
                "infrastructure_failure",
            ],
            "maximum_retries_per_pass_seed_pair": 1,
            "paired_configs": ["BL", "GPU"],
        },
        "material_passport": {
            "required_fields": [
                "contract_id",
                "contract_sha256",
                "capsule_manifest_sha256",
                "environment_manifest_sha256",
                "raw_artifact_manifest_sha256",
                "stage",
                "sealed_at_ns",
            ]
        },
    }


class FakeExternalTransport:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.telemetry_issues: dict[tuple[str, str, int], str] = {}

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        if command[:2] == ["conda", "create"]:
            clone = Path(command[command.index("--prefix") + 1])
            (clone / "bin").mkdir(parents=True)
            (clone / "bin/python").write_bytes(b"fixture-python")
            (clone / "conda-meta").mkdir()
            (clone / "conda-meta/history").write_text("fixture\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["conda", "list"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps([{"name": "pytorch", "version": "2.7.1", "build": "cu118"}]),
                "",
            )
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if command[-2] == "-c" and "torch" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "protocol_id": "C1-R1-v1.1",
                        "python": "3.11.0",
                        "pytorch": "2.7.1+cu118",
                        "torch_cuda_runtime": "11.8",
                        "cuda_available": True,
                    }
                ),
                "",
            )
        if command and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0, "NVIDIA GeForce RTX 3070, 8192, 8.6\n", ""
            )
        if len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py"):
            kind = command[2]
            output_root = Path(command[command.index("--root") + 1])
            if kind == "preflight":
                write_json(
                    output_root / "preflight/result.json",
                    {
                        "protocol_id": "C1-R1-v1.1",
                        "all_passed": True,
                        "checks": [{"check": "P4_no_other_compute_process", "passed": True}],
                        "split": {},
                    },
                )
                self._write_telemetry(output_root / "preflight/gpu_telemetry.csv", "preflight", "GPU", -1)
            elif kind == "job":
                config = command[command.index("--config") + 1]
                pass_name = command[command.index("--pass-name") + 1]
                seed = int(command[command.index("--seed") + 1])
                job_dir = output_root / "jobs" / f"{pass_name}_{config}_seed{seed}"
                write_json(
                    job_dir / "status.json",
                    {
                        "protocol_id": "C1-R1-v1.1",
                        "pass_name": pass_name,
                        "config": config,
                        "seed": seed,
                        "split": {},
                        "row_counts": {"epochs": 5, "steps": 0 if pass_name == "throughput" else 270},
                        "valid": True,
                        "invalid_reasons": [],
                        "warnings": [],
                    },
                )
                (job_dir / "per_epoch.csv").write_text(
                    "protocol_id,pass_name,config,seed,epoch,epoch_time_ns,num_steps,full_batch_count,partial_batch_count,partial_batch_size,training_examples,loss_finite\n",
                    encoding="utf-8",
                )
                if pass_name == "trace":
                    (job_dir / "per_step.csv").write_text(
                        "protocol_id,pass_name,config,seed,epoch,step,batch_size_actual,is_partial,is_first_measured_step,neg_time_ns,component_sum_ns,total_step_ns,timing_residual_ns\n",
                        encoding="utf-8",
                    )
                self._write_telemetry(
                    job_dir / "gpu_telemetry.csv",
                    pass_name,
                    config,
                    seed,
                    self.telemetry_issues.get((pass_name, config, seed), ""),
                )
            elif kind == "compute-only":
                seed = int(command[command.index("--seed") + 1])
                path = output_root / "compute_only" / f"seed{seed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("protocol_id,seed,repeat,elapsed_ns\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "runner output\n", "")
        raise AssertionError(f"unexpected external command: {command}")

    @staticmethod
    def _write_telemetry(
        path: Path, pass_name: str, config: str, seed: int, issue: str = ""
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "protocol_id", "config", "seed", "pass_name", "event", "time_ns",
            "thermal_slowdown", "other_compute_processes", "raw_gpu_query",
            "raw_process_query", "query_error",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow(
                {
                    "protocol_id": "C1-R1-v1.1",
                    "config": config,
                    "seed": seed,
                    "pass_name": pass_name,
                    "event": "after_job",
                    "time_ns": 1,
                    "thermal_slowdown": "Active" if issue == "thermal_slowdown" else "False",
                    "other_compute_processes": "999, other" if issue == "other_compute_processes" else "",
                    "raw_gpu_query": "fixture",
                    "raw_process_query": "",
                    "query_error": "query failed" if issue == "telemetry_failure" else "",
                }
            )


class X8C1CleanRoomExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.repo = self.base / "repo"
        self.root = self.base / "clean-room"
        self.active_env = self.base / "active-conda"
        self.active_env.mkdir(parents=True)

        runner_path = "src/py/experiments/c1_r1_combined_rerun.py"
        input_path = "src/py/data/FB15K237/train2id.txt"
        runner_bytes = b"print('fixture runner')\n"
        input_bytes = b"1\n0 0 0\n"
        for relative, content in [(runner_path, runner_bytes), (input_path, input_bytes)]:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        historical = self.repo / "output/results/c1_r1_combined_rerun/old.csv"
        historical.parent.mkdir(parents=True)
        historical.write_text("historical\n", encoding="utf-8")

        contract = fixture_contract(
            {
                runner_path: sha256_bytes(runner_bytes),
                input_path: sha256_bytes(input_bytes),
            }
        )
        write_json(self.repo / executor.CONTRACT_PATH, contract)

    def test_prepare_builds_verified_allowlist_capsule_and_offline_local_environment(self):
        """Catches broad/unverified copies, online setup, or root overwrites."""
        transport = FakeExternalTransport()

        result = executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )

        self.assertEqual(result["state"], "PREPARED")
        capsule_manifest = json.loads(
            (self.root / "capsule_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [entry["path"] for entry in capsule_manifest["files"]],
            [
                "src/py/data/FB15K237/train2id.txt",
                "src/py/experiments/c1_r1_combined_rerun.py",
            ],
        )
        self.assertFalse(
            (self.root / "capsule/output/results/c1_r1_combined_rerun/old.csv").exists()
        )
        self.assertTrue((self.root / "environment/conda/bin/python").is_file())
        clone_command = next(command for command in transport.commands if command[:2] == ["conda", "create"])
        self.assertIn("--offline", clone_command)
        self.assertEqual(
            clone_command[clone_command.index("--clone") + 1], str(self.active_env.resolve())
        )
        with self.assertRaises(FileExistsError):
            executor.prepare(
                self.repo,
                self.root,
                transport=transport,
                active_conda_prefix=self.active_env,
            )

    def test_preflight_and_run_follow_seed_major_serial_order_and_resume(self):
        """Catches order drift, non-capsule execution, and duplicate resumed jobs."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )

        executor.preflight(self.root, transport=transport)
        first = executor.run(self.root, transport=transport)
        runner_commands = [
            command for command in transport.commands
            if len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
        ]

        self.assertEqual(
            [
                (
                    command[2],
                    command[command.index("--pass-name") + 1] if "--pass-name" in command else "",
                    command[command.index("--config") + 1] if "--config" in command else "",
                    int(command[command.index("--seed") + 1]) if "--seed" in command else -1,
                )
                for command in runner_commands
            ],
            [
                ("preflight", "", "", -1),
                ("job", "throughput", "BL", 42),
                ("job", "throughput", "GPU", 42),
                ("job", "trace", "BL", 42),
                ("job", "trace", "GPU", 42),
                ("compute-only", "", "", 42),
            ],
        )
        self.assertTrue(all(str(self.root / "capsule") in command[1] for command in runner_commands))
        self.assertEqual(first["state"], "RAW_COMPLETE")
        executor.run(self.root, transport=transport)
        resumed_runner_commands = [
            command for command in transport.commands
            if len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
        ]
        self.assertEqual(len(resumed_runner_commands), len(runner_commands))

    def test_run_stops_and_records_invalid_when_telemetry_query_failed(self):
        """Catches acceptance of valid runner status with failed raw telemetry."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        executor.preflight(self.root, transport=transport)
        transport.telemetry_issues[("throughput", "BL", 42)] = "telemetry_failure"

        with self.assertRaisesRegex(RuntimeError, "telemetry_failure"):
            executor.run(self.root, transport=transport)

        manifest = json.loads(
            (self.root / "execution_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["jobs"][1]["state"], "INVALID")
        self.assertEqual(manifest["jobs"][1]["invalid_reason"], "telemetry_failure")
        self.assertEqual(manifest["jobs"][2]["state"], "PENDING")

    def test_remediate_retries_the_whole_invalid_pair_once_in_frozen_order(self):
        """Catches selective, reordered, destructive, or repeated remediation."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        executor.preflight(self.root, transport=transport)
        transport.telemetry_issues[("throughput", "BL", 42)] = "thermal_slowdown"
        with self.assertRaisesRegex(RuntimeError, "thermal_slowdown"):
            executor.run(self.root, transport=transport)
        transport.telemetry_issues.clear()
        prior_command_count = len(transport.commands)

        remediated = executor.remediate(
            self.root,
            pass_name="throughput",
            seed=42,
            transport=transport,
        )

        retry_commands = [
            command for command in transport.commands[prior_command_count:]
            if len(command) >= 3 and command[2] == "job"
        ]
        self.assertEqual(
            [command[command.index("--config") + 1] for command in retry_commands],
            ["BL", "GPU"],
        )
        self.assertEqual(
            [job["state"] for job in remediated["remediations"][0]["jobs"]],
            ["COMPLETED", "COMPLETED"],
        )
        self.assertFalse(remediated["jobs"][1]["analysis_eligible"])
        self.assertEqual(remediated["jobs"][2]["state"], "SKIPPED_INVALID_PAIR")
        with self.assertRaisesRegex(RuntimeError, "maximum"):
            executor.remediate(
                self.root,
                pass_name="throughput",
                seed=42,
                transport=transport,
            )

    def test_seal_binds_every_raw_artifact_and_detects_post_seal_tampering(self):
        """Catches incomplete/circular seals and post-seal byte modification."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        executor.preflight(self.root, transport=transport)
        executor.run(self.root, transport=transport)

        passport = executor.seal(self.root, sealed_at_ns=123456789)
        artifact_manifest = json.loads(
            (self.root / "raw_artifact_manifest.json").read_text(encoding="utf-8")
        )
        paths = [entry["path"] for entry in artifact_manifest["artifacts"]]
        self.assertEqual(paths, sorted(paths))
        self.assertIn("execution_manifest.json", paths)
        self.assertIn("raw/environment.json", paths)
        self.assertNotIn("raw_artifact_manifest.json", paths)
        self.assertNotIn("material_passport.json", paths)
        self.assertEqual(passport["stage"], "raw")
        self.assertEqual(passport["sealed_at_ns"], 123456789)
        self.assertTrue(executor.validate_raw_seal(self.root))

        artifact = next(
            self.root / path for path in paths if path.endswith("per_epoch.csv")
        )
        artifact.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "raw artifact.*mismatch"):
            executor.validate_raw_seal(self.root)

    def test_preflight_rejects_local_conda_clone_drift_before_gpu_execution(self):
        """Catches post-prepare mutation of the cloned runtime environment."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        (self.root / "environment/conda/conda-meta/history").write_text(
            "mutated\n", encoding="utf-8"
        )
        prior_runner_count = sum(
            len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
            for command in transport.commands
        )

        with self.assertRaisesRegex(RuntimeError, "environment clone drift"):
            executor.preflight(self.root, transport=transport)

        runner_count = sum(
            len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
            for command in transport.commands
        )
        self.assertEqual(runner_count, prior_runner_count)

    def test_status_cli_reports_the_resumable_manifest_without_gpu_execution(self):
        """Catches a missing command surface or inaccurate resume accounting."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            returncode = executor.main(["status", "--root", str(self.root)])

        observed = json.loads(output.getvalue())
        self.assertEqual(returncode, 0)
        self.assertEqual(observed["state"], "PREPARED")
        self.assertEqual(observed["jobs"], {"COMPLETED": 0, "PENDING": 6})
        self.assertFalse(observed["sealed"])


if __name__ == "__main__":
    unittest.main()
