import contextlib
import hashlib
import csv
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Callable

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
            "batch_size": 5000,
            "epochs_per_job": 5,
            "negative_samples_per_positive": 150,
            "training_examples": 267115,
        },
        "execution_matrix": {
            "primary": {
                "seeds": [42],
                "passes": ["throughput", "trace"],
                "configs": ["BL", "GPU"],
            },
            "diagnostic": {"seeds": [42], "observations_per_seed": 20},
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
        "raw_artifact_schemas": {
            "per_epoch.csv": {
                "required_columns": [
                    "protocol_id", "pass_name", "config", "seed", "epoch",
                    "epoch_time_ns", "num_steps", "full_batch_count",
                    "partial_batch_count", "partial_batch_size",
                    "training_examples", "loss_finite",
                ]
            },
            "per_step.csv": {
                "required_columns": [
                    "protocol_id", "pass_name", "config", "seed", "epoch", "step",
                    "batch_size_actual", "is_partial", "is_first_measured_step",
                    "neg_time_ns", "component_sum_ns", "total_step_ns",
                    "timing_residual_ns",
                ]
            },
            "gpu_telemetry.csv": {
                "required_columns": [
                    "protocol_id", "config", "seed", "pass_name", "event", "time_ns",
                    "thermal_slowdown", "other_compute_processes", "raw_gpu_query",
                    "raw_process_query", "query_error",
                ]
            },
        },
    }


class FakeExternalTransport:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.telemetry_issues: dict[tuple[str, str, int], str] = {}
        self.header_only_jobs: set[tuple[str, str, int]] = set()
        self.header_only_compute: set[int] = set()
        self.invocations: list[tuple[list[str], dict[str, object]]] = []
        self.after_runner: Callable[[list[str]], None] | None = None
        self.runtime_pytorch = "2.7.1+cu118"
        self.gpu_name = "NVIDIA GeForce RTX 3070"
        self.interrupt_job_once: tuple[str, str, int] | None = None
        self.malformed_preflight_telemetry = False
        self.wrapper_telemetry_issue = ""
        self.gpu_identity_failure = False

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        self.invocations.append((list(command), dict(kwargs)))
        if command[:2] == ["conda", "create"]:
            clone = Path(command[command.index("--prefix") + 1])
            (clone / "bin").mkdir(parents=True)
            (clone / "bin/python").write_bytes(b"fixture-python")
            (clone / "conda-meta").mkdir()
            (clone / "conda-meta/history").write_text("fixture\n", encoding="utf-8")
            site_packages = clone / "lib/python3.11/site-packages"
            site_packages.mkdir(parents=True)
            (site_packages / "fixture_runtime.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
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
                        "pytorch": self.runtime_pytorch,
                        "torch_cuda_runtime": "11.8",
                        "cuda_available": True,
                    }
                ),
                "",
            )
        if command and command[0] == "nvidia-smi" and any(
            "--query-compute-apps" in part for part in command
        ):
            if self.wrapper_telemetry_issue == "telemetry_failure":
                return subprocess.CompletedProcess(command, 1, "", "query failed")
            processes = (
                "999, other, 100\n"
                if self.wrapper_telemetry_issue == "other_compute_processes"
                else ""
            )
            return subprocess.CompletedProcess(command, 0, processes, "")
        if command and command[0] == "nvidia-smi" and any(
            "sw_thermal_slowdown" in part for part in command
        ):
            if self.wrapper_telemetry_issue == "telemetry_failure":
                return subprocess.CompletedProcess(command, 1, "", "query failed")
            thermal = "Active" if self.wrapper_telemetry_issue == "thermal_slowdown" else "Not Active"
            return subprocess.CompletedProcess(command, 0, f"fixture-gpu, {thermal}\n", "")
        if command and command[0] == "nvidia-smi":
            if self.gpu_identity_failure:
                return subprocess.CompletedProcess(
                    command,
                    9,
                    "",
                    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
                )
            return subprocess.CompletedProcess(
                command, 0, f"{self.gpu_name}, 8192, 8.6\n", ""
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
                if self.malformed_preflight_telemetry:
                    (output_root / "preflight/gpu_telemetry.csv").write_text(
                        "protocol_id\nC1-R1-v1.1\n", encoding="utf-8"
                    )
            elif kind == "job":
                config = command[command.index("--config") + 1]
                pass_name = command[command.index("--pass-name") + 1]
                seed = int(command[command.index("--seed") + 1])
                job_dir = output_root / "jobs" / f"{pass_name}_{config}_seed{seed}"
                if self.interrupt_job_once == (pass_name, config, seed):
                    self.interrupt_job_once = None
                    job_dir.mkdir(parents=True, exist_ok=True)
                    (job_dir / "partial.tmp").write_text("partial\n", encoding="utf-8")
                    raise KeyboardInterrupt("fixture hard interruption")
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
                self._write_primary_csvs(
                    job_dir,
                    pass_name,
                    config,
                    seed,
                    header_only=(pass_name, config, seed) in self.header_only_jobs,
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
                fields = [
                    "protocol_id", "seed", "repeat", "elapsed_ns", "loss",
                    "batch_size", "neg_num",
                ]
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    if seed not in self.header_only_compute:
                        for repeat in range(20):
                            writer.writerow(
                                {
                                    "protocol_id": "C1-R1-v1.1",
                                    "seed": seed,
                                    "repeat": repeat,
                                    "elapsed_ns": 1000 + repeat,
                                    "loss": 1.0,
                                    "batch_size": 5000,
                                    "neg_num": 150,
                                }
                            )
            if self.after_runner is not None:
                self.after_runner(command)
            return subprocess.CompletedProcess(command, 0, "runner output\n", "")
        raise AssertionError(f"unexpected external command: {command}")

    @staticmethod
    def _write_primary_csvs(
        job_dir: Path,
        pass_name: str,
        config: str,
        seed: int,
        *,
        header_only: bool,
    ) -> None:
        epoch_fields = [
            "protocol_id", "pass_name", "config", "seed", "epoch",
            "epoch_time_ns", "num_steps", "full_batch_count", "partial_batch_count",
            "partial_batch_size", "training_examples", "loss_finite",
        ]
        with (job_dir / "per_epoch.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=epoch_fields, lineterminator="\n")
            writer.writeheader()
            if not header_only:
                for epoch in range(5):
                    writer.writerow(
                        {
                            "protocol_id": "C1-R1-v1.1",
                            "pass_name": pass_name,
                            "config": config,
                            "seed": seed,
                            "epoch": epoch,
                            "epoch_time_ns": 1_000_000 + epoch,
                            "num_steps": 54,
                            "full_batch_count": 53,
                            "partial_batch_count": 1,
                            "partial_batch_size": 2115,
                            "training_examples": 267115,
                            "loss_finite": "True",
                        }
                    )
        if pass_name != "trace":
            return
        step_fields = [
            "protocol_id", "pass_name", "config", "seed", "epoch", "step",
            "batch_size_actual", "is_partial", "is_first_measured_step",
            "neg_time_ns", "component_sum_ns", "total_step_ns", "timing_residual_ns",
        ]
        with (job_dir / "per_step.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=step_fields, lineterminator="\n")
            writer.writeheader()
            if not header_only:
                for epoch in range(5):
                    for step in range(54):
                        partial = step == 53
                        writer.writerow(
                            {
                                "protocol_id": "C1-R1-v1.1",
                                "pass_name": pass_name,
                                "config": config,
                                "seed": seed,
                                "epoch": epoch,
                                "step": step,
                                "batch_size_actual": 2115 if partial else 5000,
                                "is_partial": str(partial),
                                "is_first_measured_step": str(epoch == 0 and step == 0),
                                "neg_time_ns": 1000 + step,
                                "component_sum_ns": 100,
                                "total_step_ns": 110,
                                "timing_residual_ns": 10,
                            }
                        )

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

    def test_prepare_lists_packages_with_offline_environment_not_unsupported_flag(self):
        """Catches Conda versions where ``list`` rejects the ``--offline`` argument."""
        transport = FakeExternalTransport()

        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )

        list_command = next(
            command for command in transport.commands if command[:2] == ["conda", "list"]
        )
        self.assertEqual(
            list_command,
            ["conda", "list", "--json", "--prefix", str(self.root / "environment/conda")],
        )

    def test_both_package_list_call_sites_keep_offline_controls_without_unsupported_flag(self):
        """Catches a later live-environment probe reintroducing unsupported online-prone flags."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo,
            self.root,
            transport=transport,
            active_conda_prefix=self.active_env,
        )
        _, contract = executor._load_prepared(self.root)
        executor._validate_live_environment(self.root, contract, transport)

        list_invocations = [
            (command, kwargs)
            for command, kwargs in transport.invocations
            if command[:2] == ["conda", "list"]
        ]
        self.assertEqual(len(list_invocations), 2)
        for command, kwargs in list_invocations:
            self.assertNotIn("--offline", command)
            self.assertEqual(
                command,
                ["conda", "list", "--json", "--prefix", str(self.root / "environment/conda")],
            )
            environment = kwargs["env"]
            self.assertEqual(environment["CONDA_OFFLINE"], "true")
            self.assertEqual(environment["PIP_NO_INDEX"], "1")
            self.assertEqual(environment["http_proxy"], "")
            self.assertEqual(environment["https_proxy"], "")
            self.assertEqual(environment["HTTP_PROXY"], "")
            self.assertEqual(environment["HTTPS_PROXY"], "")
            self.assertEqual(environment["ALL_PROXY"], "")

    def test_prepare_gpu_identity_failure_persists_artifact_backed_blocked_closure(self):
        """Catches loss of command, commit, probe, or stderr lineage before GPU identity fails."""
        transport = FakeExternalTransport()
        transport.gpu_identity_failure = True

        with self.assertRaisesRegex(RuntimeError, "nvidia-smi"):
            executor.prepare(
                self.repo,
                self.root,
                transport=transport,
                active_conda_prefix=self.active_env,
            )

        attempt_path = self.root / "raw/prepare_attempt.json"
        closure_path = self.root / "blocked_environment_closure.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        first_closure = closure_path.read_bytes()
        closure = json.loads(first_closure)
        self.assertEqual(attempt["state"], "BLOCKED_ENVIRONMENT")
        self.assertEqual(attempt["failure"]["stage"], "gpu_identity")
        self.assertEqual(attempt["git_head"]["stdout"].strip(), "a" * 40)
        self.assertEqual(attempt["active_prefix_probe"]["cuda_available"], True)
        self.assertEqual(attempt["clone_prefix_probe"]["pytorch"], "2.7.1+cu118")
        self.assertEqual(
            attempt["command_captures"][-1]["command"][0], "nvidia-smi"
        )
        self.assertEqual(attempt["command_captures"][-1]["returncode"], 9)
        self.assertIn("couldn't communicate", attempt["command_captures"][-1]["stderr"])
        self.assertEqual(closure["executor_commit"], "a" * 40)
        self.assertEqual(closure["runtime_probe"], attempt["clone_prefix_probe"])
        self.assertEqual(closure["failure"], attempt["failure"])

        regenerated = executor.regenerate_blocked_environment_closure(self.root)
        self.assertEqual(regenerated, closure)
        self.assertEqual(closure_path.read_bytes(), first_closure)

    def test_frozen_job_descriptors_are_exact_and_truncation_is_rejected(self):
        """Catches a mutable/truncated manifest replacing the frozen 31-job matrix."""
        root_contract = executor.load_contract(Path(__file__).resolve().parents[1])
        full_jobs = executor._build_execution_jobs(root_contract)
        self.assertEqual(len(full_jobs), 31)
        primary_jobs = [job for job in full_jobs if job["kind"] == "job"]
        diagnostic_jobs = [job for job in full_jobs if job["kind"] == "compute-only"]
        self.assertEqual(len(primary_jobs), 24)
        self.assertEqual(len(diagnostic_jobs), 6)
        self.assertEqual([job["seed"] for job in diagnostic_jobs], [42, 43, 44, 45, 46, 47])
        self.assertEqual(
            [(job.get("seed"), job.get("pass_name"), job.get("config")) for job in full_jobs[1:5]],
            [
                (42, "throughput", "BL"),
                (42, "throughput", "GPU"),
                (42, "trace", "BL"),
                (42, "trace", "GPU"),
            ],
        )
        for seed in range(42, 48):
            observed = [
                (job["pass_name"], job["config"])
                for job in primary_jobs if job["seed"] == seed
            ]
            pair = ["BL", "GPU"] if seed % 2 == 0 else ["GPU", "BL"]
            self.assertEqual(
                observed,
                [("throughput", pair[0]), ("throughput", pair[1]),
                 ("trace", pair[0]), ("trace", pair[1])],
            )
        seed43 = [
            job for job in full_jobs
            if job.get("kind") == "job" and job.get("seed") == 43
        ]
        self.assertEqual(
            [(job["pass_name"], job["config"]) for job in seed43],
            [
                ("throughput", "GPU"),
                ("throughput", "BL"),
                ("trace", "GPU"),
                ("trace", "BL"),
            ],
        )

        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        manifest_path = self.root / "execution_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete_job_descriptors = manifest["jobs"]
        manifest["jobs"] = manifest["jobs"][:1]
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(RuntimeError, "job descriptor.*drift"):
            executor.preflight(self.root, transport=transport)

        manifest["jobs"] = complete_job_descriptors
        for job in manifest["jobs"]:
            job["state"] = "COMPLETED"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(RuntimeError, "missing|not found"):
            executor.run(self.root, transport=transport)

    def test_manifest_rejects_forged_superseded_control_flow(self):
        """Catches a forged skip flag suppressing a required primary job."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        manifest_path = self.root / "execution_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["jobs"][1]["superseded_by"] = 1
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(RuntimeError, "control-flow.*drift"):
            executor.run(self.root, transport=transport)

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
        runner_envs = [
            kwargs["env"]
            for command, kwargs in transport.invocations
            if len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
        ]
        self.assertTrue(
            all(env["PYTHONDONTWRITEBYTECODE"] == "1" for env in runner_envs)
        )
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

    def test_preflight_rejects_telemetry_missing_frozen_schema(self):
        """Catches preflight acceptance when contention/query columns are absent."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        transport.malformed_preflight_telemetry = True

        with self.assertRaisesRegex(RuntimeError, "gpu_telemetry.csv schema"):
            executor.preflight(self.root, transport=transport)

    def test_run_rejects_header_only_primary_csv_despite_valid_status(self):
        """Catches trusting status row counts without validating raw CSV content."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        transport.header_only_jobs.add(("throughput", "BL", 42))

        with self.assertRaisesRegex(RuntimeError, "per_epoch.*row count"):
            executor.run(self.root, transport=transport)

    def test_compute_only_is_diagnostic_only_and_header_only_is_rejected(self):
        """Catches diagnostics masquerading as exclusivity evidence or empty data."""
        contract = executor.load_contract(Path(__file__).resolve().parents[1])
        diagnostic_jobs = [
            job for job in executor._build_execution_jobs(contract)
            if job["kind"] == "compute-only"
        ]
        self.assertTrue(diagnostic_jobs)
        self.assertTrue(all(job["evidence_scope"] == "diagnostic_only" for job in diagnostic_jobs))
        self.assertTrue(all(job["validates_gpu_exclusivity"] is False for job in diagnostic_jobs))

        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        transport.header_only_compute.add(42)
        with self.assertRaisesRegex(RuntimeError, "compute-only row count"):
            executor.run(self.root, transport=transport)

    def test_compute_only_wrapper_telemetry_stops_before_thermal_dispatch(self):
        """Catches diagnostic execution without wrapper-level GPU gating."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        transport.wrapper_telemetry_issue = "thermal_slowdown"

        with self.assertRaisesRegex(RuntimeError, "thermal_slowdown"):
            executor.run(self.root, transport=transport)

        diagnostic_commands = [
            command for command in transport.commands
            if len(command) >= 3 and command[2] == "compute-only"
        ]
        self.assertEqual(diagnostic_commands, [])
        telemetry = self.root / "raw/attempts/diagnostics_attempt0/compute_only/seed42.gpu_telemetry.csv"
        with telemetry.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["event"] for row in rows], ["before_job"])

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
        twin = self.base / "clean-room-twin"
        shutil.copytree(self.root, twin)

        passport = executor.seal(self.root)
        executor.seal(twin)
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
        self.assertGreater(passport["sealed_at_ns"], 0)
        self.assertEqual(
            (self.root / "raw_artifact_manifest.json").read_bytes(),
            (twin / "raw_artifact_manifest.json").read_bytes(),
        )
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

    def test_preflight_rejects_added_capsule_shadow_module_before_dispatch(self):
        """Catches unlisted shadow modules or bytecode entering the capsule."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        shadow = self.root / "capsule/src/py/load/torch.py"
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.write_text("raise RuntimeError('shadow')\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "capsule file-set drift"):
            executor.preflight(self.root, transport=transport)

        self.assertFalse(
            any(
                len(command) >= 3 and command[1].endswith("c1_r1_combined_rerun.py")
                for command in transport.commands
            )
        )

    def test_preflight_rejects_site_packages_byte_mutation(self):
        """Catches runtime-package mutation outside bin/python and conda-meta."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        runtime_file = (
            self.root
            / "environment/conda/lib/python3.11/site-packages/fixture_runtime.py"
        )
        runtime_file.write_text("VALUE = 2\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "environment clone drift"):
            executor.preflight(self.root, transport=transport)

    def test_environment_identity_hashes_directory_and_broken_symlinks_without_following(self):
        """Catches stat/hash traversal through common Conda-layout symlinks."""
        prefix = self.base / "symlink-env"
        prefix.mkdir()
        (prefix / "regular.txt").write_text("inside\n", encoding="utf-8")
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("outside\n", encoding="utf-8")
        (prefix / "lib64").symlink_to(outside, target_is_directory=True)
        (prefix / "broken").symlink_to("missing-target")

        entries = executor._environment_clone_entries(prefix)

        by_path = {entry["path"]: entry for entry in entries}
        self.assertEqual(set(by_path), {"broken", "lib64", "regular.txt"})
        for name, target in [("broken", "missing-target"), ("lib64", str(outside))]:
            self.assertEqual(by_path[name]["symlink_target"], target)
            self.assertEqual(
                by_path[name]["sha256"],
                hashlib.sha256(b"symlink\0" + target.encode("utf-8")).hexdigest(),
            )

    def test_run_revalidates_capsule_immediately_before_each_dispatch(self):
        """Catches capsule mutation between two jobs in the same run call."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        mutated = False

        def mutate_after_first_primary(command: list[str]) -> None:
            nonlocal mutated
            if command[2] == "job" and not mutated:
                mutated = True
                path = self.root / "capsule/src/py/load/shadow.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n", encoding="utf-8")

        transport.after_runner = mutate_after_first_primary
        with self.assertRaisesRegex(RuntimeError, "capsule file-set drift"):
            executor.run(self.root, transport=transport)

        primary_commands = [
            command for command in transport.commands
            if len(command) >= 3 and command[2] == "job"
        ]
        self.assertEqual(len(primary_commands), 1)

    def test_run_reprobes_runtime_identity_before_each_dispatch(self):
        """Catches live runtime drift between jobs despite unchanged manifest bytes."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        changed = False

        def change_probe_after_first_primary(command: list[str]) -> None:
            nonlocal changed
            if command[2] == "job" and not changed:
                changed = True
                transport.runtime_pytorch = "9.9.9+drift"

        transport.after_runner = change_probe_after_first_primary
        with self.assertRaisesRegex(RuntimeError, "live environment.*drift"):
            executor.run(self.root, transport=transport)

        primary_commands = [
            command for command in transport.commands
            if len(command) >= 3 and command[2] == "job"
        ]
        self.assertEqual(len(primary_commands), 1)

    def test_crashed_running_job_is_invalidated_without_overwriting_partial_output(self):
        """Catches redispatch into a partial output directory after hard interruption."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        transport.interrupt_job_once = ("throughput", "BL", 42)
        with self.assertRaises(KeyboardInterrupt):
            executor.run(self.root, transport=transport)
        command_count = len(
            [command for command in transport.commands if len(command) >= 3 and command[2] == "job"]
        )

        with self.assertRaisesRegex(RuntimeError, "interrupted RUNNING"):
            executor.run(self.root, transport=transport)

        manifest = json.loads(
            (self.root / "execution_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["jobs"][1]["state"], "INVALID")
        self.assertEqual(manifest["jobs"][1]["invalid_reason"], "infrastructure_failure")
        self.assertTrue(
            (
                self.root
                / "raw/attempts/throughput_seed42_attempt0/jobs/throughput_BL_seed42/partial.tmp"
            ).is_file()
        )
        self.assertEqual(
            len([command for command in transport.commands if len(command) >= 3 and command[2] == "job"]),
            command_count,
        )

    def test_interrupted_remediation_is_retained_incomplete_and_never_redispatched(self):
        """Catches more than one physical retry dispatch for a pass/seed pair."""
        transport = FakeExternalTransport()
        executor.prepare(
            self.repo, self.root, transport=transport, active_conda_prefix=self.active_env
        )
        executor.preflight(self.root, transport=transport)
        transport.telemetry_issues[("throughput", "BL", 42)] = "thermal_slowdown"
        with self.assertRaisesRegex(RuntimeError, "thermal_slowdown"):
            executor.run(self.root, transport=transport)
        transport.telemetry_issues.clear()
        transport.interrupt_job_once = ("throughput", "BL", 42)
        with self.assertRaises(KeyboardInterrupt):
            executor.remediate(
                self.root, pass_name="throughput", seed=42, transport=transport
            )
        retry_command_count = len(
            [
                command for command in transport.commands
                if len(command) >= 3
                and command[2] == "job"
                and "attempt1" in " ".join(command)
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "INCOMPLETE|maximum"):
            executor.remediate(
                self.root, pass_name="throughput", seed=42, transport=transport
            )
        closed = json.loads(
            (self.root / "execution_manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(closed["remediations"]), 1)
        self.assertEqual(closed["remediations"][0]["state"], "INCOMPLETE")
        self.assertTrue(
            (
                self.root
                / "raw/attempts/throughput_seed42_attempt1/jobs/throughput_BL_seed42/partial.tmp"
            ).is_file()
        )
        with self.assertRaisesRegex(RuntimeError, "maximum|INCOMPLETE"):
            executor.remediate(
                self.root, pass_name="throughput", seed=42, transport=transport
            )
        self.assertEqual(
            len(
                [
                    command for command in transport.commands
                    if len(command) >= 3
                    and command[2] == "job"
                    and "attempt1" in " ".join(command)
                ]
            ),
            retry_command_count,
        )

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
