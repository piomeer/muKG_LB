"""Isolated, fail-closed executor for the frozen X8 C1-R1 protocol."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


CONTRACT_PATH = Path("output/results/evidence_audit_x8_c1_r1/clean_room_contract.json")
RUNNER_PATH = Path("src/py/experiments/c1_r1_combined_rerun.py")

CommandTransport = Callable[..., subprocess.CompletedProcess[str]]


class InvalidAttempt(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def load_contract(repo_root: Path) -> dict[str, Any]:
    """Load the frozen X8 contract for a future executor invocation."""
    with (repo_root / CONTRACT_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _command_capture(
    transport: CommandTransport,
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = transport(
        command,
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    capture = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode:
        raise RuntimeError(
            f"external command failed ({result.returncode}): {' '.join(command)}"
        )
    return capture


def _offline_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CONDA_OFFLINE": "true",
            "PIP_NO_INDEX": "1",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "*",
        }
    )
    return env


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("status") != "FROZEN":
        raise RuntimeError("clean-room contract is not FROZEN")
    if contract.get("contract_id") != "X8-C1-R1-clean-room-v1":
        raise RuntimeError("unexpected clean-room contract id")
    if not contract.get("environment", {}).get("network_forbidden"):
        raise RuntimeError("clean-room contract must forbid network access")
    allowlist = contract.get("capsule", {}).get("allowlisted_paths")
    source_hashes = contract.get("source_hashes")
    if not isinstance(allowlist, list) or not isinstance(source_hashes, dict):
        raise RuntimeError("contract capsule allowlist/source hashes are malformed")
    if set(allowlist) != set(source_hashes):
        raise RuntimeError("every allowlisted path must have exactly one frozen hash")
    excluded = tuple(contract["capsule"].get("excluded_path_prefixes", []))
    if any(path == prefix or path.startswith(prefix + "/") for path in allowlist for prefix in excluded):
        raise RuntimeError("capsule allowlist intersects an excluded path prefix")
    if RUNNER_PATH.as_posix() not in source_hashes:
        raise RuntimeError("frozen runner is absent from the capsule")


def _build_execution_jobs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = [
        {"id": "preflight", "kind": "preflight", "state": "PENDING", "attempt": 0}
    ]
    primary = contract["execution_matrix"]["primary"]
    paired_order = contract["protocol"]["paired_order"]
    for seed in sorted(primary["seeds"]):
        for pass_name in primary["passes"]:
            for config in paired_order[str(seed)]:
                jobs.append(
                    {
                        "id": f"{pass_name}_{config}_seed{seed}",
                        "kind": "job",
                        "pass_name": pass_name,
                        "config": config,
                        "seed": seed,
                        "state": "PENDING",
                        "attempt": 0,
                    }
                )
    for seed in sorted(contract["execution_matrix"]["diagnostic"]["seeds"]):
        jobs.append(
            {
                "id": f"compute_only_seed{seed}",
                "kind": "compute-only",
                "seed": seed,
                "state": "PENDING",
                "attempt": 0,
            }
        )
    return jobs


def _environment_clone_entries(cloned_prefix: Path) -> list[dict[str, Any]]:
    paths = [cloned_prefix / "bin/python"]
    conda_meta = cloned_prefix / "conda-meta"
    if conda_meta.is_dir():
        paths.extend(path for path in conda_meta.rglob("*") if path.is_file())
    unique_paths = sorted(set(paths), key=lambda path: path.relative_to(cloned_prefix).as_posix())
    return [
        {
            "path": path.relative_to(cloned_prefix).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in unique_paths
    ]


def prepare(
    repo_root: Path,
    root: Path,
    *,
    transport: CommandTransport = subprocess.run,
    active_conda_prefix: Path | None = None,
) -> dict[str, Any]:
    """Create a new allowlisted capsule and an offline local Conda clone."""
    repo_root = repo_root.resolve()
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing clean-room root: {root}")

    contract_path = repo_root / CONTRACT_PATH
    contract = load_contract(repo_root)
    _validate_contract(contract)
    contract_sha256 = _sha256_file(contract_path)
    source_hashes: dict[str, str] = contract["source_hashes"]
    for relative, expected in source_hashes.items():
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"allowlisted source/input is missing: {relative}")
        actual = _sha256_file(source)
        if actual != expected:
            raise RuntimeError(
                f"source/input hash drift for {relative}: expected {expected}, got {actual}"
            )

    if active_conda_prefix is None:
        prefix_text = os.environ.get("CONDA_PREFIX", "")
        if not prefix_text:
            raise RuntimeError("CONDA_PREFIX is required to clone the active environment")
        active_conda_prefix = Path(prefix_text)
    active_conda_prefix = active_conda_prefix.resolve()
    if not active_conda_prefix.is_dir():
        raise FileNotFoundError(f"active Conda environment does not exist: {active_conda_prefix}")

    root.mkdir(parents=True)
    capsule = root / "capsule"
    capsule_files = []
    for relative in sorted(source_hashes):
        source = repo_root / relative
        destination = capsule / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        actual = _sha256_file(destination)
        if actual != source_hashes[relative]:
            raise RuntimeError(f"capsule copy hash mismatch for {relative}")
        capsule_files.append(
            {"path": relative, "bytes": destination.stat().st_size, "sha256": actual}
        )
    capsule_manifest = {
        "contract_id": contract["contract_id"],
        "source_mutation_forbidden": True,
        "files": capsule_files,
    }
    capsule_manifest_path = root / "capsule_manifest.json"
    _write_json(capsule_manifest_path, capsule_manifest)

    frozen_contract_path = root / "frozen_contract.json"
    shutil.copyfile(contract_path, frozen_contract_path)
    if _sha256_file(frozen_contract_path) != contract_sha256:
        raise RuntimeError("frozen contract copy hash mismatch")

    offline_env = _offline_environment()
    cloned_prefix = root / "environment/conda"
    clone_capture = _command_capture(
        transport,
        [
            "conda",
            "create",
            "--yes",
            "--offline",
            "--prefix",
            str(cloned_prefix),
            "--clone",
            str(active_conda_prefix),
        ],
        cwd=repo_root,
        env=offline_env,
    )
    if not (cloned_prefix / "bin/python").is_file():
        raise RuntimeError("offline Conda clone did not create bin/python")

    conda_capture = _command_capture(
        transport,
        ["conda", "list", "--offline", "--json", "--prefix", str(cloned_prefix)],
        cwd=repo_root,
        env=offline_env,
    )
    try:
        packages = json.loads(conda_capture["stdout"])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Conda package manifest was not valid JSON") from exc
    probe_code = (
        "import json,sys,torch; print(json.dumps({"
        "'protocol_id':'C1-R1-v1.1','python':sys.version,"
        "'pytorch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    runtime_capture = _command_capture(
        transport,
        [str(cloned_prefix / "bin/python"), "-c", probe_code],
        cwd=capsule,
        env=offline_env,
    )
    try:
        runtime = json.loads(runtime_capture["stdout"])
    except json.JSONDecodeError as exc:
        raise RuntimeError("cloned runtime probe was not valid JSON") from exc
    gpu_capture = _command_capture(
        transport,
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=capsule,
        env=offline_env,
    )
    identity = [part.strip() for part in gpu_capture["stdout"].strip().split(",")]
    if len(identity) != 3:
        raise RuntimeError("GPU identity query returned an unexpected shape")
    gpu = {
        "name": identity[0],
        "total_memory_mib": float(identity[1]),
        "compute_capability": identity[2],
    }
    expected_runtime = contract["environment"]["expected_runtime"]
    if (
        runtime.get("protocol_id") != contract["protocol"]["protocol_id"]
        or runtime.get("pytorch") != expected_runtime["pytorch"]
        or runtime.get("torch_cuda_runtime") != expected_runtime["torch_cuda_runtime"]
        or not runtime.get("cuda_available")
    ):
        raise RuntimeError("cloned runtime does not match the frozen environment contract")
    expected_gpu = contract["environment"]["expected_gpu"]
    if (
        gpu["name"] != expected_gpu["name"]
        or gpu["compute_capability"] != expected_gpu["compute_capability"]
        or gpu["total_memory_mib"] != float(expected_gpu["total_memory_mib"])
    ):
        raise RuntimeError("GPU identity does not match the frozen environment contract")

    git_capture = _command_capture(
        transport,
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        cwd=repo_root,
        env=offline_env,
    )
    environment_document = {
        "protocol_id": contract["protocol"]["protocol_id"],
        **runtime,
        "gpu": gpu,
        "raw_command_captures": {
            "conda_clone": clone_capture,
            "conda_list": conda_capture,
            "runtime_probe": runtime_capture,
            "gpu_identity": gpu_capture,
            "git_commit": git_capture,
        },
    }
    _write_json(root / "raw/environment.json", environment_document)
    environment_manifest = {
        "contract_id": contract["contract_id"],
        "source_prefix": str(active_conda_prefix),
        "cloned_prefix": str(cloned_prefix),
        "git_commit": git_capture["stdout"].strip(),
        "packages": packages,
        "runtime": runtime,
        "gpu": gpu,
        "network_forbidden": True,
        "clone_identity_files": _environment_clone_entries(cloned_prefix),
    }
    environment_manifest_path = root / "environment_manifest.json"
    _write_json(environment_manifest_path, environment_manifest)

    execution_manifest = {
        "contract_id": contract["contract_id"],
        "protocol_id": contract["protocol"]["protocol_id"],
        "state": "PREPARED",
        "contract_sha256": contract_sha256,
        "capsule_manifest_sha256": _sha256_file(capsule_manifest_path),
        "environment_manifest_sha256": _sha256_file(environment_manifest_path),
        "jobs": _build_execution_jobs(contract),
        "remediations": [],
    }
    _write_json(root / "execution_manifest.json", execution_manifest)
    return execution_manifest


def _load_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest_path = root / "execution_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"clean-room root is not prepared: {root}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (root / "frozen_contract.json").open(encoding="utf-8") as handle:
        contract = json.load(handle)
    _validate_contract(contract)
    checks = {
        "frozen contract": (root / "frozen_contract.json", manifest["contract_sha256"]),
        "capsule manifest": (
            root / "capsule_manifest.json",
            manifest["capsule_manifest_sha256"],
        ),
        "environment manifest": (
            root / "environment_manifest.json",
            manifest["environment_manifest_sha256"],
        ),
    }
    for label, (path, expected) in checks.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"{label} drift detected")
    with (root / "capsule_manifest.json").open(encoding="utf-8") as handle:
        capsule_manifest = json.load(handle)
    expected_entries = contract["source_hashes"]
    observed_entries = {
        entry["path"]: entry["sha256"] for entry in capsule_manifest["files"]
    }
    if observed_entries != expected_entries:
        raise RuntimeError("capsule manifest does not match the frozen source hashes")
    for relative, expected in expected_entries.items():
        path = root / "capsule" / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"capsule source/input drift detected: {relative}")
    environment_manifest = _read_json(root / "environment_manifest.json")
    cloned_prefix = root / "environment/conda"
    if environment_manifest.get("clone_identity_files") != _environment_clone_entries(
        cloned_prefix
    ):
        raise RuntimeError("environment clone drift detected")
    return manifest, contract


def _write_execution_manifest(root: Path, manifest: dict[str, Any]) -> None:
    _write_json(root / "execution_manifest.json", manifest)


def _job_output_root(root: Path, job: dict[str, Any]) -> Path:
    attempt = job.get("attempt", 0)
    if job["kind"] == "preflight":
        return root / "raw/attempts" / f"preflight_attempt{attempt}"
    if job["kind"] == "job":
        return (
            root
            / "raw/attempts"
            / f"{job['pass_name']}_seed{job['seed']}_attempt{attempt}"
        )
    return root / "raw/attempts" / f"diagnostics_attempt{attempt}"


def _runner_command(root: Path, job: dict[str, Any]) -> list[str]:
    command = [
        str(root / "environment/conda/bin/python"),
        str(root / "capsule" / RUNNER_PATH),
        job["kind"],
        "--root",
        str(_job_output_root(root, job)),
    ]
    if job["kind"] == "job":
        command.extend(
            [
                "--config",
                job["config"],
                "--pass-name",
                job["pass_name"],
                "--seed",
                str(job["seed"]),
            ]
        )
    elif job["kind"] == "compute-only":
        command.extend(["--seed", str(job["seed"])])
    return command


def _job_creation_target(root: Path, job: dict[str, Any]) -> Path:
    output_root = _job_output_root(root, job)
    if job["kind"] == "preflight":
        return output_root / "preflight"
    if job["kind"] == "job":
        return (
            output_root
            / "jobs"
            / f"{job['pass_name']}_{job['config']}_seed{job['seed']}"
        )
    return output_root / "compute_only" / f"seed{job['seed']}.csv"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _telemetry_invalid_reason(path: Path, protocol_id: str) -> str | None:
    if not path.is_file():
        return "telemetry_failure"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "telemetry_failure"
    false_values = {"", "0", "false", "no", "none", "not active", "disabled"}
    for row in rows:
        if row.get("protocol_id") != protocol_id:
            return "telemetry_failure"
        if row.get("query_error", "").strip():
            return "telemetry_failure"
        if row.get("other_compute_processes", "").strip():
            return "other_compute_processes"
        if row.get("thermal_slowdown", "").strip().lower() not in false_values:
            return "thermal_slowdown"
    return None


def _validate_runner_artifacts(
    root: Path,
    job: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    output_root = _job_output_root(root, job)
    protocol_id = contract["protocol"]["protocol_id"]
    if job["kind"] == "preflight":
        result = _read_json(output_root / "preflight/result.json")
        if result.get("protocol_id") != protocol_id or result.get("all_passed") is not True:
            raise RuntimeError("preflight did not pass the frozen protocol")
        reason = _telemetry_invalid_reason(
            output_root / "preflight/gpu_telemetry.csv", protocol_id
        )
        if reason:
            raise InvalidAttempt(reason, "preflight telemetry is invalid")
        return
    if job["kind"] == "job":
        job_dir = (
            output_root
            / "jobs"
            / f"{job['pass_name']}_{job['config']}_seed{job['seed']}"
        )
        status = _read_json(job_dir / "status.json")
        identity = (status.get("pass_name"), status.get("config"), status.get("seed"))
        expected = (job["pass_name"], job["config"], job["seed"])
        if status.get("protocol_id") != protocol_id or identity != expected:
            raise RuntimeError(f"runner status identity drift for {job['id']}")
        if status.get("valid") is not True:
            reasons = status.get("invalid_reasons") or ["infrastructure_failure"]
            joined = " ".join(map(str, reasons)).lower()
            if "thermal" in joined:
                reason = "thermal_slowdown"
            elif "other" in joined and "process" in joined:
                reason = "other_compute_processes"
            elif "telemetry" in joined or "query" in joined:
                reason = "telemetry_failure"
            else:
                reason = "infrastructure_failure"
            raise InvalidAttempt(reason, f"runner marked {job['id']} invalid: {reasons}")
        if not (job_dir / "per_epoch.csv").is_file():
            raise RuntimeError(f"missing per_epoch.csv for {job['id']}")
        if job["pass_name"] == "trace" and not (job_dir / "per_step.csv").is_file():
            raise RuntimeError(f"missing per_step.csv for {job['id']}")
        reason = _telemetry_invalid_reason(job_dir / "gpu_telemetry.csv", protocol_id)
        if reason:
            raise InvalidAttempt(reason, f"telemetry invalid for {job['id']}")
        return
    diagnostic = output_root / "compute_only" / f"seed{job['seed']}.csv"
    if not diagnostic.is_file():
        raise RuntimeError(f"missing compute-only artifact for seed {job['seed']}")


def _execute_job(
    root: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    job: dict[str, Any],
    transport: CommandTransport,
) -> None:
    if job["state"] == "COMPLETED":
        return
    if job["state"] not in {"PENDING", "RUNNING"}:
        raise RuntimeError(f"job {job['id']} requires remediation: {job['state']}")
    creation_target = _job_creation_target(root, job)
    if job["state"] == "PENDING" and creation_target.exists():
        raise FileExistsError(f"refusing to overwrite runner output: {creation_target}")
    job["state"] = "RUNNING"
    job["command"] = _runner_command(root, job)
    job["started_at_ns"] = time.time_ns()
    _write_execution_manifest(root, manifest)
    result = transport(
        job["command"],
        cwd=str(root / "capsule"),
        env={**_offline_environment(), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
    )
    job["ended_at_ns"] = time.time_ns()
    command_dir = root / "raw/commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = command_dir / f"{job['id']}_attempt{job.get('attempt', 0)}.stdout.log"
    stderr_path = command_dir / f"{job['id']}_attempt{job.get('attempt', 0)}.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    job["returncode"] = result.returncode
    job["stdout_path"] = str(stdout_path.relative_to(root))
    job["stderr_path"] = str(stderr_path.relative_to(root))
    if result.returncode:
        job["state"] = "INVALID"
        job["invalid_reason"] = "infrastructure_failure"
        _write_execution_manifest(root, manifest)
        raise RuntimeError(f"runner command failed for {job['id']}: {result.returncode}")
    try:
        _validate_runner_artifacts(root, job, contract)
    except InvalidAttempt as exc:
        job["state"] = "INVALID"
        job["invalid_reason"] = exc.reason
        _write_execution_manifest(root, manifest)
        raise
    except Exception:
        job["state"] = "INVALID"
        job["invalid_reason"] = "infrastructure_failure"
        _write_execution_manifest(root, manifest)
        raise
    job["state"] = "COMPLETED"
    _write_execution_manifest(root, manifest)


def preflight(
    root: Path,
    *,
    transport: CommandTransport = subprocess.run,
) -> dict[str, Any]:
    """Run the frozen preflight once and stop on any failure."""
    root = root.resolve()
    manifest, contract = _load_prepared(root)
    job = manifest["jobs"][0]
    if job["kind"] != "preflight":
        raise RuntimeError("execution manifest does not begin with preflight")
    _execute_job(root, manifest, contract, job, transport)
    manifest["state"] = "PREFLIGHT_PASSED"
    _write_execution_manifest(root, manifest)
    return manifest


def run(
    root: Path,
    *,
    transport: CommandTransport = subprocess.run,
) -> dict[str, Any]:
    """Resume the remaining primary and diagnostic jobs in frozen serial order."""
    root = root.resolve()
    manifest, contract = _load_prepared(root)
    if manifest["jobs"][0]["state"] != "COMPLETED":
        raise RuntimeError("successful preflight is required before run")
    if (root / "material_passport.json").exists():
        raise RuntimeError("sealed raw artifacts cannot be executed or overwritten")
    for remediation in manifest["remediations"]:
        if any(job["state"] != "COMPLETED" for job in remediation["jobs"]):
            raise RuntimeError("an incomplete remediation must be resolved before run")
    manifest["state"] = "RUNNING"
    _write_execution_manifest(root, manifest)
    for job in manifest["jobs"][1:]:
        if job.get("superseded_by") is not None:
            continue
        _execute_job(root, manifest, contract, job, transport)
    manifest["state"] = "RAW_COMPLETE"
    _write_execution_manifest(root, manifest)
    return manifest


def remediate(
    root: Path,
    *,
    pass_name: str,
    seed: int,
    transport: CommandTransport = subprocess.run,
) -> dict[str, Any]:
    """Retry one explicitly invalid BL/GPU pair, exactly once and in place."""
    root = root.resolve()
    manifest, contract = _load_prepared(root)
    if (root / "material_passport.json").exists():
        raise RuntimeError("sealed raw artifacts cannot be remediated")
    primary = contract["execution_matrix"]["primary"]
    if pass_name not in primary["passes"] or seed not in primary["seeds"]:
        raise ValueError("remediation pass/seed is outside the frozen matrix")
    existing = [
        item
        for item in manifest["remediations"]
        if item["pass_name"] == pass_name and item["seed"] == seed
    ]
    maximum = contract["retry_policy"]["maximum_retries_per_pass_seed_pair"]
    if len(existing) >= maximum:
        raise RuntimeError("maximum remediation retries reached for pass/seed pair")
    pair = [
        job
        for job in manifest["jobs"]
        if job.get("kind") == "job"
        and job.get("pass_name") == pass_name
        and job.get("seed") == seed
    ]
    order = contract["protocol"]["paired_order"][str(seed)]
    by_config = {job["config"]: job for job in pair}
    if set(by_config) != set(contract["retry_policy"]["paired_configs"]):
        raise RuntimeError("execution manifest does not contain the frozen BL/GPU pair")
    invalid_jobs = [job for job in pair if job["state"] == "INVALID"]
    if not invalid_jobs:
        raise RuntimeError("remediation requires an explicitly invalid initial attempt")
    eligible = set(contract["retry_policy"]["eligible_failure_reasons"])
    invalid_reasons = {job.get("invalid_reason") for job in invalid_jobs}
    if not invalid_reasons <= eligible:
        raise RuntimeError(f"invalid reason is not eligible for remediation: {invalid_reasons}")

    attempt = len(existing) + 1
    retry_jobs = []
    for config in order:
        retry_jobs.append(
            {
                "id": f"{pass_name}_{config}_seed{seed}_retry{attempt}",
                "kind": "job",
                "pass_name": pass_name,
                "config": config,
                "seed": seed,
                "attempt": attempt,
                "state": "PENDING",
                "analysis_eligible": True,
            }
        )
    remediation_record = {
        "pass_name": pass_name,
        "seed": seed,
        "attempt": attempt,
        "trigger_reasons": sorted(invalid_reasons),
        "jobs": retry_jobs,
    }
    manifest["remediations"].append(remediation_record)
    for job in pair:
        job["analysis_eligible"] = False
        job["superseded_by"] = attempt
        if job["state"] == "PENDING":
            job["state"] = "SKIPPED_INVALID_PAIR"
    manifest["state"] = "REMEDIATING"
    _write_execution_manifest(root, manifest)
    for job in retry_jobs:
        _execute_job(root, manifest, contract, job, transport)
    manifest["state"] = "REMEDIATED"
    _write_execution_manifest(root, manifest)
    return manifest


def _raw_artifact_paths(root: Path) -> list[Path]:
    paths = [path for path in (root / "raw").rglob("*") if path.is_file()]
    paths.append(root / "execution_manifest.json")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _raw_artifact_entries(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _raw_artifact_paths(root)
    ]


def seal(root: Path, *, sealed_at_ns: int | None = None) -> dict[str, Any]:
    """Seal a complete raw snapshot with a deterministic artifact manifest."""
    root = root.resolve()
    manifest, contract = _load_prepared(root)
    artifact_manifest_path = root / "raw_artifact_manifest.json"
    passport_path = root / "material_passport.json"
    if artifact_manifest_path.exists() or passport_path.exists():
        raise FileExistsError("refusing to overwrite an existing raw seal")
    if manifest.get("state") != "RAW_COMPLETE":
        raise RuntimeError("raw matrix is incomplete and cannot be sealed")
    for job in manifest["jobs"]:
        if job.get("superseded_by") is not None:
            continue
        if job["state"] != "COMPLETED":
            raise RuntimeError(f"raw matrix job is incomplete: {job['id']}")
        _validate_runner_artifacts(root, job, contract)
    for remediation in manifest["remediations"]:
        for job in remediation["jobs"]:
            if job["state"] != "COMPLETED":
                raise RuntimeError("raw remediation is incomplete")
            _validate_runner_artifacts(root, job, contract)

    artifact_manifest = {
        "contract_id": contract["contract_id"],
        "stage": "raw",
        "artifacts": _raw_artifact_entries(root),
    }
    _write_json(artifact_manifest_path, artifact_manifest)
    passport = {
        "contract_id": contract["contract_id"],
        "contract_sha256": manifest["contract_sha256"],
        "capsule_manifest_sha256": manifest["capsule_manifest_sha256"],
        "environment_manifest_sha256": manifest["environment_manifest_sha256"],
        "raw_artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
        "stage": "raw",
        "sealed_at_ns": time.time_ns() if sealed_at_ns is None else sealed_at_ns,
    }
    required = set(contract["material_passport"]["required_fields"])
    if not required <= set(passport):
        raise RuntimeError("material passport is missing a required field")
    _write_json(passport_path, passport)
    validate_raw_seal(root)
    return passport


def validate_raw_seal(root: Path) -> bool:
    """Validate the passport, manifest hash, exact path set, bytes, and hashes."""
    root = root.resolve()
    execution_manifest, contract = _load_prepared(root)
    artifact_manifest_path = root / "raw_artifact_manifest.json"
    passport_path = root / "material_passport.json"
    if not artifact_manifest_path.is_file() or not passport_path.is_file():
        raise RuntimeError("raw seal is incomplete")
    passport = _read_json(passport_path)
    artifact_manifest = _read_json(artifact_manifest_path)
    required = set(contract["material_passport"]["required_fields"])
    if not required <= set(passport):
        raise RuntimeError("raw material passport is missing required fields")
    expected_bindings = {
        "contract_id": contract["contract_id"],
        "contract_sha256": execution_manifest["contract_sha256"],
        "capsule_manifest_sha256": execution_manifest["capsule_manifest_sha256"],
        "environment_manifest_sha256": execution_manifest["environment_manifest_sha256"],
        "raw_artifact_manifest_sha256": _sha256_file(artifact_manifest_path),
        "stage": "raw",
    }
    for field, expected in expected_bindings.items():
        if passport.get(field) != expected:
            raise RuntimeError(f"raw artifact seal binding mismatch: {field}")
    if artifact_manifest.get("contract_id") != contract["contract_id"]:
        raise RuntimeError("raw artifact manifest contract mismatch")
    observed_entries = artifact_manifest.get("artifacts")
    if not isinstance(observed_entries, list):
        raise RuntimeError("raw artifact manifest is malformed")
    actual_entries = _raw_artifact_entries(root)
    if observed_entries != actual_entries:
        raise RuntimeError("raw artifact path/byte/hash mismatch")
    return True


def status(root: Path) -> dict[str, Any]:
    """Return a read-only resume/seal summary, validating any existing seal."""
    root = root.resolve()
    manifest, _ = _load_prepared(root)
    all_jobs = list(manifest["jobs"])
    for remediation in manifest["remediations"]:
        all_jobs.extend(remediation["jobs"])
    counts = Counter(job["state"] for job in all_jobs)
    counts.setdefault("COMPLETED", 0)
    counts.setdefault("PENDING", 0)
    sealed = (root / "material_passport.json").exists()
    if sealed:
        validate_raw_seal(root)
    return {
        "contract_id": manifest["contract_id"],
        "protocol_id": manifest["protocol_id"],
        "state": manifest["state"],
        "jobs": dict(sorted(counts.items())),
        "remediation_count": len(manifest["remediations"]),
        "sealed": sealed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--active-conda-prefix", type=Path)
    for command in ("status", "preflight", "run", "seal"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, required=True)
    remediate_parser = subparsers.add_parser("remediate")
    remediate_parser.add_argument("--root", type=Path, required=True)
    remediate_parser.add_argument(
        "--pass-name", choices=["throughput", "trace"], required=True
    )
    remediate_parser.add_argument("--seed", type=int, choices=range(42, 48), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.repo_root,
            args.root,
            active_conda_prefix=args.active_conda_prefix,
        )
    elif args.command == "status":
        result = status(args.root)
    elif args.command == "preflight":
        result = preflight(args.root)
    elif args.command == "run":
        result = run(args.root)
    elif args.command == "remediate":
        result = remediate(args.root, pass_name=args.pass_name, seed=args.seed)
    else:
        result = seal(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
