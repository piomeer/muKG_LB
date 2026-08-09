"""Isolated, fail-closed executor for the frozen X8 C1-R1 protocol."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
import shutil
import stat
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


class PrepareCommandFailure(RuntimeError):
    """An external prepare command failed after its raw capture was retained."""

    def __init__(self, stage: str, capture: dict[str, Any]):
        super().__init__(
            f"external command failed ({capture['returncode']}): "
            f"{' '.join(capture['command'])}"
        )
        self.stage = stage
        self.capture = capture


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
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def _prepare_capture_command(
    root: Path,
    attempt: dict[str, Any],
    stage: str,
    transport: CommandTransport,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Capture every prepare subprocess before allowing a failure to escape."""
    result = transport(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    capture = {
        "stage": stage,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    attempt["command_captures"].append(capture)
    _write_json(root / "raw/prepare_attempt.json", attempt)
    if result.returncode:
        raise PrepareCommandFailure(stage, capture)
    return capture


def _decode_runtime_probe(capture: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        runtime = json.loads(capture["stdout"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} runtime probe was not valid JSON") from exc
    if not isinstance(runtime, dict):
        raise RuntimeError(f"{label} runtime probe must return a JSON object")
    return runtime


def _historical_prepare_lineage(roots: list[Path]) -> list[dict[str, Any]]:
    """Describe prior attempts without inventing missing raw captures."""
    entries: list[dict[str, Any]] = []
    for historic_root in sorted((path.resolve() for path in roots), key=str):
        capture_path = historic_root / "raw/prepare_attempt.json"
        if capture_path.is_file():
            entries.append(
                {
                    "root": str(historic_root),
                    "state": "RETAINED_CAPTURE",
                    "prepare_attempt_sha256": _sha256_file(capture_path),
                }
            )
        else:
            entries.append(
                {
                    "root": str(historic_root),
                    "state": "HISTORICAL_LINEAGE_INCOMPLETE",
                    "reason": "raw/prepare_attempt.json is absent; capture cannot be reconstructed",
                }
            )
    return entries


def _blocked_environment_closure(
    root: Path, attempt: dict[str, Any]
) -> dict[str, Any]:
    """Derive a deterministic blocked closure solely from retained prepare lineage."""
    if attempt.get("state") != "BLOCKED_ENVIRONMENT" or not isinstance(
        attempt.get("failure"), dict
    ):
        raise RuntimeError("blocked closure requires a retained failed prepare attempt")
    git_head = attempt.get("git_head")
    clone_probe = attempt.get("clone_prefix_probe")
    if not isinstance(git_head, dict) or not isinstance(clone_probe, dict):
        raise RuntimeError("blocked closure requires retained Git and clone probe lineage")
    return {
        "artifact_kind": "blocked_environment_closure",
        "attempt_id": attempt["attempt_id"],
        "capsule_manifest_sha256": attempt.get("capsule_manifest_sha256"),
        "contract_id": attempt["contract_id"],
        "contract_sha256": attempt["contract_sha256"],
        "executor_commit": git_head["stdout"].strip(),
        "failure": attempt["failure"],
        "historical_prepare_lineage": attempt["historical_prepare_lineage"],
        "material_passport": None,
        "prepare_attempt_path": "raw/prepare_attempt.json",
        "prepare_attempt_sha256": _sha256_file(root / "raw/prepare_attempt.json"),
        "prepared_manifest": False,
        "runtime_probe": clone_probe,
        "verdict": "BLOCKED_ENVIRONMENT",
    }


def regenerate_blocked_environment_closure(root: Path) -> dict[str, Any]:
    """Regenerate a blocked closure from retained raw prepare capture only."""
    root = root.resolve()
    attempt = _read_json(root / "raw/prepare_attempt.json")
    closure = _blocked_environment_closure(root, attempt)
    _write_json(root / "blocked_environment_closure.json", closure)
    return closure


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
                "evidence_scope": "diagnostic_only",
                "validates_gpu_exclusivity": False,
            }
        )
    return jobs


def _job_descriptor(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "kind", "pass_name", "config", "seed",
        "evidence_scope", "validates_gpu_exclusivity",
    )
    return {key: job[key] for key in keys if key in job}


def _validate_manifest_control_flow(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> None:
    pair_counts: dict[tuple[Any, Any], int] = {}
    for item in manifest.get("remediations", []):
        pair = (item.get("pass_name"), item.get("seed"))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    maximum = contract["retry_policy"]["maximum_retries_per_pass_seed_pair"]
    if any(count > maximum for count in pair_counts.values()):
        raise RuntimeError("execution manifest remediation retry-cap drift detected")
    initial_jobs = manifest.get("jobs", [])
    for item in manifest.get("remediations", []):
        pass_name = item.get("pass_name")
        seed = item.get("seed")
        if (
            pass_name not in contract["execution_matrix"]["primary"]["passes"]
            or seed not in contract["execution_matrix"]["primary"]["seeds"]
            or item.get("attempt") != 1
            or item.get("dispatch") != 1
            or item.get("state") not in {"RUNNING", "COMPLETED", "INCOMPLETE"}
        ):
            raise RuntimeError("execution manifest remediation control-flow drift detected")
        order = contract["protocol"]["paired_order"][str(seed)]
        expected_retry_descriptors = [
            {
                "id": f"{pass_name}_{config}_seed{seed}_retry1",
                "kind": "job",
                "pass_name": pass_name,
                "config": config,
                "seed": seed,
            }
            for config in order
        ]
        actual_retry_descriptors = [
            _job_descriptor(job) for job in item.get("jobs", [])
        ]
        if actual_retry_descriptors != expected_retry_descriptors:
            raise RuntimeError("execution manifest remediation job/order drift detected")
        incomplete = item["state"] == "INCOMPLETE"
        if item.get("analysis_eligible") is incomplete:
            raise RuntimeError("execution manifest remediation eligibility drift detected")
        for retry_job in item["jobs"]:
            if (
                retry_job.get("attempt") != 1
                or retry_job.get("dispatch") != 1
                or retry_job.get("analysis_eligible") is incomplete
            ):
                raise RuntimeError("execution manifest retry control-flow drift detected")
        if item["state"] == "COMPLETED" and any(
            job.get("state") != "COMPLETED" for job in item["jobs"]
        ):
            raise RuntimeError("execution manifest completed remediation drift detected")
        pair_jobs = [
            job for job in initial_jobs
            if job.get("kind") == "job"
            and job.get("pass_name") == pass_name
            and job.get("seed") == seed
        ]
        if len(pair_jobs) != 2 or not any(job.get("state") == "INVALID" for job in pair_jobs):
            raise RuntimeError("execution manifest remediation trigger drift detected")
        trigger_reasons = set(item.get("trigger_reasons", []))
        eligible_reasons = set(contract["retry_policy"]["eligible_failure_reasons"])
        if not trigger_reasons or not trigger_reasons <= eligible_reasons:
            raise RuntimeError("execution manifest remediation reason drift detected")
    remediated_pairs = {
        (item.get("pass_name"), item.get("seed")): item
        for item in manifest.get("remediations", [])
    }
    for job in manifest.get("jobs", []):
        pair = (job.get("pass_name"), job.get("seed"))
        remediation = remediated_pairs.get(pair) if job.get("kind") == "job" else None
        if remediation is None:
            if "superseded_by" in job or "analysis_eligible" in job:
                raise RuntimeError("execution manifest control-flow field drift detected")
            continue
        if (
            job.get("superseded_by") != remediation.get("attempt")
            or job.get("analysis_eligible") is not False
        ):
            raise RuntimeError("execution manifest control-flow remediation drift detected")


def _environment_clone_entries(cloned_prefix: Path) -> list[dict[str, Any]]:
    def mutable_cache(path: Path) -> bool:
        relative = path.relative_to(cloned_prefix)
        parts = set(relative.parts)
        return (
            "__pycache__" in parts
            or ".cache" in parts
            or relative.parts[:2] in {("var", "cache"), ("pkgs", "cache")}
            or path.suffix in {".pyc", ".pyo", ".lock"}
        )

    def identity_file(path: Path) -> bool:
        try:
            mode = path.lstat().st_mode
        except OSError:
            return False
        return stat.S_ISREG(mode) or stat.S_ISLNK(mode)

    paths = sorted(
        (
            path
            for path in cloned_prefix.rglob("*")
            if identity_file(path) and not mutable_cache(path)
        ),
        key=lambda path: path.relative_to(cloned_prefix).as_posix(),
    )
    entries = []
    for path in paths:
        if path.is_symlink():
            target = os.readlink(path)
            entry = {
                "path": path.relative_to(cloned_prefix).as_posix(),
                "bytes": path.lstat().st_size,
                "sha256": hashlib.sha256(
                    b"symlink\0" + target.encode("utf-8")
                ).hexdigest(),
                "symlink_target": target,
            }
        else:
            entry = {
                "path": path.relative_to(cloned_prefix).as_posix(),
                "bytes": path.lstat().st_size,
                "sha256": _sha256_file(path),
            }
        entries.append(entry)
    return entries


def prepare(
    repo_root: Path,
    root: Path,
    *,
    transport: CommandTransport = subprocess.run,
    active_conda_prefix: Path | None = None,
    historical_incomplete_roots: list[Path] | None = None,
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
    attempt = {
        "artifact_kind": "prepare_attempt",
        "attempt_id": f"prepare-{root.name}-{time.time_ns()}",
        "started_at_ns": time.time_ns(),
        "state": "RUNNING",
        "repo_root": str(repo_root),
        "root": str(root),
        "active_conda_prefix": str(active_conda_prefix),
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
        "historical_prepare_lineage": _historical_prepare_lineage(
            historical_incomplete_roots or []
        ),
        "command_captures": [],
    }
    _write_json(root / "raw/prepare_attempt.json", attempt)
    failure_stage = "capsule"
    try:
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
        attempt["capsule_manifest_sha256"] = _sha256_file(capsule_manifest_path)

        frozen_contract_path = root / "frozen_contract.json"
        shutil.copyfile(contract_path, frozen_contract_path)
        if _sha256_file(frozen_contract_path) != contract_sha256:
            raise RuntimeError("frozen contract copy hash mismatch")

        offline_env = _offline_environment()
        cloned_prefix = root / "environment/conda"
        failure_stage = "git_head"
        git_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
            transport,
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            cwd=repo_root,
            env=offline_env,
        )
        attempt["git_head"] = git_capture
        failure_stage = "active_prefix_probe"
        active_probe_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
            transport,
            [str(active_conda_prefix / "bin/python"), "-c", _runtime_probe_code()],
            cwd=repo_root,
            env=offline_env,
        )
        attempt["active_prefix_probe"] = _decode_runtime_probe(
            active_probe_capture, "active-prefix"
        )
        _write_json(root / "raw/prepare_attempt.json", attempt)

        failure_stage = "conda_clone"
        clone_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
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

        failure_stage = "conda_list"
        conda_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
            transport,
            ["conda", "list", "--json", "--prefix", str(cloned_prefix)],
            cwd=repo_root,
            env=offline_env,
        )
        try:
            packages = json.loads(conda_capture["stdout"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Conda package manifest was not valid JSON") from exc
        failure_stage = "clone_prefix_probe"
        runtime_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
            transport,
            [str(cloned_prefix / "bin/python"), "-c", _runtime_probe_code()],
            cwd=capsule,
            env=offline_env,
        )
        runtime = _decode_runtime_probe(runtime_capture, "cloned")
        attempt["clone_prefix_probe"] = runtime
        _write_json(root / "raw/prepare_attempt.json", attempt)
        failure_stage = "gpu_identity"
        gpu_capture = _prepare_capture_command(
            root,
            attempt,
            failure_stage,
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
    except Exception as exc:
        attempt["state"] = "BLOCKED_ENVIRONMENT"
        attempt["failure"] = {
            "stage": failure_stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(root / "raw/prepare_attempt.json", attempt)
        regenerate_blocked_environment_closure(root)
        raise

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
    attempt["state"] = "PREPARED"
    attempt["environment_manifest_sha256"] = _sha256_file(environment_manifest_path)
    _write_json(root / "raw/prepare_attempt.json", attempt)
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
    expected_descriptors = [
        _job_descriptor(job) for job in _build_execution_jobs(contract)
    ]
    actual_descriptors = [_job_descriptor(job) for job in manifest.get("jobs", [])]
    if actual_descriptors != expected_descriptors:
        raise RuntimeError("execution manifest job descriptor/order drift detected")
    _validate_manifest_control_flow(manifest, contract)
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
    actual_capsule_paths = {
        path.relative_to(root / "capsule").as_posix()
        for path in (root / "capsule").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_capsule_paths != set(expected_entries):
        raise RuntimeError("capsule file-set drift detected")
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
    dispatch_suffix = (
        f"_dispatch{job['dispatch']}" if job.get("dispatch", 1) > 1 else ""
    )
    if job["kind"] == "preflight":
        return root / "raw/attempts" / f"preflight_attempt{attempt}"
    if job["kind"] == "job":
        return (
            root
            / "raw/attempts"
            / f"{job['pass_name']}_seed{job['seed']}_attempt{attempt}{dispatch_suffix}"
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
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _runtime_probe_code() -> str:
    return (
        "import json,sys,torch; print(json.dumps({"
        "'protocol_id':'C1-R1-v1.1','python':sys.version,"
        "'pytorch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available()}))"
    )


def _validate_live_environment(
    root: Path,
    contract: dict[str, Any],
    transport: CommandTransport,
) -> None:
    expected = _read_json(root / "environment_manifest.json")
    cloned_prefix = root / "environment/conda"
    offline_env = _offline_environment()
    package_capture = _command_capture(
        transport,
        ["conda", "list", "--json", "--prefix", str(cloned_prefix)],
        cwd=root / "capsule",
        env=offline_env,
    )
    runtime_capture = _command_capture(
        transport,
        [str(cloned_prefix / "bin/python"), "-c", _runtime_probe_code()],
        cwd=root / "capsule",
        env=offline_env,
    )
    gpu_capture = _command_capture(
        transport,
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=root / "capsule",
        env=offline_env,
    )
    try:
        packages = json.loads(package_capture["stdout"])
        runtime = json.loads(runtime_capture["stdout"])
        identity = [part.strip() for part in gpu_capture["stdout"].strip().split(",")]
        gpu = {
            "name": identity[0],
            "total_memory_mib": float(identity[1]),
            "compute_capability": identity[2],
        }
    except (json.JSONDecodeError, IndexError, ValueError) as exc:
        raise RuntimeError("live environment probe is malformed") from exc
    if (
        packages != expected.get("packages")
        or runtime != expected.get("runtime")
        or gpu != expected.get("gpu")
        or runtime.get("protocol_id") != contract["protocol"]["protocol_id"]
    ):
        raise RuntimeError("live environment identity drift detected")


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


def _read_csv_rows(path: Path, required_columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing raw CSV: {path.name}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise RuntimeError(f"{path.name} schema missing columns: {missing}")
        return list(reader)


def _integer(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{path.name} invalid integer {field}") from exc


def _true(value: str) -> bool:
    return value.strip().lower() in {"true", "1"}


def _validate_primary_csvs(
    job_dir: Path,
    job: dict[str, Any],
    contract: dict[str, Any],
    status: dict[str, Any],
) -> None:
    schemas = contract["raw_artifact_schemas"]
    protocol = contract["protocol"]
    protocol_id = protocol["protocol_id"]
    epochs = int(protocol["epochs_per_job"])
    batch_size = int(protocol["batch_size"])
    examples = int(protocol["training_examples"])
    full_batches, partial_size = divmod(examples, batch_size)
    steps_per_epoch = full_batches + (1 if partial_size else 0)

    epoch_path = job_dir / "per_epoch.csv"
    epoch_rows = _read_csv_rows(
        epoch_path, schemas["per_epoch.csv"]["required_columns"]
    )
    if len(epoch_rows) != epochs:
        raise RuntimeError(
            f"per_epoch.csv row count mismatch: expected {epochs}, got {len(epoch_rows)}"
        )
    for expected_epoch, row in enumerate(epoch_rows):
        identity = (
            row.get("protocol_id"),
            row.get("pass_name"),
            row.get("config"),
            _integer(row, "seed", epoch_path),
        )
        if identity != (
            protocol_id,
            job["pass_name"],
            job["config"],
            job["seed"],
        ):
            raise RuntimeError("per_epoch.csv protocol/job identity mismatch")
        invariants = {
            "epoch": (expected_epoch, _integer(row, "epoch", epoch_path)),
            "num_steps": (steps_per_epoch, _integer(row, "num_steps", epoch_path)),
            "full_batch_count": (
                full_batches,
                _integer(row, "full_batch_count", epoch_path),
            ),
            "partial_batch_count": (
                1 if partial_size else 0,
                _integer(row, "partial_batch_count", epoch_path),
            ),
            "partial_batch_size": (
                partial_size,
                _integer(row, "partial_batch_size", epoch_path),
            ),
            "training_examples": (examples, _integer(row, "training_examples", epoch_path)),
        }
        for field, (expected, observed) in invariants.items():
            if observed != expected:
                raise RuntimeError(f"per_epoch.csv invariant mismatch: {field}")
        if _integer(row, "epoch_time_ns", epoch_path) <= 0 or not _true(
            row.get("loss_finite", "")
        ):
            raise RuntimeError("per_epoch.csv timing/loss invariant mismatch")
    expected_step_rows = epochs * steps_per_epoch if job["pass_name"] == "trace" else 0
    status_counts = status.get("row_counts", {})
    if status_counts != {"epochs": epochs, "steps": expected_step_rows}:
        raise RuntimeError("status.json row_counts do not match raw CSV contract")
    if job["pass_name"] != "trace":
        return

    step_path = job_dir / "per_step.csv"
    step_rows = _read_csv_rows(
        step_path, schemas["per_step.csv"]["required_columns"]
    )
    if len(step_rows) != expected_step_rows:
        raise RuntimeError(
            f"per_step.csv row count mismatch: expected {expected_step_rows}, got {len(step_rows)}"
        )
    for index, row in enumerate(step_rows):
        epoch, step = divmod(index, steps_per_epoch)
        identity = (
            row.get("protocol_id"),
            row.get("pass_name"),
            row.get("config"),
            _integer(row, "seed", step_path),
            _integer(row, "epoch", step_path),
            _integer(row, "step", step_path),
        )
        if identity != (
            protocol_id,
            job["pass_name"],
            job["config"],
            job["seed"],
            epoch,
            step,
        ):
            raise RuntimeError("per_step.csv protocol/job/grid identity mismatch")
        partial = step == full_batches
        expected_size = partial_size if partial else batch_size
        if (
            _integer(row, "batch_size_actual", step_path) != expected_size
            or _true(row.get("is_partial", "")) != partial
            or _true(row.get("is_first_measured_step", "")) != (index == 0)
        ):
            raise RuntimeError("per_step.csv full/partial batch invariant mismatch")
        neg = _integer(row, "neg_time_ns", step_path)
        component = _integer(row, "component_sum_ns", step_path)
        total = _integer(row, "total_step_ns", step_path)
        residual = _integer(row, "timing_residual_ns", step_path)
        if neg <= 0 or component < 0 or total <= 0 or component + residual != total:
            raise RuntimeError("per_step.csv numeric timing invariant mismatch")


def _validate_compute_only_csv(
    path: Path, job: dict[str, Any], contract: dict[str, Any]
) -> None:
    required = [
        "protocol_id", "seed", "repeat", "elapsed_ns", "loss", "batch_size", "neg_num"
    ]
    rows = _read_csv_rows(path, required)
    observations = int(contract["execution_matrix"]["diagnostic"]["observations_per_seed"])
    if len(rows) != observations:
        raise RuntimeError(
            f"compute-only row count mismatch: expected {observations}, got {len(rows)}"
        )
    protocol = contract["protocol"]
    for repeat, row in enumerate(rows):
        try:
            loss = float(row["loss"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("compute-only loss is not numeric") from exc
        if (
            row.get("protocol_id") != protocol["protocol_id"]
            or _integer(row, "seed", path) != job["seed"]
            or _integer(row, "repeat", path) != repeat
            or _integer(row, "elapsed_ns", path) <= 0
            or not math.isfinite(loss)
            or _integer(row, "batch_size", path) != protocol["batch_size"]
            or _integer(row, "neg_num", path) != protocol["negative_samples_per_positive"]
        ):
            raise RuntimeError("compute-only diagnostic identity/numeric invariant mismatch")


def _compute_telemetry_path(root: Path, job: dict[str, Any]) -> Path:
    return (
        _job_output_root(root, job)
        / "compute_only"
        / f"seed{job['seed']}.gpu_telemetry.csv"
    )


def _capture_compute_telemetry(
    root: Path,
    job: dict[str, Any],
    contract: dict[str, Any],
    event: str,
    transport: CommandTransport,
) -> str | None:
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,clocks_throttle_reasons.sw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    offline_env = _offline_environment()
    captures = []
    for command in (gpu_command, process_command):
        try:
            result = transport(
                command,
                cwd=str(root / "capsule"),
                env=offline_env,
                capture_output=True,
                text=True,
            )
            captures.append(result)
        except Exception as exc:
            captures.append(subprocess.CompletedProcess(command, 1, "", repr(exc)))
    gpu_result, process_result = captures
    raw_gpu = gpu_result.stdout.strip()
    raw_process = process_result.stdout.strip()
    query_errors = []
    if gpu_result.returncode:
        query_errors.append(f"gpu_query:{gpu_result.stderr.strip()}")
    if process_result.returncode:
        query_errors.append(f"process_query:{process_result.stderr.strip()}")
    other_processes = []
    for line in raw_process.splitlines():
        try:
            pid = int(line.split(",", 1)[0].strip())
        except ValueError:
            continue
        if pid != os.getpid():
            other_processes.append(line.strip())
    thermal = raw_gpu.rsplit(",", 1)[-1].strip() if "," in raw_gpu else ""
    row = {
        "protocol_id": contract["protocol"]["protocol_id"],
        "config": "GPU",
        "seed": job["seed"],
        "pass_name": "compute_only",
        "event": event,
        "time_ns": time.time_ns(),
        "thermal_slowdown": thermal,
        "other_compute_processes": " | ".join(other_processes),
        "raw_gpu_query": raw_gpu.replace("\n", " | "),
        "raw_process_query": raw_process.replace("\n", " | "),
        "query_error": ";".join(query_errors),
    }
    path = _compute_telemetry_path(root, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = contract["raw_artifact_schemas"]["gpu_telemetry.csv"]["required_columns"]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    if row["query_error"]:
        return "telemetry_failure"
    if row["other_compute_processes"]:
        return "other_compute_processes"
    if thermal.lower() not in {"", "0", "false", "no", "none", "not active", "disabled"}:
        return "thermal_slowdown"
    return None


def _validate_compute_telemetry(
    root: Path, job: dict[str, Any], contract: dict[str, Any]
) -> None:
    path = _compute_telemetry_path(root, job)
    fields = contract["raw_artifact_schemas"]["gpu_telemetry.csv"]["required_columns"]
    rows = _read_csv_rows(path, fields)
    if [row.get("event") for row in rows] != ["before_job", "after_job"]:
        raise RuntimeError("compute-only wrapper telemetry event/count mismatch")
    for row in rows:
        if (
            row.get("protocol_id") != contract["protocol"]["protocol_id"]
            or row.get("config") != "GPU"
            or row.get("pass_name") != "compute_only"
            or _integer(row, "seed", path) != job["seed"]
            or _integer(row, "time_ns", path) <= 0
        ):
            raise RuntimeError("compute-only wrapper telemetry identity mismatch")
    reason = _telemetry_invalid_reason(path, contract["protocol"]["protocol_id"])
    if reason:
        raise InvalidAttempt(reason, "compute-only wrapper telemetry is invalid")


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
        telemetry_path = output_root / "preflight/gpu_telemetry.csv"
        _read_csv_rows(
            telemetry_path,
            contract["raw_artifact_schemas"]["gpu_telemetry.csv"]["required_columns"],
        )
        reason = _telemetry_invalid_reason(
            telemetry_path, protocol_id
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
        _validate_primary_csvs(job_dir, job, contract, status)
        telemetry_rows = _read_csv_rows(
            job_dir / "gpu_telemetry.csv",
            contract["raw_artifact_schemas"]["gpu_telemetry.csv"]["required_columns"],
        )
        for row in telemetry_rows:
            if (
                row.get("protocol_id") != protocol_id
                or row.get("pass_name") != job["pass_name"]
                or row.get("config") != job["config"]
                or _integer(row, "seed", job_dir / "gpu_telemetry.csv") != job["seed"]
            ):
                raise RuntimeError("gpu_telemetry.csv protocol/job identity mismatch")
        reason = _telemetry_invalid_reason(job_dir / "gpu_telemetry.csv", protocol_id)
        if reason:
            raise InvalidAttempt(reason, f"telemetry invalid for {job['id']}")
        return
    diagnostic = output_root / "compute_only" / f"seed{job['seed']}.csv"
    _validate_compute_only_csv(diagnostic, job, contract)
    _validate_compute_telemetry(root, job, contract)


def _execute_job(
    root: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    job: dict[str, Any],
    transport: CommandTransport,
) -> None:
    if job["state"] == "COMPLETED":
        _load_prepared(root)
        _validate_runner_artifacts(root, job, contract)
        return
    if job["state"] == "RUNNING":
        job["state"] = "INVALID"
        job["invalid_reason"] = "infrastructure_failure"
        _write_execution_manifest(root, manifest)
        raise RuntimeError(
            f"interrupted RUNNING job requires paired remediation: {job['id']}"
        )
    if job["state"] != "PENDING":
        raise RuntimeError(f"job {job['id']} requires remediation: {job['state']}")
    _load_prepared(root)
    _validate_live_environment(root, contract, transport)
    creation_target = _job_creation_target(root, job)
    if creation_target.exists():
        raise FileExistsError(f"refusing to overwrite runner output: {creation_target}")
    job["state"] = "RUNNING"
    job["command"] = _runner_command(root, job)
    job["started_at_ns"] = time.time_ns()
    _write_execution_manifest(root, manifest)
    if job["kind"] == "compute-only":
        reason = _capture_compute_telemetry(
            root, job, contract, "before_job", transport
        )
        if reason:
            job["state"] = "INVALID"
            job["invalid_reason"] = reason
            _write_execution_manifest(root, manifest)
            raise InvalidAttempt(reason, f"compute-only pre-dispatch telemetry for {job['id']}")
    result = transport(
        job["command"],
        cwd=str(root / "capsule"),
        env={**_offline_environment(), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
    )
    job["ended_at_ns"] = time.time_ns()
    if job["kind"] == "compute-only":
        reason = _capture_compute_telemetry(
            root, job, contract, "after_job", transport
        )
        if reason:
            job["state"] = "INVALID"
            job["invalid_reason"] = reason
            _write_execution_manifest(root, manifest)
            raise InvalidAttempt(reason, f"compute-only post-dispatch telemetry for {job['id']}")
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
    _validate_runner_artifacts(root, manifest["jobs"][0], contract)
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
    if existing:
        item = existing[0]
        if item.get("state") == "COMPLETED" or all(
            job.get("state") == "COMPLETED" for job in item["jobs"]
        ):
            raise RuntimeError("maximum remediation retries reached for pass/seed pair")
        item["state"] = "INCOMPLETE"
        item["analysis_eligible"] = False
        for retry_job in item["jobs"]:
            retry_job["analysis_eligible"] = False
            if retry_job["state"] == "RUNNING":
                retry_job["state"] = "INVALID"
                retry_job["invalid_reason"] = "infrastructure_failure"
            elif retry_job["state"] == "PENDING":
                retry_job["state"] = "SKIPPED_INCOMPLETE_REMEDIATION"
        manifest["state"] = "INCOMPLETE"
        _write_execution_manifest(root, manifest)
        raise RuntimeError(
            "remediation attempt is INCOMPLETE; maximum physical retry exhausted"
        )
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

    attempt = 1
    dispatch = 1
    retry_jobs = []
    for config in order:
        retry_jobs.append(
            {
                "id": (
                    f"{pass_name}_{config}_seed{seed}_retry{attempt}"
                    + (f"_dispatch{dispatch}" if dispatch > 1 else "")
                ),
                "kind": "job",
                "pass_name": pass_name,
                "config": config,
                "seed": seed,
                "attempt": attempt,
                "dispatch": dispatch,
                "state": "PENDING",
                "analysis_eligible": True,
            }
        )
    remediation_record = {
        "pass_name": pass_name,
        "seed": seed,
        "attempt": attempt,
        "dispatch": dispatch,
        "state": "RUNNING",
        "analysis_eligible": True,
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
    remediation_record["state"] = "COMPLETED"
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


def seal(root: Path) -> dict[str, Any]:
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
        "sealed_at_ns": time.time_ns(),
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
    prepare_parser.add_argument(
        "--historical-incomplete-root",
        type=Path,
        action="append",
        default=[],
        help="retain a prior root as incomplete lineage when raw prepare capture is absent",
    )
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
            historical_incomplete_roots=args.historical_incomplete_root,
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
