"""Public X8 clean-room audit contract interface.

Independent analysis and comparison are intentionally added in the audit task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


CONTRACT_PATH = Path("output/results/evidence_audit_x8_c1_r1/clean_room_contract.json")
INDEPENDENT_DIR = Path("derived/independent")
INDEPENDENT_MANIFEST = Path("independent_artifact_manifest.json")
INDEPENDENT_PASSPORT = Path("independent_material_passport.json")
INDEPENDENT_OUTPUT_NAMES = (
    "checks.json",
    "leave_one_seed_out.csv",
    "seed_level_metrics.csv",
    "summary.json",
)
FROZEN_ORIGINAL_ESTIMATES = {
    "E1": 6.013389739959145,
    "E2": 87.87705683218147,
    "E3": 3_002_619.603144654,
}


def load_contract(repo_root: Path) -> dict[str, Any]:
    """Load the frozen X8 contract for a future audit invocation."""
    with (repo_root / CONTRACT_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)


def geometric_summary(
    values: Iterable[float], *, t95: float, t97_5: float
) -> dict[str, Any]:
    materialized = list(values)
    if len(materialized) < 2 or any(value <= 0 for value in materialized):
        raise ValueError("geometric summary requires at least two positive values")
    logs = [math.log(value) for value in materialized]
    center = statistics.fmean(logs)
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "estimate": math.exp(center),
        "ci95": {
            "low": math.exp(center - t95 * standard_error),
            "high": math.exp(center + t95 * standard_error),
        },
        "ci97_5_bonferroni": {
            "low": math.exp(center - t97_5 * standard_error),
            "high": math.exp(center + t97_5 * standard_error),
        },
    }


def arithmetic_summary(values: Iterable[float], *, t95: float) -> dict[str, Any]:
    materialized = list(values)
    if len(materialized) < 2:
        raise ValueError("arithmetic summary requires at least two values")
    center = statistics.fmean(materialized)
    sample_sd = statistics.stdev(materialized)
    margin = t95 * sample_sd / math.sqrt(len(materialized))
    return {
        "estimate": center,
        "sample_sd": sample_sd,
        "ci95": {"low": center - margin, "high": center + margin},
    }


def population_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("population SD requires observations")
    center = statistics.fmean(materialized)
    return math.sqrt(
        statistics.fmean((value - center) ** 2 for value in materialized)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(
    path: Path, fields: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = [field for field in required if field not in fields]
            if missing:
                raise RuntimeError(f"{path.name} schema missing columns: {missing}")
            return list(reader)
    except OSError as exc:
        raise RuntimeError(f"missing raw CSV: {path}") from exc


def _require_fields(
    document: dict[str, Any], required: Iterable[str], *, label: str
) -> None:
    missing = [field for field in required if field not in document]
    if missing:
        raise RuntimeError(f"{label} schema missing fields: {missing}")


def _integer(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{path.name} invalid integer {field}") from exc


def _boolean(value: str, *, path: Path, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RuntimeError(f"{path.name} invalid boolean {field}")


def _artifact_entries(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]


def _raw_artifact_paths(root: Path) -> list[Path]:
    paths = [path for path in (root / "raw").rglob("*") if path.is_file()]
    paths.append(root / "execution_manifest.json")
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _load_frozen(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(root / "frozen_contract.json")
    manifest = _read_json(root / "execution_manifest.json")
    if contract.get("status") != "FROZEN":
        raise RuntimeError("clean-room contract is not FROZEN")
    if contract.get("contract_id") != "X8-C1-R1-clean-room-v1":
        raise RuntimeError("unexpected clean-room contract id")
    if contract.get("analysis_controls", {}).get("pooling_forbidden") is not True:
        raise RuntimeError("clean-room contract must forbid pooling")
    if manifest.get("contract_id") != contract["contract_id"]:
        raise RuntimeError("execution manifest contract mismatch")
    if manifest.get("protocol_id") != contract["protocol"]["protocol_id"]:
        raise RuntimeError("execution manifest protocol mismatch")
    return manifest, contract


def validate_raw_seal(root: Path) -> bool:
    """Independently validate the complete raw seal and its lineage bindings."""
    root = root.resolve()
    manifest, contract = _load_frozen(root)
    raw_manifest_path = root / "raw_artifact_manifest.json"
    passport_path = root / "material_passport.json"
    raw_manifest = _read_json(raw_manifest_path)
    passport = _read_json(passport_path)
    required = contract["material_passport"]["required_fields"]
    _require_fields(passport, required, label="raw material passport")
    if not isinstance(passport.get("sealed_at_ns"), int) or passport["sealed_at_ns"] <= 0:
        raise RuntimeError("raw material passport invalid sealed_at_ns")
    bindings = {
        "contract_id": contract["contract_id"],
        "contract_sha256": _sha256_file(root / "frozen_contract.json"),
        "capsule_manifest_sha256": _sha256_file(root / "capsule_manifest.json"),
        "environment_manifest_sha256": _sha256_file(root / "environment_manifest.json"),
        "raw_artifact_manifest_sha256": _sha256_file(raw_manifest_path),
        "stage": "raw",
    }
    for field, expected in bindings.items():
        if passport.get(field) != expected or manifest.get(field) not in {None, expected}:
            raise RuntimeError(f"raw artifact seal binding mismatch: {field}")
    if (
        raw_manifest.get("contract_id") != contract["contract_id"]
        or raw_manifest.get("stage") != "raw"
    ):
        raise RuntimeError("raw artifact manifest identity mismatch")
    actual = _artifact_entries(root, _raw_artifact_paths(root))
    if raw_manifest.get("artifacts") != actual:
        raise RuntimeError("raw artifact path/byte/hash mismatch")
    return True


def _expected_initial_ids(contract: dict[str, Any]) -> list[str]:
    ids = ["preflight"]
    primary = contract["execution_matrix"]["primary"]
    for seed in sorted(primary["seeds"]):
        for pass_name in primary["passes"]:
            for config in contract["protocol"]["paired_order"][str(seed)]:
                ids.append(f"{pass_name}_{config}_seed{seed}")
    for seed in sorted(contract["execution_matrix"].get("diagnostic", {}).get("seeds", [])):
        ids.append(f"compute_only_seed{seed}")
    return ids


def _select_analysis_jobs(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if manifest.get("state") != "RAW_COMPLETE":
        raise RuntimeError("raw execution is incomplete")
    initial = manifest.get("jobs")
    if not isinstance(initial, list) or [job.get("id") for job in initial] != _expected_initial_ids(contract):
        raise RuntimeError("execution manifest job descriptor/order drift detected")
    if initial[0].get("kind") != "preflight" or initial[0].get("state") != "COMPLETED":
        raise RuntimeError("preflight is incomplete")
    primary = contract["execution_matrix"]["primary"]
    initial_jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for job in initial[1:]:
        if job.get("kind") != "job":
            continue
        key = (job.get("pass_name"), job.get("config"), job.get("seed"))
        if key in initial_jobs:
            raise RuntimeError("duplicate initial primary attempt")
        initial_jobs[key] = job
    expected_keys = {
        (pass_name, config, seed)
        for seed in primary["seeds"]
        for pass_name in primary["passes"]
        for config in primary["configs"]
    }
    if set(initial_jobs) != expected_keys:
        raise RuntimeError("execution manifest primary matrix drift detected")

    remediations = manifest.get("remediations", [])
    if not isinstance(remediations, list):
        raise RuntimeError("execution manifest remediations are malformed")
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for remediation in remediations:
        pair = (remediation.get("pass_name"), remediation.get("seed"))
        if pair in by_pair:
            raise RuntimeError("more than one remediation for a pass/seed pair")
        if pair[0] not in primary["passes"] or pair[1] not in primary["seeds"]:
            raise RuntimeError("remediation lies outside the frozen matrix")
        if (
            remediation.get("attempt") != 1
            or remediation.get("dispatch") != 1
            or remediation.get("state") != "COMPLETED"
            or remediation.get("analysis_eligible") is not True
        ):
            raise RuntimeError("remediation is not a completed predeclared attempt")
        reasons = remediation.get("trigger_reasons")
        eligible = set(contract["retry_policy"]["eligible_failure_reasons"])
        if not isinstance(reasons, list) or not reasons or not set(reasons) <= eligible:
            raise RuntimeError("remediation trigger reason is not eligible")
        by_pair[pair] = remediation

    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for seed in primary["seeds"]:
        for pass_name in primary["passes"]:
            pair = (pass_name, seed)
            order = contract["protocol"]["paired_order"][str(seed)]
            remediation = by_pair.get(pair)
            if remediation is None:
                for config in order:
                    job = initial_jobs[(pass_name, config, seed)]
                    if (
                        job.get("attempt") != 0
                        or job.get("state") != "COMPLETED"
                        or "analysis_eligible" in job
                        or "superseded_by" in job
                    ):
                        raise RuntimeError("initial attempt is not analysis eligible")
                    selected[(pass_name, config, seed)] = job
                continue
            retry_jobs = remediation.get("jobs")
            if not isinstance(retry_jobs, list) or [job.get("config") for job in retry_jobs] != order:
                raise RuntimeError("remediation pair/order drift detected")
            for config in order:
                initial_job = initial_jobs[(pass_name, config, seed)]
                if (
                    initial_job.get("analysis_eligible") is not False
                    or initial_job.get("superseded_by") != 1
                ):
                    raise RuntimeError("superseded attempt eligibility drift detected")
            invalid_jobs = [
                initial_jobs[(pass_name, config, seed)]
                for config in order
                if initial_jobs[(pass_name, config, seed)].get("state") == "INVALID"
            ]
            if not invalid_jobs:
                raise RuntimeError("remediation lacks an explicitly invalid initial attempt")
            actual_reasons: set[str] = set()
            eligible_reasons = set(contract["retry_policy"]["eligible_failure_reasons"])
            for invalid_job in invalid_jobs:
                reasons: list[Any] = []
                if "invalid_reason" in invalid_job:
                    reasons.append(invalid_job["invalid_reason"])
                if "invalid_reasons" in invalid_job:
                    listed = invalid_job["invalid_reasons"]
                    if not isinstance(listed, list):
                        raise RuntimeError("invalid reason lineage is malformed")
                    reasons.extend(listed)
                if not reasons or any(
                    not isinstance(reason, str) or not reason.strip()
                    for reason in reasons
                ):
                    raise RuntimeError("invalid reason lineage is missing or malformed")
                normalized = {reason.strip() for reason in reasons}
                if not normalized <= eligible_reasons:
                    raise RuntimeError("invalid reason is not retry eligible")
                actual_reasons.update(normalized)
            if remediation["trigger_reasons"] != sorted(actual_reasons):
                raise RuntimeError("trigger reasons do not match actual invalid reasons")
            for config, retry_job in zip(order, retry_jobs):
                expected_id = f"{pass_name}_{config}_seed{seed}_retry1"
                if (
                    retry_job.get("id") != expected_id
                    or retry_job.get("kind") != "job"
                    or retry_job.get("pass_name") != pass_name
                    or retry_job.get("seed") != seed
                    or retry_job.get("attempt") != 1
                    or retry_job.get("dispatch") != 1
                    or retry_job.get("state") != "COMPLETED"
                    or retry_job.get("analysis_eligible") is not True
                ):
                    raise RuntimeError("retry attempt control-flow drift detected")
                selected[(pass_name, config, seed)] = retry_job
    return selected


def _job_dir(root: Path, job: dict[str, Any]) -> Path:
    suffix = f"_dispatch{job['dispatch']}" if job.get("dispatch", 1) > 1 else ""
    attempt_root = (
        root / "raw/attempts"
        / f"{job['pass_name']}_seed{job['seed']}_attempt{job.get('attempt', 0)}{suffix}"
    )
    return attempt_root / "jobs" / f"{job['pass_name']}_{job['config']}_seed{job['seed']}"


def _validate_lineage_schemas(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    schemas = contract["raw_artifact_schemas"]
    environment = _read_json(root / "raw/environment.json")
    _require_fields(
        environment, schemas["environment.json"]["required_fields"], label="environment.json"
    )
    if environment.get("protocol_id") != contract["protocol"]["protocol_id"]:
        raise RuntimeError("environment protocol mismatch")
    preflight_root = root / "raw/attempts/preflight_attempt0/preflight"
    preflight = _read_json(preflight_root / "result.json")
    _require_fields(
        preflight,
        schemas["preflight/result.json"]["required_fields"],
        label="preflight/result.json",
    )
    if (
        preflight.get("protocol_id") != contract["protocol"]["protocol_id"]
        or preflight.get("all_passed") is not True
    ):
        raise RuntimeError("preflight did not pass the frozen protocol")
    telemetry = _read_csv(
        preflight_root / "gpu_telemetry.csv",
        schemas["gpu_telemetry.csv"]["required_columns"],
    )
    if not telemetry:
        raise RuntimeError("preflight telemetry is empty")
    split = preflight.get("split")
    split_fields = (
        "declared_triples",
        "raw_triples",
        "held_out_size",
        "training_set_size",
        "split_seed",
        "split_algorithm",
        "source_path",
        "source_sha256",
        "raw_order_sha256",
        "file_order_sha256",
        "held_out_order_sha256",
        "training_order_sha256",
    )
    if not isinstance(split, dict):
        raise RuntimeError("preflight split lineage is malformed")
    _require_fields(split, split_fields, label="preflight split")
    integer_fields = (
        "declared_triples",
        "raw_triples",
        "held_out_size",
        "training_set_size",
        "split_seed",
    )
    if any(
        not isinstance(split[field], int) or isinstance(split[field], bool)
        for field in integer_fields
    ):
        raise RuntimeError("preflight split size/seed lineage is malformed")
    if (
        split["training_set_size"] != int(contract["protocol"]["training_examples"])
        or split["raw_triples"]
        != split["training_set_size"] + split["held_out_size"]
        or split["declared_triples"] != split["raw_triples"]
        or split["split_seed"] != 42
    ):
        raise RuntimeError("preflight split size/rule drift from frozen protocol")
    source_path = split["source_path"]
    source_hashes = contract.get("source_hashes", {})
    if (
        not isinstance(source_path, str)
        or source_hashes.get(source_path) != split["source_sha256"]
        or not isinstance(split["split_algorithm"], str)
        or not split["split_algorithm"].strip()
    ):
        raise RuntimeError("preflight split source/rule drift from frozen protocol")
    for field in (
        "source_sha256",
        "raw_order_sha256",
        "file_order_sha256",
        "held_out_order_sha256",
        "training_order_sha256",
    ):
        value = split[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"preflight split hash is malformed: {field}")
    return split


def _validate_job(
    root: Path,
    job: dict[str, Any],
    contract: dict[str, Any],
    expected_split: dict[str, Any],
) -> dict[str, Any]:
    schemas = contract["raw_artifact_schemas"]
    protocol = contract["protocol"]
    path = _job_dir(root, job)
    status = _read_json(path / "status.json")
    _require_fields(status, schemas["status.json"]["required_fields"], label="status.json")
    identity = (status.get("pass_name"), status.get("config"), status.get("seed"))
    expected_identity = (job["pass_name"], job["config"], job["seed"])
    if (
        status.get("protocol_id") != protocol["protocol_id"]
        or identity != expected_identity
        or status.get("valid") is not True
        or status.get("invalid_reasons") not in ([], None)
        or status.get("split") != expected_split
    ):
        raise RuntimeError("selected status identity/validity/split drift detected")
    telemetry_path = path / "gpu_telemetry.csv"
    telemetry = _read_csv(
        telemetry_path, schemas["gpu_telemetry.csv"]["required_columns"]
    )
    if not telemetry:
        raise RuntimeError("selected job telemetry is empty")
    false_values = {"", "0", "false", "no", "none", "not active", "disabled"}
    for row in telemetry:
        if (
            row.get("protocol_id") != protocol["protocol_id"]
            or row.get("pass_name") != job["pass_name"]
            or row.get("config") != job["config"]
            or _integer(row, "seed", telemetry_path) != job["seed"]
            or _integer(row, "time_ns", telemetry_path) <= 0
            or row.get("query_error", "").strip()
            or row.get("other_compute_processes", "").strip()
            or row.get("thermal_slowdown", "").strip().lower() not in false_values
        ):
            raise RuntimeError("selected job telemetry is invalid")

    epochs = int(protocol["epochs_per_job"])
    batch_size = int(protocol["batch_size"])
    examples = int(protocol["training_examples"])
    full_batches, partial_size = divmod(examples, batch_size)
    steps_per_epoch = full_batches + (1 if partial_size else 0)
    epoch_path = path / "per_epoch.csv"
    epoch_rows = _read_csv(epoch_path, schemas["per_epoch.csv"]["required_columns"])
    if len(epoch_rows) != epochs:
        raise RuntimeError("per_epoch.csv row count mismatch")
    epoch_times: list[int] = []
    for expected_epoch, row in enumerate(epoch_rows):
        if (
            row.get("protocol_id") != protocol["protocol_id"]
            or row.get("pass_name") != job["pass_name"]
            or row.get("config") != job["config"]
            or _integer(row, "seed", epoch_path) != job["seed"]
            or _integer(row, "epoch", epoch_path) != expected_epoch
            or _integer(row, "num_steps", epoch_path) != steps_per_epoch
            or _integer(row, "full_batch_count", epoch_path) != full_batches
            or _integer(row, "partial_batch_count", epoch_path) != (1 if partial_size else 0)
            or _integer(row, "partial_batch_size", epoch_path) != partial_size
            or _integer(row, "training_examples", epoch_path) != examples
            or not _boolean(row.get("loss_finite", ""), path=epoch_path, field="loss_finite")
        ):
            raise RuntimeError("per_epoch.csv identity/invariant mismatch")
        value = _integer(row, "epoch_time_ns", epoch_path)
        if value <= 0:
            raise RuntimeError("per_epoch.csv timing invariant mismatch")
        epoch_times.append(value)
    expected_steps = epochs * steps_per_epoch if job["pass_name"] == "trace" else 0
    if status.get("row_counts") != {"epochs": epochs, "steps": expected_steps}:
        raise RuntimeError("status.json row_counts mismatch")
    if job["pass_name"] != "trace":
        return {"epoch_times_ns": epoch_times}

    step_path = path / "per_step.csv"
    step_rows = _read_csv(step_path, schemas["per_step.csv"]["required_columns"])
    if len(step_rows) != expected_steps:
        raise RuntimeError("per_step.csv row count mismatch")
    full_by_epoch: dict[int, list[int]] = {epoch: [] for epoch in range(epochs)}
    e2_filter = contract["analysis"]["filters"]["E2"]
    for index, row in enumerate(step_rows):
        epoch, step = divmod(index, steps_per_epoch)
        partial = bool(partial_size and step == full_batches)
        if (
            row.get("protocol_id") != protocol["protocol_id"]
            or row.get("pass_name") != "trace"
            or row.get("config") != job["config"]
            or _integer(row, "seed", step_path) != job["seed"]
            or _integer(row, "epoch", step_path) != epoch
            or _integer(row, "step", step_path) != step
            or _integer(row, "batch_size_actual", step_path) != (partial_size if partial else batch_size)
            or _boolean(row.get("is_partial", ""), path=step_path, field="is_partial") != partial
            or _boolean(
                row.get("is_first_measured_step", ""),
                path=step_path,
                field="is_first_measured_step",
            ) != (index == 0)
        ):
            raise RuntimeError("per_step.csv identity/grid/filter invariant mismatch")
        neg = _integer(row, "neg_time_ns", step_path)
        component = _integer(row, "component_sum_ns", step_path)
        total = _integer(row, "total_step_ns", step_path)
        residual = _integer(row, "timing_residual_ns", step_path)
        if neg <= 0 or component < 0 or total <= 0 or component + residual != total:
            raise RuntimeError("per_step.csv numeric timing invariant mismatch")
        is_partial = _boolean(row["is_partial"], path=step_path, field="is_partial")
        if (
            is_partial is bool(e2_filter["is_partial"])
            and _integer(row, "batch_size_actual", step_path) == int(e2_filter["batch_size_actual"])
        ):
            full_by_epoch[epoch].append(neg)
    if any(len(values) != full_batches for values in full_by_epoch.values()):
        raise RuntimeError("full-batch analysis filter coverage mismatch")
    return {"epoch_times_ns": epoch_times, "full_neg_ns_by_epoch": full_by_epoch}


def _geometric_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return math.exp(statistics.fmean(math.log(value) for value in materialized))


def _independent_paths(root: Path, *, require_complete: bool = True) -> list[Path]:
    directory = root / INDEPENDENT_DIR
    if directory.is_symlink():
        raise RuntimeError("independent output directory must not be a symlink")
    if not directory.exists():
        if require_complete:
            raise RuntimeError("independent outputs are incomplete")
        return []
    if not directory.is_dir():
        raise RuntimeError("independent output root is not a directory")
    expected = {directory / name for name in INDEPENDENT_OUTPUT_NAMES}
    observed: set[Path] = set()
    for path in directory.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"independent output must not be a symlink: {path.name}")
        if path not in expected or not path.is_file():
            raise RuntimeError(f"unexpected independent output: {path.name}")
        observed.add(path)
    if require_complete and observed != expected:
        raise RuntimeError("independent outputs are incomplete")
    return sorted(observed, key=lambda item: item.relative_to(root).as_posix())


def validate_independent_seal(root: Path) -> bool:
    """Validate the independent seal; comparison must call this before original I/O."""
    root = root.resolve()
    validate_raw_seal(root)
    manifest, contract = _load_frozen(root)
    raw_passport = _read_json(root / "material_passport.json")
    derived_manifest_path = root / INDEPENDENT_MANIFEST
    derived_manifest = _read_json(derived_manifest_path)
    passport = _read_json(root / INDEPENDENT_PASSPORT)
    if (
        derived_manifest.get("contract_id") != contract["contract_id"]
        or derived_manifest.get("stage") != "independent"
        or derived_manifest.get("raw_artifact_manifest_sha256")
        != raw_passport["raw_artifact_manifest_sha256"]
    ):
        raise RuntimeError("independent artifact manifest identity mismatch")
    expected = _artifact_entries(root, _independent_paths(root))
    if derived_manifest.get("artifacts") != expected:
        raise RuntimeError("independent artifact path/byte/hash mismatch")
    bindings = {
        "contract_id": contract["contract_id"],
        "contract_sha256": manifest["contract_sha256"],
        "capsule_manifest_sha256": manifest["capsule_manifest_sha256"],
        "environment_manifest_sha256": manifest["environment_manifest_sha256"],
        "raw_artifact_manifest_sha256": raw_passport["raw_artifact_manifest_sha256"],
        "independent_artifact_manifest_sha256": _sha256_file(derived_manifest_path),
        "stage": "independent",
        "sealed_at_ns": raw_passport["sealed_at_ns"],
    }
    for field, expected_value in bindings.items():
        if passport.get(field) != expected_value:
            raise RuntimeError(f"independent artifact seal binding mismatch: {field}")
    return True


def run_independent(root: Path) -> dict[str, Any]:
    """Recompute every X8 estimand from the sealed raw nanosecond artifacts."""
    root = root.resolve()
    validate_raw_seal(root)
    _independent_paths(root, require_complete=False)
    manifest, contract = _load_frozen(root)
    expected_split = _validate_lineage_schemas(root, contract)
    selected = _select_analysis_jobs(manifest, contract)
    measurements = {
        key: _validate_job(root, job, contract, expected_split)
        for key, job in selected.items()
    }
    seeds = sorted(contract["execution_matrix"]["primary"]["seeds"])
    required = int(contract["analysis"]["primary_gate"]["complete_pairs_required"])
    if len(seeds) != required:
        raise RuntimeError("frozen primary seed count drift detected")
    seed_rows: list[dict[str, Any]] = []
    e1_values: list[float] = []
    e2_values: list[float] = []
    e3_values: list[float] = []
    for seed in seeds:
        throughput = {
            config: statistics.fmean(
                measurements[("throughput", config, seed)]["epoch_times_ns"]
            )
            for config in ("BL", "GPU")
        }
        trace_sd: dict[str, float] = {}
        trace_mean: dict[str, float] = {}
        for config in ("BL", "GPU"):
            by_epoch = measurements[("trace", config, seed)]["full_neg_ns_by_epoch"]
            trace_sd[config] = statistics.fmean(
                population_sd(by_epoch[epoch]) for epoch in sorted(by_epoch)
            )
            trace_mean[config] = statistics.fmean(
                value for epoch in sorted(by_epoch) for value in by_epoch[epoch]
            )
        if throughput["GPU"] <= 0 or trace_sd["GPU"] <= 0:
            raise RuntimeError("GPU denominator is not positive")
        e1 = throughput["BL"] / throughput["GPU"]
        e2 = trace_sd["BL"] / trace_sd["GPU"]
        e3 = trace_mean["GPU"]
        e1_values.append(e1)
        e2_values.append(e2)
        e3_values.append(e3)
        seed_rows.append(
            {
                "seed": seed,
                "E1_bl_epoch_mean_ns": throughput["BL"],
                "E1_gpu_epoch_mean_ns": throughput["GPU"],
                "E1_ratio": e1,
                "E2_bl_mean_epoch_population_sd_ns": trace_sd["BL"],
                "E2_gpu_mean_epoch_population_sd_ns": trace_sd["GPU"],
                "E2_ratio": e2,
                "E3_gpu_full_neg_mean_ns": e3,
                "throughput_attempt": selected[("throughput", "BL", seed)].get("attempt", 0),
                "trace_attempt": selected[("trace", "BL", seed)].get("attempt", 0),
            }
        )

    criticals = contract["analysis"]["t_critical_values"]
    e1_summary = geometric_summary(
        e1_values,
        t95=float(criticals["df5_ci95"]),
        t97_5=float(criticals["df5_ci97_5"]),
    )
    e2_summary = geometric_summary(
        e2_values,
        t95=float(criticals["df5_ci95"]),
        t97_5=float(criticals["df5_ci97_5"]),
    )
    e3_summary = arithmetic_summary(e3_values, t95=float(criticals["df5_ci95"]))
    e1_summary["seed_level_ratios"] = e1_values
    e2_summary["seed_level_ratios"] = e2_values
    e3_result = {
        "estimate_ns": e3_summary["estimate"],
        "sample_sd_ns": e3_summary["sample_sd"],
        "ci95_ns": e3_summary["ci95"],
        "seed_level_gpu_means_ns": e3_values,
    }
    direction = {
        "E1": {"count": sum(value > 1.0 for value in e1_values), "required": required},
        "E2": {"count": sum(value > 1.0 for value in e2_values), "required": required},
    }
    for item in direction.values():
        item["passed"] = item["count"] == item["required"]
    threshold = float(
        contract["analysis"]["primary_gate"]["simultaneous_lower_bound_strictly_above"]
    )
    primary_gate = (
        e1_summary["ci97_5_bonferroni"]["low"] > threshold
        and e2_summary["ci97_5_bonferroni"]["low"] > threshold
    )
    summary = {
        "contract_id": contract["contract_id"],
        "protocol_id": contract["protocol"]["protocol_id"],
        "stage": "independent",
        "status": "ANALYZED",
        "analysis_unit": "seed-level paired run",
        "pooling_performed": False,
        "complete_seed_pairs": len(seed_rows),
        "estimands": {"E1": e1_summary, "E2": e2_summary, "E3": e3_result},
        "direction_consistency": direction,
        "primary_gate_passed": primary_gate,
    }
    checks = {
        "all_passed": True,
        "checks": [
            {"check": "raw_seal_valid", "passed": True},
            {"check": "selected_attempts_predeclared", "passed": True},
            {"check": "raw_schemas_valid", "passed": True},
            {"check": "six_seed_pairs_complete", "passed": len(seed_rows) == required},
            {"check": "pooling_forbidden", "passed": summary["pooling_performed"] is False},
        ],
    }
    leave_one_rows = []
    for omitted_index, omitted_seed in enumerate(seeds):
        leave_one_rows.append(
            {
                "omitted_seed": omitted_seed,
                "remaining_seed_count": len(seeds) - 1,
                "E1_estimate": _geometric_mean(
                    value for index, value in enumerate(e1_values) if index != omitted_index
                ),
                "E2_estimate": _geometric_mean(
                    value for index, value in enumerate(e2_values) if index != omitted_index
                ),
                "E3_estimate_ns": statistics.fmean(
                    value for index, value in enumerate(e3_values) if index != omitted_index
                ),
            }
        )

    output = root / INDEPENDENT_DIR
    _write_json(output / "summary.json", summary)
    _write_json(output / "checks.json", checks)
    seed_fields = [
        "seed", "E1_bl_epoch_mean_ns", "E1_gpu_epoch_mean_ns", "E1_ratio",
        "E2_bl_mean_epoch_population_sd_ns", "E2_gpu_mean_epoch_population_sd_ns",
        "E2_ratio", "E3_gpu_full_neg_mean_ns", "throughput_attempt", "trace_attempt",
    ]
    _write_csv(output / "seed_level_metrics.csv", seed_fields, seed_rows)
    leave_fields = [
        "omitted_seed", "remaining_seed_count", "E1_estimate", "E2_estimate",
        "E3_estimate_ns",
    ]
    _write_csv(output / "leave_one_seed_out.csv", leave_fields, leave_one_rows)
    raw_passport = _read_json(root / "material_passport.json")
    derived_manifest = {
        "contract_id": contract["contract_id"],
        "stage": "independent",
        "raw_artifact_manifest_sha256": raw_passport["raw_artifact_manifest_sha256"],
        "artifacts": _artifact_entries(root, _independent_paths(root)),
    }
    _write_json(root / INDEPENDENT_MANIFEST, derived_manifest)
    independent_passport = {
        "contract_id": contract["contract_id"],
        "contract_sha256": manifest["contract_sha256"],
        "capsule_manifest_sha256": manifest["capsule_manifest_sha256"],
        "environment_manifest_sha256": manifest["environment_manifest_sha256"],
        "raw_artifact_manifest_sha256": raw_passport["raw_artifact_manifest_sha256"],
        "independent_artifact_manifest_sha256": _sha256_file(root / INDEPENDENT_MANIFEST),
        "stage": "independent",
        "sealed_at_ns": raw_passport["sealed_at_ns"],
    }
    _write_json(root / INDEPENDENT_PASSPORT, independent_passport)
    validate_independent_seal(root)
    return summary


def _inside_inclusive(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def compare_estimates(
    independent: dict[str, Any],
    original: dict[str, float],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Compare, but never pool, clean-room and frozen original point estimates."""
    status = independent.get("status")
    if status in {"INCOMPLETE", "BLOCKED_ENVIRONMENT"}:
        return {"verdict": status, "estimands": {}}
    if status != "ANALYZED":
        raise RuntimeError("independent analysis status is not comparable")
    if independent.get("pooling_performed") is not False:
        raise RuntimeError("pooling clean-room and original observations is forbidden")
    required = int(contract["analysis"]["primary_gate"]["complete_pairs_required"])
    if independent.get("complete_seed_pairs") != required:
        return {"verdict": "INCOMPLETE", "estimands": {}}
    clean = {
        "E1": float(independent["estimands"]["E1"]["estimate"]),
        "E2": float(independent["estimands"]["E2"]["estimate"]),
        "E3": float(independent["estimands"]["E3"]["estimate_ns"]),
    }
    tolerance = contract["analysis"]["numerical_fidelity_ratios"]
    comparisons: dict[str, dict[str, Any]] = {}
    for estimand in ("E1", "E2", "E3"):
        if original[estimand] <= 0:
            raise RuntimeError("frozen original estimate is not positive")
        ratio = clean[estimand] / float(original[estimand])
        lower, upper = map(float, tolerance[estimand])
        comparisons[estimand] = {
            "clean_estimate": clean[estimand],
            "original_estimate": float(original[estimand]),
            "clean_to_original_ratio": ratio,
            "inclusive_tolerance": [lower, upper],
            "within_tolerance": _inside_inclusive(ratio, lower, upper),
        }
    direction = independent.get("direction_consistency", {})
    directions_pass = all(
        direction.get(estimand, {}).get("passed") is True
        and direction[estimand].get("count") == required
        for estimand in ("E1", "E2")
    )
    supported = independent.get("primary_gate_passed") is True and directions_pass
    if not supported:
        verdict = "NOT_REPRODUCED"
    elif all(item["within_tolerance"] for item in comparisons.values()):
        verdict = "VERIFIED"
    else:
        verdict = "SUPPORTED_WITH_NUMERICAL_DRIFT"
    if verdict not in contract["analysis"]["verdict_states"]:
        raise RuntimeError("comparison produced a non-canonical verdict")
    return {
        "verdict": verdict,
        "pooling_performed": False,
        "estimands": comparisons,
    }


def run_compare(root: Path, original_root: Path) -> dict[str, Any]:
    """Enter comparison only through the validated independent-seal gate."""
    root = root.resolve()
    validate_independent_seal(root)
    original_root = original_root.resolve()
    if original_root == root or original_root.is_relative_to(root):
        raise RuntimeError("original result root must remain separate from clean-room outputs")
    manifest, contract = _load_frozen(root)
    independent = _read_json(root / INDEPENDENT_DIR / "summary.json")
    original = _read_original_estimates(original_root)
    result = compare_estimates(independent, original, contract)
    fallacies = _statistical_fallacy_scan(independent, contract)
    status_only = result["verdict"] in {"INCOMPLETE", "BLOCKED_ENVIRONMENT"}
    if not status_only and not all(item["passed"] for item in fallacies):
        result["verdict"] = "NOT_REPRODUCED"
    result.update(
        {
            "contract_id": contract["contract_id"],
            "protocol_id": manifest["protocol_id"],
            "stage": "compare",
            "independent_seal_validated_before_original_read": True,
            "fallacy_check_count": len(fallacies),
            "fallacy_scan_all_passed": all(item["passed"] for item in fallacies),
        }
    )
    output = root / "derived/comparison"
    _write_json(output / "comparison.json", result)
    _write_csv(
        output / "statistical_fallacy_scan.csv",
        ["check_id", "category", "status", "evidence", "passed"],
        fallacies,
    )
    return result


def _read_original_estimates(original_root: Path) -> dict[str, float]:
    summary = _read_json(original_root / "analysis/summary.json")
    try:
        estimates = {
            "E1": float(summary["C1.2-R1"]["geometric_mean_speedup"]),
            "E2": float(summary["C1.3-R1"]["geometric_mean_sd_compression"]),
            "E3": float(summary["C1.7-R1"]["six_run_mean_ms"]) * 1_000_000,
        }
        gates = [
            summary["C1.2-R1"]["A_gate_passed"],
            summary["C1.3-R1"]["A_gate_passed"],
            summary["C1.7-R1"]["A_gate_passed"],
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("frozen original summary schema is invalid") from exc
    if gates != [True, True, True]:
        raise RuntimeError("frozen original gates drift detected")
    for estimand, expected in FROZEN_ORIGINAL_ESTIMATES.items():
        absolute_tolerance = 1e-9 if estimand == "E3" else 1e-15
        if not math.isclose(
            estimates[estimand], expected, rel_tol=0.0, abs_tol=absolute_tolerance
        ):
            raise RuntimeError(f"frozen original estimate drift detected: {estimand}")
    return dict(FROZEN_ORIGINAL_ESTIMATES)


def _statistical_fallacy_scan(
    independent: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    required = int(contract["analysis"]["primary_gate"]["complete_pairs_required"])
    no_pooling = (
        contract["analysis_controls"].get("pooling_forbidden") is True
        and independent.get("pooling_performed") is False
    )
    seed_unit = (
        independent.get("analysis_unit") == "seed-level paired run"
        and independent.get("complete_seed_pairs") == required
    )
    attempts_predeclared = independent.get("status") == "ANALYZED"
    bonferroni = all(
        "ci97_5_bonferroni" in independent.get("estimands", {}).get(estimand, {})
        for estimand in ("E1", "E2")
    )
    frozen_paths = contract.get("status") == "FROZEN"
    definitions = [
        (
            "F01", "aggregation_reversal/no_pooling", no_pooling,
            "clean-room and original observations remain separate; effects are seed-level",
        ),
        (
            "F02", "pseudoreplication/nested_units", seed_unit,
            "seed is the independent unit; epochs and batches remain nested",
        ),
        (
            "F03", "selection/collider", attempts_predeclared,
            "only the manifest-predeclared completed attempt pair is selected",
        ),
        (
            "F04", "base_rate", True,
            "not applicable: no prevalence or classification claim is made",
        ),
        (
            "F05", "regression_to_mean", True,
            "not applicable: attempts are not selected adaptively by measured effect",
        ),
        (
            "F06", "survivorship", attempts_predeclared,
            "excluded invalid attempts remain retained in the sealed raw snapshot",
        ),
        (
            "F07", "look_elsewhere", bonferroni,
            "the frozen E1/E2 co-primary family uses Bonferroni 97.5% intervals",
        ),
        (
            "F08", "forking_paths", frozen_paths,
            "estimands, filters, seeds, and retry rules were frozen before analysis",
        ),
        (
            "F09", "correlation_not_causation", True,
            "paired runtime effects are not interpreted as an observational correlation",
        ),
        (
            "F10", "reverse_causality", True,
            "not applicable: BL/GPU configuration precedes every measured outcome",
        ),
        (
            "F11", "causal_overreach", True,
            "scope is limited to internal same-host artifact verification",
        ),
    ]
    not_applicable = {"F04", "F05", "F10"}
    return [
        {
            "check_id": check_id,
            "category": category,
            "status": (
                "NOT_APPLICABLE"
                if check_id in not_applicable
                else ("PASS" if passed else "FAIL")
            ),
            "evidence": evidence,
            "passed": passed,
        }
        for check_id, category, passed, evidence in definitions
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("independent", "compare"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.stage is not None or args.root is not None or args.original_root is not None:
            raise ValueError("--self-test cannot be combined with stage arguments")
        result = self_test()
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.stage is None or args.root is None:
        raise ValueError("--stage and --root are required")
    if args.stage == "independent" and args.original_root is not None:
        raise ValueError("original-root argument is forbidden in independent stage")
    if args.stage == "independent":
        result = run_independent(args.root)
    else:
        if args.original_root is None:
            raise ValueError("--original-root is required in compare stage")
        result = run_compare(args.root, args.original_root)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def self_test() -> dict[str, Any]:
    geometric = geometric_summary(
        [2.0] * 6,
        t95=2.570581835636314,
        t97_5=3.163381449748624,
    )
    arithmetic = arithmetic_summary(
        [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        t95=2.570581835636314,
    )
    checks = {
        "population_sd_ddof0": population_sd([10.0, 30.0]) == 10.0,
        "geometric_constant_ratio": geometric["ci97_5_bonferroni"]
        == {"low": 2.0, "high": 2.0},
        "arithmetic_mean": arithmetic["estimate"] == 15.0,
        "frozen_original_positive": all(
            value > 0 for value in FROZEN_ORIGINAL_ESTIMATES.values()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"self-test failure: {checks}")
    return {"status": "PASS", "checks": checks}


if __name__ == "__main__":
    raise SystemExit(main())
