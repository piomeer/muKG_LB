#!/usr/bin/env python3
"""Validate and analyze the C1-R1 v1.1 combined rerun artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "C1-R1-v1.1"
SEEDS = tuple(range(42, 48))
CONFIGS = ("BL", "GPU")
EPOCHS = tuple(range(5))
BATCH_SIZE = 5000
FULL_PER_EPOCH = 53
PARTIAL_SIZE = 2115
T_CRIT_DF5 = 2.570581835636314


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str],
              rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires observations")
    return sum(materialized) / len(materialized)


def population_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    center = mean(materialized)
    return math.sqrt(mean((value - center) ** 2 for value in materialized))


def sample_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized)


def arithmetic_t_ci(values: Iterable[float]) -> tuple[float, float, float]:
    materialized = list(values)
    center = mean(materialized)
    se = sample_sd(materialized) / math.sqrt(len(materialized))
    margin = T_CRIT_DF5 * se
    return center, center - margin, center + margin


def geometric_t_ci(ratios: Iterable[float]) -> tuple[float, float, float]:
    materialized = list(ratios)
    if any(value <= 0 for value in materialized):
        raise ValueError("geometric CI requires positive ratios")
    logs = [math.log(value) for value in materialized]
    center = mean(logs)
    margin = T_CRIT_DF5 * sample_sd(logs) / math.sqrt(len(logs))
    return math.exp(center), math.exp(center - margin), math.exp(center + margin)


def thermal_active(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "0", "no", "false", "not active", "[n/a]", "n/a"}


def validate_epoch_rows(rows: list[dict[str, str]], config: str,
                        seed: int, pass_name: str) -> None:
    if len(rows) != 5:
        raise AssertionError(f"{pass_name}/{config}/{seed}: expected 5 epoch rows")
    if {int(row["epoch"]) for row in rows} != set(EPOCHS):
        raise AssertionError(f"{pass_name}/{config}/{seed}: epoch IDs mismatch")
    for row in rows:
        if row["protocol_id"] != PROTOCOL_ID:
            raise AssertionError("protocol ID mismatch")
        if row["config"] != config or int(row["seed"]) != seed:
            raise AssertionError("config/seed mismatch")
        if int(row["num_steps"]) != 54:
            raise AssertionError("step count mismatch")
        if int(row["full_batch_count"]) != FULL_PER_EPOCH:
            raise AssertionError("full batch count mismatch")
        if int(row["partial_batch_count"]) != 1:
            raise AssertionError("partial batch count mismatch")
        if int(row["partial_batch_size"]) != PARTIAL_SIZE:
            raise AssertionError("partial batch size mismatch")
        if int(row["training_examples"]) != 267115:
            raise AssertionError("training coverage mismatch")
        if row["loss_finite"].lower() != "true":
            raise AssertionError("non-finite loss")


def validate_step_rows(rows: list[dict[str, str]], config: str, seed: int) -> None:
    if len(rows) != 270:
        raise AssertionError(f"trace/{config}/{seed}: expected 270 step rows")
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["protocol_id"] != PROTOCOL_ID or row["pass_name"] != "trace":
            raise AssertionError("trace protocol/pass mismatch")
        if row["config"] != config or int(row["seed"]) != seed:
            raise AssertionError("trace config/seed mismatch")
        by_epoch[int(row["epoch"])].append(row)
        component_sum = sum(int(row[field]) for field in [
            "zero_grad_ns", "neg_time_ns", "positive_tensor_build_ns",
            "forward_ns", "backward_ns", "optimizer_ns",
        ])
        if component_sum != int(row["component_sum_ns"]):
            raise AssertionError("component sum mismatch")
        if (
            int(row["total_step_ns"]) - component_sum
            != int(row["timing_residual_ns"])
        ):
            raise AssertionError("timing residual mismatch")
    if set(by_epoch) != set(EPOCHS):
        raise AssertionError("trace epoch IDs mismatch")
    for epoch, epoch_rows in by_epoch.items():
        if len(epoch_rows) != 54:
            raise AssertionError(f"epoch {epoch}: trace row count mismatch")
        full = [
            row for row in epoch_rows
            if row["is_partial"].lower() == "false"
            and int(row["batch_size_actual"]) == BATCH_SIZE
        ]
        partial = [
            row for row in epoch_rows if row["is_partial"].lower() == "true"
        ]
        if len(full) != FULL_PER_EPOCH or len(partial) != 1:
            raise AssertionError(f"epoch {epoch}: partial/full classification mismatch")
        if int(partial[0]["batch_size_actual"]) != PARTIAL_SIZE:
            raise AssertionError(f"epoch {epoch}: partial size mismatch")


def verify_stored_hashes(root: Path) -> list[str]:
    errors = []
    hash_path = root / "artifact_hashes.csv"
    if not hash_path.exists():
        return ["artifact_hashes.csv missing"]
    for row in read_csv(hash_path):
        path = root / row["path"]
        if not path.exists():
            errors.append(f"missing hashed artifact: {row['path']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"hash mismatch: {row['path']}")
        elif path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch: {row['path']}")
    return errors


def analyze(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    preflight = json.loads(
        (root / "preflight/result.json").read_text(encoding="utf-8")
    )
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise AssertionError("unexpected protocol")
    if not preflight["all_passed"]:
        raise AssertionError("preflight did not pass")
    if protocol["dataset"]["training_set_size"] != 267115:
        raise AssertionError("training set size mismatch in protocol")

    checks: list[dict[str, Any]] = []
    invalid_jobs: list[str] = []
    retry_root = root / "reruns/throughput_seed45_attempt2"
    retry_used = retry_root.exists()
    throughput_epochs: dict[tuple[str, int], list[dict[str, str]]] = {}
    trace_steps: dict[tuple[str, int], list[dict[str, str]]] = {}

    for config in CONFIGS:
        for seed in SEEDS:
            for pass_name in ("throughput", "trace"):
                if pass_name == "throughput" and seed == 45 and retry_used:
                    job_dir = retry_root / "jobs" / f"{pass_name}_{config}_seed{seed}"
                    attempt = 2
                else:
                    job_dir = root / "jobs" / f"{pass_name}_{config}_seed{seed}"
                    attempt = 1
                epoch_rows = read_csv(job_dir / "per_epoch.csv")
                validate_epoch_rows(epoch_rows, config, seed, pass_name)
                if pass_name == "throughput":
                    throughput_epochs[(config, seed)] = epoch_rows
                else:
                    steps = read_csv(job_dir / "per_step.csv")
                    validate_step_rows(steps, config, seed)
                    trace_steps[(config, seed)] = steps
                status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
                telemetry = read_csv(job_dir / "gpu_telemetry.csv")
                other_process = any(row["other_compute_processes"] for row in telemetry)
                thermal = any(thermal_active(row["thermal_slowdown"]) for row in telemetry)
                valid = bool(status["valid"]) and not other_process and not thermal
                label = f"{pass_name}_{config}_seed{seed}_attempt{attempt}"
                if not valid:
                    invalid_jobs.append(label)
                checks.append({
                    "check": f"job_{label}",
                    "passed": valid,
                    "epoch_rows": len(epoch_rows),
                    "step_rows": 0 if pass_name == "throughput" else len(steps),
                    "other_compute_process": other_process,
                    "thermal_slowdown": thermal,
                })

    hash_errors = [
        f"original:{error}" for error in verify_stored_hashes(root)
    ]
    if retry_used:
        hash_errors.extend(
            f"retry:{error}" for error in verify_stored_hashes(retry_root)
        )
        original_telemetry = read_csv(
            root / "jobs/throughput_GPU_seed45/gpu_telemetry.csv"
        )
        original_thermal = any(
            thermal_active(row["thermal_slowdown"]) for row in original_telemetry
        )
        checks.append({
            "check": "excluded_throughput_GPU_seed45_attempt1",
            "passed": original_thermal,
            "reason": "explicit thermal slowdown telemetry; one paired retry allowed",
        })
    checks.append({
        "check": "stored_artifact_hashes",
        "passed": not hash_errors,
        "errors": hash_errors,
    })

    run_rows = []
    paired_rows = []
    c12_ratios = []
    c13_ratios = []
    gpu_run_neg_means_ms = []

    c13_run_values: dict[tuple[str, int], float] = {}
    for config in CONFIGS:
        for seed in SEEDS:
            epoch_time_s = [
                int(row["epoch_time_ns"]) / 1e9
                for row in throughput_epochs[(config, seed)]
            ]
            steps = trace_steps[(config, seed)]
            epoch_full_sds_ms = []
            all_full_neg_ms = []
            all_including_partial_ms = []
            excluding_first_global_ms = []
            for epoch in EPOCHS:
                epoch_rows = [row for row in steps if int(row["epoch"]) == epoch]
                full = [
                    row for row in epoch_rows
                    if row["is_partial"].lower() == "false"
                    and int(row["batch_size_actual"]) == BATCH_SIZE
                ]
                full_values = [int(row["neg_time_ns"]) / 1e6 for row in full]
                epoch_full_sds_ms.append(population_sd(full_values))
                all_full_neg_ms.extend(full_values)
                all_including_partial_ms.extend(
                    int(row["neg_time_ns"]) / 1e6 for row in epoch_rows
                )
                excluding_first_global_ms.extend(
                    int(row["neg_time_ns"]) / 1e6
                    for row in full
                    if not (
                        row["is_first_measured_step"].lower() == "true"
                    )
                )
            c13_run = mean(epoch_full_sds_ms)
            c13_run_values[(config, seed)] = c13_run
            run_row = {
                "config": config,
                "seed": seed,
                "throughput_epoch_mean_s": mean(epoch_time_s),
                "throughput_epoch_sample_sd_s": sample_sd(epoch_time_s),
                "trace_full_neg_mean_ms": mean(all_full_neg_ms),
                "trace_full_neg_population_sd_ms": population_sd(all_full_neg_ms),
                "trace_mean_epoch_population_sd_ms": c13_run,
                "trace_including_partial_neg_population_sd_ms": population_sd(
                    all_including_partial_ms
                ),
                "trace_excluding_first_global_neg_population_sd_ms": population_sd(
                    excluding_first_global_ms
                ),
                "epoch0_avg_loss": float(
                    throughput_epochs[(config, seed)][0]["avg_loss"]
                ),
                "epoch4_avg_loss": float(
                    throughput_epochs[(config, seed)][-1]["avg_loss"]
                ),
            }
            run_rows.append(run_row)
            if config == "GPU":
                gpu_run_neg_means_ms.append(run_row["trace_full_neg_mean_ms"])

    run_index = {(row["config"], row["seed"]): row for row in run_rows}
    for seed in SEEDS:
        bl = run_index[("BL", seed)]
        gpu = run_index[("GPU", seed)]
        c12 = bl["throughput_epoch_mean_s"] / gpu["throughput_epoch_mean_s"]
        c13 = (
            bl["trace_mean_epoch_population_sd_ms"]
            / gpu["trace_mean_epoch_population_sd_ms"]
        )
        c12_ratios.append(c12)
        c13_ratios.append(c13)
        paired_rows.append({
            "seed": seed,
            "bl_throughput_epoch_mean_s": bl["throughput_epoch_mean_s"],
            "gpu_throughput_epoch_mean_s": gpu["throughput_epoch_mean_s"],
            "c1_2_paired_speedup": c12,
            "bl_mean_epoch_neg_population_sd_ms": (
                bl["trace_mean_epoch_population_sd_ms"]
            ),
            "gpu_mean_epoch_neg_population_sd_ms": (
                gpu["trace_mean_epoch_population_sd_ms"]
            ),
            "c1_3_paired_sd_compression": c13,
        })

    c12_gm, c12_low, c12_high = geometric_t_ci(c12_ratios)
    c13_gm, c13_low, c13_high = geometric_t_ci(c13_ratios)
    c17_mean, c17_low, c17_high = arithmetic_t_ci(gpu_run_neg_means_ms)
    all_jobs_valid = not invalid_jobs
    hashes_valid = not hash_errors
    complete_pairs = len(c12_ratios) == len(c13_ratios) == 6
    c12_a = (
        all_jobs_valid and hashes_valid and complete_pairs and c12_low > 1.0
    )
    c13_a = (
        all_jobs_valid and hashes_valid and complete_pairs and c13_low > 1.0
    )
    c17_a = all_jobs_valid and hashes_valid and len(gpu_run_neg_means_ms) == 6
    warnings = []
    if any(value > 10 for value in c12_ratios):
        warnings.append("C1.2 paired speedup exceeds 10x; inspect trace and loss diagnostics")
    for row in run_rows:
        if row["epoch4_avg_loss"] >= row["epoch0_avg_loss"]:
            warnings.append(
                f"{row['config']} seed {row['seed']} final loss did not fall below epoch-0"
            )

    summary = {
        "protocol_id": PROTOCOL_ID,
        "environment": {
            "hostname": environment["hostname"],
            "pytorch": environment["pytorch"],
            "torch_cuda_runtime": environment["torch_cuda_runtime"],
            "gpu": environment["gpu"],
        },
        "validity": {
            "preflight_passed": preflight["all_passed"],
            "all_jobs_valid": all_jobs_valid,
            "invalid_jobs": invalid_jobs,
            "artifact_hashes_valid": hashes_valid,
            "hash_errors": hash_errors,
            "complete_six_pairs": complete_pairs,
            "protocol_allowed_retry_used": retry_used,
        },
        "C1.2-R1": {
            "paired_speedups": c12_ratios,
            "geometric_mean_speedup": c12_gm,
            "ci95_low": c12_low,
            "ci95_high": c12_high,
            "A_gate_passed": c12_a,
        },
        "C1.3-R1": {
            "estimand": (
                "paired compression of the run mean of five within-epoch "
                "population SDs, full 5000-example batches only"
            ),
            "paired_sd_compressions": c13_ratios,
            "geometric_mean_sd_compression": c13_gm,
            "ci95_low": c13_low,
            "ci95_high": c13_high,
            "A_gate_passed": c13_a,
        },
        "C1.7-R1": {
            "gpu_run_full_batch_neg_means_ms": gpu_run_neg_means_ms,
            "six_run_mean_ms": c17_mean,
            "six_run_sample_sd_ms": sample_sd(gpu_run_neg_means_ms),
            "ci95_low_ms": c17_low,
            "ci95_high_ms": c17_high,
            "A_gate_passed": c17_a,
        },
        "warnings": warnings,
        "checks": checks,
    }
    analysis_dir = root / "analysis"
    write_csv(
        analysis_dir / "run_level_metrics.csv",
        list(run_rows[0].keys()),
        run_rows,
    )
    write_csv(
        analysis_dir / "paired_metrics.csv",
        list(paired_rows[0].keys()),
        paired_rows,
    )
    write_json(analysis_dir / "summary.json", summary)
    write_json(analysis_dir / "checks.json", {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
    })
    return summary


def self_test() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert abs(population_sd([1.0, 2.0, 3.0]) - math.sqrt(2 / 3)) < 1e-12
    assert abs(sample_sd([1.0, 2.0, 3.0]) - 1.0) < 1e-12
    gm, low, high = geometric_t_ci([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert gm == low == high == 2.0
    center, low, high = arithmetic_t_ci([3.0] * 6)
    assert center == low == high == 3.0
    fixture = [
        {"is_partial": "False", "batch_size_actual": "5000", "value": 1},
        {"is_partial": "True", "batch_size_actual": "2115", "value": 100},
    ]
    selected = [
        row for row in fixture
        if row["is_partial"].lower() == "false"
        and int(row["batch_size_actual"]) == BATCH_SIZE
    ]
    assert [row["value"] for row in selected] == [1]
    print("C1-R1 analysis self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/results/c1_r1_combined_rerun"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    summary = analyze(args.root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
