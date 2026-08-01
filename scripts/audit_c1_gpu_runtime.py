#!/usr/bin/env python3
"""Recompute and audit C1 GPU-runtime evidence without running experiments.

The script is intentionally CPU-only and uses the Python standard library. It
reads frozen repository artifacts, validates their schemas and provenance, then
writes deterministic machine-readable audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


AUDIT_VERSION = "1.0"
AUDIT_DATE = "2026-08-01"
DEFAULT_OUTPUT = "output/results/evidence_audit_part2"

METRIC_FIELDS = [
    "claim_id",
    "metric_id",
    "protocol",
    "statistic",
    "unit",
    "n_primary",
    "n_comparator",
    "primary_label",
    "primary_value",
    "comparator_label",
    "comparator_value",
    "ratio_label",
    "ratio_value",
    "rounding_lower",
    "rounding_upper",
    "source_paths",
    "notes",
]


SOURCE_SPECS = [
    {
        "source_id": "phase8_cpu_trace",
        "path": "output/results/unified_runtime/runtime_trace_CPU.csv",
        "kind": "csv",
        "precision": "full_precision_float",
        "required": [
            "epoch",
            "step",
            "neg_time_ms",
            "fwd_time_ms",
            "bwd_time_ms",
            "opt_time_ms",
            "total_step_ms",
        ],
    },
    {
        "source_id": "phase8_gpu_trace",
        "path": "output/results/unified_runtime/runtime_trace_GPU.csv",
        "kind": "csv",
        "precision": "full_precision_float",
        "required": [
            "epoch",
            "step",
            "neg_time_ms",
            "fwd_time_ms",
            "bwd_time_ms",
            "opt_time_ms",
            "total_step_ms",
        ],
    },
    {
        "source_id": "phase8_cpu_epoch",
        "path": "output/results/unified_runtime/epoch_summary_CPU.csv",
        "kind": "csv",
        "precision": "full_precision_float",
        "required": ["epoch", "avg_loss", "total_time_s"],
    },
    {
        "source_id": "phase8_gpu_epoch",
        "path": "output/results/unified_runtime/epoch_summary_GPU.csv",
        "kind": "csv",
        "precision": "full_precision_float",
        "required": ["epoch", "avg_loss", "total_time_s"],
    },
    {
        "source_id": "phase9_step1_results",
        "path": "output/results/phase9_step1/results.csv",
        "kind": "csv",
        "precision": "mixed_full_precision",
        "required": [
            "config",
            "epoch_1_loss",
            "epoch_2_loss",
            "mrr_sample",
            "hits10_sample",
        ],
    },
    {
        "source_id": "phase9_step2_summary",
        "path": "output/results/phase9_step2/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1s_memory_rounded_1MiB",
        "required": [
            "config",
            "final_loss",
            "mrr",
            "hits10",
            "avg_epoch_time_s",
            "gpu_mem_mb",
        ],
    },
    {
        "source_id": "phase9_step2_bl",
        "path": "output/results/phase9_step2/BL/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1s_quality_rounded_0.0001",
        "required": [
            "Epoch",
            "Loss",
            "MRR",
            "Hits@10",
            "Time (s)",
            "GPU Mem (MB)",
        ],
    },
    {
        "source_id": "phase9_step2_gpu",
        "path": "output/results/phase9_step2/GPU/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1s_quality_rounded_0.0001",
        "required": [
            "epoch",
            "avg_loss",
            "mrr",
            "hits10",
            "epoch_time_s",
            "gpu_mem_mb",
        ],
    },
    {
        "source_id": "phase9_step3_bl",
        "path": "output/results/phase9_step3/BL/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1",
        "required": [
            "epoch",
            "avg_loss",
            "mrr",
            "hits10",
            "epoch_time_s",
            "neg_time_mean_ms",
            "neg_time_std_ms",
            "step_time_mean_ms",
            "step_time_std_ms",
        ],
    },
    {
        "source_id": "phase9_step3_gpu",
        "path": "output/results/phase9_step3/GPU/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1",
        "required": [
            "epoch",
            "avg_loss",
            "mrr",
            "hits10",
            "epoch_time_s",
            "neg_time_mean_ms",
            "neg_time_std_ms",
            "step_time_mean_ms",
            "step_time_std_ms",
        ],
    },
    {
        "source_id": "phase9_step3_cbp_gpu",
        "path": "output/results/phase9_step3/CBP+GPU/summary.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1",
        "required": [
            "epoch",
            "avg_loss",
            "mrr",
            "hits10",
            "epoch_time_s",
            "neg_time_mean_ms",
            "neg_time_std_ms",
            "step_time_mean_ms",
            "step_time_std_ms",
        ],
    },
    {
        "source_id": "phase10_gpu_repeats",
        "path": "output/results/phase10_step2_5/gpu_repeats.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1",
        "required": [
            "config",
            "run",
            "seed",
            "final_loss",
            "final_mrr",
            "final_hits10",
            "epoch_time_s",
            "mean_neg_ms",
            "std_neg_ms",
            "mean_step_ms",
            "std_step_ms",
        ],
    },
    {
        "source_id": "phase10_cpu_repeats",
        "path": "output/results/phase10_step2_5/cpu_repeats.csv",
        "kind": "csv",
        "precision": "timing_rounded_0.1",
        "required": [
            "config",
            "run",
            "seed",
            "epoch_time_s",
            "mean_neg_ms",
            "std_neg_ms",
            "mean_step_ms",
            "std_step_ms",
        ],
    },
    {
        "source_id": "phase6_breakdown",
        "path": "output/results/training_time_breakdown.md",
        "kind": "csv",
        "precision": "timing_rounded_0.001ms_percentage_rounded_0.01",
        "required": ["stage", "time_ms", "pct"],
    },
    {
        "source_id": "gpu_sampler_validation",
        "path": "output/results/gpu_sampler/validation.csv",
        "kind": "csv",
        "precision": "full_precision_float",
        "required": ["step", "time_ms"],
    },
    {
        "source_id": "profiling_analyzer",
        "path": "analyze_profiling.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "sampler_validator",
        "path": "src/py/experiments/validate_gpu_sampler.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "phase8_driver",
        "path": "src/py/experiments/run_unified_runtime_validation.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "phase9_step1_driver",
        "path": "src/py/experiments/phase9_step1_alignment.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "phase9_step2_driver",
        "path": "src/py/experiments/phase9_step2_benchmark.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "phase9_step3_driver",
        "path": "src/py/experiments/phase9_step3_ablation.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "phase10_driver",
        "path": "src/py/experiments/phase10_step2_5_validation.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "gpu_sampler",
        "path": "src/py/load/gpu_sampler.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "stale_sampler_validator",
        "path": "src/py/experiments/validate_gpu_sampler_full.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "paper_asset_generator",
        "path": "generate_paper_assets.py",
        "kind": "python",
        "precision": "not_applicable",
    },
    {
        "source_id": "semantic_alignment_report",
        "path": "docs/semantic_alignment_report.md",
        "kind": "markdown",
        "precision": "narrative",
    },
    {
        "source_id": "figure4_gpu_runtime_trace",
        "path": "paper_assets/figures/fig4_gpu_runtime_trace.pdf",
        "kind": "pdf",
        "precision": "derived_figure",
    },
    {
        "source_id": "figure5_benchmark_bars",
        "path": "paper_assets/figures/fig5_benchmark_bars.pdf",
        "kind": "pdf",
        "precision": "derived_figure",
    },
    {
        "source_id": "figure6_ablation_variance",
        "path": "paper_assets/figures/fig6_ablation_variance.pdf",
        "kind": "pdf",
        "precision": "derived_figure",
    },
]


CLAIM_GRADES = [
    {
        "claim_id": "C1.1",
        "grade": "B",
        "paper_disposition": "excluded",
        "reason": (
            "The full-precision trace is recomputable, but the Phase 8 CPU "
            "comparator simultaneously corrupts head and tail and is not the "
            "original Bernoulli/global-collision CPU sampler named by the claim."
        ),
        "paper_safe_wording": (
            "Audit-only: the Phase 8 synthetic CPU validation sampler and the "
            "GPU tail-only sampler recorded different component timings under "
            "non-equivalent sampling procedures."
        ),
        "remediation": (
            "If a component-level speedup is required, benchmark the frozen "
            "original CPU sampler and GPU sampler with unrounded per-call traces, "
            "matched batches, explicit warm-up, and at least three independent runs."
        ),
    },
    {
        "claim_id": "C1.2",
        "grade": "C",
        "paper_disposition": "hold_pending_reexperiment",
        "reason": (
            "The 25.1s and 4.4s summaries reproduce approximately 5.7x, but "
            "Phase 9 is a single run and Phase 10 repeat outputs were rounded "
            "before statistical analysis."
        ),
        "paper_safe_wording": (
            "Audit-only: the stored Phase 9 summaries report mean epoch times "
            "of 25.1s for BL and 4.4s for GPU."
        ),
        "remediation": (
            "Repeat matched BL/GPU runs at three or more seeds, retain full "
            "precision per-epoch timing, freeze warm-up handling, and report "
            "repeat-level speedup uncertainty."
        ),
    },
    {
        "claim_id": "C1.3",
        "grade": "C",
        "paper_disposition": "hold_pending_reexperiment",
        "reason": (
            "The 142.5x ratio is computed from rounded final-epoch within-run "
            "batch standard deviations; per-step traces are absent, the short "
            "final batch is included, and independent-run uncertainty is missing."
        ),
        "paper_safe_wording": (
            "Audit-only: the rounded final-epoch Phase 9 summaries contain "
            "28.5ms for BL and 0.2ms for GPU."
        ),
        "remediation": (
            "Record unrounded per-step timings and batch sizes for matched "
            "independent runs; report full-size-batch dispersion per run and "
            "between-run uncertainty separately."
        ),
    },
    {
        "claim_id": "C1.4",
        "grade": "B",
        "paper_disposition": "excluded",
        "reason": (
            "The Phase 8 step-time trace is recomputable, but it uses the same "
            "synthetic CPU comparator as C1.1 and is not an original-CPU "
            "end-to-end comparison."
        ),
        "paper_safe_wording": (
            "Audit-only: Phase 8 recorded step times for two non-equivalent "
            "validation sampling paths."
        ),
        "remediation": (
            "Use the matched original CPU/GPU repeat protocol required by C1.2 "
            "and retain unrounded step-component traces."
        ),
    },
    {
        "claim_id": "C1.5",
        "grade": "C",
        "paper_disposition": "excluded",
        "reason": (
            "The Phase 9 Step 1 evaluator masks the target entity itself with "
            "infinity, producing meaningless MRR/Hits@10; two epochs cannot "
            "establish non-inferiority between semantically different samplers."
        ),
        "paper_safe_wording": "No quality-equivalence or non-inferiority claim is allowed.",
        "remediation": (
            "Use a corrected filtered evaluator on the official test split, "
            "freeze a non-inferiority margin before running, train to convergence, "
            "and run sufficient independent seeds for the chosen test."
        ),
    },
    {
        "claim_id": "C1.6",
        "grade": "C",
        "paper_disposition": "hold_pending_reexperiment",
        "reason": (
            "The stored memory values are rounded whole-training peak allocations "
            "and do not isolate memory attributable to the sampler."
        ),
        "paper_safe_wording": "No sampler-only VRAM overhead claim is allowed.",
        "remediation": (
            "Measure allocated and reserved VRAM immediately before and after "
            "sampler generation with reset peaks, matched batches, warm-up, and "
            "at least three independent runs."
        ),
    },
    {
        "claim_id": "C1.7",
        "grade": "C",
        "paper_disposition": "hold_pending_reexperiment",
        "reason": (
            "The 2.9--3.4ms range is present in rounded epoch summaries, but "
            "unrounded per-step traces and repeat-level uncertainty are absent."
        ),
        "paper_safe_wording": (
            "Audit-only: Phase 9 rounded epoch summaries report GPU "
            "negative-sampling means between 2.9ms and 3.4ms."
        ),
        "remediation": (
            "Preserve unrounded per-step traces in the matched repeat protocol "
            "for C1.2/C1.3 and summarize steady-state full-size batches."
        ),
    },
    {
        "claim_id": "C1.8",
        "grade": "C",
        "paper_disposition": "excluded",
        "reason": (
            "The values come from five training epochs and a fixed 200-sample "
            "held-out subset of shuffled training triples, not a full official "
            "test/convergence protocol."
        ),
        "paper_safe_wording": "The sampled MRR values are retained only as audit history.",
        "remediation": (
            "Only revisit quality after the corrected full-convergence protocol "
            "defined for C1.5; do not promote the current descriptive values."
        ),
    },
    {
        "claim_id": "C1.9",
        "grade": "C",
        "paper_disposition": "hold_pending_reexperiment",
        "reason": (
            "Phase 6 includes Collate/Tensor Construction while the Phase 8 "
            "per-step trace uses different component boundaries and omits a "
            "comparable Collate field; the pre/post shares cannot be combined."
        ),
        "paper_safe_wording": (
            "Phase 6 and Phase 8 component shares may be described separately "
            "under their own measurement definitions, not as a pre/post shift."
        ),
        "remediation": (
            "Instrument CPU and GPU paths in one driver with identical, exhaustive "
            "timing regions whose shares sum to the same end-to-end denominator."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing output/, src/, and docs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <repo-root>/{DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic in-memory fixture tests and exit.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: Sequence[str] = ()) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        return list(reader), fields


def floats(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ratio_bounds_from_rounding(
    numerator: float, denominator: float, resolution: float
) -> tuple[float, float]:
    half = resolution / 2.0
    numerator_low = numerator - half
    numerator_high = numerator + half
    denominator_low = denominator - half
    denominator_high = denominator + half
    if denominator_low <= 0:
        raise ValueError("Rounded denominator interval crosses zero")
    return numerator_low / denominator_high, numerator_high / denominator_low


def select_phase8_rows(
    rows: Sequence[dict[str, str]],
    *,
    exclude_partial: bool,
    exclude_first_observation: bool,
) -> list[dict[str, str]]:
    selected = list(rows)
    if exclude_partial:
        max_step_by_epoch: dict[int, int] = {}
        for row in selected:
            epoch = int(row["epoch"])
            max_step_by_epoch[epoch] = max(max_step_by_epoch.get(epoch, -1), int(row["step"]))
        selected = [
            row
            for row in selected
            if int(row["step"]) != max_step_by_epoch[int(row["epoch"])]
        ]
    if exclude_first_observation and selected:
        first_key = min((int(row["epoch"]), int(row["step"])) for row in selected)
        selected = [
            row
            for row in selected
            if (int(row["epoch"]), int(row["step"])) != first_key
        ]
    return selected


def number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Non-finite output value: {value}")
    return format(value, ".12g")


def metric_row(
    *,
    claim_id: str,
    metric_id: str,
    protocol: str,
    statistic: str,
    unit: str,
    n_primary: int | None,
    n_comparator: int | None,
    primary_label: str,
    primary_value: float | None,
    comparator_label: str = "",
    comparator_value: float | None = None,
    ratio_label: str = "",
    ratio_value: float | None = None,
    rounding_lower: float | None = None,
    rounding_upper: float | None = None,
    source_paths: Sequence[str] = (),
    notes: str = "",
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "metric_id": metric_id,
        "protocol": protocol,
        "statistic": statistic,
        "unit": unit,
        "n_primary": number(n_primary),
        "n_comparator": number(n_comparator),
        "primary_label": primary_label,
        "primary_value": number(primary_value),
        "comparator_label": comparator_label,
        "comparator_value": number(comparator_value),
        "ratio_label": ratio_label,
        "ratio_value": number(ratio_value),
        "rounding_lower": number(rounding_lower),
        "rounding_upper": number(rounding_upper),
        "source_paths": ";".join(source_paths),
        "notes": notes,
    }


def check(
    check_id: str,
    claim_ids: Sequence[str],
    status: str,
    severity: str,
    evidence: Sequence[str],
    detail: str,
) -> dict[str, Any]:
    if status not in {"PASS", "FAIL", "UNVERIFIABLE"}:
        raise ValueError(f"Invalid check status: {status}")
    return {
        "check_id": check_id,
        "claim_ids": list(claim_ids),
        "status": status,
        "severity": severity,
        "evidence": list(evidence),
        "detail": detail,
    }


def build_manifest(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    manifest: list[dict[str, Any]] = []
    csv_cache: dict[str, list[dict[str, str]]] = {}
    for spec in SOURCE_SPECS:
        path = repo_root / spec["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Required audit source missing: {spec['path']}")
        entry: dict[str, Any] = {
            "source_id": spec["source_id"],
            "path": spec["path"],
            "kind": spec["kind"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "precision": spec["precision"],
        }
        if spec["kind"] == "csv":
            rows, fields = read_csv(path, spec.get("required", ()))
            entry["row_count"] = len(rows)
            entry["columns"] = fields
            csv_cache[spec["source_id"]] = rows
        elif spec["kind"] in {"python", "markdown"}:
            entry["line_count"] = len(path.read_text(encoding="utf-8").splitlines())
        manifest.append(entry)
    return manifest, csv_cache


def add_phase8_metrics(
    metrics: list[dict[str, str]],
    cpu_rows: Sequence[dict[str, str]],
    gpu_rows: Sequence[dict[str, str]],
) -> None:
    views = [
        ("all_batches", False, False, False),
        ("full_size_batches", True, False, False),
        ("full_size_exclude_gpu_first", True, False, True),
    ]
    sources = [
        "output/results/unified_runtime/runtime_trace_CPU.csv",
        "output/results/unified_runtime/runtime_trace_GPU.csv",
    ]
    for view_name, exclude_partial, exclude_cpu_first, exclude_gpu_first in views:
        cpu_view = select_phase8_rows(
            cpu_rows,
            exclude_partial=exclude_partial,
            exclude_first_observation=exclude_cpu_first,
        )
        gpu_view = select_phase8_rows(
            gpu_rows,
            exclude_partial=exclude_partial,
            exclude_first_observation=exclude_gpu_first,
        )
        for claim_id, column, metric_name in [
            ("C1.1", "neg_time_ms", "negative_sampling_time"),
            ("C1.4", "total_step_ms", "recorded_step_time"),
        ]:
            cpu_mean = statistics.mean(floats(cpu_view, column))
            gpu_mean = statistics.mean(floats(gpu_view, column))
            metrics.append(
                metric_row(
                    claim_id=claim_id,
                    metric_id=f"phase8_{metric_name}_{view_name}",
                    protocol=f"Phase 8 {view_name}",
                    statistic="arithmetic_mean",
                    unit="ms",
                    n_primary=len(cpu_view),
                    n_comparator=len(gpu_view),
                    primary_label="synthetic_cpu_validation_sampler",
                    primary_value=cpu_mean,
                    comparator_label="gpu_tail_only_sampler",
                    comparator_value=gpu_mean,
                    ratio_label="synthetic_cpu/gpu",
                    ratio_value=cpu_mean / gpu_mean,
                    source_paths=sources,
                    notes=(
                        "The CPU comparator is not the original Bernoulli/global-"
                        "collision sampler. The final batch in each epoch is partial "
                        "when excluded; only the first GPU observation is treated "
                        "as CUDA warm-up in the steady-state view."
                    ),
                )
            )

    cpu_full = select_phase8_rows(
        cpu_rows, exclude_partial=True, exclude_first_observation=False
    )
    gpu_steady = select_phase8_rows(
        gpu_rows, exclude_partial=True, exclude_first_observation=True
    )
    for label, claim_id, rows in [
        ("synthetic_cpu_validation_sampler", "C1.9", cpu_full),
        ("gpu_tail_only_sampler", "C1.9", gpu_steady),
    ]:
        neg_total = sum(floats(rows, "neg_time_ms"))
        step_total = sum(floats(rows, "total_step_ms"))
        metrics.append(
            metric_row(
                claim_id=claim_id,
                metric_id=f"phase8_recorded_neg_share_{label}",
                protocol="Phase 8 full-size batches; GPU first observation excluded",
                statistic="ratio_of_sums",
                unit="percent",
                n_primary=len(rows),
                n_comparator=None,
                primary_label=label,
                primary_value=100.0 * neg_total / step_total,
                source_paths=sources,
                notes=(
                    "Share uses only the recorded neg/fwd/bwd/opt step denominator; "
                    "it is not directly comparable to the Phase 6 exhaustive profile."
                ),
            )
        )


def build_metrics(
    cache: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    add_phase8_metrics(metrics, cache["phase8_cpu_trace"], cache["phase8_gpu_trace"])

    phase9_bl = cache["phase9_step2_bl"]
    phase9_gpu = cache["phase9_step2_gpu"]
    bl_epoch = floats(phase9_bl, "Time (s)")
    gpu_epoch = floats(phase9_gpu, "epoch_time_s")
    bl_mean = statistics.mean(bl_epoch)
    gpu_mean = statistics.mean(gpu_epoch)
    lower, upper = ratio_bounds_from_rounding(bl_mean, gpu_mean, 0.1)
    metrics.append(
        metric_row(
            claim_id="C1.2",
            metric_id="phase9_step2_epoch_time",
            protocol="Phase 9 Step 2; five epochs per configuration",
            statistic="mean_of_rounded_epoch_observations",
            unit="s",
            n_primary=len(bl_epoch),
            n_comparator=len(gpu_epoch),
            primary_label="BL_original_cpu",
            primary_value=bl_mean,
            comparator_label="GPU_tail_only",
            comparator_value=gpu_mean,
            ratio_label="BL/GPU",
            ratio_value=bl_mean / gpu_mean,
            rounding_lower=lower,
            rounding_upper=upper,
            source_paths=[
                "output/results/phase9_step2/BL/summary.csv",
                "output/results/phase9_step2/GPU/summary.csv",
            ],
            notes=(
                "Each stored epoch time was rounded to 0.1s before this mean; "
                "the interval is the possible ratio range induced by rounding alone."
            ),
        )
    )

    phase10_bl = [
        row for row in cache["phase10_cpu_repeats"] if row["config"] == "BL"
    ]
    phase10_gpu = [
        row for row in cache["phase10_gpu_repeats"] if row["config"] == "GPU"
    ]
    bl_repeat = floats(phase10_bl, "epoch_time_s")
    gpu_repeat = floats(phase10_gpu, "epoch_time_s")
    bl_repeat_mean = statistics.mean(bl_repeat)
    gpu_repeat_mean = statistics.mean(gpu_repeat)
    lower, upper = ratio_bounds_from_rounding(
        bl_repeat_mean, gpu_repeat_mean, 0.1
    )
    metrics.append(
        metric_row(
            claim_id="C1.2",
            metric_id="phase10_repeat_epoch_time_context",
            protocol="Phase 10 Step 2.5 separate rounded repeat summaries",
            statistic="mean_across_rounded_run_summaries",
            unit="s",
            n_primary=len(bl_repeat),
            n_comparator=len(gpu_repeat),
            primary_label="BL_original_cpu",
            primary_value=bl_repeat_mean,
            comparator_label="GPU_tail_only",
            comparator_value=gpu_repeat_mean,
            ratio_label="BL/GPU",
            ratio_value=bl_repeat_mean / gpu_repeat_mean,
            rounding_lower=lower,
            rounding_upper=upper,
            source_paths=[
                "output/results/phase10_step2_5/cpu_repeats.csv",
                "output/results/phase10_step2_5/gpu_repeats.csv",
            ],
            notes=(
                f"BL sample std={sample_std(bl_repeat):.12g}s; "
                f"GPU sample std={sample_std(gpu_repeat):.12g}s. "
                "Full precision was discarded before storage, so zero dispersion "
                "in the GPU column is not evidence of zero runtime variance."
            ),
        )
    )

    step3_bl = cache["phase9_step3_bl"]
    step3_gpu = cache["phase9_step3_gpu"]
    final_bl = max(step3_bl, key=lambda row: int(row["epoch"]))
    final_gpu = max(step3_gpu, key=lambda row: int(row["epoch"]))
    bl_final_std = float(final_bl["neg_time_std_ms"])
    gpu_final_std = float(final_gpu["neg_time_std_ms"])
    lower, upper = ratio_bounds_from_rounding(bl_final_std, gpu_final_std, 0.1)
    metrics.append(
        metric_row(
            claim_id="C1.3",
            metric_id="phase9_step3_final_epoch_neg_std",
            protocol="Phase 9 Step 3 final epoch; all batches including final partial batch",
            statistic="population_std_within_epoch_from_rounded_summary",
            unit="ms",
            n_primary=1,
            n_comparator=1,
            primary_label="BL_original_cpu",
            primary_value=bl_final_std,
            comparator_label="GPU_tail_only",
            comparator_value=gpu_final_std,
            ratio_label="BL/GPU",
            ratio_value=bl_final_std / gpu_final_std,
            rounding_lower=lower,
            rounding_upper=upper,
            source_paths=[
                "output/results/phase9_step3/BL/summary.csv",
                "output/results/phase9_step3/GPU/summary.csv",
            ],
            notes=(
                "The generator uses numpy.std with ddof=0. Per-step data are not "
                "stored, so the partial-batch effect cannot be removed. The wide "
                "ratio interval is caused by the denominator rounded to 0.2ms."
            ),
        )
    )

    bl_epoch_stds = floats(step3_bl, "neg_time_std_ms")
    gpu_epoch_stds = floats(step3_gpu, "neg_time_std_ms")
    metrics.append(
        metric_row(
            claim_id="C1.3",
            metric_id="phase9_step3_mean_epoch_neg_std",
            protocol="Phase 9 Step 3; ten rounded within-epoch dispersions",
            statistic="mean_of_epoch_population_stds",
            unit="ms",
            n_primary=len(bl_epoch_stds),
            n_comparator=len(gpu_epoch_stds),
            primary_label="BL_original_cpu",
            primary_value=statistics.mean(bl_epoch_stds),
            comparator_label="GPU_tail_only",
            comparator_value=statistics.mean(gpu_epoch_stds),
            ratio_label="BL/GPU",
            ratio_value=statistics.mean(bl_epoch_stds) / statistics.mean(gpu_epoch_stds),
            source_paths=[
                "output/results/phase9_step3/BL/summary.csv",
                "output/results/phase9_step3/GPU/summary.csv",
            ],
            notes=(
                "Epochs are repeated measurements within one training run, not "
                "independent experimental repeats."
            ),
        )
    )

    gpu_means = floats(step3_gpu, "neg_time_mean_ms")
    gpu_post_warmup = [
        float(row["neg_time_mean_ms"]) for row in step3_gpu if int(row["epoch"]) > 0
    ]
    for metric_id, values, note in [
        (
            "phase9_step3_gpu_neg_mean_min",
            gpu_means,
            "Minimum over all ten rounded epoch summaries.",
        ),
        (
            "phase9_step3_gpu_neg_mean_max",
            gpu_means,
            "Maximum over all ten rounded epoch summaries.",
        ),
        (
            "phase9_step3_gpu_post_epoch0_min",
            gpu_post_warmup,
            "Minimum after excluding epoch 0 as a warm-up sensitivity view.",
        ),
        (
            "phase9_step3_gpu_post_epoch0_max",
            gpu_post_warmup,
            "Maximum after excluding epoch 0 as a warm-up sensitivity view.",
        ),
    ]:
        value = min(values) if metric_id.endswith("min") else max(values)
        metrics.append(
            metric_row(
                claim_id="C1.7",
                metric_id=metric_id,
                protocol="Phase 9 Step 3 GPU run",
                statistic="minimum" if metric_id.endswith("min") else "maximum",
                unit="ms",
                n_primary=len(values),
                n_comparator=None,
                primary_label="GPU_tail_only",
                primary_value=value,
                source_paths=["output/results/phase9_step3/GPU/summary.csv"],
                notes=note + " Underlying per-step observations are absent.",
            )
        )

    step1 = {row["config"]: row for row in cache["phase9_step1_results"]}
    for config in ["CPU_original", "GPU_v2"]:
        metrics.append(
            metric_row(
                claim_id="C1.5",
                metric_id=f"phase9_step1_bugged_quality_{config.lower()}",
                protocol="Phase 9 Step 1; two epochs; broken filtered evaluator",
                statistic="stored_bugged_evaluation",
                unit="MRR",
                n_primary=1,
                n_comparator=None,
                primary_label=config,
                primary_value=float(step1[config]["mrr_sample"]),
                source_paths=["output/results/phase9_step1/results.csv"],
                notes=(
                    f"Stored Hits@10={step1[config]['hits10_sample']}. The true "
                    "entity is masked to infinity, so this value is not meaningful."
                ),
            )
        )

    final_bl2 = max(phase9_bl, key=lambda row: int(row["Epoch"]))
    final_gpu2 = max(phase9_gpu, key=lambda row: int(row["epoch"]))
    metrics.append(
        metric_row(
            claim_id="C1.8",
            metric_id="phase9_step2_sampled_final_mrr",
            protocol="Phase 9 Step 2; epoch 4; 200 sampled held-out training triples",
            statistic="stored_sampled_MRR",
            unit="MRR",
            n_primary=200,
            n_comparator=200,
            primary_label="BL_original_cpu",
            primary_value=float(final_bl2["MRR"]),
            comparator_label="GPU_tail_only",
            comparator_value=float(final_gpu2["mrr"]),
            source_paths=[
                "output/results/phase9_step2/BL/summary.csv",
                "output/results/phase9_step2/GPU/summary.csv",
            ],
            notes=(
                f"Stored Hits@10: BL={final_bl2['Hits@10']}, "
                f"GPU={final_gpu2['hits10']}. This is not an official full-test "
                "or full-convergence protocol."
            ),
        )
    )

    main_summary = {row["config"]: row for row in cache["phase9_step2_summary"]}
    bl_mem = float(main_summary["BL"]["gpu_mem_mb"])
    gpu_mem = float(main_summary["GPU"]["gpu_mem_mb"])
    metrics.append(
        metric_row(
            claim_id="C1.6",
            metric_id="phase9_step2_whole_training_peak_memory",
            protocol="Phase 9 Step 2 whole-training configuration peak",
            statistic="stored_rounded_peak_memory",
            unit="MiB",
            n_primary=1,
            n_comparator=1,
            primary_label="BL_original_cpu",
            primary_value=bl_mem,
            comparator_label="GPU_tail_only",
            comparator_value=gpu_mem,
            ratio_label="GPU_minus_BL",
            ratio_value=gpu_mem - bl_mem,
            source_paths=["output/results/phase9_step2/summary.csv"],
            notes=(
                "Values are rounded to whole MiB and include model, optimizer, "
                "positive/negative tensors, and sampler allocations; they are not "
                "sampler-only deltas."
            ),
        )
    )

    breakdown = {
        row["stage"]: (float(row["time_ms"]), float(row["pct"]))
        for row in cache["phase6_breakdown"]
        if row["stage"] != "---"
    }
    for stage, (time_ms, pct) in breakdown.items():
        metrics.append(
            metric_row(
                claim_id="C1.9",
                metric_id=f"phase6_profile_{stage.lower().replace(' ', '_')}",
                protocol="Phase 6 training-time breakdown",
                statistic="stored_component_total_and_share",
                unit="percent",
                n_primary=1,
                n_comparator=None,
                primary_label=stage,
                primary_value=pct,
                source_paths=["output/results/training_time_breakdown.md"],
                notes=f"Stored component time={time_ms:.12g}ms.",
            )
        )

    return metrics


def build_checks(
    repo_root: Path,
    manifest: Sequence[dict[str, Any]],
    cache: dict[str, list[dict[str, str]]],
    metrics: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    phase8_code = (repo_root / "src/py/experiments/run_unified_runtime_validation.py").read_text()
    step1_code = (repo_root / "src/py/experiments/phase9_step1_alignment.py").read_text()
    step2_code = (repo_root / "src/py/experiments/phase9_step2_benchmark.py").read_text()
    step3_code = (repo_root / "src/py/experiments/phase9_step3_ablation.py").read_text()
    phase10_code = (repo_root / "src/py/experiments/phase10_step2_5_validation.py").read_text()
    stale_code = (repo_root / "src/py/experiments/validate_gpu_sampler_full.py").read_text()
    sampler_code = (repo_root / "src/py/load/gpu_sampler.py").read_text()

    metric_map = {row["metric_id"]: row for row in metrics}
    phase9_ratio = float(metric_map["phase9_step2_epoch_time"]["ratio_value"])
    phase9_final_ratio = float(
        metric_map["phase9_step3_final_epoch_neg_std"]["ratio_value"]
    )
    all_manifest_valid = all(entry["bytes"] > 0 for entry in manifest)
    cbp_gpu_rows = cache["phase9_step3_cbp_gpu"]
    quality_inconsistent = any(
        float(row["hits10"]) == 1.0 and float(row["mrr"]) < 0.1
        for row in cbp_gpu_rows
    )

    return [
        check(
            "source_manifest_complete",
            [f"C1.{idx}" for idx in range(1, 10)],
            "PASS" if all_manifest_valid else "FAIL",
            "critical",
            [entry["path"] for entry in manifest],
            "All required sources exist, are non-empty, and satisfy their declared CSV schemas.",
        ),
        check(
            "phase8_trace_recomputable",
            ["C1.1", "C1.4"],
            "PASS",
            "info",
            [
                "output/results/unified_runtime/runtime_trace_CPU.csv",
                "output/results/unified_runtime/runtime_trace_GPU.csv",
            ],
            "Full-precision Phase 8 per-step traces support deterministic sensitivity recomputation.",
        ),
        check(
            "phase8_cpu_comparator_identity",
            ["C1.1", "C1.4"],
            "FAIL",
            "critical",
            ["src/py/experiments/run_unified_runtime_validation.py"],
            (
                "The CPU function generates both a replacement head and replacement "
                "tail for every negative and has no all_triples_set/global collision "
                "check. It is not the original CPU sampler named by the claims."
            ),
        ),
        check(
            "phase8_gpu_sampler_semantics",
            ["C1.1", "C1.4"],
            "PASS" if "pos_heads.repeat_interleave" in sampler_code else "FAIL",
            "high",
            ["src/py/load/gpu_sampler.py"],
            "The current GPU sampler is tail-only and filters candidates against batch pos_tails.",
        ),
        check(
            "figure4_comparative_lineage",
            ["C1.1", "C1.4"],
            "FAIL",
            "high",
            [
                "generate_paper_assets.py",
                "paper_assets/figures/fig4_gpu_runtime_trace.pdf",
            ],
            (
                "Fig.4 reads only runtime_trace_GPU.md and plots the GPU component "
                "stack; it contains no CPU series and cannot evidence a CPU/GPU ratio."
            ),
        ),
        check(
            "phase9_epoch_speedup_reproduced",
            ["C1.2"],
            "PASS" if math.isclose(phase9_ratio, 25.1 / 4.4) else "FAIL",
            "info",
            [
                "output/results/phase9_step2/BL/summary.csv",
                "output/results/phase9_step2/GPU/summary.csv",
            ],
            f"The mean of stored rounded epochs reproduces BL/GPU={phase9_ratio:.12g}x.",
        ),
        check(
            "figure5_data_lineage",
            ["C1.2"],
            "FAIL",
            "high",
            [
                "generate_paper_assets.py",
                "paper_assets/figures/fig5_benchmark_bars.pdf",
            ],
            (
                "Fig.5 bar heights are hardcoded as 25.1, 25.3, 4.4, and 4.7 "
                "rather than loaded from the cited summary CSV."
            ),
        ),
        check(
            "phase9_process_stable_seed",
            ["C1.2", "C1.3", "C1.7", "C1.8"],
            "FAIL" if "hash(label)" in step2_code or "hash(label)" in step3_code else "PASS",
            "high",
            [
                "src/py/experiments/phase9_step2_benchmark.py",
                "src/py/experiments/phase9_step3_ablation.py",
            ],
            "Python hash(label) is process-dependent unless PYTHONHASHSEED is externally frozen.",
        ),
        check(
            "phase9_torch_seed",
            ["C1.2", "C1.3", "C1.7", "C1.8"],
            "FAIL"
            if "torch.manual_seed" not in step2_code
            or "torch.manual_seed" not in step3_code
            else "PASS",
            "high",
            [
                "src/py/experiments/phase9_step2_benchmark.py",
                "src/py/experiments/phase9_step3_ablation.py",
            ],
            "The Phase 9 Step 2/3 drivers do not set a Torch seed before model initialization.",
        ),
        check(
            "phase10_unrounded_repeat_precision",
            ["C1.2", "C1.3", "C1.7"],
            "FAIL" if ':.1f' in phase10_code else "PASS",
            "critical",
            [
                "src/py/experiments/phase10_step2_5_validation.py",
                "output/results/phase10_step2_5/gpu_repeats.csv",
                "output/results/phase10_step2_5/cpu_repeats.csv",
            ],
            "Phase 10 writes timing values with one-decimal formatting before downstream statistics.",
        ),
        check(
            "phase9_final_epoch_variance_ratio_reproduced",
            ["C1.3"],
            "PASS" if math.isclose(phase9_final_ratio, 142.5) else "FAIL",
            "info",
            [
                "output/results/phase9_step3/BL/summary.csv",
                "output/results/phase9_step3/GPU/summary.csv",
            ],
            f"The stored rounded final-epoch values reproduce {phase9_final_ratio:.12g}x.",
        ),
        check(
            "figure6_raw_lineage",
            ["C1.3", "C1.7"],
            "UNVERIFIABLE",
            "high",
            [
                "generate_paper_assets.py",
                "paper_assets/figures/fig6_ablation_variance.pdf",
            ],
            (
                "Fig.6 reads only the last row of each rounded Phase 9 Step 3 "
                "summary; absent per-step traces prevent raw-data verification."
            ),
        ),
        check(
            "phase9_variance_raw_trace",
            ["C1.3", "C1.7"],
            "UNVERIFIABLE",
            "critical",
            ["output/results/phase9_step3/"],
            "No Phase 9 Step 3 per-step raw trace exists; only rounded per-epoch summaries are stored.",
        ),
        check(
            "phase9_variance_partial_batch_control",
            ["C1.3"],
            "FAIL" if "np.std(neg_times)" in step3_code else "UNVERIFIABLE",
            "high",
            ["src/py/experiments/phase9_step3_ablation.py"],
            "numpy.std is applied to all batches, including the short final batch; batch size is not stored per observation.",
        ),
        check(
            "phase9_step1_quality_evaluator",
            ["C1.5"],
            "FAIL" if "scores[known_t] = float('inf')" in step1_code else "PASS",
            "critical",
            ["src/py/experiments/phase9_step1_alignment.py"],
            "The evaluator masks the target tail itself, causing the saved MRR/Hits@10 values to be meaningless.",
        ),
        check(
            "phase9_step2_quality_scope",
            ["C1.8"],
            "FAIL",
            "critical",
            ["src/py/experiments/phase9_step2_benchmark.py"],
            "Evaluation uses 200 triples sampled from a shuffled training pool after only five epochs, not an official full-test convergence protocol.",
        ),
        check(
            "phase9_step3_quality_internal_consistency",
            ["C1.5", "C1.8"],
            "FAIL" if quality_inconsistent else "PASS",
            "high",
            ["output/results/phase9_step3/CBP+GPU/summary.csv"],
            "CBP+GPU stores Hits@10=1.0 with MRR below 0.1, an impossible metric combination under the stated rank definitions.",
        ),
        check(
            "sampler_only_memory_measurement",
            ["C1.6"],
            "UNVERIFIABLE",
            "critical",
            [
                "output/results/phase9_step2/summary.csv",
                "output/results/gpu_sampler/validation.csv",
            ],
            (
                "The validation CSV contains timing only. Phase 9 stores rounded "
                "whole-training peaks; sampler-only allocated/reserved deltas are absent."
            ),
        ),
        check(
            "legacy_sampler_validator_interface",
            ["C1.5", "C1.6"],
            "FAIL"
            if "neg_h, neg_t, corrupt_mask = gpu_sampler.generate" in stale_code
            else "PASS",
            "high",
            [
                "src/py/experiments/validate_gpu_sampler_full.py",
                "src/py/load/gpu_sampler.py",
            ],
            "The legacy validator expects three return values and head/tail corruption, while the current sampler returns two tail-only tensors.",
        ),
        check(
            "bottleneck_component_denominator_alignment",
            ["C1.9"],
            "FAIL",
            "critical",
            [
                "output/results/training_time_breakdown.md",
                "output/results/unified_runtime/runtime_trace_GPU.csv",
            ],
            "Phase 6 includes Collate/Tensor Construction, while Phase 8 records neg/fwd/bwd/opt under a different denominator.",
        ),
    ]


def write_outputs(
    output_dir: Path,
    manifest: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, str]],
    checks: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "audit": "Phase X Step 2 — C1 GPU Runtime",
        "audit_version": AUDIT_VERSION,
        "audit_date": AUDIT_DATE,
        "source_count": len(manifest),
        "sources": list(manifest),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "recomputed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    grade_counts: dict[str, int] = {}
    for claim in CLAIM_GRADES:
        grade_counts[claim["grade"]] = grade_counts.get(claim["grade"], 0) + 1
    status_counts: dict[str, int] = {}
    for item in checks:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    checks_payload = {
        "audit": "Phase X Step 2 — C1 GPU Runtime",
        "audit_version": AUDIT_VERSION,
        "audit_date": AUDIT_DATE,
        "strict_a_rule": (
            "A requires unrounded raw observations, a frozen and symmetric "
            "estimand, at least three independent repeats, repeat-level "
            "uncertainty, valid code, and paper wording limited to that protocol."
        ),
        "summary": {
            "claim_count": len(CLAIM_GRADES),
            "grade_counts": grade_counts,
            "check_count": len(checks),
            "check_status_counts": status_counts,
        },
        "claim_grades": CLAIM_GRADES,
        "checks": list(checks),
    }
    (output_dir / "audit_checks.json").write_text(
        json.dumps(checks_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_self_test() -> None:
    fixture = [
        {"epoch": "0", "step": "0", "value": "100"},
        {"epoch": "0", "step": "1", "value": "10"},
        {"epoch": "0", "step": "2", "value": "5"},
        {"epoch": "1", "step": "0", "value": "10"},
        {"epoch": "1", "step": "1", "value": "10"},
        {"epoch": "1", "step": "2", "value": "5"},
    ]
    all_rows = select_phase8_rows(
        fixture, exclude_partial=False, exclude_first_observation=False
    )
    full_rows = select_phase8_rows(
        fixture, exclude_partial=True, exclude_first_observation=False
    )
    steady_rows = select_phase8_rows(
        fixture, exclude_partial=True, exclude_first_observation=True
    )
    assert len(all_rows) == 6
    assert len(full_rows) == 4
    assert len(steady_rows) == 3
    assert statistics.mean(float(row["value"]) for row in full_rows) == 32.5
    assert statistics.mean(float(row["value"]) for row in steady_rows) == 10.0
    assert statistics.pstdev([1.0, 2.0, 3.0]) == math.sqrt(2.0 / 3.0)
    assert sample_std([1.0, 2.0, 3.0]) == 1.0
    low, high = ratio_bounds_from_rounding(25.1, 4.4, 0.1)
    assert math.isclose(low, 25.05 / 4.45)
    assert math.isclose(high, 25.15 / 4.35)
    assert math.isclose(25.1 / 4.4, 5.704545454545454)
    print("audit_c1_gpu_runtime self-test: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / DEFAULT_OUTPUT
    )
    manifest, cache = build_manifest(repo_root)
    metrics = build_metrics(cache)
    checks = build_checks(repo_root, manifest, cache, metrics)
    write_outputs(output_dir, manifest, metrics, checks)
    print(f"C1 evidence audit outputs written to {output_dir}")
    print(f"sources={len(manifest)} metrics={len(metrics)} checks={len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
