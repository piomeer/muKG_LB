#!/usr/bin/env python3
"""Deterministic, read-only audit of the C3 offline cost-model evidence.

The audit intentionally does not import experiment drivers, access CUDA, train,
or use the network.  It reconstructs statistics from existing CSV/Markdown
artifacts and inspects source provenance with AST/text checks.  All derived
outputs omit wall-clock timestamps so repeated runs are byte-identical.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AUDIT_VERSION = "1.0"
DEFAULT_OUTPUT = "output/results/evidence_audit_part4"

SOURCE_PATHS = [
    "docs/evidence_audit_part1_claim_inventory.md",
    "docs/phase_x_x0_research_freeze.md",
    "docs/cost_model.md",
    "scripts/validate_weight_assumption.py",
    "scripts/analyze_weight_validation.py",
    "scripts/fit_cost_model.py",
    "scripts/generate_validation_report.py",
    "src/py/load/features.py",
    "src/py/load/cost_model.py",
    "src/py/data/FB15K237/train2id.txt",
    "src/py/data/FB15K237/entity2id.txt",
    "output/results/weight_validation.md",
    "output/results/cost_model_data.md",
    "output/results/cost_model_summary.md",
    "output/results/phase10_step2_5/validation_results.md",
    "output/results/runtime_attribution/runtime_attribution.csv",
    "output/results/runtime_attribution/attribution_report.md",
    "output/results/runtime_attribution/attribution_interpretation.md",
    "output/results/gpu_cost_model/benchmark.csv",
    "output/results/gpu_cost_model/benchmark.md",
    "output/results/entity_features.npz",
    "output/results/cost_table.npy",
    "output/results/hub_analysis.md",
    "scripts/validate_b1_correlation.py",
    "scripts/plot_corrected_B_correlation.py",
]

METRIC_FIELDS = [
    "metric_id",
    "claim_id",
    "phase",
    "configuration",
    "statistic",
    "value",
    "unit",
    "n",
    "filter",
    "source_paths",
    "derivation",
    "paper_use",
]

LINEAGE_FIELDS = [
    "node",
    "variable",
    "definition",
    "producer",
    "consumer",
    "unit",
    "sampling_unit",
    "dataset_scope",
    "leakage_status",
    "provenance_status",
    "audit_note",
]

CLAIM_FIELDS = [
    "claim_id",
    "original_claim_id",
    "grade",
    "disposition",
    "evidence_chain",
    "paper_safe_wording",
    "upgrade_condition",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_python(path: Path) -> ast.Module:
    return ast.parse(read_text(path), filename=path.as_posix())


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read comma-separated artifacts and reject Markdown separator rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clean: list[dict[str, str]] = []
    for row in rows:
        values = list(row.values())
        if not values or all((value or "").strip() == "---" for value in values):
            continue
        clean.append({key: (value or "").strip() for key, value in row.items()})
    return clean


def pearson_r(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size != ya.size or xa.size < 2:
        return float("nan")
    x0 = xa - xa.mean()
    y0 = ya - ya.mean()
    denominator = float(np.sqrt(np.dot(x0, x0) * np.dot(y0, y0)))
    return float(np.dot(x0, y0) / denominator) if denominator else float("nan")


def r_squared(x: Iterable[float], y: Iterable[float]) -> float:
    value = pearson_r(x, y)
    return float(value * value)


def ols_r_squared(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size < 2 or xa.size != ya.size:
        return float("nan")
    design = np.column_stack([np.ones(xa.size), xa])
    beta, *_ = np.linalg.lstsq(design, ya, rcond=None)
    fitted = design @ beta
    total = float(np.sum((ya - ya.mean()) ** 2))
    return float(1.0 - np.sum((ya - fitted) ** 2) / total) if total else float("nan")


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    return float(np.mean(np.abs(np.asarray(list(actual)) - np.asarray(list(predicted)))))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    delta = np.asarray(list(actual)) - np.asarray(list(predicted))
    return float(np.sqrt(np.mean(delta * delta)))


def filter_complete_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not bool(row.get("is_partial", False))
        and int(row.get("batch_size_actual", 5000)) == 5000
    ]


def linear_crossover_n(n1: float, d1: float, n2: float, d2: float) -> float:
    """Linear interpolation for d(n)=cpu_time-gpu_time crossing zero."""
    if d1 == d2:
        return float("nan")
    return float(n1 + (0.0 - d1) * (n2 - n1) / (d2 - d1))


def run_cpu_fixtures(repo_root: Path | None = None) -> dict[str, Any]:
    del repo_root
    x = [-3.0, -1.0, 1.0, 3.0]
    y = [-3.0, 1.0, -1.0, 3.0]
    rows = [
        {"batch_size_actual": 5000, "is_partial": False, "time_ms": 1.2},
        {"batch_size_actual": 5000, "is_partial": False, "time_ms": 2.3},
        {"batch_size_actual": 5000, "is_partial": False, "time_ms": 3.4567},
        {"batch_size_actual": 2500, "is_partial": True, "time_ms": 4.56},
    ]
    rounded_rows = [
        row for row in rows if abs(row["time_ms"] - round(row["time_ms"], 1)) < 1e-12
    ]
    return {
        "pearson_r": pearson_r(x, y),
        "r_squared": r_squared(x, y),
        "ols_r_squared": ols_r_squared(x, y),
        "full_rows": len(rows),
        "complete_rows": len(filter_complete_rows(rows)),
        "rounded_rows": len(rounded_rows),
        "crossover_linear_n": linear_crossover_n(150000.0, -1.0, 300000.0, 1.0),
    }


def _metric(
    metric_id: str,
    claim_id: str,
    phase: str,
    configuration: str,
    statistic: str,
    value: float,
    unit: str,
    n: int | str,
    row_filter: str,
    source_paths: str,
    derivation: str,
    paper_use: str,
) -> dict[str, str]:
    return {
        "metric_id": metric_id,
        "claim_id": claim_id,
        "phase": phase,
        "configuration": configuration,
        "statistic": statistic,
        "value": f"{value:.12g}",
        "unit": unit,
        "n": str(n),
        "filter": row_filter,
        "source_paths": source_paths,
        "derivation": derivation,
        "paper_use": paper_use,
    }


def _runtime_metrics(repo_root: Path) -> tuple[list[dict[str, str]], int]:
    path = repo_root / "output/results/runtime_attribution/runtime_attribution.csv"
    rows = read_rows(path)
    metrics: list[dict[str, str]] = []
    cbp = [row for row in rows if row["config"] == "CBP"]
    max_idx = max(int(row["batch_idx"]) for row in cbp)
    filters = {
        "all": cbp,
        "exclude_first_batch": [row for row in cbp if int(row["batch_idx"]) != 0],
        "exclude_last_short_batch": [row for row in cbp if int(row["batch_idx"]) != max_idx],
        "exclude_first_and_last": [
            row for row in cbp if int(row["batch_idx"]) not in {0, max_idx}
        ],
    }
    source = "output/results/runtime_attribution/runtime_attribution.csv"
    for name, subset in filters.items():
        weights = [float(row["batch_weight"]) for row in subset]
        times = [float(row["neg_sampling_time"]) for row in subset]
        metrics.append(
            _metric(
                f"c3_2_cbp_weight_neg_r_{name}",
                "C3.2",
                "Phase 6",
                "CBP",
                "pearson_r",
                pearson_r(weights, times),
                "unitless",
                len(subset),
                name,
                source,
                "Pearson correlation(batch_weight, neg_sampling_time)",
                "descriptive-only",
            )
        )
    return metrics, len(rows)


def _build_metrics(repo_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    metrics: list[dict[str, str]] = []
    weight_rows = read_rows(repo_root / "output/results/weight_validation.md")
    candidate = [float(row["avg_candidate_size"]) for row in weight_rows]
    times = [float(row["actual_time_ms"]) for row in weight_rows]
    theoretical = [float(row["theoretical_weight"]) for row in weight_rows]
    retries = [float(row["avg_retry"]) for row in weight_rows]
    metrics.extend(
        [
            _metric("c3_1_r_candidate_size_time", "C3.1-R1", "Phase 5.5", "synthetic", "pearson_r", pearson_r(candidate, times), "unitless", len(weight_rows), "separator_rows_removed", "output/results/weight_validation.md", "candidate_size vs actual_time_ms", "synthetic-only"),
            _metric("c3_1_r2_candidate_size_time", "C3.1-R1", "Phase 5.5", "synthetic", "r_squared", r_squared(candidate, times), "unitless", len(weight_rows), "separator_rows_removed", "output/results/weight_validation.md", "Pearson r squared; not an independent OLS fit", "synthetic-only"),
            _metric("c3_1_ols_r2_candidate_size_time", "C3.1-R1", "Phase 5.5", "synthetic", "ols_r_squared", ols_r_squared(candidate, times), "unitless", len(weight_rows), "separator_rows_removed", "output/results/weight_validation.md", "OLS with intercept on same synthetic rows", "synthetic-only"),
            _metric("c3_1_r_theoretical_weight_time", "C3.1-R1", "Phase 5.5", "synthetic", "pearson_r", pearson_r(theoretical, times), "unitless", len(weight_rows), "separator_rows_removed", "output/results/weight_validation.md", "theoretical_weight vs actual_time_ms", "historical-diagnostic"),
            _metric("c3_1_r_avg_retry_time", "C3.1-R1", "Phase 5.5", "synthetic", "pearson_r", pearson_r(retries, times), "unitless", len(weight_rows), "separator_rows_removed", "output/results/weight_validation.md", "avg_retry vs actual_time_ms", "historical-diagnostic"),
        ]
    )

    cost_rows = read_rows(repo_root / "output/results/cost_model_data.md")
    metrics.append(_metric("c3_1_cost_probe_rows", "C3.1-L", "Phase 5", "probe", "numeric_rows", len(cost_rows), "rows", len(cost_rows), "separator_rows_removed", "output/results/cost_model_data.md", "CSV rows after separator removal", "lineage-only"))

    runtime_metrics, runtime_count = _runtime_metrics(repo_root)
    metrics.extend(runtime_metrics)

    bench_rows = read_rows(repo_root / "output/results/gpu_cost_model/benchmark.csv")
    differences = [float(row["cpu_time_ms"]) - float(row["gpu_total_ms"]) for row in bench_rows]
    crossover = float("nan")
    for left, right, d_left, d_right in zip(bench_rows, bench_rows[1:], differences, differences[1:]):
        if d_left <= 0 < d_right:
            crossover = linear_crossover_n(float(left["N"]), d_left, float(right["N"]), d_right)
            break
    metrics.append(_metric("c3_4_microbenchmark_crossover_n", "C3.4", "Phase 7 Step 3", "CPU-vs-GPU", "linear_interpolated_crossover", crossover, "samples", len(bench_rows), "adjacent measured points", "output/results/gpu_cost_model/benchmark.csv", "linear interpolation of cpu_time_ms - gpu_total_ms", "context-only"))

    feature_path = repo_root / "output/results/entity_features.npz"
    cost_path = repo_root / "output/results/cost_table.npy"
    with np.load(feature_path) as feature_data:
        candidate_array = feature_data["candidate_size"]
    cost_array = np.load(cost_path)
    metrics.extend(
        [
            _metric("c3_6_feature_entries", "C3.6", "artifact", "entity_features", "entry_count", len(candidate_array), "entries", len(candidate_array), "all", "output/results/entity_features.npz", "candidate_size array length", "implementation-only"),
            _metric("c3_6_cost_table_entries", "C3.6", "artifact", "cost_table", "entry_count", len(cost_array), "entries", len(cost_array), "all", "output/results/cost_table.npy", "cost_table array length", "implementation-only"),
            _metric("c3_6_cost_table_data_bytes", "C3.6", "artifact", "cost_table", "data_bytes", cost_array.nbytes, "bytes", len(cost_array), "all", "output/results/cost_table.npy", "float32 ndarray nbytes, excluding .npy header", "implementation-only"),
        ]
    )
    facts = {
        "weight_validation_numeric_rows": len(weight_rows),
        "cost_model_probe_numeric_rows": len(cost_rows),
        "runtime_attribution_rows": runtime_count,
        "gpu_microbenchmark_rows": len(bench_rows),
        "runtime_attribution_cbp_rows": len([row for row in read_rows(repo_root / "output/results/runtime_attribution/runtime_attribution.csv") if row["config"] == "CBP"]),
        "gpu_crossover_samples": crossover,
        "feature_candidate_size_shape": list(candidate_array.shape),
        "feature_candidate_size_dtype": str(candidate_array.dtype),
        "cost_table_shape": list(cost_array.shape),
        "cost_table_dtype": str(cost_array.dtype),
        "cost_table_data_bytes": int(cost_array.nbytes),
    }
    return metrics, facts


def _source_manifest(repo_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        path = repo_root / relative
        entry: dict[str, Any] = {"path": relative, "exists": path.is_file()}
        if path.is_file():
            entry.update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
        else:
            entry.update({"sha256": None, "bytes": None})
        manifest.append(entry)
    return manifest


def _source_text(repo_root: Path, relative: str) -> str:
    path = repo_root / relative
    return read_text(path) if path.is_file() else ""


def _lineage_rows() -> list[dict[str, str]]:
    rows = [
        ["triples", "train2id", "FB15k-237 training triples", "scripts/fit_cost_model.py", "FeatureExtractor", "triples", "triple", "historical full train; split hash absent", "unknown", "missing dataset/split metadata", "Current probe loads train2id.txt directly."],
        ["FeatureExtractor", "candidate_size", "union of head_to_tails[e] and tail_to_heads[e]", "src/py/load/features.py", "CostModel", "count", "entity", "cache scope not recorded", "relation-type mismatch", "cache provenance absent", "Relation id is not used in neighbor union."],
        ["CostModel", "cost_table", "min(max_try, 1/(1-neg_num/candidate_size))*b3_const", "src/py/load/cost_model.py", "Scheduler", "declared ms", "entity", "depends on feature cache", "formula not measured against target", "deterministic construction only", "Formula is an implementation mapping, not validated prediction."],
        ["Scheduler", "batch_weight", "sum of unique entity costs in a batch", "src/py/load/cost_model.py", "BatchProvider", "declared ms", "batch", "historical runtime layout", "target unit mismatch unresolved", "partial", "Unique-entity aggregation is not the same as measured sampler work."],
        ["Runtime", "neg_sampling_time", "measured negative-sampling timer", "output/results/runtime_attribution/runtime_attribution.csv", "C3.2", "ms", "batch", "single Phase 6 epoch", "no out-of-sample split", "rounded/limited", "Descriptive association only."],
    ]
    return [dict(zip(LINEAGE_FIELDS, row)) for row in rows]


def _audit_checks(repo_root: Path, facts: dict[str, Any]) -> dict[str, Any]:
    feature_source = _source_text(repo_root, "src/py/load/features.py")
    validation_source = _source_text(repo_root, "scripts/validate_weight_assumption.py")
    phase10_source = _source_text(repo_root, "scripts/generate_validation_report.py") + _source_text(repo_root, "output/results/phase10_step2_5/validation_results.md")
    cost_source = _source_text(repo_root, "src/py/load/cost_model.py")
    hub_source = _source_text(repo_root, "scripts/plot_corrected_B_correlation.py") + _source_text(repo_root, "output/results/hub_analysis.md")
    manifest = _source_manifest(repo_root)
    hub_rows = read_rows(repo_root / "output/results/hub_analysis.md")
    hub_values = {row.get("hub_entity_count", "") for row in hub_rows}
    attribution_rows = read_rows(repo_root / "output/results/runtime_attribution/runtime_attribution.csv")
    return {
        "source_paths_all_present": all(bool(row["exists"]) for row in manifest),
        "synthetic_candidate_construction": "range(min(c_size" in validation_source and ("d*1.5" in validation_source.replace(" ", "") or "degree*1.5" in validation_source.replace(" ", "")),
        "phase10_target_is_deterministic_cost_table": "cost_table" in phase10_source and "pre-computed expected cost" in phase10_source,
        "candidate_size_ignores_relation_type": "head_to_tails[h]" in feature_source and "tail_to_heads[t]" in feature_source and "head_to_tails.get(e, set()) | tail_to_heads.get(e, set())" in feature_source,
        "feature_cache_lacks_provenance_metadata": "dataset_hash" not in feature_source and "split_hash" not in feature_source and "config_hash" not in feature_source,
        "cost_model_is_deterministic_formula": "def build_cost_table" in cost_source and "np.random" not in cost_source,
        "hub_count_two_value_warning_present": len(hub_values) == 2 and hub_values == {"4230", "6000"} and ("only 2" in hub_source or "2 个不同值" in hub_source),
        "weight_validation_numeric_rows": facts["weight_validation_numeric_rows"],
        "cost_model_probe_numeric_rows": facts["cost_model_probe_numeric_rows"],
        "runtime_attribution_rows": facts["runtime_attribution_rows"],
        "gpu_microbenchmark_rows": facts["gpu_microbenchmark_rows"],
        "runtime_attribution_single_epoch_warning": len(attribution_rows) == 546 and not {"seed", "run", "repeat"}.intersection(attribution_rows[0]),
        "runtime_attribution_raw_repeat_artifact_present": False,
        "phase10_circularity_warning": True,
        "partial_batch_metadata_present_in_runtime_csv": False,
        "rounded_runtime_artifact": True,
        "c3_predictive_gate_passed": False,
    }


def _claim_verdicts() -> list[dict[str, str]]:
    return [
        {"claim_id": "C3.1-L", "original_claim_id": "C3.1", "grade": "D", "disposition": "RETRACTED", "evidence_chain": "Part1 inventory → legacy R²=0.9008/455 wording → Phase5.5 source and artifact mismatch", "paper_safe_wording": "Do not report R²=0.9008, 455 measured observations, or 90% explained variance.", "upgrade_condition": "Reconstruct the measured target and sampling unit before any new claim."},
        {"claim_id": "C3.1-R1", "original_claim_id": "C3.1", "grade": "C", "disposition": "SYNTHETIC_ONLY", "evidence_chain": "validate_weight_assumption.py → weight_validation.md (400 numeric rows) → independent r/r² recomputation", "paper_safe_wording": "In a synthetic validation, candidate_size was descriptively associated with measured time; this is not out-of-sample predictive validation.", "upgrade_condition": "Use the frozen CPU sampler, real candidate provenance, held-out complete batches, and independent seed uncertainty."},
        {"claim_id": "C3.2", "original_claim_id": "C3.2", "grade": "C", "disposition": "DESCRIPTIVE_HOLD", "evidence_chain": "runtime_attribution.py → runtime_attribution.csv (546 rows) → CBP r sensitivity recomputation", "paper_safe_wording": "Within one Phase6 CBP layout, batch weight and measured negative-sampling time show a descriptive association; no causal or predictive interpretation is supported.", "upgrade_condition": "Independent seed-grouped runs, unrounded measurements, complete-batch estimand, and held-out evaluation."},
        {"claim_id": "C3.3", "original_claim_id": "C3.3", "grade": "A", "disposition": "IMPLEMENTATION_FACT", "evidence_chain": "features.py + cost_model.py AST → deterministic CPU fixture → cost_table artifact", "paper_safe_wording": "Given a supplied feature array and fixed constants, the prototype deterministically constructs a lookup table without an online profiler.", "upgrade_condition": "None for implementation fact; predictive validity requires the separate batch-level audit protocol."},
        {"claim_id": "C3.4", "original_claim_id": "C3.4", "grade": "C", "disposition": "TRANSFER_TO_C1_CONTEXT", "evidence_chain": "gpu_cost_microbench.py → benchmark.csv (7 aggregate points) → crossover recomputation", "paper_safe_wording": "The stored microbenchmark provides contextual CPU/GPU timing crossover evidence under its stated, non-equivalent operations.", "upgrade_condition": "Raw repeated timings and matched operations if the crossover is promoted beyond design context."},
        {"claim_id": "C3.5", "original_claim_id": "C3.5", "grade": "D", "disposition": "RETRACTED", "evidence_chain": "historical hub_count artifact → corrected analysis warning → two-value/short-batch confounding", "paper_safe_wording": "Do not use hub_entity_count correlation as cost-model evidence.", "upgrade_condition": "None; replace with a separately audited variable and estimand."},
        {"claim_id": "C3.6", "original_claim_id": "C3.6", "grade": "A", "disposition": "IMPLEMENTATION_FACT", "evidence_chain": "cost_model.py array subscript → cost_table.npy dtype/shape/nbytes inspection", "paper_safe_wording": "The current artifact contains 14,505 float32 entries (58,020 data bytes, excluding the .npy header) and supports constant-time array lookup.", "upgrade_condition": "None for storage/access fact; dataset and split provenance must be added before interpreting the table empirically."},
    ]


def build_audit(repo_root: Path) -> dict[str, Any]:
    metrics, facts = _build_metrics(repo_root)
    checks = _audit_checks(repo_root, facts)
    return {
        "audit_version": AUDIT_VERSION,
        "estimand": {
            "unit": "complete batch",
            "outcome": "negative-sampling time from sampler call to target-device tensor availability",
            "primary_target": "held-out batch prediction",
            "current_evidence_status": "not established",
        },
        "source_manifest": {"sources": _source_manifest(repo_root)},
        "variable_lineage": _lineage_rows(),
        "recomputed_metrics": metrics,
        "claim_verdicts": _claim_verdicts(),
        "audit_checks": checks,
        "facts": facts,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_outputs(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(repo_root)
    paths = {
        "source_manifest": output_dir / "source_manifest.json",
        "variable_lineage": output_dir / "variable_lineage.csv",
        "recomputed_metrics": output_dir / "recomputed_metrics.csv",
        "claim_verdicts": output_dir / "claim_verdicts.csv",
        "audit_checks": output_dir / "audit_checks.json",
    }
    paths["source_manifest"].write_text(json.dumps(audit["source_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["audit_checks"].write_text(json.dumps({"audit_version": AUDIT_VERSION, **audit["audit_checks"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["variable_lineage"], LINEAGE_FIELDS, audit["variable_lineage"])
    _write_csv(paths["recomputed_metrics"], METRIC_FIELDS, audit["recomputed_metrics"])
    _write_csv(paths["claim_verdicts"], CLAIM_FIELDS, audit["claim_verdicts"])
    return paths


def render_report(audit: dict[str, Any]) -> str:
    metrics = {row["metric_id"]: row["value"] for row in audit["recomputed_metrics"]}
    lines = [
        "# Phase X Part 4 — C3 Cost Model Evidence Audit",
        "",
        "This report is generated from existing artifacts only. It does not run training, GPU code, CPU experiments, network retrieval, or runtime modifications.",
        "",
        "## Frozen estimand",
        "",
        "The candidate rescue estimand is held-out prediction of complete-batch negative-sampling time under the frozen CPU baseline sampler. Current artifacts do not establish this estimand.",
        "",
        "## Claim verdicts",
        "",
    ]
    for claim in audit["claim_verdicts"]:
        lines.extend([
            f"### {claim['claim_id']} — {claim['grade']} ({claim['disposition']})",
            "",
            f"Evidence: {claim['evidence_chain']}",
            "",
            f"Paper-safe wording: {claim['paper_safe_wording']}",
            "",
            f"Upgrade condition: {claim['upgrade_condition']}",
            "",
        ])
    lines.extend([
        "## Recomputed values",
        "",
        f"- Synthetic `candidate_size` Pearson r: {metrics.get('c3_1_r_candidate_size_time')} (400 numeric rows).",
        f"- Synthetic `candidate_size` r²: {metrics.get('c3_1_r2_candidate_size_time')}; this is not an independently established predictive R².",
        f"- Synthetic theoretical-weight Pearson r: {metrics.get('c3_1_r_theoretical_weight_time')}.",
        f"- CBP runtime attribution rows: {audit['facts']['runtime_attribution_cbp_rows']}; the report is single-layout and descriptive.",
        f"- CBP weight/time r sensitivity: all={metrics.get('c3_2_cbp_weight_neg_r_all')}, exclude-first={metrics.get('c3_2_cbp_weight_neg_r_exclude_first_batch')}, exclude-last={metrics.get('c3_2_cbp_weight_neg_r_exclude_last_short_batch')}, exclude-both={metrics.get('c3_2_cbp_weight_neg_r_exclude_first_and_last')}.",
        f"- Stored GPU microbenchmark crossover interpolation: {audit['facts']['gpu_crossover_samples']} samples; contextual only.",
        f"- Cost-table artifact: {audit['facts']['cost_table_shape']} {audit['facts']['cost_table_dtype']}, {audit['facts']['cost_table_data_bytes']} data bytes excluding header.",
        "",
        "## X7 propagation corrections",
        "",
        "Remove or quarantine legacy `R²=0.9008`, `90% explained`, `455 sampled observations`, relation-type candidate-pool wording, and any claim that the Phase 10 bootstrap validates measured runtime prediction.",
        "",
        "## X6.5 minimum rescue protocol",
        "",
        "Use unrounded complete-batch observations, frozen feature/cache provenance, independent seed groups, no within-run random split, held-out R² as primary metric, MAE/RMSE/Pearson/Spearman as secondary metrics, and intercept-only/simple-feature baselines. At least three independent repeats are required; ten seed groups are recommended.",
        "",
        "## Gate result",
        "",
        "Current predictive C3 does not pass the X0 A/B contribution gate. C3.3 and C3.6 remain implementation facts. Any rescue experiment requires X5.5 approval and belongs to X6.5.",
        "",
    ])
    return "\n".join(lines)


def run_self_test() -> None:
    expected = run_cpu_fixtures()
    assert math.isclose(expected["pearson_r"], 0.8, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(expected["r_squared"], 0.64, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(expected["ols_r_squared"], 0.64, rel_tol=0, abs_tol=1e-12)
    assert expected["complete_rows"] == 3
    assert expected["rounded_rows"] == 2
    assert math.isclose(expected["crossover_linear_n"], 225000.0, rel_tol=0, abs_tol=1e-9)
    print("C3 audit self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (repo_root / DEFAULT_OUTPUT)
    write_outputs(repo_root, output_dir)
    audit = build_audit(repo_root)
    report_path = repo_root / "docs/evidence_audit_part4_c3_cost_model.md"
    report_path.write_text(render_report(audit), encoding="utf-8")
    print(f"C3 audit outputs written to {output_dir}")
    print(f"C3 audit report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
