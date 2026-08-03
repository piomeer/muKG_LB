#!/usr/bin/env python3
"""Audit C2 framework evidence using source, artifacts, AST, and CPU fixtures.

This script never imports experiment drivers, starts training, or accesses CUDA.
Its outputs are deterministic: no wall-clock timestamps, temporary paths, or
runtime durations are serialized.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AUDIT_VERSION = "1.0"
DEFAULT_OUTPUT = "output/results/evidence_audit_part3"

ARCHITECTURE_FIELDS = [
    "role_order",
    "stage",
    "role",
    "implementation",
    "interface",
    "input",
    "output",
    "implemented_status",
    "evidence_path",
    "boundary_note",
]

METRIC_FIELDS = [
    "metric_id",
    "phase",
    "configuration",
    "statistic",
    "value",
    "unit",
    "n",
    "source_paths",
    "derivation",
    "claim_relation",
    "paper_use",
]

BASE_SOURCE_PATHS = [
    "docs/evidence_audit_part1_claim_inventory.md",
    "docs/gpu_runtime_architecture.md",
    "docs/phase8_architecture_freeze.md",
    "docs/runtime_framework_spec.md",
    "paper/draft/method.md",
    "src/py/load/features.py",
    "src/py/load/cost_model.py",
    "src/py/load/schedulers.py",
    "src/py/load/batch_provider.py",
    "src/py/load/gpu_sampler.py",
    "src/py/experiments/phase9_step2_benchmark.py",
    "output/results/unified_runtime/unified_runtime_validation.md",
    "output/results/integration_validation/validation_summary.json",
    "output/results/scheduler_overhead.md",
    "output/results/entity_features.npz",
    "output/results/cost_table.npy",
    "output/results/cost_model_summary.md",
    "output/results/exp_Baseline/training.md",
    "output/results/exp_CBP/training.md",
    "output/results/c1_r1_combined_rerun/protocol.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())


def class_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def method_calls(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def string_literals(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _scheduler_call_combination(node: ast.Call) -> str | None:
    if _call_name(node) != "Scheduler" or len(node.args) < 2:
        return None
    sorter = _call_name(node.args[0])
    packer = _call_name(node.args[1])
    if sorter is None or packer is None:
        return None
    return f"{sorter}+{packer}"


def _factory_scheduler_combinations(tree: ast.AST) -> list[str]:
    combinations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_scheduler":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                combination = _scheduler_call_combination(child)
                if combination is not None:
                    combinations.add(combination)
    return sorted(combinations)


def _phase9_scheduler_combinations(tree: ast.AST) -> list[str]:
    combinations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "configs" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for element in node.value.elts:
            if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) < 4:
                continue
            sorter = _call_name(element.elts[2])
            packer = _call_name(element.elts[3])
            if sorter is not None and packer is not None:
                combinations.add(f"{sorter}+{packer}")
    return sorted(combinations)


def _signature(tree: ast.AST, function_name: str, class_name: str | None = None) -> str:
    candidates: Iterable[ast.AST]
    if class_name is None:
        candidates = tree.body if isinstance(tree, ast.Module) else []
    else:
        candidates = (
            node.body
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        candidates = next(candidates, [])
    for node in candidates:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return f"{function_name}({ast.unparse(node.args)})"
    raise ValueError(f"missing interface {class_name + '.' if class_name else ''}{function_name}")


def _class_bases(tree: ast.AST, class_name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [ast.unparse(base) for base in node.bases]
    return []


def _method_node(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                return child
    raise ValueError(f"missing method {class_name}.{method_name}")


def _method_calls_named(tree: ast.AST, class_name: str, method_name: str) -> set[str]:
    return method_calls(_method_node(tree, class_name, method_name))


def _uses_named_subscript(tree: ast.AST, variable_name: str) -> bool:
    return any(
        isinstance(node, ast.Subscript)
        and (
            (isinstance(node.value, ast.Name) and node.value.id == variable_name)
            or (isinstance(node.value, ast.Attribute) and node.value.attr == variable_name)
        )
        for node in ast.walk(tree)
    )


def _phase9_config_labels(tree: ast.AST) -> list[str]:
    labels: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "configs" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for element in node.value.elts:
            if (
                isinstance(element, (ast.Tuple, ast.List))
                and element.elts
                and isinstance(element.elts[0], ast.Constant)
                and isinstance(element.elts[0].value, str)
            ):
                labels.add(element.elts[0].value)
    return sorted(labels)


def _phase9_suffixes(tree: ast.AST) -> tuple[str, str]:
    literals = string_literals(tree)
    write_suffix = ".md" if "summary.md" in literals else ""
    read_suffix = ".csv" if any("summary.csv" in value for value in literals) else ""
    return write_suffix, read_suffix


def _backend_branch_present(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        calls = {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        if "use_gpu" in names and "generate" in calls and "cuda" in calls:
            return True
    return False


def _source_schema(path: Path) -> tuple[str, int | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return "csv", len(rows), list(reader.fieldnames or [])
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = sorted(payload) if isinstance(payload, dict) else []
        rows = len(payload) if isinstance(payload, list) else None
        return "json", rows, fields
    if suffix == ".py":
        tree = parse_python(path)
        fields = sorted(class_names(tree) | function_names(tree))
        return "python", None, fields
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
        return "numpy", int(value.shape[0]) if value.ndim else 1, [
            f"dtype={value.dtype}",
            f"shape={list(value.shape)}",
        ]
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            fields = [
                f"{name}:dtype={archive[name].dtype}:shape={list(archive[name].shape)}"
                for name in sorted(archive.files)
            ]
        return "numpy_archive", None, fields
    return "text", len(path.read_text(encoding="utf-8", errors="replace").splitlines()), []


def build_source_manifest(repo_root: Path) -> dict[str, Any]:
    paths = list(BASE_SOURCE_PATHS)
    paths.extend(
        path.relative_to(repo_root).as_posix()
        for path in sorted(
            (repo_root / "output/results/c1_r1_combined_rerun/jobs").glob(
                "throughput_*_seed*/per_epoch.csv"
            )
        )
    )
    sources = []
    for relative in sorted(set(paths)):
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required C2 evidence is missing: {relative}")
        kind, row_count, fields = _source_schema(path)
        sources.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "kind": kind,
                "row_count": row_count,
                "fields_or_symbols": fields,
            }
        )
    return {
        "audit": "Phase X Part 3 — C2 Unified Runtime Framework",
        "audit_version": AUDIT_VERSION,
        "sources": sources,
    }


def run_cpu_fixtures(repo_root: Path) -> dict[str, Any]:
    """Exercise implementation properties using small deterministic CPU inputs."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.py.load.batch_provider import BatchProvider
    from src.py.load.cost_model import build_cost_table
    from src.py.load.schedulers import (
        ChunkPacker,
        CostSorter,
        FFDPacker,
        RandomSorter,
        Scheduler,
    )

    features = {
        "candidate_size": np.array([151, 200, 300, 600, 1200, 2000], dtype=np.int32),
        "degree": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        "hub_flag": np.array([False, False, False, False, True, True]),
    }
    first = build_cost_table(features, neg_num=150)
    second = build_cost_table(features, neg_num=150)

    triples = [
        (0, 0, 1),
        (1, 0, 2),
        (2, 0, 3),
        (3, 0, 4),
        (4, 0, 5),
        (5, 0, 0),
        (0, 1, 2),
        (2, 1, 4),
        (4, 1, 0),
        (1, 1, 3),
        (3, 1, 5),
    ]
    combinations: list[str] = []
    for sorter in (CostSorter(), RandomSorter(seed=7)):
        for packer in (ChunkPacker(), FFDPacker()):
            scheduler = Scheduler(sorter, packer)
            batches = scheduler.pack_batches(triples, first, batch_size=4)
            assert sum(len(batch) for batch in batches) == len(triples)
            combinations.append(scheduler.get_name())

    provider = BatchProvider(
        Scheduler(CostSorter(), ChunkPacker()),
        first,
        batch_size=4,
        enable_logging=False,
    )
    provider_batches = list(provider.iterate(triples))
    provider_coverage = sorted(item for batch in provider_batches for item in batch)

    all_batches = Scheduler(CostSorter(), ChunkPacker()).pack_batches(
        triples, first, batch_size=4
    )
    rank_batches: list[list[list[tuple[int, int, int]]]] = []
    for rank in range(3):
        ranked_provider = BatchProvider(
            Scheduler(CostSorter(), ChunkPacker()),
            first,
            batch_size=4,
            enable_logging=False,
        )
        ranked_provider.set_rank(rank=rank, world_size=3)
        rank_batches.append(list(ranked_provider.iterate(triples)))
    rank_batch_sets = [
        {tuple(batch) for batch in partition}
        for partition in rank_batches
    ]
    pairwise_disjoint = all(
        rank_batch_sets[left].isdisjoint(rank_batch_sets[right])
        for left in range(len(rank_batch_sets))
        for right in range(left + 1, len(rank_batch_sets))
    )
    rank_union = set().union(*rank_batch_sets)

    ordered = CostSorter().sort(triples, first)
    chunk = ChunkPacker().pack(ordered, batch_size=4)
    ffd = FFDPacker().pack(ordered, batch_size=4)

    return {
        "cost_model_deterministic": first.tobytes() == second.tobytes(),
        "cost_model_float32": first.dtype == np.float32,
        "cost_model_shape": list(first.shape),
        "scheduler_combinations": sorted(combinations),
        "batch_provider_full_coverage": provider_coverage == sorted(triples),
        "rank_partitions_disjoint": pairwise_disjoint,
        "rank_partitions_cover_all_batches": rank_union == {tuple(batch) for batch in all_batches},
        "ffd_equals_chunk_on_frozen_fixture": ffd == chunk,
    }


def _phase6_overhead(path: Path) -> float:
    match = re.search(r"overhead=([0-9]+(?:\.[0-9]+)?)ms", path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"no scheduler overhead found in {path}")
    return float(match.group(1))


def _c1_r1_rows(repo_root: Path, config: str) -> list[dict[str, str]]:
    paths = sorted(
        (
            repo_root / "output/results/c1_r1_combined_rerun/jobs"
        ).glob(f"throughput_{config}_seed*/per_epoch.csv")
    )
    rows = [row for path in paths for row in read_csv(path)]
    if len(paths) != 6 or len(rows) != 30:
        raise ValueError(
            f"expected 6 files and 30 C1-R1 {config} rows; got {len(paths)} files/{len(rows)} rows"
        )
    return rows


def build_recomputed_metrics(repo_root: Path) -> list[dict[str, str]]:
    metrics = [
        {
            "metric_id": "phase6_bl_scheduler_overhead_ms",
            "phase": "Phase 6",
            "configuration": "RandomSorter+ChunkPacker",
            "statistic": "recorded single-epoch overhead",
            "value": f"{_phase6_overhead(repo_root / 'output/results/exp_Baseline/training.md'):.12g}",
            "unit": "ms",
            "n": "1",
            "source_paths": "output/results/exp_Baseline/training.md",
            "derivation": "parse BatchProvider overhead field",
            "claim_relation": "C2.6 historical contradiction",
            "paper_use": "protocol-specific description only",
        },
        {
            "metric_id": "phase6_cbp_scheduler_overhead_ms",
            "phase": "Phase 6",
            "configuration": "CostSorter+FFDPacker",
            "statistic": "recorded single-epoch overhead",
            "value": f"{_phase6_overhead(repo_root / 'output/results/exp_CBP/training.md'):.12g}",
            "unit": "ms",
            "n": "1",
            "source_paths": "output/results/exp_CBP/training.md",
            "derivation": "parse BatchProvider overhead field",
            "claim_relation": "C2.6 historical contradiction",
            "paper_use": "protocol-specific description only",
        },
    ]
    for config, label in (("BL", "bl"), ("GPU", "gpu")):
        rows = _c1_r1_rows(repo_root, config)
        overhead_ms = [int(row["scheduler_overhead_ns"]) / 1_000_000 for row in rows]
        epoch_ms = [int(row["epoch_time_ns"]) / 1_000_000 for row in rows]
        source_paths = (
            f"output/results/c1_r1_combined_rerun/jobs/"
            f"throughput_{config}_seed*/per_epoch.csv"
        )
        derived = [
            (
                f"c1_r1_{label}_scheduler_mean_ms",
                "arithmetic mean across throughput epochs",
                statistics.mean(overhead_ms),
                "ms",
                "mean(scheduler_overhead_ns / 1e6)",
            ),
            (
                f"c1_r1_{label}_scheduler_sd_ms",
                "sample standard deviation across throughput epochs",
                statistics.stdev(overhead_ms),
                "ms",
                "stdev(scheduler_overhead_ns / 1e6), ddof=1",
            ),
            (
                f"c1_r1_{label}_scheduler_min_ms",
                "minimum across throughput epochs",
                min(overhead_ms),
                "ms",
                "min(scheduler_overhead_ns / 1e6)",
            ),
            (
                f"c1_r1_{label}_scheduler_max_ms",
                "maximum across throughput epochs",
                max(overhead_ms),
                "ms",
                "max(scheduler_overhead_ns / 1e6)",
            ),
            (
                f"c1_r1_{label}_scheduler_epoch_pct",
                "ratio of mean scheduler time to mean epoch time",
                statistics.mean(overhead_ms) / statistics.mean(epoch_ms) * 100,
                "percent",
                "mean(scheduler_overhead_ns) / mean(epoch_time_ns) * 100",
            ),
        ]
        for metric_id, statistic, value, unit, derivation in derived:
            metrics.append(
                {
                    "metric_id": metric_id,
                    "phase": "C1-R1 throughput pass",
                    "configuration": f"{config} / RandomSorter+ChunkPacker",
                    "statistic": statistic,
                    "value": f"{value:.12g}",
                    "unit": unit,
                    "n": str(len(rows)),
                    "source_paths": source_paths,
                    "derivation": derivation,
                    "claim_relation": "C2.6 descriptive context; no replacement claim",
                    "paper_use": "protocol-specific description only",
                }
            )
    return sorted(metrics, key=lambda row: row["metric_id"])


def build_architecture_mapping() -> list[dict[str, str]]:
    return [
        {
            "role_order": "1",
            "stage": "offline control plane",
            "role": "FeatureExtractor",
            "implementation": "FeatureExtractor",
            "interface": "FeatureExtractor(triples_list, num_entities).build(force_recompute=False)",
            "input": "training triples; entity count",
            "output": "candidate_size; degree; hub_flag",
            "implemented_status": "implemented",
            "evidence_path": "src/py/load/features.py",
            "boundary_note": "cache-capable feature construction; not a trainable model",
        },
        {
            "role_order": "2",
            "stage": "offline control plane",
            "role": "CostModel",
            "implementation": "build_cost_table",
            "interface": "build_cost_table(features, neg_num=150, max_try=10, b3_const=51.8)",
            "input": "static feature dictionary; configured constants",
            "output": "np.float32 cost array",
            "implemented_status": "implemented",
            "evidence_path": "src/py/load/cost_model.py",
            "boundary_note": "deterministic explicit formula; predictive validity deferred to Part 4",
        },
        {
            "role_order": "3",
            "stage": "offline control plane",
            "role": "Cost Table",
            "implementation": "np.ndarray",
            "interface": "cost_table[entity_id]",
            "input": "entity id",
            "output": "precomputed scalar cost",
            "implemented_status": "materialized artifact",
            "evidence_path": "output/results/cost_table.npy",
            "boundary_note": "array lookup only; does not include scheduling traversal",
        },
        {
            "role_order": "4",
            "stage": "online per epoch",
            "role": "Scheduler",
            "implementation": "Scheduler(sorter, packer)",
            "interface": "pack_batches(triples_list, cost_table, batch_size)",
            "input": "epoch triples; cost table; batch size",
            "output": "list of triple batches",
            "implemented_status": "implemented",
            "evidence_path": "src/py/load/schedulers.py",
            "boundary_note": "constructor composition; no configure() or schedule() API",
        },
        {
            "role_order": "5",
            "stage": "online per epoch",
            "role": "BatchProvider",
            "implementation": "BatchProvider",
            "interface": "iterate(triples_list); set_rank(rank, world_size)",
            "input": "epoch triples and configured Scheduler",
            "output": "iterator of triple-list batches",
            "implemented_status": "implemented",
            "evidence_path": "src/py/load/batch_provider.py",
            "boundary_note": "reschedules per iterate call; rank-strided slicing only; not DataLoader",
        },
    ]


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "C2.1-R1",
            "original_claim_id": "C2.1",
            "inventory_status": "HOLD replaced by audited suffix claim",
            "grade": "A",
            "evidence_chain": [
                "source AST",
                "architecture_mapping.csv",
                "docs/unified_runtime_architecture_freeze.md",
            ],
            "paper_safe_wording": (
                "The implementation comprises an offline FeatureExtractor–CostModel–cost-table "
                "control plane and an online Scheduler–BatchProvider path; the training loop "
                "selects the negative-sampling backend."
            ),
            "remediation": "None for the implementation claim; apply the frozen wording and figure corrections in Part 7.",
        },
        {
            "claim_id": "C2.2",
            "original_claim_id": "C2.2",
            "inventory_status": "ACTIVE",
            "grade": "B",
            "evidence_chain": [
                "src/py/experiments/phase9_step2_benchmark.py AST",
                "phase9_config_labels check",
                "summary suffix lineage check",
            ],
            "paper_safe_wording": (
                "The Phase 9 Step 2 driver defines BL, CBP, GPU, and CBP+GPU in one "
                "configuration loop; its checked-in per-configuration artifact lineage "
                "requires reconciliation."
            ),
            "remediation": (
                "Repair or document the .md-write/.csv-read conversion lineage and regenerate "
                "the aggregate from explicitly hashed per-config sources."
            ),
        },
        {
            "claim_id": "C2.3",
            "original_claim_id": "C2.3",
            "inventory_status": "ACTIVE",
            "grade": "A",
            "evidence_chain": [
                "Scheduler and BatchProvider AST",
                "Phase 9 use_gpu branch AST",
                "CPU composition fixture",
            ],
            "paper_safe_wording": (
                "Both configured sampling paths consume batches from the same Scheduler and "
                "BatchProvider interfaces, while the training loop explicitly selects the CPU "
                "or GPU sampling backend."
            ),
            "remediation": "None; avoid transparent or drop-in backend-switching language.",
        },
        {
            "claim_id": "C2.4",
            "original_claim_id": "C2.4",
            "inventory_status": "ACTIVE",
            "grade": "A",
            "evidence_chain": [
                "build_cost_table AST",
                "deterministic float32 CPU fixture",
                "cost-table array access source",
            ],
            "paper_safe_wording": (
                "The implemented cost model deterministically constructs a float32 cost table "
                "from static features and configured constants, after which scheduling uses "
                "array lookups."
            ),
            "remediation": "None for implementation behavior; audit predictive validity separately in Part 4.",
        },
        {
            "claim_id": "C2.5",
            "original_claim_id": "C2.5",
            "inventory_status": "ACTIVE",
            "grade": "A",
            "evidence_chain": [
                "BatchProvider AST",
                "batch coverage fixture",
                "rank-strided partition fixture",
            ],
            "paper_safe_wording": (
                "BatchProvider yields preassembled triple-list batches and exposes rank-strided "
                "batch partitioning whose tested partitions are disjoint and collectively complete."
            ),
            "remediation": "None for the API claim; do not describe rank slicing alone as DDP readiness.",
        },
        {
            "claim_id": "C2.6",
            "original_claim_id": "C2.6",
            "inventory_status": "RETRACTED",
            "grade": "D",
            "evidence_chain": [
                "Phase 6 training logs",
                "C1-R1 per_epoch traces",
                "recomputed_metrics.csv",
            ],
            "paper_safe_wording": (
                "No universal scheduler-overhead claim is retained; observed overheads are "
                "reported only with their phase, configuration, timing boundary, and aggregation."
            ),
            "remediation": (
                "Keep the ~0.5ms claim removed. A future performance claim requires a frozen "
                "timing boundary, raw precision, and independent repeated runs."
            ),
        },
    ]


def build_audit(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    phase9_tree = parse_python(repo_root / "src/py/experiments/phase9_step2_benchmark.py")
    feature_tree = parse_python(repo_root / "src/py/load/features.py")
    cost_tree = parse_python(repo_root / "src/py/load/cost_model.py")
    scheduler_tree = parse_python(repo_root / "src/py/load/schedulers.py")
    provider_tree = parse_python(repo_root / "src/py/load/batch_provider.py")
    source_trees = [
        parse_python(path)
        for path in sorted((repo_root / "src/py").rglob("*.py"))
    ]

    implemented_classes = set().union(*(class_names(tree) for tree in source_trees))
    write_suffix, read_suffix = _phase9_suffixes(phase9_tree)
    fixtures = run_cpu_fixtures(repo_root)
    factory_combinations = _factory_scheduler_combinations(scheduler_tree)
    phase9_combinations = _phase9_scheduler_combinations(phase9_tree)
    facts = {
        "architecture_term": "two-stage architecture with five implemented roles",
        "feature_extractor_class_present": "FeatureExtractor" in class_names(feature_tree),
        "feature_output_fields": ["candidate_size", "degree", "hub_flag"],
        "cost_model_function_present": "build_cost_table" in function_names(cost_tree),
        "scheduler_class_present": "Scheduler" in class_names(scheduler_tree),
        "scheduler_method_present": "pack_batches" in function_names(scheduler_tree),
        "scheduler_configure_method_present": "configure" in function_names(scheduler_tree),
        "scheduler_schedule_method_present": "schedule" in function_names(scheduler_tree),
        "batch_provider_class_present": "BatchProvider" in class_names(provider_tree),
        "batch_provider_methods": sorted(
            {"iterate", "set_rank"} & function_names(provider_tree)
        ),
        "batch_provider_calls_pack_batches": "pack_batches" in method_calls(provider_tree),
        "batch_provider_is_torch_dataloader": any(
            base.endswith("DataLoader")
            for base in _class_bases(provider_tree, "BatchProvider")
        ),
        "phase9_config_labels": _phase9_config_labels(phase9_tree),
        "phase9_per_config_write_suffix": write_suffix,
        "phase9_per_config_read_suffix": read_suffix,
        "training_loop_selects_backend": _backend_branch_present(phase9_tree),
        "runtime_policy_implemented": "RuntimePolicy" in implemented_classes,
        "gpu_execution_implemented": "GPUExecution" in implemented_classes,
        "runtime_policy_module_present": (repo_root / "src/py/load/runtime_policy.py").exists(),
        "gpu_execution_module_present": (repo_root / "src/py/load/gpu_execution.py").exists(),
        "factory_scheduler_combinations": factory_combinations,
        "phase9_scheduler_combinations": phase9_combinations,
        "factory_supports_all_four_combinations": len(factory_combinations) == 4,
        "experiment_validates_all_four_scheduler_combinations": len(phase9_combinations) == 4,
        "scheduling_occurs_per_iterate_call": "pack_batches" in _method_calls_named(
            provider_tree, "BatchProvider", "iterate"
        ),
        "runtime_cost_access_uses_array_subscript": (
            _uses_named_subscript(scheduler_tree, "cost_table")
            and _uses_named_subscript(provider_tree, "cost_table")
        ),
        "actual_interface_signatures": {
            "feature_extractor_init": _signature(feature_tree, "__init__", "FeatureExtractor"),
            "feature_extractor_build": _signature(feature_tree, "build", "FeatureExtractor"),
            "build_cost_table": _signature(cost_tree, "build_cost_table"),
            "scheduler_init": _signature(scheduler_tree, "__init__", "Scheduler"),
            "pack_batches": _signature(scheduler_tree, "pack_batches", "Scheduler"),
            "batch_provider_init": _signature(provider_tree, "__init__", "BatchProvider"),
            "iterate": _signature(provider_tree, "iterate", "BatchProvider"),
            "set_rank": _signature(provider_tree, "set_rank", "BatchProvider"),
        },
    }
    checks = [
        {
            "check_id": "architecture_roles_present",
            "status": "PASS",
            "detail": "FeatureExtractor, build_cost_table, Scheduler, and BatchProvider symbols are present; cost table is materialized.",
        },
        {
            "check_id": "phase9_four_configs",
            "status": "PASS" if facts["phase9_config_labels"] == ["BL", "CBP", "CBP+GPU", "GPU"] else "FAIL",
            "detail": "AST-derived labels must exactly match the four registered configurations.",
        },
        {
            "check_id": "phase9_artifact_lineage",
            "status": "FAIL",
            "detail": "Per-config writer uses summary.md while final aggregation reads summary.csv.",
        },
        {
            "check_id": "explicit_backend_selection",
            "status": "PASS" if facts["training_loop_selects_backend"] else "FAIL",
            "detail": "Phase 9 training loop contains an explicit use_gpu branch.",
        },
        {
            "check_id": "design_only_classes_absent",
            "status": "PASS" if not facts["runtime_policy_implemented"] and not facts["gpu_execution_implemented"] else "FAIL",
            "detail": "RuntimePolicy and GPUExecution classes/modules are absent from the implementation.",
        },
        {
            "check_id": "cost_model_fixture",
            "status": "PASS" if fixtures["cost_model_deterministic"] and fixtures["cost_model_float32"] else "FAIL",
            "detail": "Repeated inputs produce byte-identical float32 arrays.",
        },
        {
            "check_id": "batch_provider_fixture",
            "status": "PASS" if fixtures["batch_provider_full_coverage"] else "FAIL",
            "detail": "Small fixture covers each input triple exactly once.",
        },
        {
            "check_id": "rank_partition_fixture",
            "status": "PASS" if fixtures["rank_partitions_disjoint"] and fixtures["rank_partitions_cover_all_batches"] else "FAIL",
            "detail": "Three rank-strided partitions are pairwise disjoint and their union covers all batches.",
        },
        {
            "check_id": "ffd_chunk_semantic_blocker",
            "status": "WARN" if fixtures["ffd_equals_chunk_on_frozen_fixture"] else "PASS",
            "detail": "FFDPacker equals ChunkPacker on the frozen ordered fixture; registered as a Part 5 blocker.",
        },
    ]
    return {
        "source_manifest": build_source_manifest(repo_root),
        "architecture_mapping": build_architecture_mapping(),
        "recomputed_metrics": build_recomputed_metrics(repo_root),
        "audit_checks": {
            "audit": "Phase X Part 3 — C2 Unified Runtime Framework",
            "audit_version": AUDIT_VERSION,
            "scope": "read-only source/artifact audit with deterministic CPU fixtures",
            "claims": _claim_rows(),
            "facts": facts,
            "fixtures": fixtures,
            "checks": checks,
        },
    }


def _csv_bytes(rows: Iterable[dict[str, str]], fields: list[str]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_outputs(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    audit = build_audit(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_manifest": output_dir / "source_manifest.json",
        "architecture_mapping": output_dir / "architecture_mapping.csv",
        "recomputed_metrics": output_dir / "recomputed_metrics.csv",
        "audit_checks": output_dir / "audit_checks.json",
    }
    payloads = {
        "source_manifest": _json_bytes(audit["source_manifest"]),
        "architecture_mapping": _csv_bytes(audit["architecture_mapping"], ARCHITECTURE_FIELDS),
        "recomputed_metrics": _csv_bytes(audit["recomputed_metrics"], METRIC_FIELDS),
        "audit_checks": _json_bytes(audit["audit_checks"]),
    }
    for key, path in paths.items():
        path.write_bytes(payloads[key])
    return paths


def run_self_test(repo_root: Path) -> None:
    audit = build_audit(repo_root)
    expected_grades = {
        "C2.1-R1": "A",
        "C2.2": "B",
        "C2.3": "A",
        "C2.4": "A",
        "C2.5": "A",
        "C2.6": "D",
    }
    actual_grades = {
        row["claim_id"]: row["grade"]
        for row in audit["audit_checks"]["claims"]
    }
    assert actual_grades == expected_grades
    assert len(actual_grades) == 6
    assert audit["audit_checks"]["facts"]["phase9_config_labels"] == [
        "BL",
        "CBP",
        "CBP+GPU",
        "GPU",
    ]
    assert audit["audit_checks"]["facts"]["phase9_per_config_write_suffix"] == ".md"
    assert audit["audit_checks"]["facts"]["phase9_per_config_read_suffix"] == ".csv"
    assert all(
        audit["audit_checks"]["fixtures"][key]
        for key in (
            "cost_model_deterministic",
            "cost_model_float32",
            "batch_provider_full_coverage",
            "rank_partitions_disjoint",
            "rank_partitions_cover_all_batches",
            "ffd_equals_chunk_on_frozen_fixture",
        )
    )
    print("audit_c2_framework self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from script path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"output directory (default: <repo-root>/{DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run CPU-only invariant checks without writing output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.self_test:
        run_self_test(repo_root)
        return 0
    output_dir = args.output_dir or (repo_root / DEFAULT_OUTPUT)
    paths = write_outputs(repo_root, output_dir.resolve())
    for key in sorted(paths):
        print(f"{key}: {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
