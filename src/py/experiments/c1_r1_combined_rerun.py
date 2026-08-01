#!/usr/bin/env python3
"""C1-R1 v1.1 combined rerun for C1.2/C1.3/C1.7.

The controller launches every measured job in a fresh process.  The experiment
is runtime-only: it does not evaluate link-prediction quality and it does not
modify the training or sampler implementations under test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.py.load.batch_provider import BatchProvider
from src.py.load.cost_model import build_cost_table
from src.py.load.gpu_sampler import GPUNegativeSampler
from src.py.load.schedulers import ChunkPacker, RandomSorter, Scheduler


PROTOCOL_ID = "C1-R1-v1.1"
DEFAULT_ROOT = REPO / "output/results/c1_r1_combined_rerun"
DATA_PATH = REPO / "src/py/data/FB15K237/train2id.txt"
FEATURE_PATH = REPO / "output/results/entity_features.npz"
BATCH_SIZE = 5000
NEG_NUM = 150
EPOCHS = 5
SEEDS = tuple(range(42, 48))
EXPECTED_RAW = 272_115
EXPECTED_TRAIN = 267_115
EXPECTED_FULL_BATCHES = 53
EXPECTED_PARTIAL_SIZE = 2_115
CSV_DIALECT = {"lineterminator": "\n"}

EPOCH_FIELDS = [
    "protocol_id", "pass_name", "config", "seed", "epoch",
    "epoch_time_ns", "scheduler_overhead_ns", "num_steps",
    "full_batch_count", "partial_batch_count", "partial_batch_size",
    "training_examples", "avg_loss", "loss_finite",
    "loss_change_from_epoch0", "peak_allocated_bytes", "peak_reserved_bytes",
]
STEP_FIELDS = [
    "protocol_id", "pass_name", "config", "seed", "epoch", "step",
    "batch_size_actual", "is_partial", "is_first_measured_step",
    "zero_grad_ns", "neg_time_ns", "positive_tensor_build_ns",
    "forward_ns", "backward_ns", "optimizer_ns", "component_sum_ns",
    "total_step_ns", "timing_residual_ns", "loss",
]
TELEMETRY_FIELDS = [
    "protocol_id", "config", "seed", "pass_name", "event", "epoch",
    "time_ns", "timestamp", "gpu_index", "name", "uuid", "pstate",
    "clock_graphics_mhz", "clock_memory_mhz", "temperature_c",
    "power_draw_w", "power_limit_w", "utilization_gpu_pct",
    "throttle_active", "thermal_slowdown", "other_compute_processes",
    "raw_gpu_query", "raw_process_query", "query_error",
]


class SimpleTransE(torch.nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int = 400):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(n_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(n_relations, dim)

    def forward(self, heads: torch.Tensor, rels: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        return torch.norm(
            self.ent_embeddings(heads)
            + self.rel_embeddings(rels)
            - self.ent_embeddings(tails),
            p=2,
            dim=-1,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(items: Iterable[tuple[int, int, int]]) -> str:
    digest = hashlib.sha256()
    for h, r, t in items:
        digest.update(f"{h} {r} {t}\n".encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, **CSV_DIALECT)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, **CSV_DIALECT)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_split() -> tuple[list[tuple[int, int, int]],
                          list[tuple[int, int, int]], dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        declared = int(handle.readline().strip())
        disk_rows = [
            tuple(map(int, line.split())) for line in handle if line.strip()
        ]
    # OpenKE train2id.txt stores (head, tail, relation); MuKG and every
    # Phase-9 experiment operate on (head, relation, tail).
    converted = [(head, relation, tail) for head, tail, relation in disk_rows]
    # read_kge_dataset() returns a set and KG stores list(set); preserve that
    # Phase-9 loader lineage before applying Random(42).shuffle.
    loader_set: set[tuple[int, int, int]] = set()
    for triple in converted:
        loader_set.add(triple)
    raw = list(set(loader_set))
    if declared != EXPECTED_RAW or len(raw) != EXPECTED_RAW:
        raise RuntimeError(f"unexpected training rows: declared={declared}, actual={len(raw)}")
    shuffled = list(raw)
    random.Random(42).shuffle(shuffled)
    held_out = shuffled[:5000]
    training = shuffled[5000:]
    if len(training) != EXPECTED_TRAIN:
        raise RuntimeError(f"unexpected post-split training size: {len(training)}")
    metadata = {
        "source_path": str(DATA_PATH.relative_to(REPO)),
        "source_sha256": sha256_file(DATA_PATH),
        "declared_triples": declared,
        "raw_triples": len(raw),
        "split_algorithm": "Python random.Random(42).shuffle; first 5000 held out",
        "disk_column_order": ["head", "tail", "relation"],
        "in_memory_column_order": ["head", "relation", "tail"],
        "loader_order": (
            "incrementally populated set, then list(set(loader_set)), matching "
            "read_kge_dataset + KG"
        ),
        "file_order_sha256": canonical_hash(converted),
        "split_seed": 42,
        "held_out_size": len(held_out),
        "training_set_size": len(training),
        "raw_order_sha256": canonical_hash(raw),
        "held_out_order_sha256": canonical_hash(held_out),
        "training_order_sha256": canonical_hash(training),
        "historical_alignment": (
            "Exact Phase 9 Step 2/3 split rule and sizes: 272115 -> "
            "5000 held out + 267115 training"
        ),
    }
    return raw, training, metadata


def get_dimensions(raw: list[tuple[int, int, int]]) -> tuple[int, int]:
    # Phase 9 reads validation/test after train, so kgs.entities_num covers all
    # dataset splits (14541), even though train touches only IDs 0..14504.
    entity_path = DATA_PATH.with_name("entity2id.txt")
    relation_path = DATA_PATH.with_name("relation2id.txt")
    with entity_path.open("r", encoding="utf-8") as handle:
        n_entities = int(handle.readline().strip())
    with relation_path.open("r", encoding="utf-8") as handle:
        n_relations = int(handle.readline().strip())
    if (n_entities, n_relations) != (14541, 237):
        raise RuntimeError(f"unexpected dimensions: {(n_entities, n_relations)}")
    if max(max(h, t) for h, _, t in raw) >= n_entities:
        raise RuntimeError("training entity ID exceeds declared entity count")
    if max(r for _, r, _ in raw) >= n_relations:
        raise RuntimeError("training relation ID exceeds declared relation count")
    return n_entities, n_relations


def load_cost_table(n_entities: int) -> np.ndarray:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_PATH} is required; the audit runner does not create or alter feature caches"
        )
    with np.load(FEATURE_PATH) as data:
        features = {
            "candidate_size": np.array(data["candidate_size"], copy=True),
            "degree": np.array(data["degree"], copy=True),
            "hub_flag": np.array(data["hub_flag"], copy=True),
        }
    # This is the historical Phase-9 cache: it covers the 14505 entities
    # touched by train.  BatchProvider only indexes training entities, while
    # the sampler/model use the all-splits entity count (14541).
    if len(features["candidate_size"]) != 14505 or n_entities != 14541:
        raise RuntimeError("feature cache does not match the frozen Phase-9 lineage")
    return build_cost_table(features, neg_num=NEG_NUM)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def original_cpu_neg_sampling(
    batch: list[tuple[int, int, int]],
    all_triples_set: set[tuple[int, int, int]],
    n_entities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads: list[int] = []
    tails: list[int] = []
    for h, r, t in batch:
        for _ in range(NEG_NUM):
            if random.random() < 0.5:
                for _try in range(10):
                    candidate = random.randint(0, n_entities - 1)
                    if candidate != h and (candidate, r, t) not in all_triples_set:
                        heads.append(candidate)
                        tails.append(t)
                        break
                else:
                    heads.append(h)
                    tails.append(t)
            else:
                for _try in range(10):
                    candidate = random.randint(0, n_entities - 1)
                    if candidate != t and (h, r, candidate) not in all_triples_set:
                        heads.append(h)
                        tails.append(candidate)
                        break
                else:
                    heads.append(h)
                    tails.append(t)
    return torch.tensor(heads, dtype=torch.long), torch.tensor(tails, dtype=torch.long)


def create_model(n_entities: int, n_relations: int) -> tuple[SimpleTransE, Any]:
    model = SimpleTransE(n_entities, n_relations).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def make_provider(cost_table: np.ndarray) -> BatchProvider:
    return BatchProvider(
        Scheduler(RandomSorter(seed=42), ChunkPacker()),
        cost_table,
        BATCH_SIZE,
        enable_logging=False,
    )


def tensors_and_loss(
    config: str,
    batch: list[tuple[int, int, int]],
    model: SimpleTransE,
    sampler: GPUNegativeSampler | None,
    all_triples_set: set[tuple[int, int, int]],
    n_entities: int,
) -> torch.Tensor:
    if config == "GPU":
        assert sampler is not None
        neg_h, neg_t = sampler.generate(batch)
    else:
        neg_h, neg_t = original_cpu_neg_sampling(batch, all_triples_set, n_entities)
        neg_h = neg_h.cuda()
        neg_t = neg_t.cuda()
    pos_h = torch.tensor([triple[0] for triple in batch], dtype=torch.long, device="cuda")
    pos_r = torch.tensor([triple[1] for triple in batch], dtype=torch.long, device="cuda")
    pos_t = torch.tensor([triple[2] for triple in batch], dtype=torch.long, device="cuda")
    pos_scores = model(pos_h, pos_r, pos_t)
    neg_scores = model(neg_h, pos_r.repeat_interleave(NEG_NUM), neg_t)
    return torch.mean(torch.clamp(
        pos_scores[:, None] - neg_scores.view(-1, NEG_NUM) + 1.0,
        min=0,
    ))


def warmup_training(
    config: str,
    seed: int,
    batches: list[list[tuple[int, int, int]]],
    all_triples_set: set[tuple[int, int, int]],
    n_entities: int,
    n_relations: int,
) -> list[dict[str, Any]]:
    seed_everything(seed)
    model, optimizer = create_model(n_entities, n_relations)
    sampler = GPUNegativeSampler(n_entities, NEG_NUM) if config == "GPU" else None
    rows = []
    for index, batch in enumerate(batches[:3]):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        optimizer.zero_grad()
        loss = tensors_and_loss(
            config, batch, model, sampler, all_triples_set, n_entities
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - start
        rows.append({
            "step": index,
            "elapsed_ns": elapsed,
            "loss": float(loss.detach().cpu()),
            "finite": bool(torch.isfinite(loss).item()),
        })
    del optimizer, model, sampler, loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return rows


def _parse_number(value: str) -> float | None:
    cleaned = value.strip().split()[0] if value.strip() else ""
    if cleaned in {"", "[N/A]", "N/A", "Not"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def telemetry_snapshot(config: str, seed: int, pass_name: str,
                       event: str, epoch: int | str = "") -> dict[str, Any]:
    query_fields = [
        "timestamp", "index", "name", "uuid", "pstate", "clocks.gr", "clocks.mem",
        "temperature.gpu", "power.draw", "power.limit", "utilization.gpu",
        "clocks_throttle_reasons.active",
        "clocks_throttle_reasons.sw_thermal_slowdown",
    ]
    gpu_cmd = [
        "nvidia-smi", f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    process_cmd = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    error = ""
    try:
        gpu_result = subprocess.run(gpu_cmd, check=True, capture_output=True, text=True)
        raw_gpu = gpu_result.stdout.strip()
    except Exception as exc:
        raw_gpu = ""
        error = f"gpu_query:{exc}"
    try:
        process_result = subprocess.run(
            process_cmd, check=True, capture_output=True, text=True
        )
        raw_process = process_result.stdout.strip()
    except Exception as exc:
        raw_process = ""
        error = f"{error};process_query:{exc}".strip(";")
    values = [item.strip() for item in raw_gpu.splitlines()[0].split(",")] if raw_gpu else []
    values += [""] * (len(query_fields) - len(values))
    other_processes = []
    for line in raw_process.splitlines():
        pid_text = line.split(",", 1)[0].strip()
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid():
            other_processes.append(line.strip())
    return {
        "protocol_id": PROTOCOL_ID,
        "config": config,
        "seed": seed,
        "pass_name": pass_name,
        "event": event,
        "epoch": epoch,
        "time_ns": time.time_ns(),
        "timestamp": values[0],
        "gpu_index": values[1],
        "name": values[2],
        "uuid": values[3],
        "pstate": values[4],
        "clock_graphics_mhz": _parse_number(values[5]),
        "clock_memory_mhz": _parse_number(values[6]),
        "temperature_c": _parse_number(values[7]),
        "power_draw_w": _parse_number(values[8]),
        "power_limit_w": _parse_number(values[9]),
        "utilization_gpu_pct": _parse_number(values[10]),
        "throttle_active": values[11],
        "thermal_slowdown": values[12],
        "other_compute_processes": " | ".join(other_processes),
        "raw_gpu_query": raw_gpu.replace("\n", " | "),
        "raw_process_query": raw_process.replace("\n", " | "),
        "query_error": error,
    }


def validate_batches(batches: list[list[tuple[int, int, int]]]) -> None:
    validate_batch_sizes([len(batch) for batch in batches])


def validate_batch_sizes(sizes: list[int]) -> None:
    if len(sizes) != 54:
        raise RuntimeError(f"expected 54 batches, got {len(sizes)}")
    if sizes.count(BATCH_SIZE) != EXPECTED_FULL_BATCHES:
        raise RuntimeError(f"expected 53 full batches, got {sizes.count(BATCH_SIZE)}")
    if sizes[-1] != EXPECTED_PARTIAL_SIZE:
        raise RuntimeError(f"expected final partial size 2115, got {sizes[-1]}")
    if sum(sizes) != EXPECTED_TRAIN:
        raise RuntimeError(f"batch coverage mismatch: {sum(sizes)}")


def run_job(root: Path, config: str, pass_name: str, seed: int) -> None:
    raw, training, split_metadata = load_split()
    n_entities, n_relations = get_dimensions(raw)
    all_triples_set = set(raw)
    cost_table = load_cost_table(n_entities)
    job_dir = root / "jobs" / f"{pass_name}_{config}_seed{seed}"
    job_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = job_dir / "gpu_telemetry.csv"
    append_csv(
        telemetry_path, TELEMETRY_FIELDS,
        telemetry_snapshot(config, seed, pass_name, "before_job"),
    )
    provider_for_layout = make_provider(cost_table)
    batches = list(provider_for_layout.iterate(training))
    validate_batches(batches)
    warmup_rows = warmup_training(
        config, seed, batches, all_triples_set, n_entities, n_relations
    )
    write_csv(
        job_dir / "warmup.csv",
        ["step", "elapsed_ns", "loss", "finite"],
        warmup_rows,
    )
    if not all(row["finite"] for row in warmup_rows):
        raise RuntimeError("non-finite warm-up loss")
    append_csv(
        telemetry_path, TELEMETRY_FIELDS,
        telemetry_snapshot(config, seed, pass_name, "after_warmup"),
    )

    # Disposable warm-up state is gone; reset all RNGs before the measured model.
    seed_everything(seed)
    model, optimizer = create_model(n_entities, n_relations)
    sampler = GPUNegativeSampler(n_entities, NEG_NUM) if config == "GPU" else None
    provider = make_provider(cost_table)
    torch.cuda.reset_peak_memory_stats()
    epoch_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    epoch0_loss: float | None = None

    for epoch in range(EPOCHS):
        append_csv(
            telemetry_path, TELEMETRY_FIELDS,
            telemetry_snapshot(config, seed, pass_name, "before_epoch", epoch),
        )
        model.train()
        loss_sum = torch.zeros((), device="cuda")
        step_count = 0
        observed_sizes: list[int] = []
        torch.cuda.synchronize()
        epoch_start = time.perf_counter_ns()

        for step, batch in enumerate(provider.iterate(training)):
            observed_sizes.append(len(batch))
            if pass_name == "throughput":
                optimizer.zero_grad()
                loss = tensors_and_loss(
                    config, batch, model, sampler, all_triples_set, n_entities
                )
                loss.backward()
                optimizer.step()
            else:
                torch.cuda.synchronize()
                step_start = time.perf_counter_ns()

                start = time.perf_counter_ns()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                zero_grad_ns = time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                if config == "GPU":
                    assert sampler is not None
                    neg_h, neg_t = sampler.generate(batch)
                else:
                    neg_h, neg_t = original_cpu_neg_sampling(
                        batch, all_triples_set, n_entities
                    )
                    neg_h = neg_h.cuda()
                    neg_t = neg_t.cuda()
                torch.cuda.synchronize()
                neg_time_ns = time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                pos_h = torch.tensor(
                    [triple[0] for triple in batch], dtype=torch.long, device="cuda"
                )
                pos_r = torch.tensor(
                    [triple[1] for triple in batch], dtype=torch.long, device="cuda"
                )
                pos_t = torch.tensor(
                    [triple[2] for triple in batch], dtype=torch.long, device="cuda"
                )
                torch.cuda.synchronize()
                positive_tensor_build_ns = time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                pos_scores = model(pos_h, pos_r, pos_t)
                neg_scores = model(neg_h, pos_r.repeat_interleave(NEG_NUM), neg_t)
                loss = torch.mean(torch.clamp(
                    pos_scores[:, None] - neg_scores.view(-1, NEG_NUM) + 1.0,
                    min=0,
                ))
                torch.cuda.synchronize()
                forward_ns = time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                loss.backward()
                torch.cuda.synchronize()
                backward_ns = time.perf_counter_ns() - start

                start = time.perf_counter_ns()
                optimizer.step()
                torch.cuda.synchronize()
                optimizer_ns = time.perf_counter_ns() - start
                total_step_ns = time.perf_counter_ns() - step_start
                component_sum_ns = sum([
                    zero_grad_ns, neg_time_ns, positive_tensor_build_ns,
                    forward_ns, backward_ns, optimizer_ns,
                ])
                step_rows.append({
                    "protocol_id": PROTOCOL_ID,
                    "pass_name": pass_name,
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "batch_size_actual": len(batch),
                    "is_partial": len(batch) != BATCH_SIZE,
                    "is_first_measured_step": epoch == 0 and step == 0,
                    "zero_grad_ns": zero_grad_ns,
                    "neg_time_ns": neg_time_ns,
                    "positive_tensor_build_ns": positive_tensor_build_ns,
                    "forward_ns": forward_ns,
                    "backward_ns": backward_ns,
                    "optimizer_ns": optimizer_ns,
                    "component_sum_ns": component_sum_ns,
                    "total_step_ns": total_step_ns,
                    "timing_residual_ns": total_step_ns - component_sum_ns,
                    "loss": float(loss.detach().cpu()),
                })

            if pass_name == "trace" and not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite loss at epoch={epoch}, step={step}")
            loss_sum.add_(loss.detach())
            step_count += 1

        torch.cuda.synchronize()
        epoch_time_ns = time.perf_counter_ns() - epoch_start
        validate_batch_sizes(observed_sizes)
        avg_loss = float((loss_sum / step_count).cpu())
        if not math.isfinite(avg_loss):
            raise RuntimeError(f"non-finite epoch average loss at epoch={epoch}")
        if epoch0_loss is None:
            epoch0_loss = avg_loss
        epoch_rows.append({
            "protocol_id": PROTOCOL_ID,
            "pass_name": pass_name,
            "config": config,
            "seed": seed,
            "epoch": epoch,
            "epoch_time_ns": epoch_time_ns,
            "scheduler_overhead_ns": round(provider.get_scheduler_overhead_ms() * 1e6),
            "num_steps": step_count,
            "full_batch_count": observed_sizes.count(BATCH_SIZE),
            "partial_batch_count": sum(size != BATCH_SIZE for size in observed_sizes),
            "partial_batch_size": observed_sizes[-1],
            "training_examples": sum(observed_sizes),
            "avg_loss": avg_loss,
            "loss_finite": math.isfinite(avg_loss),
            "loss_change_from_epoch0": avg_loss - epoch0_loss,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        print(
            f"{pass_name} {config} seed={seed} epoch={epoch} "
            f"time={epoch_time_ns / 1e9:.6f}s loss={avg_loss:.6f}",
            flush=True,
        )
        append_csv(
            telemetry_path, TELEMETRY_FIELDS,
            telemetry_snapshot(config, seed, pass_name, "after_epoch", epoch),
        )

    write_csv(job_dir / "per_epoch.csv", EPOCH_FIELDS, epoch_rows)
    if pass_name == "trace":
        write_csv(job_dir / "per_step.csv", STEP_FIELDS, step_rows)
    warnings = []
    if epoch_rows[-1]["avg_loss"] >= epoch_rows[0]["avg_loss"]:
        warnings.append("final average loss is not below epoch-0 average loss")
    other_process_rows = []
    with telemetry_path.open("r", encoding="utf-8", newline="") as handle:
        other_process_rows = [
            row for row in csv.DictReader(handle) if row["other_compute_processes"]
        ]
    status = {
        "protocol_id": PROTOCOL_ID,
        "config": config,
        "pass_name": pass_name,
        "seed": seed,
        "valid": not other_process_rows,
        "invalid_reasons": (
            ["other GPU compute process observed outside timed epochs"]
            if other_process_rows else []
        ),
        "warnings": warnings,
        "split": split_metadata,
        "row_counts": {
            "epochs": len(epoch_rows),
            "steps": len(step_rows),
        },
    }
    write_json(job_dir / "status.json", status)
    append_csv(
        telemetry_path, TELEMETRY_FIELDS,
        telemetry_snapshot(config, seed, pass_name, "after_job"),
    )


def memory_record(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def preflight(root: Path) -> None:
    preflight_dir = root / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=False)
    raw, training, split_metadata = load_split()
    n_entities, n_relations = get_dimensions(raw)
    all_triples_set = set(raw)
    cost_table = load_cost_table(n_entities)
    batches = list(make_provider(cost_table).iterate(training))
    validate_batches(batches)
    full_batch = batches[0]
    total_vram = torch.cuda.get_device_properties(0).total_memory
    checks: list[dict[str, Any]] = []
    telemetry_rows = [
        telemetry_snapshot("GPU", -1, "preflight", "before_preflight")
    ]
    try:
        # P1: sampler-only allocation and shape/device/count.
        seed_everything(42)
        sampler = GPUNegativeSampler(n_entities, NEG_NUM)
        for _ in range(3):
            sampler.generate(full_batch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        start = time.perf_counter_ns()
        neg_h, neg_t = sampler.generate(full_batch)
        torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - start
        p1 = {
            "check": "P1_gpu_sampler_only",
            "passed": (
                neg_h.shape == neg_t.shape == (BATCH_SIZE * NEG_NUM,)
                and neg_h.device.type == neg_t.device.type == "cuda"
                and torch.cuda.max_memory_reserved() < 0.9 * total_vram
            ),
            "elapsed_ns": elapsed_ns,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "delta_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() - baseline_allocated
            ),
            "delta_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() - baseline_reserved
            ),
            "total_vram_bytes": total_vram,
            "peak_reserved_fraction": torch.cuda.max_memory_reserved() / total_vram,
            "neg_heads_shape": list(neg_h.shape),
            "neg_tails_shape": list(neg_t.shape),
            "device": str(neg_t.device),
        }
        checks.append(p1)
        del neg_h, neg_t, sampler
        torch.cuda.empty_cache()

        # P2/P3: one complete full-size training step for each configuration.
        for check_name, config in [
            ("P2_gpu_full_training_step", "GPU"),
            ("P3_bl_full_training_step", "BL"),
        ]:
            seed_everything(42)
            model, optimizer = create_model(n_entities, n_relations)
            sampler = GPUNegativeSampler(n_entities, NEG_NUM) if config == "GPU" else None
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter_ns()
            optimizer.zero_grad()
            loss = tensors_and_loss(
                config, full_batch, model, sampler, all_triples_set, n_entities
            )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - start
            finite = bool(torch.isfinite(loss).item())
            peak_reserved = torch.cuda.max_memory_reserved()
            checks.append({
                "check": check_name,
                "passed": finite and peak_reserved < 0.9 * total_vram,
                "elapsed_ns": elapsed_ns,
                "loss": float(loss.detach().cpu()),
                "loss_finite": finite,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": peak_reserved,
                "total_vram_bytes": total_vram,
                "peak_reserved_fraction": peak_reserved / total_vram,
                "batch_size": len(full_batch),
                "negative_count": len(full_batch) * NEG_NUM,
            })
            del model, optimizer, sampler, loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        telemetry_rows.append(
            telemetry_snapshot("GPU", -1, "preflight", "after_preflight")
        )
        other_process = any(row["other_compute_processes"] for row in telemetry_rows)
        checks.append({
            "check": "P4_no_other_compute_process",
            "passed": not other_process,
            "observations": [
                row["other_compute_processes"] for row in telemetry_rows
                if row["other_compute_processes"]
            ],
        })
        all_passed = all(check["passed"] for check in checks)
        result = {
            "protocol_id": PROTOCOL_ID,
            "all_passed": all_passed,
            "checks": checks,
            "split": split_metadata,
        }
        write_json(preflight_dir / "result.json", result)
        write_csv(preflight_dir / "gpu_telemetry.csv", TELEMETRY_FIELDS, telemetry_rows)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if not all_passed:
            raise RuntimeError("preflight failed; full experiment is forbidden")
    except Exception as exc:
        failure = {
            "protocol_id": PROTOCOL_ID,
            "all_passed": False,
            "checks": checks,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "memory_at_failure": memory_record("failure"),
            "split": split_metadata,
        }
        write_json(preflight_dir / "failure.json", failure)
        raise


def compute_only(root: Path, seed: int) -> None:
    raw, training, _ = load_split()
    n_entities, n_relations = get_dimensions(raw)
    cost_table = load_cost_table(n_entities)
    batch = list(make_provider(cost_table).iterate(training))[0]
    seed_everything(seed)
    model, optimizer = create_model(n_entities, n_relations)
    sampler = GPUNegativeSampler(n_entities, NEG_NUM)
    neg_h, neg_t = sampler.generate(batch)
    pos_h = torch.tensor([x[0] for x in batch], dtype=torch.long, device="cuda")
    pos_r = torch.tensor([x[1] for x in batch], dtype=torch.long, device="cuda")
    pos_t = torch.tensor([x[2] for x in batch], dtype=torch.long, device="cuda")

    def one_step() -> tuple[int, float]:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        optimizer.zero_grad()
        pos_scores = model(pos_h, pos_r, pos_t)
        neg_scores = model(neg_h, pos_r.repeat_interleave(NEG_NUM), neg_t)
        loss = torch.mean(torch.clamp(
            pos_scores[:, None] - neg_scores.view(-1, NEG_NUM) + 1.0, min=0
        ))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return time.perf_counter_ns() - start, float(loss.detach().cpu())

    for _ in range(3):
        one_step()
    rows = []
    for repeat in range(20):
        elapsed, loss = one_step()
        rows.append({
            "protocol_id": PROTOCOL_ID,
            "seed": seed,
            "repeat": repeat,
            "elapsed_ns": elapsed,
            "loss": loss,
            "batch_size": BATCH_SIZE,
            "neg_num": NEG_NUM,
        })
    write_csv(
        root / "compute_only" / f"seed{seed}.csv",
        ["protocol_id", "seed", "repeat", "elapsed_ns", "loss",
         "batch_size", "neg_num"],
        rows,
    )


def protocol_document(split: dict[str, Any]) -> dict[str, Any]:
    source_paths = [
        DATA_PATH,
        FEATURE_PATH,
        REPO / "src/py/load/gpu_sampler.py",
        REPO / "src/py/load/batch_provider.py",
        REPO / "src/py/load/schedulers.py",
        Path(__file__).resolve(),
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at": "generated at run start; wall-clock timestamp is stored in environment.json",
        "claims": {
            "C1.2-R1": "paired end-to-end throughput epoch speedup",
            "C1.3-R1": "paired full-batch within-epoch negative-time standard-deviation compression",
            "C1.7-R1": "GPU full-batch negative-sampling mean and stability",
        },
        "dataset": split,
        "model": {
            "name": "SimpleTransE",
            "embedding_dim": 400,
            "margin": 1.0,
            "optimizer": "Adam",
            "learning_rate": 0.001,
        },
        "configs": {
            "BL": (
                "RandomSorter(42)+ChunkPacker; original CPU Bernoulli(0.5) "
                "head/tail corruption; global train-triple collision set; H2D included"
            ),
            "GPU": (
                "RandomSorter(42)+ChunkPacker; tail-only GPUNegativeSampler; "
                "batch pos_tails filtering"
            ),
            "semantic_disclosure": (
                "BL and GPU samplers are intentionally non-equivalent runtime paths; "
                "this protocol makes no quality-equivalence claim"
            ),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "negative_samples_per_positive": NEG_NUM,
            "epochs_per_job": EPOCHS,
            "independent_seeds": list(SEEDS),
            "paired_order": {
                str(seed): ["BL", "GPU"] if seed % 2 == 0 else ["GPU", "BL"]
                for seed in SEEDS
            },
            "jobs_are_separate_processes": True,
            "warmup": (
                "three full training steps on a disposable model/optimizer; "
                "discard, clear cache, reseed, then construct measured model"
            ),
        },
        "passes": {
            "throughput": {
                "epoch_boundary": (
                    "CUDA synchronize; timer starts before BatchProvider.iterate() "
                    "executes and stops after final optimizer step and synchronize"
                ),
                "per_step_synchronization": False,
                "partial_batch_included": True,
                "loss_readback": "GPU scalar accumulated; one CPU read per epoch",
            },
            "trace": {
                "per_component_synchronization": True,
                "neg_time_ns": (
                    "sampler call start until negative tensors are ready on target GPU; "
                    "BL includes Python sampling, CPU tensor construction and H2D"
                ),
                "positive_tensor_build_ns": (
                    "positive h/r/t Python lists to GPU tensors; identical code in BL/GPU"
                ),
                "partial_filter_for_primary_analysis": (
                    "is_partial == False AND batch_size_actual == 5000"
                ),
            },
            "compute_only": (
                "diagnostic only: fixed full batch and prebuilt GPU tensors, "
                "3 warmups + 20 forward/backward/optimizer observations per seed"
            ),
        },
        "partial_batch_assertions": {
            "training_examples": EXPECTED_TRAIN,
            "batches_per_epoch": 54,
            "full_batches": EXPECTED_FULL_BATCHES,
            "partial_batches": 1,
            "partial_batch_size": EXPECTED_PARTIAL_SIZE,
        },
        "preflight": {
            "P1": "sampler-only full 5000x150 generation",
            "P2": "GPU full training step",
            "P3": "BL full training step",
            "vram_limit": "peak reserved strictly below 90% of physical VRAM",
            "failure_rule": "stop; never reduce batch size",
        },
        "statistics": {
            "independent_unit": "seed-level paired run",
            "C1.2": (
                "mean five throughput epochs per config/seed; paired ratios; "
                "geometric mean and two-sided 95% t CI on log ratios"
            ),
            "C1.3": (
                "ddof=0 within each trace epoch on full batches; mean five epoch "
                "standard deviations per run; paired ratios; log-scale t CI"
            ),
            "C1.7": (
                "GPU full-batch neg-time mean per run; six-run arithmetic mean, "
                "sample SD and two-sided 95% t CI"
            ),
            "t_critical_df5": 2.570581835636314,
            "A_gate_C1.2_C1.3": (
                "six valid pairs, complete raw data, valid protocol, lower 95% CI > 1"
            ),
        },
        "diagnostics": {
            "loss": "finite required; final >= epoch0 is warning only",
            "speedup_over_10x": "warning only",
            "telemetry": (
                "before/after warmup, before/after each epoch, after job; "
                "never queried inside timed epoch"
            ),
        },
        "source_files": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }


def environment_document() -> dict[str, Any]:
    commands = {
        "nvidia_smi_q_clock_power_temperature": [
            "nvidia-smi", "-q", "-d", "CLOCK,POWER,TEMPERATURE",
        ],
        "nvidia_smi_driver": ["nvidia-smi"],
        "nvidia_smi_identity": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    }
    captures = {}
    for key, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True)
        captures[key] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    identity = captures["nvidia_smi_identity"]["stdout"].strip().split(",")
    if len(identity) != 3:
        raise RuntimeError("unable to collect GPU identity without creating a CUDA context")
    return {
        "protocol_id": PROTOCOL_ID,
        "captured_time_ns": time.time_ns(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": {
            "name": identity[0].strip(),
            "total_memory_mib_reported_by_nvidia_smi": float(identity[1].strip()),
            "compute_capability": identity[2].strip(),
        },
        "historical_environment_difference": (
            "This rerun uses the currently installed PyTorch/CUDA stack; historical "
            "Phase 9 documentation referenced an older stack. Results replace, not "
            "pool with, historical rounded observations."
        ),
        "raw_command_captures": captures,
    }


def artifact_hashes(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.name == "artifact_hashes.csv":
            continue
        rows.append({
            "path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def controller(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment root: {root}")
    root.mkdir(parents=True)
    _, _, split = load_split()
    write_json(root / "protocol.json", protocol_document(split))
    write_json(root / "environment.json", environment_document())
    script = Path(__file__).resolve()
    python = sys.executable
    manifest: list[dict[str, Any]] = []

    def launch(kind: str, config: str = "", seed: int | None = None,
               pass_name: str = "") -> None:
        label = kind
        command = [python, str(script), kind, "--root", str(root)]
        if config:
            command += ["--config", config]
            label += f"_{config}"
        if pass_name:
            command += ["--pass-name", pass_name]
            label += f"_{pass_name}"
        if seed is not None:
            command += ["--seed", str(seed)]
            label += f"_seed{seed}"
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout_path = log_dir / f"{label}.stdout.log"
        stderr_path = log_dir / f"{label}.stderr.log"
        start_ns = time.time_ns()
        print(f"START {label}", flush=True)
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle, \
                stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle:
            result = subprocess.run(
                command, cwd=REPO, stdout=stdout_handle, stderr=stderr_handle
            )
        end_ns = time.time_ns()
        manifest.append({
            "label": label,
            "kind": kind,
            "config": config,
            "seed": "" if seed is None else seed,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "elapsed_ns": end_ns - start_ns,
            "returncode": result.returncode,
            "stdout_path": str(stdout_path.relative_to(root)),
            "stderr_path": str(stderr_path.relative_to(root)),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        })
        write_csv(
            root / "run_manifest.csv",
            ["label", "kind", "config", "seed", "start_time_ns", "end_time_ns",
             "elapsed_ns", "returncode", "stdout_path", "stderr_path",
             "stdout_sha256", "stderr_sha256"],
            manifest,
        )
        if result.returncode:
            raise RuntimeError(
                f"{label} failed with return code {result.returncode}; "
                f"see {stderr_path}"
            )
        print(f"DONE {label} elapsed={(end_ns - start_ns) / 1e9:.1f}s", flush=True)

    launch("preflight")
    for pass_name in ("throughput", "trace"):
        for seed in SEEDS:
            order = ("BL", "GPU") if seed % 2 == 0 else ("GPU", "BL")
            for config in order:
                launch("job", config=config, seed=seed, pass_name=pass_name)
    for seed in SEEDS:
        launch("compute-only", seed=seed)
    write_csv(
        root / "artifact_hashes.csv",
        ["path", "bytes", "sha256"],
        artifact_hashes(root),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("all", "preflight"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    job = subparsers.add_parser("job")
    job.add_argument("--root", type=Path, required=True)
    job.add_argument("--config", choices=["BL", "GPU"], required=True)
    job.add_argument("--pass-name", choices=["throughput", "trace"], required=True)
    job.add_argument("--seed", type=int, choices=SEEDS, required=True)
    compute = subparsers.add_parser("compute-only")
    compute.add_argument("--root", type=Path, required=True)
    compute.add_argument("--seed", type=int, choices=SEEDS, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.command == "all":
        controller(root)
    elif args.command == "preflight":
        preflight(root)
    elif args.command == "job":
        run_job(root, args.config, args.pass_name, args.seed)
    elif args.command == "compute-only":
        compute_only(root, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
