"""
Phase 6 - Node 3, Stage C+D: BatchProvider (Adapter) + Verification Logging
=======================================================================
Zero-intrusion adapter between Scheduler and DataLoader.

Design principle:
    BatchProvider wraps a Scheduler and presents an iterator interface
    that the training layer consumes. The original DataLoader is completely
    unaware of the scheduling layer — it simply receives pre-assembled batches.

    For DDP: each rank gets a BatchProvider instance, each reading from
    its assigned partition of the scheduled batches.

Stage D logging probes emit:
    1. batch_runtime_variance.csv    — actual per-batch cost (latency)
    2. scheduler_overhead.csv        — scheduler wall time
    3. batch_weight_distribution.csv — theoretical cost distribution
"""

import csv
import os
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from src.py.load.schedulers import Scheduler


LOG_DIR = "output/results/"


class BatchProvider:
    """
    Adapter: wraps Scheduler + cost_table, presents batches to training loop.

    Usage:
        provider = BatchProvider(scheduler, cost_table, batch_size)
        for batch in provider.iterate(epoch_triples):
            # batch is a list of (h, r, t) triples
            # DataLoader consumes it as-is
            ...

    For DDP:
        provider = BatchProvider(scheduler, cost_table, batch_size)
        provider.set_rank(rank=0, world_size=4)  # each rank gets 1/N batches
        for batch in provider.iterate(epoch_triples):
            # Only batches for this rank
            ...
    """

    def __init__(self,
                 scheduler: Scheduler,
                 cost_table: np.ndarray,
                 batch_size: int,
                 enable_logging: bool = True):
        self.scheduler = scheduler
        self.cost_table = cost_table
        self.batch_size = batch_size
        self.enable_logging = enable_logging

        # DDP rank/world_size
        self._rank = 0
        self._world_size = 1

        # Logging accumulators
        self._log_batch_costs = []
        self._scheduler_overhead = 0.0

    def set_rank(self, rank: int, world_size: int):
        """Configure DDP mode: each rank processes a subset of batches."""
        self._rank = rank
        self._world_size = world_size

    def iterate(self, triples_list: List[Tuple[int, int, int]]
                ) -> Iterator[List[Tuple[int, int, int]]]:
        """
        Main entry point: yield batches for one epoch.

        The scheduler runs ONCE per epoch (at first iteration).
        Subsequent iterations reuse the cached batch layout.

        Args:
            triples_list: Full list of (head, rel, tail) for one epoch.

        Yields:
            batch: List of triples for one training step.
        """
        # ── Schedule ONCE per epoch ──
        sched_start = time.perf_counter()
        all_batches = self.scheduler.pack_batches(
            triples_list, self.cost_table, self.batch_size
        )
        self._scheduler_overhead = (time.perf_counter() - sched_start) * 1000.0

        # ── Slice for DDP rank ──
        if self._world_size > 1:
            my_batches = all_batches[self._rank::self._world_size]
        else:
            my_batches = all_batches

        # ── Compute batch weights for logging ──
        batch_weights = []
        for batch in my_batches:
            w = sum(max(
                float(self.cost_table[h]) if h < len(self.cost_table) else 0.0,
                float(self.cost_table[t]) if t < len(self.cost_table) else 0.0
            ) for h, r, t in batch)
            batch_weights.append(w)

        self._log_batch_costs = batch_weights

        # ── Logging probes (Stage D) ──
        if self.enable_logging and self._rank == 0:
            self._emit_logs(all_batches)

        # ── Yield batches ──
        for batch in my_batches:
            yield batch

    def get_scheduler_overhead_ms(self) -> float:
        """Return scheduler overhead for last epoch (ms)."""
        return self._scheduler_overhead

    def get_batch_weight_stats(self) -> dict:
        """Return weight statistics for last epoch."""
        if not self._log_batch_costs:
            return {}
        costs = np.array(self._log_batch_costs)
        return {
            "mean": float(costs.mean()),
            "std": float(costs.std()),
            "cv": float(costs.std() / max(costs.mean(), 1e-10)),
            "min": float(costs.min()),
            "max": float(costs.max()),
            "n_batches": len(costs),
        }

    # ── Stage D: Logging Probes ───────────────────────────────────────

    def _emit_logs(self, all_batches):
        """Emit Stage D verification logs."""
        os.makedirs(LOG_DIR, exist_ok=True)

        # 1. batch_weight_distribution.csv
        w_path = os.path.join(LOG_DIR, "batch_weight_distribution.csv")
        with open(w_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_id", "n_triples", "total_weight_ms"])
            for i, batch in enumerate(all_batches):
                w = sum(max(
                    float(self.cost_table[h]) if h < len(self.cost_table) else 0.0,
                    float(self.cost_table[t]) if t < len(self.cost_table) else 0.0
                ) for h, r, t in batch)
                writer.writerow([i, len(batch), round(w, 3)])
        print(f"[BatchProvider] Log: {w_path} ({len(all_batches)} batches)")

        # 2. scheduler_overhead.csv
        o_path = os.path.join(LOG_DIR, "scheduler_overhead.csv")
        with open(o_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scheduler", "overhead_ms", "n_triples", "n_batches"])
            writer.writerow([
                self.scheduler.get_name(),
                round(self._scheduler_overhead, 3),
                sum(len(b) for b in all_batches),
                len(all_batches),
            ])
        print(f"[BatchProvider] Log: {o_path} "
              f"(overhead={self._scheduler_overhead:.3f}ms)")


# ═══════════════════════════════════════════════════════════════════════
# Integration example (for reference, not executed here)
# ═══════════════════════════════════════════════════════════════════════
#
# # In main_FB15K237.py:
#
# from src.py.load.features import FeatureExtractor
# from src.py.load.cost_model import build_cost_table
# from src.py.load.schedulers import create_scheduler
# from src.py.load.batch_provider import BatchProvider
#
# # 1. Extract features (one-time, cached to disk)
# extractor = FeatureExtractor(triples_list, num_entities)
# features = extractor.build()
#
# # 2. Map features → cost table
# cost_table = build_cost_table(features)
#
# # 3. Create scheduler
# scheduler = create_scheduler("ffd")  # or "random" for baseline
#
# # 4. Wrap in BatchProvider
# provider = BatchProvider(scheduler, cost_table, batch_size)
#
# # 5. In training loop:
# for batch in provider.iterate(epoch_triples):
#     # batch is a list of (h, r, t)
#     # Feed to DataLoader.collate_fn as before
#     ...