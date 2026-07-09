"""
Phase 6 - Node 2: Scheduler Policy Polymorphism
================================================
Strategy pattern for batch scheduling policies.

Architecture:
    BaseScheduler (abstract interface)
        ├── RandomScheduler  (baseline: no cost awareness)
        └── FFDScheduler     (core CBP strategy: cost-aware FFD packing)

Design principle:
    "Mechanism-Policy Separation"
    - Mechanism: BaseScheduler defines the pack_batches() contract.
    - Policy: RandomScheduler / FFDScheduler are interchangeable policies
              implementing different packing strategies.
    
    This allows easy addition of new schedulers (e.g., ML-based, RL-based)
    without modifying the framework core.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseScheduler(ABC):
    """
    Abstract base class for all batch scheduling policies.

    Interface: pack_batches(triples_list, cost_table, batch_size) → List[List[Tuple]]

    The scheduler receives the full list of triples for one epoch and returns
    a list of batches, each being a list of triples. The framework guarantees
    that all triples are covered exactly once per epoch.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @abstractmethod
    def pack_batches(self,
                     triples_list: List[Tuple[int, int, int]],
                     cost_table: np.ndarray,
                     batch_size: int) -> List[List[Tuple[int, int, int]]]:
        """
        Partition triples_list into batches.

        Args:
            triples_list: Full list of (head, rel, tail) for one epoch.
            cost_table:   Pre-computed cost_table from CostEstimator.
                          Shape: (num_entities,) — expected_cost per entity.
            batch_size:   Target number of triples per batch.

        Returns:
            List of batches, where each batch is a list of triples.
            Total coverage: sum(len(b) for b in output) == len(triples_list)
        """
        pass

    def get_name(self) -> str:
        """Human-readable scheduler name (for logging & experiment tracking)."""
        return self.__class__.__name__


class RandomScheduler(BaseScheduler):
    """
    Baseline scheduler: random shuffle, no cost awareness.

    This is the "do-nothing" baseline for ablation studies.
    It replicates the current behavior of PyTorchTrainDataLoader
    (random shuffle without any cost-aware packing).
    """

    def pack_batches(self,
                     triples_list: List[Tuple[int, int, int]],
                     cost_table: np.ndarray,
                     batch_size: int) -> List[List[Tuple[int, int, int]]]:
        shuffled = list(triples_list)
        random.shuffle(shuffled)

        batches = []
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            batches.append(batch)
        return batches


class FFDScheduler(BaseScheduler):
    """
    Core CBP scheduling policy: First Fit Decreasing with cost awareness.

    How it works:
        1. Compute per-triple cost = max(cost_table[head], cost_table[tail]).
        2. Sort triples by cost descending (Decreasing).
        3. For each triple, place it into the first bin (batch) that has
           room and would not exceed the target batch size (First Fit).

    Theoretical basis:
        The cost variance across batches in DDP directly translates to
        AllReduce synchronization waiting time. By minimizing batch-level
        cost variance, we eliminate the "wooden barrel effect" where
        one slow batch holds all GPUs hostage.

    Complexity: O(N log N + N × B)
        N = number of triples, B = number of batches.
        The FFD heuristic is guaranteed to use at most ⌈11/9 × OPT⌉ bins.
    """

    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        super().__init__(seed)
        self.verbose = verbose

    def pack_batches(self,
                     triples_list: List[Tuple[int, int, int]],
                     cost_table: np.ndarray,
                     batch_size: int) -> List[List[Tuple[int, int, int]]]:
        if len(triples_list) == 0:
            return []

        # Step 1: Compute per-triple cost
        triples_with_cost = []
        for h, r, t in triples_list:
            h_cost = float(cost_table[h]) if h < len(cost_table) else 0.0
            t_cost = float(cost_table[t]) if t < len(cost_table) else 0.0
            cost = max(h_cost, t_cost)
            triples_with_cost.append((cost, h, r, t))

        # Step 2: Sort descending by cost
        triples_with_cost.sort(key=lambda x: x[0], reverse=True)

        # Step 3: First Fit Decreasing
        n_batches = (len(triples_with_cost) + batch_size - 1) // batch_size
        batches = [[] for _ in range(n_batches)]
        batch_costs = [0.0] * n_batches

        placed = 0
        for cost, h, r, t in triples_with_cost:
            # First Fit: find the first batch with room
            placed_idx = None
            for b_idx in range(n_batches):
                if len(batches[b_idx]) < batch_size:
                    placed_idx = b_idx
                    break

            if placed_idx is None:
                # Should not happen if n_batches is correctly computed
                # Fallback: create new batch
                batches.append([(h, r, t)])
                n_batches += 1
            else:
                batches[placed_idx].append((h, r, t))
                batch_costs[placed_idx] += cost

            placed += 1

        if self.verbose:
            costs = [sum(max(float(cost_table[h]), float(cost_table[t]))
                        for h, r, t in batch) for batch in batches]
            print(f"[FFDScheduler] {len(batches)} batches, "
                  f"cost: mean={np.mean(costs):.1f}±{np.std(costs):.1f}ms, "
                  f"CV={np.std(costs)/max(np.mean(costs), 1e-6):.3f}")

        return batches


# ── Utility: factory function ────────────────────────────────────────────
def create_scheduler(scheduler_type: str, **kwargs) -> BaseScheduler:
    """
    Factory: create a scheduler by type name.

    Args:
        scheduler_type: One of "random", "ffd", "ffd_scheduler", etc.
        **kwargs: Passed to the scheduler constructor.

    Returns:
        BaseScheduler instance.

    Raises:
        ValueError: If scheduler_type is unknown.
    """
    mapping = {
        "random": RandomScheduler,
        "ffd": FFDScheduler,
        "ffd_scheduler": FFDScheduler,
        "cost_aware": FFDScheduler,
    }
    cls = mapping.get(scheduler_type.lower().strip())
    if cls is None:
        raise ValueError(f"Unknown scheduler type: '{scheduler_type}'. "
                         f"Available: {list(mapping.keys())}")
    return cls(**kwargs)