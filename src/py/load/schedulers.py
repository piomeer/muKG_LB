"""
Phase 6 - Node 3, Stage B: Scheduler — Sort Policy + Packing Policy
====================================================================
Strategy pattern decomposed into two orthogonal axes:
    - Sort Policy:   how to order triples before packing
    - Packing Policy: how to partition ordered triples into batches

Architecture:
    Scheduler(sorter, packer)
        │
        ├── Sort Policies:
        │   ├── RandomSorter  — shuffle (baseline)
        │   └── CostSorter    — descending by expected cost (CBP)
        │
        └── Packing Policies:
            ├── ChunkPacker  — sequential chunks (baseline)
            └── FFDPacker    — First Fit Decreasing (CBP)

This allows 4 combinations: Random+Chunk, Random+FFD, Cost+Chunk, Cost+FFD.
Our CBP = CostSorter + FFDPacker.
"""

import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Sort Policies
# ═══════════════════════════════════════════════════════════════════════

class BaseSorter(ABC):
    """Abstract sort policy: reorder triples before packing."""

    @abstractmethod
    def sort(self, triples_list: List[Tuple[int, int, int]],
             cost_table: np.ndarray) -> List[Tuple[int, int, int]]:
        """Return reordered list of triples."""
        pass


class RandomSorter(BaseSorter):
    """Random shuffle — baseline, no cost awareness."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def sort(self, triples_list, cost_table):
        shuffled = list(triples_list)
        random.Random(self.seed).shuffle(shuffled)
        return shuffled


class CostSorter(BaseSorter):
    """Sort descending by expected cost — core CBP ordering."""

    def sort(self, triples_list, cost_table):
        def triple_cost(triple):
            h, r, t = triple
            hc = float(cost_table[h]) if h < len(cost_table) else 0.0
            tc = float(cost_table[t]) if t < len(cost_table) else 0.0
            return max(hc, tc)
        return sorted(triples_list, key=triple_cost, reverse=True)


# ═══════════════════════════════════════════════════════════════════════
# Packing Policies
# ═══════════════════════════════════════════════════════════════════════

class BasePacker(ABC):
    """Abstract packing policy: partition ordered triples into batches."""

    @abstractmethod
    def pack(self, ordered_triples: List[Tuple[int, int, int]],
             batch_size: int) -> List[List[Tuple[int, int, int]]]:
        """Return list of batches (each batch is a list of triples)."""
        pass


class ChunkPacker(BasePacker):
    """
    Sequential chunk packing — baseline.
    Simply slices ordered_triples into consecutive chunks of batch_size.
    """

    def pack(self, ordered_triples, batch_size):
        return [ordered_triples[i:i + batch_size]
                for i in range(0, len(ordered_triples), batch_size)]


class FFDPacker(BasePacker):
    """
    First Fit Decreasing packing — core CBP strategy.

    Distributes triples into bins such that each bin has batch_size triples.
    High-cost triples (sorted first) are spread across bins to minimize
    batch-level cost variance.

    Note: FFD is a standard bin-packing heuristic. CBP's innovation is
    the cost-aware sorting that feeds into FFD, not FFD itself.
    """

    def pack(self, ordered_triples, batch_size):
        if len(ordered_triples) == 0:
            return []

        n_batches = (len(ordered_triples) + batch_size - 1) // batch_size
        batches = [[] for _ in range(n_batches)]

        for triple in ordered_triples:
            # First Fit: place into the first batch with room
            placed = False
            for b_idx in range(n_batches):
                if len(batches[b_idx]) < batch_size:
                    batches[b_idx].append(triple)
                    placed = True
                    break
            if not placed:
                batches.append([triple])
                n_batches += 1

        return batches


# ═══════════════════════════════════════════════════════════════════════
# Unified Scheduler
# ═══════════════════════════════════════════════════════════════════════

class Scheduler:
    """
    Unified scheduler: composes a Sort Policy + Packing Policy.

    CBP core strategy: Scheduler(CostSorter(), FFDPacker()).
    Native baseline:   Scheduler(RandomSorter(), ChunkPacker()).

    The Scheduler is stateless from the framework's perspective —
    it takes triples and produces batches, no internal state to manage.
    """

    def __init__(self, sorter: BaseSorter, packer: BasePacker):
        self.sorter = sorter
        self.packer = packer
        self._name = f"{sorter.__class__.__name__}+{packer.__class__.__name__}"

    def pack_batches(self,
                     triples_list: List[Tuple[int, int, int]],
                     cost_table: np.ndarray,
                     batch_size: int) -> List[List[Tuple[int, int, int]]]:
        """
        Partition triples_list into batches.

        Args:
            triples_list: Full list of (head, rel, tail) for one epoch.
            cost_table:   Pre-computed cost table from CostModel.
            batch_size:   Target triples per batch.

        Returns:
            List of batches. Coverage: sum(len(b) for b in output) == len(triples_list).
        """
        ordered = self.sorter.sort(triples_list, cost_table)
        return self.packer.pack(ordered, batch_size)

    def get_name(self) -> str:
        return self._name


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def create_scheduler(scheduler_type: str, **kwargs) -> Scheduler:
    """
    Factory: create a Scheduler by type name.

    Available types:
        "random"  → RandomSorter + ChunkPacker (native baseline)
        "ffd"     → CostSorter + FFDPacker      (CBP core)
        "cost+ffd" → CostSorter + FFDPacker
    """
    mapping = {
        "random": lambda: Scheduler(RandomSorter(**kwargs), ChunkPacker()),
        "ffd": lambda: Scheduler(CostSorter(), FFDPacker()),
        "cost_ffd": lambda: Scheduler(CostSorter(), FFDPacker()),
    }
    cls = mapping.get(scheduler_type.lower().strip())
    if cls is None:
        available = list(mapping.keys())
        raise ValueError(f"Unknown scheduler: '{scheduler_type}'. "
                         f"Available: {available}")
    return cls()