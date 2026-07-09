"""
Phase 6 - Node 1: Offline Cost Estimator
=========================================
Static pre-computation of entity-wise expected negative sampling costs.

Design principle:
  Cost estimation is entirely offline — computed once per dataset load,
  cached to disk as `cost_table.npy`. Runtime lookup is O(1).

Academic narrative:
  CBP's core insight: batch sampling time is dominated by candidate pool size,
  not entity degree (validated: candidate_size vs actual_time R=0.9008).
  This module pre-computes expected_cost per entity from neighbor_dict,
  feeding the downstream scheduler with zero runtime overhead.
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


COST_CACHE_DIR = "output/results/"
COST_CACHE_PATH = os.path.join(COST_CACHE_DIR, "cost_table.npy")
NEIGHBOR_CACHE_PATH = os.path.join(COST_CACHE_DIR, "neighbor_dict.pkl")


class CostEstimator:
    """
    Offline pre-computation of entity-level expected negative sampling costs.

    Core formula (validated by Phase 5.5, 400 batch, R=0.9008):

        candidate_size(e) = |neighbor_dict.get(e, entities_list)|

        P_collision = N_neg / candidate_size(e)
        E_retry    = min(max_try, 1 / (1 - P_collision))
        expected_cost(e) = E_retry * B3_const

    where B3_const ≈ 51.8 ms (the fixed cost of set-difference collision check).
    """

    def __init__(self, triples_list: List[Tuple[int, int, int]],
                 num_entities: int,
                 neg_num: int = 150,
                 max_try: int = 10,
                 b3_const: float = 51.8):
        self.triples_list = triples_list
        self.num_entities = num_entities
        self.neg_num = neg_num
        self.max_try = max_try
        self.b3_const = b3_const

        self.neighbor_dict: Dict[int, List[int]] = {}
        self.cost_table: np.ndarray = None  # shape: (num_entities,)
        self._built = False

    # ── Public API ──────────────────────────────────────────────────────

    def build(self, force_recompute: bool = False) -> np.ndarray:
        """
        Build cost table. Loads from cache if available, computes if not.

        Args:
            force_recompute: If True, ignore cache and recompute.

        Returns:
            cost_table: numpy array of shape (num_entities,) with expected_cost per entity.
        """
        if not force_recompute:
            loaded = self._load_cache()
            if loaded is not None:
                return loaded

        self._compute_neighbor_dict()
        self._compute_cost_table()
        self._save_cache()
        self._built = True
        return self.cost_table

    def get_cost(self, entity_id: int) -> float:
        """O(1) runtime cost lookup."""
        if not self._built and self.cost_table is None:
            raise RuntimeError("CostEstimator not built. Call .build() first.")
        if entity_id < 0 or entity_id >= self.num_entities:
            return self.b3_const  # fallback for OOB entities
        return float(self.cost_table[entity_id])

    def get_candidate_size(self, entity_id: int) -> int:
        """O(1) candidate pool size lookup."""
        if not self._built and self.cost_table is None:
            raise RuntimeError("CostEstimator not built. Call .build() first.")
        if entity_id < 0 or entity_id >= self.num_entities:
            return self.num_entities
        neighbors = self.neighbor_dict.get(entity_id, list(range(self.num_entities)))
        return len(neighbors)

    # ── Private: Computation ─────────────────────────────────────────────

    def _compute_neighbor_dict(self):
        """
        Build neighbor_dict: for each entity, collect all entities that co-occur
        as head/tail in the same triple (relation-type candidate narrowing).
        """
        print("[CostEstimator] Building neighbor_dict...")
        head_to_tails = defaultdict(set)
        tail_to_heads = defaultdict(set)
        for h, r, t in self.triples_list:
            head_to_tails[h].add(t)
            tail_to_heads[t].add(h)

        # For each entity, its candidate pool = all entities reachable via
        # any relation it participates in (head or tail).
        all_entities = set(range(self.num_entities))
        for e in range(self.num_entities):
            neighbors = head_to_tails.get(e, set()) | tail_to_heads.get(e, set())
            if len(neighbors) > 0:
                self.neighbor_dict[e] = list(neighbors)
            # Entities with no neighbors will use full entity list at runtime

        n_with_neighbors = len(self.neighbor_dict)
        print(f"[CostEstimator] neighbor_dict built: {n_with_neighbors}/{self.num_entities} entities have narrowed pools")

    def _compute_cost_table(self):
        """Compute expected_cost per entity using the geometric retry model."""
        print("[CostEstimator] Computing cost_table...")
        self.cost_table = np.zeros(self.num_entities, dtype=np.float32)

        for e in range(self.num_entities):
            neighbors = self.neighbor_dict.get(e, list(range(self.num_entities)))
            c_size = len(neighbors)

            # Geometric retry expectation
            if c_size <= self.neg_num:
                # Candidate pool is too small — max retries guaranteed
                e_retry = float(self.max_try)
            else:
                p_collision = self.neg_num / c_size
                e_retry = 1.0 / (1.0 - p_collision)
                e_retry = min(float(self.max_try), e_retry)

            self.cost_table[e] = e_retry * self.b3_const

        print(f"[CostEstimator] cost_table computed: "
              f"mean={self.cost_table.mean():.2f}ms, "
              f"max={self.cost_table.max():.2f}ms, "
              f"min={self.cost_table[self.cost_table > 0].min() if np.any(self.cost_table > 0) else 0:.2f}ms")

    def compute_batch_weight(self, batch_triples: List[Tuple[int, int, int]]) -> float:
        """
        Compute expected total sampling cost for a batch of triples.

        This is the function that the scheduler will call during FFD packing.

        Args:
            batch_triples: List of (head, relation, tail) triples in the batch.

        Returns:
            total_expected_cost: Predicted total sampling time in milliseconds.
        """
        total = 0.0
        seen = set()
        for h, r, t in batch_triples:
            # Count unique entities only (cost is per entity, not per triple)
            if h not in seen:
                total += self.get_cost(h)
                seen.add(h)
            if t not in seen:
                total += self.get_cost(t)
                seen.add(t)
        return total

    # ── Private: Cache ───────────────────────────────────────────────────

    def _save_cache(self):
        """Persist cost_table and neighbor_dict to disk."""
        os.makedirs(COST_CACHE_DIR, exist_ok=True)
        np.save(COST_CACHE_PATH, self.cost_table)
        with open(NEIGHBOR_CACHE_PATH, "wb") as f:
            pickle.dump(self.neighbor_dict, f)
        print(f"[CostEstimator] Cache saved: {COST_CACHE_PATH}, {NEIGHBOR_CACHE_PATH}")

    def _load_cache(self) -> Optional[np.ndarray]:
        """Load cached cost_table if available."""
        if os.path.exists(COST_CACHE_PATH) and os.path.exists(NEIGHBOR_CACHE_PATH):
            try:
                self.cost_table = np.load(COST_CACHE_PATH)
                with open(NEIGHBOR_CACHE_PATH, "rb") as f:
                    self.neighbor_dict = pickle.load(f)
                self._built = True
                print(f"[CostEstimator] Cache loaded: {COST_CACHE_PATH} "
                      f"({len(self.neighbor_dict)} entities in neighbor_dict)")
                return self.cost_table
            except Exception as e:
                print(f"[CostEstimator] Cache load failed: {e}. Recomputing.")
        return None

    def clear_cache(self):
        """Delete cached files."""
        for path in [COST_CACHE_PATH, NEIGHBOR_CACHE_PATH]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[CostEstimator] Cache cleared: {path}")


# ── Standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with FB15k-237
    from src.py.load.read import read_kge_dataset
    triples_path = "src/py/data/FB15K237/train2id.txt"
    triples, ents, rels = read_kge_dataset(triples_path)
    num_entities = max(max(h, t) for h, r, t in triples) + 1

    estimator = CostEstimator(list(triples), num_entities, neg_num=150, max_try=10)
    cost_table = estimator.build(force_recompute=True)

    # Print top-10 most expensive entities
    top_indices = np.argsort(cost_table)[-10:][::-1]
    print("\nTop-10 Most Expensive Entities:")
    for idx in top_indices:
        c_size = estimator.get_candidate_size(int(idx))
        print(f"  Entity {int(idx):6d}: cost={cost_table[idx]:.2f}ms, "
              f"candidate_size={c_size}")

    # Test batch weight computation
    sample_batch = list(triples)[:100]
    weight = estimator.compute_batch_weight(sample_batch)
    print(f"\nSample batch weight (100 triples): {weight:.2f}ms")