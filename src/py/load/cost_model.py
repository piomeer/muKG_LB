"""
Phase 6 - Node 3, Stage A: Cost Model — Pure function from features to cost
=============================================================================
Stateless cost mapping: given entity features → expected sampling cost.

Design principle:
    CostModel is a PURE function. It reads pre-computed features from
    FeatureExtractor and applies a mathematical formula to produce cost_table.
    
    To swap cost models (e.g., GPU sampling cost → different constants),
    simply replace this module. No feature re-extraction required.
"""

import numpy as np
from typing import Optional, Tuple


def build_cost_table(features: dict,
                     neg_num: int = 150,
                     max_try: int = 10,
                     b3_const: float = 51.8) -> np.ndarray:
    """
    Build cost table from pre-computed features.

    Core formula (validated Phase 5.5, R=0.9008):

        candidate_size(e) = features['candidate_size'][e]
        P_collision       = N_neg / candidate_size(e)
        E_retry           = min(max_try, 1 / (1 - P_collision))
        expected_cost(e)  = E_retry * B3_const

    Args:
        features:   Dict from FeatureExtractor.build()
        neg_num:    Number of negative samples per triple.
        max_try:    Maximum sampling retries.
        b3_const:   Fixed cost of set-difference collision check (ms).

    Returns:
        cost_table: np.ndarray of shape (num_entities,) — expected cost per entity.
    """
    candidate_size = features["candidate_size"]
    num_entities = len(candidate_size)

    cost_table = np.zeros(num_entities, dtype=np.float32)

    for e in range(num_entities):
        c_size = int(candidate_size[e])

        if c_size <= neg_num:
            e_retry = float(max_try)
        else:
            p_collision = neg_num / c_size
            e_retry = 1.0 / (1.0 - p_collision)
            e_retry = min(float(max_try), e_retry)

        cost_table[e] = e_retry * b3_const

    print(f"[CostModel] cost_table built: "
          f"mean={cost_table.mean():.2f}ms, "
          f"max={cost_table.max():.2f}ms, "
          f"min={cost_table[cost_table > 0].min() if np.any(cost_table > 0) else 0:.2f}ms, "
          f"shape={cost_table.shape}")

    return cost_table


def compute_batch_weight(batch_triples, cost_table: np.ndarray) -> float:
    """
    Compute total expected cost for a batch of triples.

    Args:
        batch_triples: List of (head, relation, tail).
        cost_table:    Pre-computed cost_table from build_cost_table().

    Returns:
        total_expected_cost in milliseconds.
    """
    total = 0.0
    seen = set()
    for h, r, t in batch_triples:
        if h not in seen and h < len(cost_table):
            total += float(cost_table[h])
            seen.add(h)
        if t not in seen and t < len(cost_table):
            total += float(cost_table[t])
            seen.add(t)
    return total


def compute_triple_cost(triple, cost_table: np.ndarray) -> float:
    """O(1) per-triple cost = max(head_cost, tail_cost)."""
    h, r, t = triple
    hc = float(cost_table[h]) if h < len(cost_table) else 0.0
    tc = float(cost_table[t]) if t < len(cost_table) else 0.0
    return max(hc, tc)