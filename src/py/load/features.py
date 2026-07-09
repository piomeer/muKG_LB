"""
Phase 6 - Node 3, Stage A: Feature Extraction
===============================================
Topological feature extraction — purely data-level, cost-model agnostic.

Design principle:
    FeatureExtractor extracts and persists graph topological features (e.g.,
    candidate_size, degree) as `entity_features.npy`. This is a ONE-TIME cost.
    
    Future cost models (GPU sampling, Bloom Filter, etc.) can reuse these
    features by applying different mapping functions — no need to re-traverse
    the graph.
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


FEATURE_CACHE_DIR = "output/results/"
FEATURE_CACHE_PATH = os.path.join(FEATURE_CACHE_DIR, "entity_features.npz")
NEIGHBOR_CACHE_PATH = os.path.join(FEATURE_CACHE_DIR, "neighbor_dict.pkl")


class FeatureExtractor:
    """
    Extract topological features from KG triples.

    Output entity_features dict contains:
        - candidate_size:  |neighbor_dict[e]| — size of narrowed candidate pool
        - degree:          number of occurrences as head or tail
        - hub_flag:        whether entity is in top 10% by degree

    All features are cached to disk; extraction is a one-time O(V+E) cost.
    """

    def __init__(self, triples_list: List[Tuple[int, int, int]],
                 num_entities: int):
        self.triples_list = triples_list
        self.num_entities = num_entities
        self.features: dict = {}
        self.neighbor_dict: Dict[int, List[int]] = {}

    def build(self, force_recompute: bool = False) -> dict:
        """
        Extract features. Loads from cache if available.

        Returns:
            features dict with keys:
                'candidate_size': np.ndarray (num_entities,)
                'degree':         np.ndarray (num_entities,)
                'hub_flag':       np.ndarray (num_entities,) bool
        """
        if not force_recompute:
            loaded = self._load_cache()
            if loaded is not None:
                return loaded

        self._compute_neighbor_dict()
        self._compute_degrees()
        self._compute_features()
        self._save_cache()
        return self.features

    def get_feature(self, entity_id: int, feature_name: str = "candidate_size") -> float:
        """O(1) feature lookup."""
        if feature_name not in self.features:
            raise KeyError(f"Unknown feature: {feature_name}. "
                           f"Available: {list(self.features.keys())}")
        if entity_id < 0 or entity_id >= self.num_entities:
            return 0.0
        return float(self.features[feature_name][entity_id])

    # ── Private ──────────────────────────────────────────────────────────

    def _compute_neighbor_dict(self):
        """Build neighbor_dict from triples (relation-type candidate narrowing)."""
        print("[FeatureExtractor] Building neighbor_dict...")
        head_to_tails = defaultdict(set)
        tail_to_heads = defaultdict(set)
        for h, r, t in self.triples_list:
            head_to_tails[h].add(t)
            tail_to_heads[t].add(h)

        for e in range(self.num_entities):
            neighbors = head_to_tails.get(e, set()) | tail_to_heads.get(e, set())
            if len(neighbors) > 0:
                self.neighbor_dict[e] = list(neighbors)

        n = len(self.neighbor_dict)
        print(f"[FeatureExtractor] neighbor_dict: {n}/{self.num_entities} entities have narrowed pools")

    def _compute_degrees(self):
        """Compute entity degree (head + tail occurrences)."""
        self._degree = np.zeros(self.num_entities, dtype=np.int32)
        for h, r, t in self.triples_list:
            self._degree[h] += 1
            self._degree[t] += 1

    def _compute_features(self):
        """Assemble all features into a single dict."""
        candidate_size = np.full(self.num_entities, self.num_entities, dtype=np.int32)
        for e, neighbors in self.neighbor_dict.items():
            candidate_size[e] = len(neighbors)

        # Hub flag: top 10% by degree
        sorted_deg = np.sort(self._degree)[::-1]
        hub_threshold = sorted_deg[max(1, len(sorted_deg) * 10 // 100) - 1]
        hub_flag = self._degree >= hub_threshold

        self.features = {
            "candidate_size": candidate_size,
            "degree": self._degree,
            "hub_flag": hub_flag,
        }
        print(f"[FeatureExtractor] Features computed: "
              f"candidate_size mean={candidate_size.mean():.1f}, "
              f"degree mean={self._degree.mean():.1f}, "
              f"hub_count={hub_flag.sum()}")

    def _save_cache(self):
        os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
        np.savez_compressed(FEATURE_CACHE_PATH,
                            candidate_size=self.features["candidate_size"],
                            degree=self.features["degree"],
                            hub_flag=self.features["hub_flag"])
        with open(NEIGHBOR_CACHE_PATH, "wb") as f:
            pickle.dump(self.neighbor_dict, f)
        print(f"[FeatureExtractor] Cache saved: {FEATURE_CACHE_PATH}")

    def _load_cache(self) -> Optional[dict]:
        if os.path.exists(FEATURE_CACHE_PATH) and os.path.exists(NEIGHBOR_CACHE_PATH):
            try:
                data = np.load(FEATURE_CACHE_PATH)
                with open(NEIGHBOR_CACHE_PATH, "rb") as f:
                    self.neighbor_dict = pickle.load(f)
                self.features = {
                    "candidate_size": data["candidate_size"],
                    "degree": data["degree"],
                    "hub_flag": data["hub_flag"],
                }
                print(f"[FeatureExtractor] Cache loaded: {FEATURE_CACHE_PATH}")
                return self.features
            except Exception as e:
                print(f"[FeatureExtractor] Cache load failed: {e}.")
        return None