#!/usr/bin/env python3
"""
Phase 5 - Step 2: MuKG Runtime Cost Model Probe Script

Loads FB15k-237 training data, generates 500 mini-batches with diverse feature
distributions, measures actual negative sampling time, and fits a
multivariate linear regression model to predict T_sampling.

Outputs:
  - output/results/cost_model_data.csv     (batch-level features + target)
  - output/results/cost_model_summary.txt  (regression coefficients, R²)
"""

import csv
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "src/py/data/FB15K237/"
TRAIN_PATH = os.path.join(DATA_DIR, "train2id.txt")
ENTITY_PATH = os.path.join(DATA_DIR, "entity2id.txt")
RELATION_PATH = os.path.join(DATA_DIR, "relation2id.txt")

OUT_DIR = "output/results/"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_BATCHES = 500
BATCH_SIZE = 5000
NEG_TRIPLE_NUM = 150
MAX_TRY = 10
HUB_PERCENTILE = 10  # Top 10% entities considered Hub

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Data Loading ───────────────────────────────────────────────────────────
def load_triples(path):
    """Load train2id.txt format: first line = count, then 'h r t' per line."""
    triples = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  # Skip count line
        parts = line.strip().split()
        if len(parts) == 3:
            h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            triples.append((h, r, t))
    print(f"[DATA] Loaded {len(triples)} training triples")
    return triples


def compute_entity_degrees(triples, num_entities):
    """Compute degree (count of appearances as head or tail) for each entity."""
    deg = Counter()
    for h, r, t in triples:
        deg[h] += 1
        deg[t] += 1
    return deg


def compute_hub_threshold(degree_counter, percentile=10):
    """Return the degree threshold above which an entity is considered a Hub."""
    if len(degree_counter) == 0:
        return 0
    sorted_deg = sorted(degree_counter.values(), reverse=True)
    idx = max(1, len(sorted_deg) * percentile // 100)
    return sorted_deg[idx - 1]


def load_entity_count(path):
    with open(path, "r") as f:
        first_line = f.readline().strip()
    return int(first_line)


# ── Mini-Batch Generator ───────────────────────────────────────────────────
class BatchFeatureGenerator:
    """
    Generates mini-batches with controlled feature diversity by sampling
    from different regions of the entity degree distribution.
    """

    def __init__(self, triples, entity_degrees, hub_threshold, num_entities):
        self.triples = triples
        self.all_triples_set = set(triples)
        self.entity_degrees = entity_degrees
        self.hub_threshold = hub_threshold
        self.num_entities = num_entities
        self.entities_list = list(range(num_entities))

        # Pre-compute which entities are Hub (degree >= threshold)
        self.is_hub = np.array(
            [entity_degrees.get(e, 0) >= hub_threshold for e in range(num_entities)]
        )

        # Index triples by head and tail for faster candidate lookup
        self.head_to_triples = defaultdict(set)
        self.tail_to_triples = defaultdict(set)
        for h, r, t in triples:
            self.head_to_triples[h].add((h, r, t))
            self.tail_to_triples[t].add((h, r, t))

        # Pre-compute entity degree list for quick avg computation
        self.degree_list = np.array([entity_degrees.get(e, 0) for e in range(num_entities)])

    def _sample_batch_with_hub_ratio(self, target_hub_ratio):
        """
        Create a batch where approximately `target_hub_ratio` fraction of
        entities are Hub entities.
        """
        hub_indices = np.where(self.is_hub)[0]
        non_hub_indices = np.where(~self.is_hub)[0]

        n_hub = max(1, int(BATCH_SIZE * target_hub_ratio))
        n_non_hub = BATCH_SIZE - n_hub

        # Clamp to available entities
        n_hub = min(n_hub, len(hub_indices))
        n_non_hub = min(n_non_hub, len(non_hub_indices))

        selected = []
        if n_hub > 0:
            selected.extend(np.random.choice(hub_indices, n_hub, replace=False).tolist())
        if n_non_hub > 0:
            selected.extend(
                np.random.choice(non_hub_indices, n_non_hub, replace=False).tolist()
            )

        # Sample triples that involve these entities
        batch_triples = []
        used_set = set()
        # Shuffle selected entities to randomize which entity we try first
        sel_shuffled = list(selected)
        random.shuffle(sel_shuffled)

        for ent in sel_shuffled:
            # Get triples where this entity is head or tail
            candidate_triples = list(self.head_to_triples.get(ent, set()) |
                                     self.tail_to_triples.get(ent, set()))
            random.shuffle(candidate_triples)
            for t in candidate_triples:
                if t not in used_set and len(batch_triples) < BATCH_SIZE:
                    batch_triples.append(t)
                    used_set.add(t)
                    if len(batch_triples) >= BATCH_SIZE:
                        break
            if len(batch_triples) >= BATCH_SIZE:
                break

        # If we don't have enough triples through entity filtering, sample randomly
        if len(batch_triples) < BATCH_SIZE:
            remaining = random.sample(
                [t for t in self.triples if t not in used_set],
                min(BATCH_SIZE - len(batch_triples),
                    len(self.triples) - len(used_set))
            )
            batch_triples.extend(remaining)

        return batch_triples[:BATCH_SIZE]

    def generate_batches(self):
        """
        Generate NUM_BATCHES batches with diverse hub ratios.
        Strategy: sweep hub_ratio from 0.0 to 1.0 in steps, plus random.
        """
        # Systematic sweep of hub ratios
        hub_ratios = []
        # Linear sweep
        for i in range(20):
            hub_ratios.append(i / 20.0)
        # Add extra density near extremes (0-0.2 and 0.8-1.0)
        for i in range(10):
            hub_ratios.append(np.random.uniform(0.0, 0.2))
            hub_ratios.append(np.random.uniform(0.8, 1.0))
        # Add random
        for _ in range(NUM_BATCHES - len(hub_ratios)):
            hub_ratios.append(np.random.uniform(0.0, 1.0))

        hub_ratios = hub_ratios[:NUM_BATCHES]

        batches = []
        for i, ratio in enumerate(hub_ratios):
            batch_triples = self._sample_batch_with_hub_ratio(ratio)
            batches.append(batch_triples)
            if (i + 1) % 100 == 0:
                print(f"[BATCH] Generated {i+1}/{NUM_BATCHES} batches...")

        return batches


# ── Cost Measurement (Replicating the real sampling logic) ─────────────────
def measure_sampling_cost(batch_triples, all_triples_set, entity_degrees,
                          hub_threshold, num_entities, neg_num, max_try):
    """
    Replicates the exact negative sampling logic from `_deep_profiled_neg_sampling`
    in pytorch_dataloader.py (B1-B5), measures per-batch timing and collects features.

    Returns:
        features: dict of input variables for this batch
        time_ms: total negative sampling time in ms
    """
    entities_list = list(range(num_entities))
    neg_batch = []

    # ── Compute Batch Features ──
    batch_entities = set()
    for h, r, t in batch_triples:
        batch_entities.add(h)
        batch_entities.add(t)

    # Hub count: entities in batch whose degree >= hub_threshold
    hub_count = sum(1 for e in batch_entities if entity_degrees.get(e, 0) >= hub_threshold)

    # Avg degree of batch entities
    avg_degree = np.mean([entity_degrees.get(e, 0) for e in batch_entities]) if batch_entities else 0.0

    # Max degree
    max_degree = max([entity_degrees.get(e, 0) for e in batch_entities], default=0)

    # Unique entities in batch
    unique_count = len(batch_entities)

    # ── Measure B1-B5 (replicate real logic with timers) ──
    t_start = time.perf_counter()

    total_candidate_size = 0
    total_collision_ops = 0
    total_samples_attempted = 0
    total_retries = 0

    b1_time = 0.0
    b2_time = 0.0
    b3_time = 0.0
    b4_time = 0.0
    b5_time = 0.0

    for head, relation, tail in batch_triples:
        neg_triples = []
        nums_to_sample = neg_num
        head_candidates = entities_list  # No neighbor dict in base case
        tail_candidates = entities_list

        for i in range(max_try):
            # ── B1: Random Sampling ──
            t0 = time.perf_counter()
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
            b1_time += time.perf_counter() - t0

            total_samples_attempted += nums_to_sample
            total_candidate_size += len(head_candidates if corrupt_head_prob else tail_candidates)

            # ── B2: Candidate Construction ──
            t0 = time.perf_counter()
            if corrupt_head_prob:
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            b2_time += time.perf_counter() - t0

            if i == max_try - 1:
                # ── B5: Final append ──
                t0 = time.perf_counter()
                neg_triples += list(i_neg_triples)
                b5_time += time.perf_counter() - t0
                break
            else:
                # ── B3: Collision Check (set difference) ──
                t0 = time.perf_counter()
                filtered = list(i_neg_triples - all_triples_set)
                b3_time += time.perf_counter() - t0
                total_collision_ops += len(i_neg_triples)

                # ── B5: extend ──
                t0 = time.perf_counter()
                neg_triples += filtered
                b5_time += time.perf_counter() - t0

            # ── B4: Retry check ──
            t0 = time.perf_counter()
            if len(neg_triples) == neg_num:
                b4_time += time.perf_counter() - t0
                break
            else:
                nums_to_sample = neg_num - len(neg_triples)
                b4_time += time.perf_counter() - t0

        total_retries += i + 1

        # ── B5: extend neg_batch ──
        t0 = time.perf_counter()
        neg_batch.extend(neg_triples)
        b5_time += time.perf_counter() - t0

    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000.0

    # Collision rate: fraction of generated candidates that collided
    total_generated = total_samples_attempted
    total_accepted = len(neg_batch)
    collision_rate = (total_generated - total_accepted) / max(total_generated, 1)

    features = {
        "hub_count": hub_count,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "unique_entities": unique_count,
        "avg_retry": round(total_retries / len(batch_triples), 4),
        "total_retries": total_retries,
        "avg_candidate_size": round(total_candidate_size / max(len(batch_triples), 1), 1),
        "collision_rate": round(collision_rate, 4),
        "total_collision_ops": total_collision_ops,
        "total_samples_attempted": total_samples_attempted,
        # Sub-stage times (ms)
        "b1_time_ms": round(b1_time * 1000.0, 3),
        "b2_time_ms": round(b2_time * 1000.0, 3),
        "b3_time_ms": round(b3_time * 1000.0, 3),
        "b4_time_ms": round(b4_time * 1000.0, 3),
        "b5_time_ms": round(b5_time * 1000.0, 3),
    }

    return features, total_time_ms


# ── Multivariate Linear Regression ─────────────────────────────────────────
def fit_regression(X, y, feature_names):
    """
    Fit multivariate linear regression: y = X @ beta + intercept
    Uses closed-form OLS.

    Returns: (coeffs, intercept, r_squared, adj_r_squared)
    """
    n = X.shape[0]
    # Add bias term
    X_with_bias = np.column_stack([np.ones(n), X])

    # OLS: beta = (X^T X)^{-1} X^T y
    try:
        beta = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("[REGRESSION] LinAlgError, using pseudo-inverse")
        beta = np.linalg.pinv(X_with_bias) @ y

    intercept = beta[0]
    coeffs = beta[1:]

    # R²
    y_pred = X_with_bias @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-10)

    # Adjusted R²
    p = X.shape[1]
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / max((n - p - 1), 1)

    return coeffs, intercept, r_squared, adj_r_squared, y_pred


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MuKG Runtime Cost Model Fitting (Phase 5 - Step 2)")
    print("=" * 60)

    # 1. Load data
    print("\n── Loading FB15k-237 data ──")
    triples = load_triples(TRAIN_PATH)
    num_entities = load_entity_count(ENTITY_PATH)
    print(f"  Entities: {num_entities}")

    # 2. Compute degrees and hub threshold
    entity_degrees = compute_entity_degrees(triples, num_entities)
    hub_threshold = compute_hub_threshold(entity_degrees, HUB_PERCENTILE)
    print(f"  Hub threshold (Top {HUB_PERCENTILE}%): degree >= {hub_threshold}")
    hub_count_total = sum(1 for d in entity_degrees.values() if d >= hub_threshold)
    print(f"  Hub entities: {hub_count_total} / {num_entities} ({100*hub_count_total/num_entities:.1f}%)")

    # 3. Generate batches
    print(f"\n── Generating {NUM_BATCHES} mini-batches ──")
    generator = BatchFeatureGenerator(triples, entity_degrees, hub_threshold, num_entities)
    batches = generator.generate_batches()
    print(f"  Generated {len(batches)} batches (batch_size={BATCH_SIZE})")

    # 4. Measure sampling cost for each batch
    print(f"\n── Measuring sampling cost for {len(batches)} batches ──")
    all_records = []
    all_features_list = []

    for idx, batch in enumerate(batches):
        features, time_ms = measure_sampling_cost(
            batch, set(triples), entity_degrees,
            hub_threshold, num_entities, NEG_TRIPLE_NUM, MAX_TRY
        )
        features["batch_id"] = idx
        features["T_sampling_ms"] = round(time_ms, 3)
        all_records.append(features)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(batches)}] T_sampling={time_ms:.1f}ms, "
                  f"hub={features['hub_count']}, "
                  f"collision_rate={features['collision_rate']:.3f}")

    # 5. Save CSV
    csv_path = os.path.join(OUT_DIR, "cost_model_data.csv")
    fieldnames = list(all_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[SAVED] {csv_path} ({len(all_records)} rows)")

    # 6. Fit regression models
    print("\n" + "=" * 60)
    print("  REGRESSION ANALYSIS")
    print("=" * 60)

    # Extract feature matrix
    X_names_candidates = [
        "hub_count",
        "avg_degree",
        "max_degree",
        "unique_entities",
        "avg_retry",
        "collision_rate",
        "total_collision_ops",
        "total_samples_attempted",
        "avg_candidate_size",
    ]

    X = np.array([[r[name] for name in X_names_candidates] for r in all_records])
    y = np.array([r["T_sampling_ms"] for r in all_records])

    # --- Full Model ---
    print("\n── Full Model (all features) ──")
    coeffs, intercept, r2, adj_r2, y_pred = fit_regression(X, y, X_names_candidates)
    print(f"  R² = {r2:.6f}")
    print(f"  Adjusted R² = {adj_r2:.6f}")
    print(f"  Intercept = {intercept:.4f} ms")
    print(f"\n  Coefficients:")
    for name, c in zip(X_names_candidates, coeffs):
        print(f"    {name:30s}: {c:+.6f}  ms/unit")

    # --- Feature Importance: Standardized Coefficients (Beta Weights) ---
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    _, _, _, _, _ = fit_regression(X_std, y, X_names_candidates)
    coeffs_std, _, r2_std, _, _ = fit_regression(X_std, y, X_names_candidates)
    print(f"\n── Standardized Coefficients (Beta Weights) ──")
    importance = sorted(zip(X_names_candidates, coeffs_std),
                        key=lambda x: abs(x[1]), reverse=True)
    for name, c in importance:
        bar = "█" * int(abs(c) * 10)
        print(f"    {name:30s}: β = {c:+.4f}  {bar}")

    # --- Reduced Model: Top 3 features only ---
    print("\n── Reduced Model (Top 3 features by |β|) ──")
    top3_names = [name for name, _ in importance[:3]]
    top3_indices = [X_names_candidates.index(n) for n in top3_names]
    X_top3 = X[:, top3_indices]
    coeffs_r, intercept_r, r2_r, adj_r2_r, _ = fit_regression(X_top3, y, top3_names)
    print(f"  Features: {top3_names}")
    print(f"  R² = {r2_r:.6f}")
    print(f"  Adjusted R² = {adj_r2_r:.6f}")
    print(f"  Intercept = {intercept_r:.4f} ms")
    for name, c in zip(top3_names, coeffs_r):
        print(f"    {name:30s}: {c:+.6f} ms/unit")

    # --- Single-feature R² (for feature importance ranking) ---
    print(f"\n── Single-Feature R² Ranking ──")
    single_r2s = []
    for i, name in enumerate(X_names_candidates):
        X_single = X[:, i:i+1]
        _, _, r2_s, _, _ = fit_regression(X_single, y, [name])
        single_r2s.append((name, r2_s))
    single_r2s.sort(key=lambda x: x[1], reverse=True)
    for name, r2_s in single_r2s:
        bar = "█" * int(r2_s * 50)
        print(f"    {name:30s}: R² = {r2_s:.4f}  {bar}")

    # --- Residual Analysis ---
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    print(f"\n── Residual Metrics ──")
    print(f"  RMSE = {rmse:.4f} ms")
    print(f"  MAE  = {mae:.4f} ms")
    print(f"  Mean T_sampling = {np.mean(y):.4f} ms")
    print(f"  CV (RMSE/Mean)  = {rmse / max(np.mean(y), 1e-10):.4f}")

    # 7. Save summary text
    summary_path = os.path.join(OUT_DIR, "cost_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MuKG Runtime Cost Model Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: FB15k-237 ({num_entities} entities, {len(triples)} triples)\n")
        f.write(f"Batches: {NUM_BATCHES}, Batch Size: {BATCH_SIZE}\n")
        f.write(f"Neg Triple Num: {NEG_TRIPLE_NUM}, Max Try: {MAX_TRY}\n")
        f.write(f"Hub Threshold (Top {HUB_PERCENTILE}%): degree >= {hub_threshold}\n\n")

        f.write("Full Model (all features):\n")
        f.write(f"  R² = {r2:.6f}, Adjusted R² = {adj_r2:.6f}\n")
        f.write(f"  Intercept = {intercept:.4f} ms\n")
        f.write(f"  Equation: T_sampling = {intercept:.4f}\n")
        for name, c in zip(X_names_candidates, coeffs):
            f.write(f"    + ({c:+.6f}) × {name}\n")

        f.write(f"\nStandardized Coefficients (Feature Importance):\n")
        for name, c in importance:
            f.write(f"  {name:30s}: β = {c:+.4f}\n")

        f.write(f"\nReduced Model (Top 3):\n")
        f.write(f"  Features: {top3_names}\n")
        f.write(f"  R² = {adj_r2_r:.6f}\n")
        f.write(f"  Equation: T_sampling = {intercept_r:.4f}\n")
        for name, c in zip(top3_names, coeffs_r):
            f.write(f"    + ({c:+.6f}) × {name}\n")

        f.write(f"\nResidual Metrics:\n")
        f.write(f"  RMSE = {rmse:.4f} ms\n")
        f.write(f"  MAE  = {mae:.4f} ms\n")
        f.write(f"  Mean T_sampling = {np.mean(y):.4f} ms\n")
        f.write(f"  CV = {rmse / max(np.mean(y), 1e-10):.4f}\n")
    print(f"\n[SAVED] {summary_path}")

    print("\n[DONE] Cost model fitting complete.")
    print(f"  CSV:  {csv_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()