#!/usr/bin/env python3
"""
Phase 5.5: Algorithm Validation — DDBP Weight Assumption Validation

Tests the core hypothesis: theoretical batch weight (computed from entity degrees
and candidate pool size) correlates strongly with actual negative sampling time.

Requires R > 0.85 to proceed to Phase 6.
"""
import csv, math, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np

DATA_DIR = "src/py/data/FB15K237/"
TRAIN_PATH = os.path.join(DATA_DIR, "train2id.txt")
OUT_DIR = "output/results/"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_BATCHES = 400
BATCH_SIZE = 5000
NEG_NUM = 150
MAX_TRY = 10
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Load data ──
def load_triples(path):
    triples = []
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    print(f"[DATA] Loaded {len(triples)} training triples")
    return triples

triples = load_triples(TRAIN_PATH)
all_triples_set = set(triples)
num_entities = max(max(h, t) for h, r, t in triples) + 1
entities_list = list(range(num_entities))
print(f"[DATA] {num_entities} entities")

# ── Degree table ──
deg = Counter()
for h, r, t in triples:
    deg[h] += 1
    deg[t] += 1
print(f"[DEGREE] Mean={np.mean(list(deg.values())):.1f}, Max={max(deg.values())}")

# ── Simulation core: force Regime 2 with narrowed candidate pools ──
def simulate_candidate_size(entity_id, entity_degrees):
    """Simulate neighbor-dict narrowing: candidate pool size proportional to degree."""
    d = entity_degrees.get(entity_id, 1)
    # Simulate narrowed pool: between 20 and 3000 (always < 5000 for Regime 2)
    narrowed = min(max(int(d * 1.5), 20), 3000)
    return narrowed, min(narrowed, num_entities)

def theoretical_weight_for_batch(batch_triples, entity_degrees):
    """
    Compute theoretical batch weight = sum of E[retry] factors.
    weight ~ sum over triples of max(degree) * (1/(1-N_neg/candidate_size))
    """
    total_weight = 0.0
    candidate_sizes = []
    for h, r, t in batch_triples:
        d = max(entity_degrees.get(h, 1), entity_degrees.get(t, 1))
        c_size, _ = simulate_candidate_size(h if d == entity_degrees.get(h, 1) else t, entity_degrees)
        candidate_sizes.append(c_size)
        # Theoretical retry expectation (geometric)
        p_coll = NEG_NUM / max(c_size, 1)
        e_retry = 1.0 / max(1.0 - p_coll, 0.01)
        e_retry = min(MAX_TRY, e_retry)
        # Weight: d drives B1+B2 cost, e_retry drives B3+B4 cost
        weight = (d / max(c_size, 1)) * e_retry
        total_weight += weight
    avg_cs = np.mean(candidate_sizes) if candidate_sizes else num_entities
    return total_weight, avg_cs

def measure_actual_sampling_time(batch_triples):
    """Measure actual B1-B5 negative sampling time (ms)."""
    neg_batch = []
    t0 = time.perf_counter()
    for head, relation, tail in batch_triples:
        nums_to_sample = NEG_NUM
        c_size, candidates_full = simulate_candidate_size(head, deg)
        head_candidates = list(range(min(c_size, num_entities)))
        c_size_t, _ = simulate_candidate_size(tail, deg)
        tail_candidates = list(range(min(c_size_t, num_entities)))

        for i in range(MAX_TRY):
            corrupt_head = np.random.binomial(1, 0.5)
            if corrupt_head:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg = {(head, relation, t2) for t2 in neg_tails}
            if i == MAX_TRY - 1:
                neg_batch += list(i_neg)
                break
            filtered = list(i_neg - all_triples_set)
            neg_batch += filtered
            if len(neg_batch) >= NEG_NUM * (len(batch_triples) - len(neg_batch)//NEG_NUM):
                pass
            if len(neg_batch) == NEG_NUM * len(batch_triples):
                break
            nums_to_sample = NEG_NUM * (len(batch_triples) - len(neg_batch)//max(len(batch_triples),1))
            # simpler: break when done
        # simplified retry loop
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def measure_actual_sampling_time_v2(batch_triples):
    """More accurate measurement: replicate the real B1-B5 logic with narrowed pools."""
    t_start = time.perf_counter()
    total_retries = 0
    total_needed = len(batch_triples) * NEG_NUM
    
    for head, relation, tail in batch_triples:
        neg_triples = []
        nums_to_sample = NEG_NUM
        
        # Force Regime 2: artificially narrow candidate pools
        d_h = deg.get(head, 1)
        d_t = deg.get(tail, 1)
        c_size_h = min(max(int(d_h * 1.5), 20), 3000)
        c_size_t = min(max(int(d_t * 1.5), 20), 3000)
        head_candidates = list(range(min(c_size_h, num_entities)))
        tail_candidates = list(range(min(c_size_t, num_entities)))
        
        for i in range(MAX_TRY):
            corrupt_head = np.random.binomial(1, 0.5)
            if corrupt_head:
                k = min(nums_to_sample, len(head_candidates))
                if k <= 0:
                    total_retries += (i + 1)
                    break
                neg_heads = random.sample(head_candidates, k)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                k = min(nums_to_sample, len(tail_candidates))
                if k <= 0:
                    total_retries += (i + 1)
                    break
                neg_tails = random.sample(tail_candidates, k)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            
            if i == MAX_TRY - 1:
                neg_triples += list(i_neg_triples)
                total_retries += (i + 1)
                break
            else:
                filtered = list(i_neg_triples - all_triples_set)
                neg_triples += filtered
            
            if len(neg_triples) >= NEG_NUM:
                total_retries += (i + 1)
                break
            else:
                nums_to_sample = NEG_NUM - len(neg_triples)
    
    t_end = time.perf_counter()
    return (t_end - t_start) * 1000.0, total_retries / max(len(batch_triples), 1)


def sample_entity_by_degree_strategy(strategy="mixed"):
    """Sample entities with controlled degree distribution."""
    sorted_ents = sorted(deg.items(), key=lambda x: x[1])
    n = len(sorted_ents)
    
    if strategy == "hub_heavy":
        # Top 20% highest-degree entities
        start = int(n * 0.8)
        pool = [e for e, d in sorted_ents[start:]]
    elif strategy == "long_tail":
        # Bottom 60% lowest-degree entities
        end = int(n * 0.6)
        pool = [e for e, d in sorted_ents[:end]]
    elif strategy == "mixed":
        # 40% hub, 40% mid, 20% long-tail
        hub_start = int(n * 0.8)
        mid_start = int(n * 0.4)
        mid_end = int(n * 0.8)
        tail_end = int(n * 0.4)
        pool = ( [e for e, d in sorted_ents[hub_start:]] +
                 [e for e, d in sorted_ents[mid_start:mid_end]] +
                 [e for e, d in sorted_ents[:tail_end]] )
    else:  # uniform
        pool = [e for e, d in sorted_ents]
    
    return pool

def build_batch_from_entities(entity_pool, triples_list):
    """Build a batch of triples by sampling entities, then finding triples involving them."""
    selected = random.sample(entity_pool, min(BATCH_SIZE, len(entity_pool)))
    # Map entities to triples
    head_index = defaultdict(list)
    tail_index = defaultdict(list)
    for h, r, t in triples_list:
        head_index[h].append((h, r, t))
        tail_index[t].append((h, r, t))
    
    batch_triples = []
    used = set()
    for e in selected:
        candidates = head_index.get(e, []) + tail_index.get(e, [])
        for t in candidates:
            if t not in used and len(batch_triples) < BATCH_SIZE:
                batch_triples.append(t)
                used.add(t)
                if len(batch_triples) >= BATCH_SIZE:
                    break
        if len(batch_triples) >= BATCH_SIZE:
            break
    
    # Fill remaining if needed
    if len(batch_triples) < BATCH_SIZE:
        remaining = random.sample(
            [t for t in triples_list if t not in used],
            min(BATCH_SIZE - len(batch_triples), len(triples_list))
        )
        batch_triples.extend(remaining)
    
    return batch_triples[:BATCH_SIZE]

# ── Main validation ──
print("\n" + "=" * 60)
print("  Phase 5.5: DDBP Weight Assumption Validation")
print("=" * 60)

# Build entity pools for different strategies
hub_pool = sample_entity_by_degree_strategy("hub_heavy")
tail_pool = sample_entity_by_degree_strategy("long_tail")
mixed_pool = sample_entity_by_degree_strategy("mixed")
uniform_pool = sample_entity_by_degree_strategy("uniform")

print(f"\n[POOLS] Hub({len(hub_pool)}), Tail({len(tail_pool)}), Mixed({len(mixed_pool)})")

# Generate batches with diverse distributions
strategies = ["hub_heavy"] * 100 + ["long_tail"] * 100 + ["mixed"] * 100 + ["uniform"] * 100
strategies = strategies[:NUM_BATCHES]
random.shuffle(strategies)

pool_map = {
    "hub_heavy": hub_pool, "long_tail": tail_pool,
    "mixed": mixed_pool, "uniform": uniform_pool
}

results = []
print(f"\n── Generating {NUM_BATCHES} batches and measuring... ──")

for i, strat in enumerate(strategies):
    pool = pool_map[strat]
    batch_triples = build_batch_from_entities(pool, triples)
    
    # Theoretical weight
    weight, avg_cs = theoretical_weight_for_batch(batch_triples, deg)
    
    # Actual sampling time (with narrowed pools to force Regime 2)
    actual_ms, avg_retry = measure_actual_sampling_time_v2(batch_triples)
    
    results.append({
        "batch_id": i,
        "strategy": strat,
        "theoretical_weight": round(weight, 4),
        "actual_time_ms": round(actual_ms, 3),
        "avg_candidate_size": round(avg_cs, 1),
        "avg_retry": round(avg_retry, 4),
    })
    
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{NUM_BATCHES}] ...")

# Save CSV
csv_path = os.path.join(OUT_DIR, "weight_validation.md")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\n[SAVED] {csv_path}")

# ── Correlation Analysis ──
weights = np.array([r["theoretical_weight"] for r in results])
times = np.array([r["actual_time_ms"] for r in results])

def pearson_r(x, y):
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxy = sum(a*b for a,b in zip(x,y))
    sx2 = sum(a*a for a in x)
    sy2 = sum(b*b for b in y)
    d = math.sqrt((n*sx2-sx*sx)*(n*sy2-sy*sy))
    return (n*sxy - sx*sy)/d if d else 0

R = pearson_r(weights.tolist(), times.tolist())
R2 = R * R

print("\n" + "=" * 60)
print("  CORRELATION RESULTS")
print("=" * 60)
print(f"\n  Theoretical Weight vs Actual Sampling Time")
print(f"  ─────────────────────────────────────────")
print(f"  Pearson R  = {R:.6f}")
print(f"  R²         = {R2:.6f}")
print(f"  N batches  = {len(results)}")
print(f"\n  Per-strategy R:")
for strat in ["hub_heavy", "long_tail", "mixed", "uniform"]:
    sub = [r for r in results if r["strategy"] == strat]
    if len(sub) >= 3:
        w_sub = np.array([r["theoretical_weight"] for r in sub])
        t_sub = np.array([r["actual_time_ms"] for r in sub])
        r_sub = pearson_r(w_sub.tolist(), t_sub.tolist())
        print(f"    {strat:15s}: R = {r_sub:.4f}")

print(f"\n  Mean actual_time: {np.mean(times):.2f} ms")
print(f"  Mean theoretical_weight: {np.mean(weights):.4f}")
print(f"  Avg candidate_size range: [{min(r['avg_candidate_size'] for r in results):.0f}, "
      f"{max(r['avg_candidate_size'] for r in results):.0f}]")
print(f"  Avg retry range: [{min(r['avg_retry'] for r in results):.3f}, "
      f"{max(r['avg_retry'] for r in results):.3f}]")

# Decision
print("\n" + "=" * 60)
print("  DECISION")
print("=" * 60)
if R > 0.85:
    print(f"  ✅ R = {R:.4f} > 0.85 → Hypothesis VALIDATED")
    print(f"  ▶ Proceeding to Phase 6: DDBP Implementation")
else:
    print(f"  ❌ R = {R:.4f} ≤ 0.85 → Hypothesis REJECTED")
    print(f"  ▶ Need to revisit Cost Model before Phase 6")
print("=" * 60)

# Save summary
summary_path = os.path.join(OUT_DIR, "weight_validation_summary.md")
with open(summary_path, "w") as f:
    f.write(f"Phase 5.5: Weight Assumption Validation\n")
    f.write(f"========================================\n")
    f.write(f"Pearson R  = {R:.6f}\n")
    f.write(f"R²         = {R2:.6f}\n")
    f.write(f"N batches  = {len(results)}\n")
    f.write(f"Decision: {'VALIDATED' if R > 0.85 else 'REJECTED'}\n")
print(f"[SAVED] {summary_path}")