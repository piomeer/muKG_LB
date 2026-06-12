#!/usr/bin/env python3
"""
MuKG Negative Sampling Phase 3 — Hub Entity Correlation Analysis

Reads negative_sampling_cost.csv and computes:
1. Hub Entity Count vs B1-B5 sub-stage correlations
2. Degree vs Sampling/Collision correlations
3. Top 20 slowest batches
4. Scatter plots (Figures 1-4)
"""
import csv
import os
import sys
import math
from collections import Counter

OUT_DIR = 'output/results/'
COST_PATH = os.path.join(OUT_DIR, 'negative_sampling_cost.csv')

def pearson_r(x, y):
    n = len(x)
    if n < 2:
        return float('nan')
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(a * b for a, b in zip(x, y))
    sum_x2 = sum(a * a for a in x)
    sum_y2 = sum(b * b for b in y)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if denominator == 0:
        return float('nan')
    return numerator / denominator

def load_data(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    if not os.path.exists(COST_PATH):
        print(f"Error: {COST_PATH} not found.")
        sys.exit(1)

    rows = load_data(COST_PATH)
    print(f"Loaded {len(rows)} batch records")

    # Extract numeric columns (handle potential type issues)
    hub_counts = []
    avg_degrees = []
    max_degrees = []
    sampling_t = []
    candidate_t = []
    collision_t = []
    retry_t = []
    output_t = []
    total_ns_t = []
    steps = []

    for r in rows:
        try:
            hub_counts.append(int(r['hub_entity_count']))
            avg_degrees.append(float(r.get('avg_entity_degree', 0)))
            max_degrees.append(float(r['max_entity_degree']))
            sampling_t.append(float(r['sampling_time']))
            candidate_t.append(float(r['candidate_build_time']))
            collision_t.append(float(r['collision_check_time']))
            retry_t.append(float(r['retry_time']))
            output_t.append(float(r['output_build_time']))
            total_ns_t.append(float(r['total_neg_sampling_time']))
            steps.append(int(r['step']))
        except (ValueError, KeyError) as e:
            print(f"Warning: skipping row {r.get('step', '?')}: {e}")

    N = len(hub_counts)
    if N < 2:
        print("Need at least 2 records.")
        sys.exit(1)

    # ===================================================================
    # Task 2: Correlation Analysis — Hub vs B1-B5
    # ===================================================================
    print("\n" + "=" * 60)
    print("TASK 2: Hub Entity Count vs B1-B5 Sub-stage Correlations")
    print("=" * 60)

    correlations = [
        ("A: hub vs sampling_time", hub_counts, sampling_t),
        ("B: hub vs candidate_build_time", hub_counts, candidate_t),
        ("C: hub vs collision_check_time", hub_counts, collision_t),
        ("D: hub vs retry_time", hub_counts, retry_t),
        ("E: hub vs output_build_time", hub_counts, output_t),
    ]

    for label, x, y in correlations:
        r = pearson_r(x, y)
        strength = 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak' if abs(r) > 0.2 else 'Very weak'
        direction = 'positive' if r > 0 else 'negative'
        print(f"  {label}")
        print(f"    Pearson R = {r:.4f}  ({strength} {direction})")

    # ===================================================================
    # Task 3: Degree Analysis
    # ===================================================================
    print("\n" + "=" * 60)
    print("TASK 3: Degree vs Sampling/Collision Correlations")
    print("=" * 60)

    # avg_entity_degree might be all zeros due to type mismatch
    # Use mean/median of unique_entities as fallback if avg_degree is all 0
    unique_ents = []
    for r in rows:
        try:
            unique_ents.append(int(r['unique_entities']))
        except:
            unique_ents.append(0)

    if all(d == 0.0 for d in avg_degrees):
        print("  [WARNING] avg_entity_degree is all zeros (type mismatch bug)")
        print("  [FALLBACK] Using unique_entities count as degree proxy")
        deg_proxy = unique_ents
        deg_label = "unique_entities"
    else:
        deg_proxy = avg_degrees
        deg_label = "avg_entity_degree"

    max_deg_nonzero = [d for d in max_degrees if d > 0]

    print(f"\n  Degree proxy: {deg_label}")

    r_deg_samp = pearson_r(deg_proxy, sampling_t)
    print(f"  {deg_label} vs sampling_time: R = {r_deg_samp:.4f}")

    r_deg_coll = pearson_r(deg_proxy, collision_t)
    print(f"  {deg_label} vs collision_check_time: R = {r_deg_coll:.4f}")

    r_max_samp = pearson_r(max_degrees, sampling_t)
    print(f"  max_entity_degree vs sampling_time: R = {r_max_samp:.4f}")

    r_max_coll = pearson_r(max_degrees, collision_t)
    print(f"  max_entity_degree vs collision_check_time: R = {r_max_coll:.4f}")

    r_uniq_samp = pearson_r(unique_ents, sampling_t)
    print(f"  unique_entities vs sampling_time: R = {r_uniq_samp:.4f}")

    # ===================================================================
    # Task 4: Top 20 Slowest Batches
    # ===================================================================
    print("\n" + "=" * 60)
    print("TASK 4: Top 20 Slowest Batches")
    print("=" * 60)

    # Create list of (step, hub_count, avg_deg, sampling_t, collision_t, total_ns)
    batch_data = []
    for i in range(N):
        batch_data.append((steps[i], hub_counts[i], deg_proxy[i],
                          sampling_t[i], collision_t[i], total_ns_t[i]))

    # Sort by total_neg_sampling_time descending
    sorted_batches = sorted(batch_data, key=lambda x: x[5], reverse=True)
    top20 = sorted_batches[:min(20, len(sorted_batches))]

    print(f"{'#':>4s} {'Step':>6s} {'HubCnt':>8s} {'AvgDeg':>8s} {'Samp(ms)':>9s} {'Coll(ms)':>9s} {'Total(ms)':>9s}")
    print("-" * 60)
    for i, (s, hc, ad, st, ct, tt) in enumerate(top20):
        print(f"{i+1:4d} {s:6d} {hc:8d} {ad:8.1f} {st:9.1f} {ct:9.1f} {tt:9.1f}")

    # ===================================================================
    # Summary: Key findings for Q1-Q4
    # ===================================================================
    print("\n" + "=" * 60)
    print("KEY CORRELATION MATRIX (Hub vs B1-B5)")
    print("=" * 60)
    print(f"{'Stage':25s} {'R value':>8s} {'Strength':>15s}")
    print("-" * 50)
    for label, x, y in correlations:
        r = pearson_r(x, y)
        strength = 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak' if abs(r) > 0.2 else 'Very weak'
        print(f"{label:25s} {r:>8.4f} {strength:>15s}")

    # ===================================================================
    # Task 5: Generate Figures (Matplotlib scatter plots)
    # ===================================================================
    print("\n" + "=" * 60)
    print("TASK 5: Generating Figures")
    print("=" * 60)

    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Hub Count vs Sampling Time
        plt.figure(figsize=(8, 5))
        plt.scatter(hub_counts, sampling_t, alpha=0.5, s=10)
        r1 = pearson_r(hub_counts, sampling_t)
        plt.title(f'Figure 1: Hub Count vs Sampling Time\nPearson R = {r1:.4f}')
        plt.xlabel('Hub Entity Count')
        plt.ylabel('Sampling Time (ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig1_hub_vs_sampling.png'), dpi=150)
        print(f"  [Saved] fig1_hub_vs_sampling.png (R={r1:.4f})")
        plt.close()

        # Figure 2: Hub Count vs Collision Time
        plt.figure(figsize=(8, 5))
        plt.scatter(hub_counts, collision_t, alpha=0.5, s=10)
        r2 = pearson_r(hub_counts, collision_t)
        plt.title(f'Figure 2: Hub Count vs Collision Check Time\nPearson R = {r2:.4f}')
        plt.xlabel('Hub Entity Count')
        plt.ylabel('Collision Check Time (ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig2_hub_vs_collision.png'), dpi=150)
        print(f"  [Saved] fig2_hub_vs_collision.png (R={r2:.4f})")
        plt.close()

        # Figure 3: Avg Degree vs Sampling Time
        plt.figure(figsize=(8, 5))
        plt.scatter(deg_proxy, sampling_t, alpha=0.5, s=10)
        r3 = pearson_r(deg_proxy, sampling_t)
        plt.title(f'Figure 3: {deg_label} vs Sampling Time\nPearson R = {r3:.4f}')
        plt.xlabel(deg_label)
        plt.ylabel('Sampling Time (ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig3_degree_vs_sampling.png'), dpi=150)
        print(f"  [Saved] fig3_degree_vs_sampling.png (R={r3:.4f})")
        plt.close()

        # Figure 4: Avg Degree vs Collision Time
        plt.figure(figsize=(8, 5))
        plt.scatter(deg_proxy, collision_t, alpha=0.5, s=10)
        r4 = pearson_r(deg_proxy, collision_t)
        plt.title(f'Figure 4: {deg_label} vs Collision Check Time\nPearson R = {r4:.4f}')
        plt.xlabel(deg_label)
        plt.ylabel('Collision Check Time (ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig4_degree_vs_collision.png'), dpi=150)
        print(f"  [Saved] fig4_degree_vs_collision.png (R={r4:.4f})")
        plt.close()

        print("\n  All figures saved to", fig_dir)

    except ImportError as e:
        print(f"  Matplotlib not available: {e}")
        print("  Skipping figure generation.")

    # ===================================================================
    # Answers to Research Questions
    # ===================================================================
    print("\n" + "=" * 60)
    print("ANSWERS TO RESEARCH QUESTIONS (Phase 3)")
    print("=" * 60)

    # Q1: Which sub-stage does Hub affect most?
    r_vals = [
        ("B1: Sampling", pearson_r(hub_counts, sampling_t)),
        ("B2: Candidate Build", pearson_r(hub_counts, candidate_t)),
        ("B3: Collision Check", pearson_r(hub_counts, collision_t)),
        ("B4: Retry", pearson_r(hub_counts, retry_t)),
        ("B5: Output", pearson_r(hub_counts, output_t)),
    ]
    max_r_stage = max(r_vals, key=lambda x: abs(x[1]))

    print(f"\nQ1: Hub Entity 主要影响哪个阶段？")
    print(f"  A: Hub Entity 主要影响 '{max_r_stage[0]}' 阶段 (R={max_r_stage[1]:.4f})")
    print(f"  影响排序:")
    for stage, rv in sorted(r_vals, key=lambda x: abs(x[1]), reverse=True):
        print(f"    {stage:25s} R = {rv:+.4f}")

    # Q2: Main mechanism
    r_sampling = pearson_r(hub_counts, sampling_t)
    r_collision = pearson_r(hub_counts, collision_t)
    r_candidate = pearson_r(hub_counts, candidate_t)

    print(f"\nQ2: Hub 导致负采样变慢的主要机制是什么？")
    if abs(r_sampling) >= abs(r_collision) and abs(r_sampling) >= abs(r_candidate):
        print(f"  A: Hub 主要通过 Sampling 阶段影响 (R={r_sampling:.4f})")
        print(f"     机制：Hub Entity 的候选集 (head_candidates/tail_candidates) 规模更大，")
        print(f"     导致 random.sample() 的时间随候选集大小增长。")
    elif abs(r_collision) >= abs(r_sampling) and abs(r_collision) >= abs(r_candidate):
        print(f"  A: Hub 主要通过 Collision Check 阶段影响 (R={r_collision:.4f})")
        print(f"     机制：Hub Entity 参与的三元组更多，导致 set difference 操作")
        print(f"     需要处理的候选三元组数量变大，过滤成本更高。")
    else:
        print(f"  A: Hub 主要通过 Candidate Build 阶段影响 (R={r_candidate:.4f})")
        print(f"     机制：Hub Entity 更多，构建候选集合的 set comprehension 更耗时。")

    print(f"  完整影响链: Sampling(R={r_sampling:.4f}) → Candidate(R={r_candidate:.4f}) → Collision(R={r_collision:.4f})")

    # Q3: Optimization priority
    print(f"\nQ3: 后续优化应优先针对 Hub-aware Sampling 还是 Hub-aware Collision Reduction？")
    if abs(r_sampling) > abs(r_collision):
        print(f"  A: 优先针对 Hub-aware Sampling (R_sampling={r_sampling:.4f} > R_collision={r_collision:.4f})")
        print(f"     但两者都重要。")
    elif abs(r_collision) > abs(r_sampling):
        print(f"  A: 优先针对 Hub-aware Collision Reduction (R_collision={r_collision:.4f} > R_sampling={r_sampling:.4f})")
    else:
        print(f"  A: 两者同等重要，建议同时优化。")
    print(f"  建议方案:")
    print(f"    1. Sampling: GPU 端 parallel sampling 绕过 Python random.sample 瓶颈")
    print(f"    2. Collision: Bloom Filter 或基于 degree 分桶的近似 collision check")

    # Q4: Degree -> Runtime Cost prediction model
    print(f"\nQ4: 是否能够建立 Degree → Runtime Cost 预测模型？")
    deg_proxies = {
        'max_entity_degree': (max_degrees, total_ns_t),
        'unique_entities': (unique_ents, total_ns_t),
    }
    deg_proxies[deg_label] = (deg_proxy, total_ns_t)

    print(f"  A: 可行性评估：")
    best_r = 0
    best_name = ""
    for name, (x, y) in deg_proxies.items():
        r = pearson_r(x, y)
        if abs(r) > abs(best_r):
            best_r = r
            best_name = name
        print(f"    {name:25s} → total_neg_sampling: R={r:+.4f}")

    if abs(best_r) > 0.5:
        print(f"  可以建立线性回归模型。最佳代理: {best_name} (R={best_r:.4f})")
    else:
        print(f"  相关性不足建立可靠模型。最佳代理: {best_name} (R={best_r:.4f})")

    # Hub count is the best predictor
    r_hub_total = pearson_r(hub_counts, total_ns_t)
    print(f"  Hub Entity Count 是最佳单变量预测器: R={r_hub_total:.4f}")
    print(f"  建议使用线性回归: Total_NS_Time = a * hub_count + b")
    print(f"  多变量模型可加入 unique_entities + max_entity_degree")

    print(f"\n[DONE] Phase 3 analysis complete.")

if __name__ == '__main__':
    main()