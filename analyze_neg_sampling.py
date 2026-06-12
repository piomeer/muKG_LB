#!/usr/bin/env python3
"""
MuKG Negative Sampling Deep Profiling Analysis (Phase 2)

Reads negative_sampling_cost.csv and computes:
- Correlation A: hub_entity_count vs total_neg_sampling_time
- Correlation B: avg_entity_degree vs total_neg_sampling_time
- Correlation C: avg_retry vs total_neg_sampling_time
- Correlation D: collision_check_time vs total_neg_sampling_time
- Negative Sampling Runtime Breakdown
"""
import csv
import os
import sys
import math

OUT_DIR = 'output/results/'
COST_PATH = os.path.join(OUT_DIR, 'negative_sampling_cost.csv')
BREAKDOWN_PATH = os.path.join(OUT_DIR, 'negative_sampling_breakdown.csv')

def pearson_r(x, y):
    """Compute Pearson correlation coefficient R."""
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

def load_cost_data(path):
    """Load negative_sampling_cost.csv into list of dicts."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    if not os.path.exists(COST_PATH):
        print(f"Error: {COST_PATH} not found. Run the experiment first.")
        print("Run: python3 -u src/py/experiments/main_FB15K237.py 2>&1 | tee experiment_output.log")
        sys.exit(1)
    
    rows = load_cost_data(COST_PATH)
    print(f"Loaded {len(rows)} batch records from {COST_PATH}")
    
    if len(rows) < 2:
        print("Need at least 2 records for correlation analysis.")
        sys.exit(1)
    
    # Extract columns
    epochs = [int(r['epoch']) for r in rows]
    steps = [int(r['step']) for r in rows]
    batch_sizes = [int(r['batch_size']) for r in rows]
    
    # B1-B5 sub-stage times
    sampling_t = [float(r['sampling_time']) for r in rows]
    candidate_t = [float(r['candidate_build_time']) for r in rows]
    collision_t = [float(r['collision_check_time']) for r in rows]
    retry_sub_t = [float(r['retry_time']) for r in rows]
    output_t = [float(r['output_build_time']) for r in rows]
    total_ns_t = [float(r['total_neg_sampling_time']) for r in rows]
    
    # Graph structure
    hub_counts = [int(r['hub_entity_count']) for r in rows]
    avg_degrees = [float(r['avg_entity_degree']) for r in rows]
    avg_retries = [float(r['avg_retry']) for r in rows]
    max_retries = [float(r['max_retry']) for r in rows]
    
    # === Task 6: Runtime Breakdown ===
    print("\n" + "="*60)
    print("Negative Sampling Runtime Breakdown (Task 6)")
    print("="*60)
    
    total_b1 = sum(sampling_t)
    total_b2 = sum(candidate_t)
    total_b3 = sum(collision_t)
    total_b4 = sum(retry_sub_t)
    total_b5 = sum(output_t)
    total_sum_b = total_b1 + total_b2 + total_b3 + total_b4 + total_b5
    total_ns = sum(total_ns_t)
    
    components = [
        ('B1: Sampling', total_b1),
        ('B2: Candidate Build', total_b2),
        ('B3: Collision Check', total_b3),
        ('B4: Retry', total_b4),
        ('B5: Output Build', total_b5),
    ]
    
    print(f"{'Component':25s} {'Time(ms)':>12s} {'% of NS':>10s}")
    print("-" * 50)
    for label, val in components:
        pct = (val / total_ns * 100) if total_ns > 0 else 0
        print(f"{label:25s} {val:>12.1f} {pct:>9.2f}%")
    print("-" * 50)
    print(f"{'Total (B1-B5 sum)':25s} {total_sum_b:>12.1f} {(total_sum_b/total_ns*100) if total_ns>0 else 0:>9.2f}%")
    print(f"{'Total (neg_sampling)':25s} {total_ns:>12.1f} {'100.00%':>10s}")
    
    # Save breakdown
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BREAKDOWN_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Component', 'Time_ms', 'Ratio_pct'])
        for label, val in components:
            pct = (val / total_ns * 100) if total_ns > 0 else 0
            w.writerow([label, round(val, 3), round(pct, 2)])
        w.writerow(['Total (B1-B5)', round(total_sum_b, 3), round(total_sum_b/total_ns*100, 2) if total_ns>0 else 0])
        w.writerow(['Total (neg_sampling)', round(total_ns, 3), 100.0])
    print(f"\n[Saved] {BREAKDOWN_PATH}")
    
    # === Correlation Analysis ===
    print("\n" + "="*60)
    print("Correlation Analysis (Task 5)")
    print("="*60)
    
    # A: hub_entity_count vs total_neg_sampling_time
    r_a = pearson_r(hub_counts, total_ns_t)
    print(f"\nA: hub_entity_count vs total_neg_sampling_time")
    print(f"   Pearson R = {r_a:.4f}")
    print(f"   Interpretation: {'Strong' if abs(r_a) > 0.7 else 'Moderate' if abs(r_a) > 0.4 else 'Weak' if abs(r_a) > 0.2 else 'Very weak'} {'positive' if r_a > 0 else 'negative'} correlation")
    
    # B: avg_entity_degree vs total_neg_sampling_time
    r_b = pearson_r(avg_degrees, total_ns_t)
    print(f"\nB: avg_entity_degree vs total_neg_sampling_time")
    print(f"   Pearson R = {r_b:.4f}")
    print(f"   Interpretation: {'Strong' if abs(r_b) > 0.7 else 'Moderate' if abs(r_b) > 0.4 else 'Weak' if abs(r_b) > 0.2 else 'Very weak'} {'positive' if r_b > 0 else 'negative'} correlation")
    
    # C: avg_retry vs total_neg_sampling_time
    r_c = pearson_r(avg_retries, total_ns_t)
    print(f"\nC: avg_retry vs total_neg_sampling_time")
    print(f"   Pearson R = {r_c:.4f}")
    print(f"   Interpretation: {'Strong' if abs(r_c) > 0.7 else 'Moderate' if abs(r_c) > 0.4 else 'Weak' if abs(r_c) > 0.2 else 'Very weak'} {'positive' if r_c > 0 else 'negative'} correlation")
    
    # D: collision_check_time vs total_neg_sampling_time
    r_d = pearson_r(collision_t, total_ns_t)
    print(f"\nD: collision_check_time vs total_neg_sampling_time")
    print(f"   Pearson R = {r_d:.4f}")
    print(f"   Interpretation: {'Strong' if abs(r_d) > 0.7 else 'Moderate' if abs(r_d) > 0.4 else 'Weak' if abs(r_d) > 0.2 else 'Very weak'} {'positive' if r_d > 0 else 'negative'} correlation")
    
    # Additional: hub_top1_pct vs neg_sampling
    if 'hub_top1_pct_count' in rows[0]:
        hub_t1 = [int(r.get('hub_top1_pct_count', 0)) for r in rows]
        r_t1 = pearson_r(hub_t1, total_ns_t)
        print(f"\n[Extra] hub_top1_pct_count vs total_neg_sampling_time")
        print(f"   Pearson R = {r_t1:.4f}")
    
    # === Summary Statistics ===
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    print(f"\n  Total steps analyzed: {len(rows)}")
    print(f"  Epoch range: {min(epochs)} - {max(epochs)}")
    print(f"  Batch size: {min(batch_sizes)} - {max(batch_sizes)}")
    print(f"  Neg sampling time per step: mean={sum(total_ns_t)/len(total_ns_t)/len(rows)*len(total_ns_t):.1f}ms (total={total_ns/1000:.2f}s)")
    print(f"    sampling:      mean={sum(sampling_t)/len(rows):.2f}ms per step")
    print(f"    candidate:     mean={sum(candidate_t)/len(rows):.2f}ms per step")
    print(f"    collision:     mean={sum(collision_t)/len(rows):.2f}ms per step")
    print(f"    retry:         mean={sum(retry_sub_t)/len(rows):.2f}ms per step")
    print(f"    output:        mean={sum(output_t)/len(rows):.2f}ms per step")
    print(f"  Avg hub count: {sum(hub_counts)/len(hub_counts):.1f} per batch (Top 10%)")
    print(f"  Avg entity degree: {sum(avg_degrees)/len(avg_degrees):.1f}")
    print(f"  Avg retry: {sum(avg_retries)/len(avg_retries):.3f}")
    print(f"  Max retry observed: {max(max_retries)}")
    
    print("\n" + "="*60)
    print("ANSWERS TO RESEARCH QUESTIONS")
    print("="*60)
    
    # Q1
    max_component = max(components, key=lambda x: x[1])
    q1 = f"Negative Sampling 内部 '{max_component[0]}' 阶段耗时最高，占总负采样时间的 {max_component[1]/total_ns*100:.1f}%"
    print(f"\nQ1: Negative Sampling 内部哪个阶段耗时最高？")
    print(f"  A: {q1}")
    
    # Q2
    q2_collision_pct = total_b3 / total_ns * 100
    q2 = f"Collision Check 占总负采样时间 {q2_collision_pct:.1f}%"
    print(f"\nQ2: Collision Check 是否是主要瓶颈？")
    print(f"  A: {q2}")
    
    # Q3
    print(f"\nQ3: Retry Count 是否显著影响运行时间？")
    q3_cls = '显著' if abs(r_c) > 0.3 else '不显著'
    print(f"  A: avg_retry 与总时间的相关系数 R={r_c:.4f}，相关性{q3_cls}。")
    q3_cls_max = '显著' if abs(pearson_r(max_retries, total_ns_t)) > 0.3 else '不显著'
    print(f"     最大重试次数与总时间的相关系数 R={pearson_r(max_retries, total_ns_t):.4f}，相关性{q3_cls_max}。")
    
    # Q4
    print(f"\nQ4: Hub Entity 与运行时间是否存在相关性？")
    q4_cls = '强' if abs(r_a) > 0.7 else '中等' if abs(r_a) > 0.4 else '弱'
    q4_dir = '正' if r_a > 0 else '负'
    print(f"  A: hub_entity_count 与总负采样时间的 Pearson R={r_a:.4f}，{q4_cls}{q4_dir}相关。")
    print(f"     avg_entity_degree 与总负采样时间的 Pearson R={r_b:.4f}。")
    
    # Q5
    print(f"\nQ5: 后续优化最应该针对哪个模块？")
    bottleneck = max(components, key=lambda x: x[1])
    print(f"  A: 最应该优化 '{bottleneck[0]}' 模块，占负采样时间的 {bottleneck[1]/total_ns*100:.1f}%。")
    components_sorted = sorted(components, key=lambda x: x[1], reverse=True)
    for i, (label, val) in enumerate(components_sorted):
        print(f"     #{i+1}: {label} ({val/total_ns*100:.1f}%)")
    
    print("\n[Analysis complete]")

if __name__ == '__main__':
    main()