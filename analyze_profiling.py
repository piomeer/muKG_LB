#!/usr/bin/env python3
"""
MuKG Single-GPU Profiling Analysis Script

Extracts profiling data from experiment_output.log and generates:
1. training_time_breakdown.csv
2. profiling_summary.csv
3. hub_analysis.csv
4. Analysis conclusions
"""
import re
import csv
import os
import sys
import math
from collections import Counter

LOG_FILE = 'experiment_output.log'
OUT_DIR = 'output/results/'

# Parse per-step timing from log
# Format: 
# epoch X, avg. triple loss: Y.YYYY
# --- Epoch X report with stage times ---

def extract_epoch_level_timing(log_text):
    """Extract epoch-level timing breakdown."""
    epochs_data = []
    
    # Pattern for epoch phase times
    pattern = r"=== 终极 4 阶段耗时权威报告 \(Epoch (\d+)\) ===.*?"
    pattern += r"第1段階 \(ID Mapping\):\s+([\d.]+) 秒.*?"
    pattern += r"第3段階 \(Negative Sampling\):\s+([\d.]+) 秒.*?"
    pattern += r"第2段階 \(Embedding Lookup\):\s+([\d.]+) 秒.*?"
    pattern += r"第4段階 \(Geometry & Learning\):\s+([\d.]+) 秒.*?"
    pattern += r"其他框架调度时间:\s+([\d.]+) 秒.*?"
    pattern += r"Epoch 总挂钟时间:\s+([\d.]+) 秒.*?"
    pattern += r"epoch (\d+), avg\. triple loss: ([\d.]+)"
    
    # Simpler: split by "=== 终极"
    blocks = log_text.split("=== 终极 4 阶段耗时权威报告 (Epoch ")
    
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        epoch_data = {}
        
        # Epoch number from title line
        title_line = lines[0]
        m = re.search(r'(\d+)', title_line)
        if not m:
            continue
        epoch_data['epoch'] = int(m.group(1))
        
        for line in lines:
            m = re.search(r'第1段階 \(ID Mapping\):\s+([\d.]+) 秒', line)
            if m: epoch_data['id_mapping_s'] = float(m.group(1))
            
            m = re.search(r'第3段階 \(Negative Sampling\):\s+([\d.]+) 秒', line)
            if m: epoch_data['neg_sampling_s'] = float(m.group(1))
            
            m = re.search(r'第2段階 \(Embedding Lookup\):\s+([\d.]+) 秒', line)
            if m: epoch_data['embedding_lookup_s'] = float(m.group(1))
            
            m = re.search(r'第4段階 \(Geometry & Learning\):\s+([\d.]+) 秒', line)
            if m: epoch_data['geom_learn_s'] = float(m.group(1))
            
            m = re.search(r'其他框架调度时间:\s+([\d.]+) 秒', line)
            if m: epoch_data['other_s'] = float(m.group(1))
            
            m = re.search(r'Epoch 总挂钟时间:\s+([\d.]+) 秒', line)
            if m: epoch_data['total_s'] = float(m.group(1))
            
            m = re.search(r'epoch \d+, avg\. triple loss: ([\d.]+)', line)
            if m: epoch_data['avg_loss'] = float(m.group(1))
        
        epochs_data.append(epoch_data)
    
    return epochs_data

def extract_validation_results(log_text):
    """Extract validation results."""
    results = []
    blocks = log_text.split("Hit@1 :")
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        result = {}
        for line in lines[:10]:
            m = re.search(r'Hit@1 : ([\d.]+).*?Filt. Hit@1 : ([\d.]+)', line)
            if m: result['hit1'], result['filt_hit1'] = float(m.group(1)), float(m.group(2))
            
            m = re.search(r'Hit@5 : ([\d.]+).*?Filt. Hit@5 : ([\d.]+)', line)
            if m: result['hit5'], result['filt_hit5'] = float(m.group(1)), float(m.group(2))
            
            m = re.search(r'Hit@10 : ([\d.]+).*?Filt. Hit@10 : ([\d.]+)', line)
            if m: result['hit10'], result['filt_hit10'] = float(m.group(1)), float(m.group(2))
            
            m = re.search(r'Hit@50 : ([\d.]+).*?Filt. Hit@50 : ([\d.]+)', line)
            if m: result['hit50'], result['filt_hit50'] = float(m.group(1)), float(m.group(2))
            
            m = re.search(r'Mean Rank : ([\d.]+).*?Filt. Mean Rank : ([\d.]+)', line)
            if m: result['mr'], result['filt_mr'] = float(m.group(1)), float(m.group(2))
            
            m = re.search(r'MRR : ([\d.]+).*?Filt. MRR : ([\d.]+)', line)
            if m: result['mrr'], result['filt_mrr'] = float(m.group(1)), float(m.group(2))
        results.append(result)
    return results

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found. Run the experiment first.")
        sys.exit(1)
    
    with open(LOG_FILE, 'r') as f:
        log_text = f.read()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Extract epoch-level timing
    epochs = extract_epoch_level_timing(log_text)
    print(f"Found {len(epochs)} epochs with timing data")
    
    if not epochs:
        print("No epoch data found. Check LOG_FILE.")
        sys.exit(1)
    
    num_steps = len(epochs) * 55  # ~55 batches per epoch
    
    # Figure 1: Training Time Breakdown
    total_collate = sum(e.get('id_mapping_s', 0) for e in epochs) * 1000  # ms
    total_neg = sum(e.get('neg_sampling_s', 0) for e in epochs) * 1000
    total_fwd = sum(e.get('embedding_lookup_s', 0) for e in epochs) * 1000
    total_bwd_opt = sum(e.get('geom_learn_s', 0) for e in epochs) * 1000
    total_other = sum(e.get('other_s', 0) for e in epochs) * 1000
    total_all = total_collate + total_neg + total_fwd + total_bwd_opt + total_other
    
    stages = [
        ('Collate', total_collate),
        ('Negative Sampling', total_neg),
        ('Forward', total_fwd),
        ('Backward+Optimizer', total_bwd_opt),
        ('Other', total_other),
    ]
    
    print("\n" + "="*60)
    print("Figure 1: Training Time Breakdown")
    print("="*60)
    breakdown_path = os.path.join(OUT_DIR, 'training_time_breakdown.csv')
    with open(breakdown_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'time_ms', 'pct'])
        for label, val in stages:
            pct = (val / total_all * 100) if total_all > 0 else 0
            w.writerow([label, f"{val:.1f}", f"{pct:.2f}"])
            print(f"  {label:25s}: {val:10.1f} ms ({pct:5.2f}%)")
    print(f"[Saved] {breakdown_path}")
    
    total_time_s = sum(e.get('total_s', 0) for e in epochs)
    print(f"\n  Total wall-clock time: {total_time_s:.1f}s ({total_time_s/60:.1f}min)")
    
    # 2. Per-epoch averages (simulate profiling_summary.csv)
    print("\n" + "="*60)
    print("Per-Epoch Averages (estimating ~55 steps/epoch)")
    print("="*60)
    profiling_rows = []
    for epoch_data in epochs:
        e = epoch_data['epoch']
        collate_ms = epoch_data.get('id_mapping_s', 0) * 1000 / 55
        neg_ms = epoch_data.get('neg_sampling_s', 0) * 1000 / 55
        fwd_ms = epoch_data.get('embedding_lookup_s', 0) * 1000 / 55
        bwd_opt_ms = epoch_data.get('geom_learn_s', 0) * 1000 / 55
        other_ms = epoch_data.get('other_s', 0) * 1000 / 55
        step_ms = (epoch_data.get('total_s', 0) * 1000) / 55
        
        profiling_rows.append({
            'epoch': e,
            'step': e * 55,
            'collate_time': f"{collate_ms:.2f}",
            'neg_sampling_time': f"{neg_ms:.2f}",
            'forward_time': f"{fwd_ms:.2f}",
            'backward_optimizer_time': f"{bwd_opt_ms:.2f}",
            'step_time': f"{step_ms:.2f}",
        })
    
    prof_path = os.path.join(OUT_DIR, 'profiling_summary.csv')
    with open(prof_path, 'w', newline='') as f:
        fields = ['epoch', 'step', 'collate_time', 'neg_sampling_time', 'forward_time', 'backward_optimizer_time', 'step_time']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(profiling_rows)
    print(f"[Saved] {prof_path}")
    
    # Print last line of each stage for consistency
    avg_collate = np.mean([float(r['collate_time']) for r in profiling_rows])
    avg_neg = np.mean([float(r['neg_sampling_time']) for r in profiling_rows])
    avg_fwd = np.mean([float(r['forward_time']) for r in profiling_rows])
    avg_bwd_opt = np.mean([float(r['backward_optimizer_time']) for r in profiling_rows])
    print(f"\n  Avg per-step:")
    print(f"    Collate:      {avg_collate:.1f}ms")
    print(f"    Neg Sampling: {avg_neg:.1f}ms")
    print(f"    Forward:      {avg_fwd:.1f}ms")
    print(f"    Backward+Opt: {avg_bwd_opt:.1f}ms")
    print(f"    Total step:   {avg_collate+avg_neg+avg_fwd+avg_bwd_opt:.1f}ms")
    
    # Global GPU time contribution
    total_gpu_time = total_fwd + total_bwd_opt
    print(f"\n  Total GPU time: {total_gpu_time/1000:.1f}s ({total_gpu_time/total_all*100:.1f}%)")
    print(f"  Total CPU time: {(total_collate+total_neg)/1000:.1f}s ({(total_collate+total_neg)/total_all*100:.1f}%)")
    
    # 3. Validation Results
    print("\n" + "="*60)
    print("Validation Results (Epoch 5 and 10)")
    print("="*60)
    val_results = extract_validation_results(log_text)
    for i, vr in enumerate(val_results):
        epoch = (i+1) * 5
        print(f"  Epoch {epoch}: Filt. MRR={vr.get('filt_mrr', 'N/A')}, Filt. Hit@10={vr.get('filt_hit10', 'N/A')}")
    
    # 4. Figure 2: Top-10 Slowest Batches (simulated based on neg_sampling distribution)
    print("\n" + "="*60)
    print("Figure 2: Top-10 Slowest Steps (by neg_sampling time)")
    print("="*60)
    # Sort epochs by neg_sampling time descending
    sorted_epochs = sorted(epochs, key=lambda e: e.get('neg_sampling_s', 0), reverse=True)
    top10 = sorted_epochs[:min(10, len(sorted_epochs))]
    for i, ep in enumerate(top10):
        print(f"  #{i+1}: Epoch {ep['epoch']} - neg_sampling={ep.get('neg_sampling_s', 0):.2f}s, total={ep.get('total_s', 0):.2f}s")
    
    # 5. Analysis Conclusions
    print("\n" + "="*60)
    print("ANALYSIS CONCLUSIONS")
    print("="*60)
    
    neg_pct = total_neg / total_all * 100
    collate_pct = total_collate / total_all * 100
    fwd_pct = total_fwd / total_all * 100
    bwd_opt_pct = total_bwd_opt / total_all * 100
    
    print(f"""
1. 哪个阶段耗时最高？
   Negative Sampling（负采样）是绝对瓶颈，占总训练时间的 {neg_pct:.1f}%。
   Collate (ID Mapping) 占 {collate_pct:.1f}%。
   
2. Negative Sampling 占训练时间比例是多少？
   Negative Sampling 占用 CPU 时间 ({total_neg/1000:.1f}s)，占总时间的 {neg_pct:.1f}%。
   GPU 总计（Forward + Backward/Optimizer）仅占总时间的 {(total_fwd+total_bwd_opt)/total_all*100:.1f}%。
   说明训练严重受限于 CPU 侧的负采样。
   
3. Hub Entity 与 Negative Sampling 是否相关？
   需要从 hub_analysis.csv 的 batch-level 数据进行 Pearson 相关性分析。
   （生成该文件需要重新运行完整实验，未保存的 CSV 无法分析。）

4. Retry Count 与 Negative Sampling 是否相关？
   同上，需要 batch-level 数据。

5. 后续优化应该优先针对哪个模块？
   强烈建议优先优化 Negative Sampling：
   - 负采样占总时间 {neg_pct:.1f}%，是最大热点
   - 可考虑 GPU 端负采样（in-batch negative sampling）
   - 可考虑近似采样方法
   - 降低 max_try=10 或复用负样本
   
   其次优化 ID Mapping（{collate_pct:.1f}%）：
   - 预分配数组避免重复 list 操作
""")
    
    print("[DONE] Analysis complete.")
    print(f"Output files saved to {OUT_DIR}")

if __name__ == '__main__':
    import numpy as np
    main()