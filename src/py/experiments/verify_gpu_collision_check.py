#!/usr/bin/env python3
"""
Phase 7 Step 2: GPU Vectorized Collision Check Feasibility Verification
Measure: GPU random generation + isin filtering vs CPU numpy randint
"""
import torch
import time
import numpy as np

# 参数模拟实际 batch
batch_size = 5000
neg_num = 150
total_neg = batch_size * neg_num  # 750,000
n_entities = 14541  # FB15k-237
device = 'cuda'

# 模拟正样本的尾实体（假设替换尾实体）
pos_tails = torch.randint(0, n_entities, (batch_size,), device='cpu')
pos_tails_gpu = pos_tails.to(device)

# 生成候选负样本（远多于需要量，以应对碰撞）
oversample_factor = 1.2  # 多生成 20%
num_candidates = int(total_neg * oversample_factor)

print(f"Generating {num_candidates} candidate negatives on GPU...")
torch.cuda.synchronize()
t0 = time.perf_counter()

# 在 GPU 上生成随机候选尾实体
candidates = torch.randint(0, n_entities, (num_candidates,), device=device)

# 向量化碰撞检查：排除等于正样本尾实体的候选
mask = ~torch.isin(candidates, pos_tails_gpu)   # [num_candidates]
valid_candidates = candidates[mask]

# 如果不够 total_neg，再追加生成（此处简化，打印信息）
torch.cuda.synchronize()
t1 = time.perf_counter()

cpu_time = 0.0
# 对比 CPU 生成相同数量的随机 int
t2 = time.perf_counter()
_ = np.random.randint(0, n_entities, size=total_neg, dtype=np.int64)
t3 = time.perf_counter()
cpu_time = t3 - t2

print(f"GPU vectorized collision check + generation: {t1 - t0:.4f} seconds")
print(f"CPU numpy randint generation (no check): {cpu_time:.4f} seconds")
print(f"Valid negatives obtained: {valid_candidates.size(0)} / needed: {total_neg}")
if valid_candidates.size(0) >= total_neg:
    print("✅ GPU vectorized method can produce enough valid negatives!")
else:
    print("⚠️ Need to increase oversampling factor or implement retry.")