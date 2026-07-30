#!/usr/bin/env python3
"""
验证：B1 与 hub_count 的 Pearson R=0.816 到底是怎么来的

两个问题：
1. 为什么 B1 (random.sample) 和 hub 数有相关性？
2. 为什么重试时 B1 也被计时了？
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── 1. 读取数据 ──
df = pd.read_csv('output/results/negative_sampling_cost.csv')
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print()

# ── 2. 关键列 ──
hub = df['hub_entity_count'].values
b1 = df['sampling_time'].values        # B1: 采样时间
b2 = df['candidate_build_time'].values  # B2: 候选构建
b3 = df['collision_check_time'].values  # B3: 碰撞检查
b4 = df['retry_time'].values            # B4: 重试逻辑
avg_retry = df['avg_retry'].values

# ── 3. 原始 R 值（和之前一样） ──
print("=" * 60)
print("【原始 R 值】")
print("=" * 60)
r_b1, p_b1 = stats.pearsonr(hub, b1)
r_b2, p_b2 = stats.pearsonr(hub, b2)
r_b3, p_b3 = stats.pearsonr(hub, b3)
r_b4, p_b4 = stats.pearsonr(hub, b4)
r_retry, p_retry = stats.pearsonr(hub, avg_retry)
print(f"hub_count vs B1 (sampling):       R = {r_b1:.4f}  p = {p_b1:.2e}")
print(f"hub_count vs B2 (candidate build): R = {r_b2:.4f}  p = {p_b2:.2e}")
print(f"hub_count vs B3 (collision check): R = {r_b3:.4f}  p = {p_b3:.2e}")
print(f"hub_count vs B4 (retry logic):     R = {r_b4:.4f}  p = {p_b4:.2e}")
print(f"hub_count vs avg_retry:            R = {r_retry:.4f}  p = {p_retry:.2e}")
print()

# ── 4. 关键洞察：avg_retry 对 B1 的放大效应 ──
# B1 被计入了每次重试。所以 B1_total = B1_first_pass + B1_retries
# avg_retry 约 1.28~1.30，意味着每个三元组平均有 1.3 次 B1 调用
print("=" * 60)
print("【avg_retry 对 B1 的放大效应】")
print("=" * 60)
avg_retry_mean = avg_retry.mean()
print(f"avg_retry 均值: {avg_retry_mean:.4f}")
print(f"avg_retry 范围: {avg_retry.min():.4f} ~ {avg_retry.max():.4f}")

# 如果没有重试，B1 的总时间 = 第一次采样的时间
# 有重试时，每次重试也调用 random.sample（虽然 nums_to_sample 更少）
# 但 time.perf_counter 的开销每次都一样
print()
print("【数值示例：一个三元组的 B1 累积】")
print("  假设 batch_size=3000, neg_num=150, avg_retry=1.3")
print(f"  B1 总时间 / batch ≈ {b1.mean():.1f} ms")
print(f"  包含 {3000 * avg_retry_mean:.0f} 次 random.sample 调用")
print(f"  每次调用的平均耗时 ≈ {b1.mean() / (3000 * avg_retry_mean):.4f} ms")
print()

# ── 5. 按 epoch 分别计算 R ──
print("=" * 60)
print("【按 Epoch 分别计算 R（检查是否是 epoch 混淆）】")
print("=" * 60)
for epoch in sorted(df['epoch'].unique()):
    mask = df['epoch'] == epoch
    if mask.sum() < 5:
        continue
    r_ep, p_ep = stats.pearsonr(hub[mask], b1[mask])
    print(f"  Epoch {epoch}: n={mask.sum()}, R(B1 vs hub) = {r_ep:.4f}")
print()

# ── 6. B1 按 avg_retry 分组 ──
print("=" * 60)
print("【按 avg_retry 分组查看 B1 均值】")
print("=" * 60)
bins = [1.0, 1.25, 1.28, 1.30, 1.32, 1.35, 10.0]
labels = ['1.00-1.25', '1.25-1.28', '1.28-1.30', '1.30-1.32', '1.32-1.35', '>1.35']
df['retry_bin'] = pd.cut(avg_retry, bins=bins, labels=labels)
for label in labels:
    mask = df['retry_bin'] == label
    if mask.sum() > 0:
        print(f"  avg_retry {label}: n={mask.sum()}, "
              f"B1_mean={b1[mask].mean():.2f}ms, "
              f"hub_mean={hub[mask].mean():.0f}")
print()

# ── 7. 偏相关：控制 avg_retry 后的 B1 vs hub ──
print("=" * 60)
print("【偏相关：控制 avg_retry 后 B1 vs hub 的净相关】")
print("=" * 60)
# 残差法：B1_resid = B1 被 avg_retry 解释后剩余的部分
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(avg_retry.reshape(-1, 1), b1)
b1_pred = lr.predict(avg_retry.reshape(-1, 1))
b1_resid = b1 - b1_pred
r_partial, p_partial = stats.pearsonr(hub, b1_resid)
print(f"  B1_residual (移除 avg_retry 影响后) vs hub: R = {r_partial:.4f}")
print(f"  解释：如果 R 明显下降 → avg_retry 是因果链")
print()

# ── 8. 分位数展示 ──
print("=" * 60)
print("【按 hub_count 五等分展示】")
print("=" * 60)
df['hub_quintile'] = pd.qcut(hub, 5, labels=['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)'])
for q in ['Q1(低)', 'Q2', 'Q3', 'Q4', 'Q5(高)']:
    mask = df['hub_quintile'] == q
    print(f"  {q}: hub范围={hub[mask].min():.0f}-{hub[mask].max():.0f}, "
          f"B1={b1[mask].mean():.2f}ms, B2={b2[mask].mean():.2f}ms, "
          f"B3={b3[mask].mean():.2f}ms, avg_retry={avg_retry[mask].mean():.3f}")

print()
print("=" * 60)
print("【关键结论】")
print("=" * 60)
print("""
1. B1 在每次重试中都被计时：
   - 第一次尝试：random.sample(candidates, 150) → 计入 B1
   - 如果碰撞检查后缺 3 个，重试：random.sample(candidates, 3) → 也计入 B1
   - avg_retry=1.3 意味着每个三元组平均被采 1.3 次
   - 重试时 nums_to_sample 虽然少，但 random.sample 的函数调用开销 + time.perf_counter 开销一样

2. 如果 B1 的相关性来自 avg_retry 的放大效应：
   - 控制 avg_retry 后的偏相关会显著下降
   - 如果没下降 → 有其他未知因素

3. B1 的真正瓶颈不是 hub 数，而是 Python random.sample 的调用开销本身：
   - 3000 三元组 × 1.3 次 = 3900 次函数调用 / batch
   - 每调用 ~0.03ms → 合计 ~120ms
""")