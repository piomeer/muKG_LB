#!/usr/bin/env python3
"""
PPT Page 10 — 实体访问长尾分布与缓存命中率组合图
左图: エンティティアクセス分布（ロングテール）
右图: キャッシュヒット率 vs 上位エンティティ数

数据源:
  - output/results/hub_reuse_analysis.csv  (实体出现次数)
  - output/results/cache_feasibility.csv   (Cache 理论命中率)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ── 1. 日文字体设置 ──
JP_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
try:
    fp = fm.FontProperties(fname=JP_FONT_PATH)
    plt.rcParams['font.family'] = fp.get_name()
except Exception:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 2. 读取数据 ──
df_reuse = pd.read_csv('output/results/hub_reuse_analysis.csv')
df_cache = pd.read_csv('output/results/cache_feasibility.csv')

print(f"hub_reuse_analysis.csv: {len(df_reuse)} rows")
print(f"  列名: {list(df_reuse.columns)}")
print(f"  occurrence_count 范围: {df_reuse['occurrence_count'].min()} ~ {df_reuse['occurrence_count'].max()}")
print(f"  实体数: {len(df_reuse)}")

print(f"\ncache_feasibility.csv: {len(df_cache)} rows")
print(f"  列名: {list(df_cache.columns)}")
for _, row in df_cache.iterrows():
    print(f"  Top {row['top_k']}: cache_hit_rate = {row['cache_hit_rate']:.4f} ({row['cache_hit_rate']*100:.2f}%)")

# ── 3. 计算关键统计 ──
# 长尾：出现次数 <= 10 的实体
long_tail_count = (df_reuse['occurrence_count'] <= 10).sum()
total_entities = len(df_reuse)
print(f"\n长尾实体 (<= 10 回): {long_tail_count} / {total_entities} ({long_tail_count/total_entities*100:.1f}%)")

# Top 100, Top 1000 出现次数占比
for k in [100, 1000]:
    top_k_occur = df_reuse.head(k)['occurrence_count'].sum()
    total_occur = df_reuse['occurrence_count'].sum()
    print(f"Top {k}: {top_k_occur} / {total_occur} = {top_k_occur/total_occur*100:.2f}%")

# ── 4. 绘图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ===== 左图: 实体访问长尾分布 =====
ax1 = axes[0]
ranks = df_reuse['rank'].values
occurrences = df_reuse['occurrence_count'].values

# 对数散点图（降采样以加速绘图，但保持所有数据点形状）
# 取前 2000 个点全显示，其余采样
max_points = 5000
if len(ranks) > max_points:
    step = len(ranks) // max_points
    idx = np.arange(0, len(ranks), step)
    # 确保包含第一个点
    if idx[0] != 0:
        idx = np.concatenate([[0], idx])
else:
    idx = np.arange(len(ranks))

ax1.scatter(ranks[idx], occurrences[idx], s=3, alpha=0.5, color='#4285F4', edgecolors='none')

ax1.set_xscale('log')
ax1.set_yscale('log')

# 标注 Top 100, Top 1000
y_max = occurrences[0]
for k, color in [(100, '#EA4335'), (1000, '#FBBC04')]:
    ax1.axvline(x=k, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    # 找到第 k 个实体的出现次数
    k_occur = occurrences[min(k-1, len(occurrences)-1)]
    ax1.axhline(y=k_occur, color=color, linestyle=':', linewidth=1.0, alpha=0.5)
    # 标注文字
    ax1.text(k, y_max * 0.8, f'Top {k}', fontsize=10, color=color,
             fontweight='bold', ha='right', rotation=90, va='top')

# 标注占比信息
total_occur = occurrences.sum()
for k in [100, 1000]:
    k_occur_sum = occurrences[:k].sum()
    ratio = k_occur_sum / total_occur * 100
    k_occur_val = occurrences[min(k-1, len(occurrences)-1)]
    ax1.annotate(f'Top {k}\n{k_occur_sum:,} 回\n({ratio:.1f}%)',
                 xy=(k, k_occur_val),
                 xytext=(k * 2.5, k_occur_val * 2),
                 fontsize=9, color='#333',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_title('エンティティアクセス分布（ロングテール）', fontsize=14, fontweight='bold')
ax1.set_xlabel('エンティティランク（対数）', fontsize=12)
ax1.set_ylabel('出現回数（対数）', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0.8, len(ranks) * 1.5)
ax1.set_ylim(0.5, occurrences[0] * 5)

# ===== 右图: 缓存命中率累积曲线 =====
ax2 = axes[1]
top_k_vals = df_cache['top_k'].values
hit_rates = df_cache['cache_hit_rate'].values * 100  # 转为百分比

# 柱状图
colors_bar = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_k_vals)))
bars = ax2.bar(range(len(top_k_vals)), hit_rates, color=colors_bar, width=0.6,
               edgecolor='navy', linewidth=0.8, alpha=0.85)

# 柱顶标注
for i, (bar, rate) in enumerate(zip(bars, hit_rates)):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 红色水平虚线：Top 1000 = 37.97%
top1000_rate = df_cache[df_cache['top_k'] == 1000]['cache_hit_rate'].values[0] * 100
ax2.axhline(y=top1000_rate, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(len(top_k_vals) - 0.5, top1000_rate + 0.5,
         f'Top1000 で {top1000_rate:.2f}%', fontsize=10, color='red',
         fontweight='bold', ha='right', va='bottom')

# X 轴标签
ax2.set_xticks(range(len(top_k_vals)))
ax2.set_xticklabels([f'Top {int(k)}' for k in top_k_vals], fontsize=11)
ax2.set_xlim(-0.6, len(top_k_vals) - 0.4)

ax2.set_title('キャッシュヒット率 vs 上位エンティティ数', fontsize=14, fontweight='bold')
ax2.set_xlabel('上位エンティティ数', fontsize=12)
ax2.set_ylabel('キャッシュヒット率 (%)', fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(pad=2.5)

# ── 5. 保存 ──
out_path = 'output/figs/hub_reuse_and_cache.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 图片已保存: {out_path}")
plt.close()

# ── 6. 报告关键数值 ──
print("\n===== 关键数值报告 =====")
print(f"左图 - 长尾分布:")
print(f"  Top 1 出现次数: {occurrences[0]}")
print(f"  Top 100 出现次数合计: {occurrences[:100].sum()} ({occurrences[:100].sum()/total_occur*100:.2f}%)")
print(f"  Top 1000 出现次数合计: {occurrences[:1000].sum()} ({occurrences[:1000].sum()/total_occur*100:.2f}%)")
print(f"  长尾实体 (<=10次): {long_tail_count}/{total_entities} ({long_tail_count/total_entities*100:.1f}%)")
print(f"右图 - 缓存命中率:")
for _, row in df_cache.iterrows():
    print(f"  Top {int(row['top_k'])}: {row['cache_hit_rate']*100:.2f}%")