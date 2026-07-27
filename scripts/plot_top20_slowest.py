#!/usr/bin/env python3
"""
Top 20 Slowest Batches Hub Count — 竖向条形图
用于 PPT Page 8

数据源: output/results/negative_sampling_cost.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ── 1. 日文字体设置 (同 Page 7) ──
JP_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
try:
    fp = fm.FontProperties(fname=JP_FONT_PATH)
    plt.rcParams['font.family'] = fp.get_name()
except Exception:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 2. 读取数据 ──
df = pd.read_csv('output/results/negative_sampling_cost.csv')
print(f"数据行数: {len(df)}")
print(f"列名: {list(df.columns)}")

# ── 3. 按 sampling_time 降序排序，取前 20 ──
df_sorted = df.sort_values('sampling_time', ascending=False).reset_index(drop=True)
top20 = df_sorted.head(20).copy()
top20['rank'] = range(1, 21)  # 1～20

# ── 4. 打印结果 ──
print("\n===== Top 20 Slowest Batches (by sampling_time) =====")
print(f"{'Rank':<6}{'sampling_time(ms)':<18}{'hub_entity_count':<16}{'collision_check(ms)':<20}{'candidate_build(ms)':<20}")
print("="*80)
for _, row in top20.iterrows():
    print(f"{int(row['rank']):<6}{row['sampling_time']:<18.3f}{row['hub_entity_count']:<16.0f}{row['collision_check_time']:<20.3f}{row['candidate_build_time']:<20.3f}")

# ── 5. hub_count 最大值 ──
max_hub = df['hub_entity_count'].max()
mean_hub_top20 = top20['hub_entity_count'].mean()
print(f"\nhub_count 最大值（全局）: {max_hub:.0f}")
print(f"Top 20 hub_count 平均值: {mean_hub_top20:.1f}")
print(f"Top 20 hub_count 范围: {top20['hub_entity_count'].min():.0f} ~ {top20['hub_entity_count'].max():.0f}")

# ── 6. 绘图 ──
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    top20['rank'],
    top20['hub_entity_count'],
    color='coral',
    alpha=0.8,
    edgecolor='darkred',
    linewidth=0.5,
    width=0.7,
)

# 水平虚线：最大 hub 数
ax.axhline(y=max_hub, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(20.5, max_hub + 30, f'最大ハブ数 ({int(max_hub):,})',
        fontsize=11, color='gray', ha='right', va='bottom')

# 在每个柱子上方标注数值
for bar, val in zip(bars, top20['hub_entity_count']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f'{int(val):,}', ha='center', va='bottom', fontsize=8, color='black', rotation=45)

# 轴标签
ax.set_xlabel('バッチランク', fontsize=13)
ax.set_ylabel('ハブ数', fontsize=13)
ax.set_title('最も遅い上位20バッチのハブ数', fontsize=15, fontweight='bold')

ax.set_xticks(top20['rank'])
ax.set_xticklabels(top20['rank'], fontsize=10)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, max_hub * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# ── 7. 保存 ──
out_path = 'output/figs/top20_slowest_hub_count.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 图片已保存: {out_path}")
plt.close()