#!/usr/bin/env python3
"""
Hub Correlation Analysis — 横向三子图散点图
用于 PPT Page 7: Hub数 vs B1/B2/B3 耗时

数据源: output/results/negative_sampling_cost.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 1. 日文字体设置 ──
# Noto Sans CJK JP (系统已安装)
JP_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# 尝试使用该字体
import matplotlib.font_manager as fm
try:
    fp = fm.FontProperties(fname=JP_FONT_PATH)
    font_name = fp.get_name()
    plt.rcParams['font.family'] = font_name
except Exception:
    # 备选: 尝试 'Noto Sans CJK JP'
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 2. 读取数据 ──
df = pd.read_csv('output/results/negative_sampling_cost.csv')
print(f"数据行数: {len(df)}")
print(f"列名: {list(df.columns)}")

# ── 3. 确认列存在 ──
x_col = 'hub_entity_count'
y_cols = {
    'sampling_time': 'サンプリング時間',
    'collision_check_time': '衝突チェック時間',
    'candidate_build_time': '候補構築時間',
}

for c in [x_col] + list(y_cols.keys()):
    assert c in df.columns, f"列 {c} 不存在！"

# 检查时间单位 — 数值看起来已经是 ms
# profiling_summary 中的 collate_time=353.562 (ms)
# negative_sampling_cost 中的 sampling_time=120.377 (同样量级)
# 确认是毫秒
print(f"\n{x_col} 范围: {df[x_col].min():.0f} ~ {df[x_col].max():.0f}")
for yc in y_cols:
    print(f"{yc} 范围: {df[yc].min():.4f} ~ {df[yc].max():.4f} (单位: ms)")

# ── 4. 计算 Pearson R ──
print("\n===== Pearson 相关系数 =====")
results = {}
for yc, ylabel in y_cols.items():
    r, p = stats.pearsonr(df[x_col], df[yc])
    results[yc] = (r, p)
    print(f"{ylabel} vs {x_col}: R = {r:.4f}, p = {p:.4e}")

# ── 5. 绘图 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 配色
palette = ['#4285F4', '#EA4335', '#34A853']  # 蓝/红/绿

titles = {
    'sampling_time': 'ハブ数 vs サンプリング時間',
    'collision_check_time': 'ハブ数 vs 衝突チェック時間',
    'candidate_build_time': 'ハブ数 vs 候補構築時間',
}
y_axis_labels = {
    'sampling_time': 'サンプリング時間 (ms)',
    'collision_check_time': '衝突チェック時間 (ms)',
    'candidate_build_time': '候補構築時間 (ms)',
}

for idx, (yc, ylabel) in enumerate(y_cols.items()):
    ax = axes[idx]
    r_val, p_val = results[yc]
    
    # regplot: 散点 + 回归线
    sns.regplot(
        data=df,
        x=x_col,
        y=yc,
        ax=ax,
        scatter_kws={'alpha': 0.4, 's': 15, 'color': palette[idx]},
        line_kws={'color': 'black', 'linewidth': 1.5},
        ci=95,
    )
    
    # 标注 R 值
    r_text = f'Pearson R = {r_val:.3f}'
    ax.text(0.05, 0.92, r_text, transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 标题与轴标签
    ax.set_title(titles[yc], fontsize=14, fontweight='bold')
    ax.set_xlabel('ハブ数', fontsize=12)
    ax.set_ylabel(y_axis_labels[yc], fontsize=12)
    ax.tick_params(labelsize=11)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(pad=2.0)

# ── 6. 保存 ──
out_path = 'output/figs/hub_correlation_analysis.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存: {out_path}")
plt.close()

# ── 7. 最终汇总 ──
print("\n===== 结果汇总 =====")
for yc, ylabel in y_cols.items():
    r, p = results[yc]
    print(f"  {ylabel}: R = {r:.3f}, p = {p:.4e}")