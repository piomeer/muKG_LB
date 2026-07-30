#!/usr/bin/env python3
"""
修正版 B1/B2/B3 相关性分析
问题：hub_entity_count 只有 2 个不同值（4230 和 6000），R=0.816 是伪相关
修正：使用有真正变异的特征（unique_entities, total_retry, avg_retry）

输出：
  1. output/figs/correlation_heatmap.png — 相关矩阵热图（所有变量）
  2. output/figs/unique_entities_vs_B.png — unique_entities vs B1/B2/B3
  3. 控制台输出偏相关分析结果
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from scipy import stats
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None
import warnings
warnings.filterwarnings('ignore')

# ── 1. 日文字体 ──
JP_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
try:
    fp = fm.FontProperties(fname=JP_FONT_PATH)
    plt.rcParams['font.family'] = fp.get_name()
except Exception:
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# ── 2. 读取数据 ──
df = pd.read_csv('output/results/negative_sampling_cost.csv')
print(f"总行数: {len(df)}")

# ── 3. 选择有足够变异的列 ──
feature_cols = ['unique_entities', 'total_retry', 'avg_retry', 'hub_entity_count']
target_cols = ['sampling_time', 'candidate_build_time', 'collision_check_time', 'retry_time', 'total_neg_sampling_time']

# 变异性检查
print("\n===== 变异性检查 =====")
for c in feature_cols + target_cols:
    vals = df[c].values
    print(f"  {c:<28}: unique={len(np.unique(vals)):>4}, std={vals.std():>8.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

# ── 4. 所有相关性的相关矩阵 ──
print("\n===== 相关矩阵 (Pearson R) =====")
all_cols = target_cols + feature_cols
corr_matrix = df[all_cols].corr()
print(corr_matrix.round(3).to_string())

# ── 5. 偏相关分析（控制 avg_retry） ──
print("\n===== 偏相关：控制 avg_retry 后 B 阶段 vs 其他特征 =====")
if LinearRegression is not None:
    for feature in ['unique_entities', 'total_retry']:
        for target in ['sampling_time', 'collision_check_time', 'candidate_build_time']:
            lr = LinearRegression()
            X_ctrl = df[['avg_retry']].values
            y_target = df[target].values
            y_feat = df[feature].values
            lr.fit(X_ctrl, y_target)
            resid_target = y_target - lr.predict(X_ctrl)
            lr.fit(X_ctrl, y_feat)
            resid_feat = y_feat - lr.predict(X_ctrl)
            r_partial, p_partial = stats.pearsonr(resid_feat, resid_target)
            r_raw, _ = stats.pearsonr(df[feature], df[target])
            print(f"  {feature} vs {target}: raw R={r_raw:.4f}, partial R (ctrl avg_retry)={r_partial:.4f}")

# ── 6. 图 1: 相关矩阵热图 ──
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
cmap = sns.diverging_palette(250, 30, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap=cmap,
            vmin=-1.0, vmax=1.0, center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Pearson R'},
            ax=ax, annot_kws={'fontsize': 9})
ax.set_title('Negative Sampling 相関行列 (Pearson R)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('output/figs/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("\n✅ 热图已保存: output/figs/correlation_heatmap.png")

# ── 7. 图 2: unique_entities vs B1/B2/B3（横向三子图） ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
x_col = 'unique_entities'
y_cols = ['sampling_time', 'collision_check_time', 'candidate_build_time']
labels_jp = {
    'sampling_time': 'サンプリング時間 (B1)',
    'collision_check_time': '衝突チェック時間 (B3)',
    'candidate_build_time': '候補構築時間 (B2)',
}
titles_jp = {
    'sampling_time': 'ユニークエンティティ数 vs サンプリング時間',
    'collision_check_time': 'ユニークエンティティ数 vs 衝突チェック時間',
    'candidate_build_time': 'ユニークエンティティ数 vs 候補構築時間',
}
palette = ['#4285F4', '#EA4335', '#34A853']

results_unique_entities = {}
for idx, (yc, title) in enumerate(titles_jp.items()):
    ax = axes[idx]
    r_val, p_val = stats.pearsonr(df[x_col], df[yc])
    results_unique_entities[yc] = (r_val, p_val)
    
    sns.regplot(data=df, x=x_col, y=yc, ax=ax,
                scatter_kws={'alpha': 0.4, 's': 15, 'color': palette[idx]},
                line_kws={'color': 'black', 'linewidth': 1.5},
                ci=95)
    
    r_text = f'R = {r_val:.3f}'
    ax.text(0.05, 0.92, r_text, transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('ユニークエンティティ数', fontsize=12)
    ax.set_ylabel(f'{labels_jp[yc]} (ms)', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(pad=2.0)
fig.savefig('output/figs/unique_entities_vs_B.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ 散点图已保存: output/figs/unique_entities_vs_B.png")

# ── 8. 输出修正后的 R 值 ──
print("\n" + "=" * 60)
print("【修正后结论】")
print("=" * 60)
print("""
原值: hub_entity_count 只有 2 个不同值 (4230, 6000)，不能用做回归分析。

真实有变异的特征:
  unique_entities (唯一实体数): n_unique=123, std=97.89
  total_retry (总重试次数): n_unique=131, std=121.41
  avg_retry (平均重试次数): n_unique=131, std=0.01

修正后的 R 值:
  unique_entities vs B1: R={:.4f}
  unique_entities vs B2: R={:.4f}
  unique_entities vs B3: R={:.4f}
  total_retry vs B1: R={:.4f}
  total_retry vs B2: R={:.4f}
  total_retry vs B3: R={:.4f}

真实因果链:
  更多唯一实体 → 更大的 collision check 工作集 → 更多重试 → B1/B2/B3 时间增加
""".format(
    results_unique_entities['sampling_time'][0],
    results_unique_entities['candidate_build_time'][0],
    results_unique_entities['collision_check_time'][0],
    stats.pearsonr(df['total_retry'], df['sampling_time'])[0],
    stats.pearsonr(df['total_retry'], df['candidate_build_time'])[0],
    stats.pearsonr(df['total_retry'], df['collision_check_time'])[0],
))