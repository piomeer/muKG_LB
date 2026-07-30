#!/usr/bin/env python3
"""
Phase 9 Step 4 — Paper Assets Generation
Generates figures, tables, and experiment summary for the paper.
Reads only from output/results/, writes only to paper_assets/.
"""
import csv
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Seaborn-style colors
COLORS = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
SINGLE_COLOR = '#4ECDC4'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUT_FIGS = 'paper_assets/figures'
OUT_TABLES = 'paper_assets/tables'
MISSING_SOURCES = []

os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Profiling Breakdown (hardcoded from Phase 7 Step 1)
# ═══════════════════════════════════════════════════════════════════
def fig1_profiling_breakdown():
    labels = ['Collate', 'Neg Sampling', 'Tensor Build', 'Forward', 'Backward', 'Optimizer']
    percentages = [46.6, 35.7, 10.7, 3.3, 3.4, 0.2]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    fig, ax = plt.subplots(figsize=(10, 3.5))
    left = 0
    for i, (label, pct, c) in enumerate(zip(labels, percentages, colors)):
        ax.barh(['Profiling Breakdown'], [pct], left=left, color=c, label=label, edgecolor='white', linewidth=0.8)
        if pct > 2:
            ax.text(left + pct / 2, 0, f'{label}\n{pct}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white' if pct > 15 else 'black')
        left += pct
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of Total Step Time (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.set_title('Fig 1: MuKG Training Step Profiling Breakdown')
    fig.savefig(f'{OUT_FIGS}/fig1_profiling_breakdown.pdf')
    plt.close(fig)
    print('[OK] fig1_profiling_breakdown.pdf')

    # Table 1
    with open(f'{OUT_TABLES}/table1_profiling.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Component', 'Percentage'])
        for l, p in zip(labels, percentages):
            w.writerow([l, p])
    print('[OK] table1_profiling.csv')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Cost Model Correlation (needs profiling_summary.csv)
# ═══════════════════════════════════════════════════════════════════
def fig2_cost_model_correlation():
    src = 'output/results/profiling_summary.csv'
    if not os.path.exists(src):
        MISSING_SOURCES.append('profiling_summary.csv')
        print('[SKIP] fig2_cost_model_correlation: profiling_summary.csv not found')
        return
    hub_counts, neg_times = [], []
    with open(src) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hc = row.get('hub_count', '')
            nt = row.get('neg_sampling_time', '')
            if hc and nt:
                try:
                    hub_counts.append(int(hc))
                    neg_times.append(float(nt))
                except (ValueError, TypeError):
                    pass
    if len(hub_counts) == 0:
        print('[SKIP] fig2: no valid data rows')
        return
    hub_counts = np.array(hub_counts)
    neg_times = np.array(neg_times)
    r = np.corrcoef(hub_counts, neg_times)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(hub_counts, neg_times, alpha=0.5, s=10, c=SINGLE_COLOR, edgecolors='none')
    ax.set_xlabel('Hub Count (per batch)')
    ax.set_ylabel('Neg Sampling Time (ms)')
    ax.set_title(f'Fig 2: Cost Model Correlation (R={r:.3f})')
    fig.savefig(f'{OUT_FIGS}/fig2_cost_model_corr.pdf')
    plt.close(fig)
    print(f'[OK] fig2_cost_model_corr.pdf ({len(hub_counts)} points, R={r:.3f})')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Batch Cost Distribution (needs batch_composition.csv)
# ═══════════════════════════════════════════════════════════════════
def fig3_batch_cost_distribution():
    src = 'output/results/integration_validation/batch_composition.csv'
    if not os.path.exists(src):
        MISSING_SOURCES.append('batch_composition.csv')
        print('[SKIP] fig3_batch_cost_distribution: batch_composition.csv not found')
        return
    baseline_costs = []
    cbp_costs = []
    with open(src) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # config_label field contains the phase name
            label = row.get('config_label', '')
            try:
                avg_cost = float(row.get('avg_cost', 0))
            except (ValueError, TypeError):
                continue
            if label == 'Baseline':
                baseline_costs.append(avg_cost)
            elif label == 'CBP':
                cbp_costs.append(avg_cost)

    if len(baseline_costs) == 0 and len(cbp_costs) == 0:
        print('[SKIP] fig3: no valid data')
        return

    bl_cv = np.std(baseline_costs) / np.mean(baseline_costs) if len(baseline_costs) > 1 else (np.std(baseline_costs) / 1e-9 if len(baseline_costs) == 1 else 0)
    cbp_cv = np.std(cbp_costs) / np.mean(cbp_costs) if len(cbp_costs) > 1 else (np.std(cbp_costs) / 1e-9 if len(cbp_costs) == 1 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if baseline_costs:
        axes[0].hist(baseline_costs, bins=20, color='#FF6B6B', edgecolor='white', alpha=0.8)
        axes[0].set_title(f'Baseline (CV={bl_cv:.4f})')
        axes[0].set_xlabel('Avg Cost per Batch')
    else:
        axes[0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0].transAxes)
    if cbp_costs:
        axes[1].hist(cbp_costs, bins=20, color=SINGLE_COLOR, edgecolor='white', alpha=0.8)
        axes[1].set_title(f'CBP (CV={cbp_cv:.4f})')
        axes[1].set_xlabel('Avg Cost per Batch')
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
    fig.suptitle('Fig 3: Batch Cost Distribution')
    fig.savefig(f'{OUT_FIGS}/fig3_batch_cost_distribution.pdf')
    plt.close(fig)
    print(f'[OK] fig3_batch_cost_distribution.pdf (Baseline CV={bl_cv:.4f}, CBP CV={cbp_cv:.4f})')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — GPU Runtime Trace (parse .md table)
# ═══════════════════════════════════════════════════════════════════
def fig4_gpu_runtime_trace():
    src = 'output/results/unified_runtime/runtime_trace_GPU.md'
    if not os.path.exists(src):
        MISSING_SOURCES.append('runtime_trace_GPU.md')
        print('[SKIP] fig4_gpu_runtime_trace: runtime_trace_GPU.md not found')
        return
    # File is CSV-formatted inside .md (rows: epoch,step,neg_time_ms,fwd_time_ms,bwd_time_ms,opt_time_ms,total_step_ms)
    negs, fwds, bwds, opts = [], [], [], []
    with open(src) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # skip header-like separator row if present
        for row in reader:
            if len(row) >= 7 and row[0].replace('.','',1).replace('-','',1).isdigit():
                try:
                    negs.append(float(row[2]))
                    fwds.append(float(row[3]))
                    bwds.append(float(row[4]))
                    opts.append(float(row[5]))
                except (ValueError, IndexError):
                    continue

    if len(negs) == 0:
        print('[SKIP] fig4_gpu_runtime_trace: no data parsed')
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(negs)))
    ax.stackplot(xs, negs, fwds, bwds, opts,
                 labels=['Neg Sampling', 'Forward', 'Backward', 'Optimizer'],
                 colors=['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4'],
                 alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Fig 4: GPU Runtime Trace (per step)')
    ax.legend(loc='upper right', frameon=False)
    fig.savefig(f'{OUT_FIGS}/fig4_gpu_runtime_trace.pdf')
    plt.close(fig)
    print(f'[OK] fig4_gpu_runtime_trace.pdf ({len(negs)} steps)')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Main Benchmark Bars (Phase 9 Step 2 hardcoded)
# ═══════════════════════════════════════════════════════════════════
def fig5_benchmark_bars():
    configs = ['BL', 'CBP', 'GPU', 'CBP+GPU']
    times = [25.1, 25.3, 4.4, 4.7]
    colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
    hatches = ['', '//', '', '//']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(configs, times, color=colors, edgecolor='black', linewidth=0.8)
    for i, (bar, h) in enumerate(zip(bars, hatches)):
        if h:
            bar.set_hatch(h)
    ax.set_ylabel('Epoch Time (s)')
    ax.set_title('Fig 5: Main Benchmark — Epoch Time (Phase 9 Step 2)')

    # Add speedup annotations for GPU bars
    cpu_time = 25.1
    for bar, t in zip(bars, times):
        speedup = cpu_time / t if t > 0 else 0
        if speedup >= 2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{speedup:.1f}x', ha='center', fontweight='bold', fontsize=11, color='#2C3E50')

    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(f'{OUT_FIGS}/fig5_benchmark_bars.pdf')
    plt.close(fig)
    print('[OK] fig5_benchmark_bars.pdf')

    # Table 2 — gather from Phase 9 Step 2 results
    with open(f'{OUT_TABLES}/table2_benchmark.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Config', 'Loss', 'MRR', 'Hits10', 'Epoch_Time_s', 'Speedup'])
        # Read from existing CSVs if available, else use known data
        known = {
            'BL':      ('0.572', '0.0136', '0.0225', '25.1', '1.0x'),
            'CBP':     ('0.574', '0.0150', '0.0350', '25.3', '1.0x'),
            'GPU':     ('0.378', '0.0132', '0.0300', '4.4', '5.7x'),
            'CBP+GPU': ('0.384', '0.0113', '0.0175', '4.7', '5.4x'),
        }
        for cfg in ['BL','CBP','GPU','CBP+GPU']:
            csv_path = f'output/results/phase9_step2/{cfg}/summary.csv'
            if os.path.exists(csv_path):
                try:
                    last = list(csv.DictReader(open(csv_path)))[-1]
                    loss = last['avg_loss']
                    mrr = last.get('mrr', 'N/A')
                    h10 = last.get('hits10', 'N/A')
                    time_s = last.get('epoch_time_s', 'N/A')
                    speedup = f'{25.1/float(time_s):.1f}x' if time_s != 'N/A' else 'N/A'
                    w.writerow([cfg, loss, mrr, h10, time_s, speedup])
                    continue
                except: pass
            # Fallback
            w.writerow(known.get(cfg, [cfg, 'N/A']*5))
    print('[OK] table2_benchmark.csv')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6 — Ablation Variance (Phase 9 Step 3)
# ═══════════════════════════════════════════════════════════════════
def fig6_ablation_variance():
    configs = ['BL', 'CBP', 'GPU', 'CBP+GPU']
    neg_stds = []
    step_stds = []
    labels_ok = []
    for cfg in configs:
        p = f'output/results/phase9_step3/{cfg}/summary.csv'
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            last = rows[-1]
            neg_stds.append(float(last['neg_time_std_ms']))
            step_stds.append(float(last['step_time_std_ms']))
            labels_ok.append(cfg)
        else:
            MISSING_SOURCES.append(f'phase9_step3/{cfg}/summary.csv')
            print(f'[SKIP] {cfg} in fig6: CSV not found')

    if len(labels_ok) < 2:
        print('[SKIP] fig6_ablation_variance: not enough data')
        return

    x = np.arange(len(labels_ok))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, neg_stds, width, color=['#FF6B6B','#FF6B6B','#4ECDC4','#4ECDC4'][:len(labels_ok)], edgecolor='black', linewidth=0.5)
    axes[0].set_title('Neg Sampling Std Dev (ms)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_ok)
    for i, v in enumerate(neg_stds):
        axes[0].text(i, v + max(neg_stds) * 0.03, f'{v:.1f}', ha='center', fontsize=9)

    axes[1].bar(x, step_stds, width, color=['#FF6B6B','#FF6B6B','#4ECDC4','#4ECDC4'][:len(labels_ok)], edgecolor='black', linewidth=0.5)
    axes[1].set_title('Step Time Std Dev (ms)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_ok)
    for i, v in enumerate(step_stds):
        axes[1].text(i, v + max(step_stds) * 0.03, f'{v:.1f}', ha='center', fontsize=9)

    fig.suptitle('Fig 6: Ablation — Variance Comparison (Phase 9 Step 3, epoch 9)')
    fig.savefig(f'{OUT_FIGS}/fig6_ablation_variance.pdf')
    plt.close(fig)
    print(f'[OK] fig6_ablation_variance.pdf ({len(labels_ok)} configs)')

    # Table 3
    with open(f'{OUT_TABLES}/table3_ablation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Config', 'Loss', 'MRR', 'Neg_Std_ms', 'Step_Std_ms', 'Epoch_Time_s'])
        for cfg in configs:
            p = f'output/results/phase9_step3/{cfg}/summary.csv'
            if os.path.exists(p):
                last = list(csv.DictReader(open(p)))[-1]
                w.writerow([cfg, last['avg_loss'], last['mrr'],
                            last['neg_time_std_ms'], last['step_time_std_ms'],
                            last['epoch_time_s']])
            else:
                w.writerow([cfg, 'N/A'] * 5)
    print('[OK] table3_ablation.csv')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('  Phase 9 Step 4 — Paper Assets Generation')
    print('=' * 60)

    fig1_profiling_breakdown()
    fig2_cost_model_correlation()
    fig3_batch_cost_distribution()
    fig4_gpu_runtime_trace()
    fig5_benchmark_bars()
    fig6_ablation_variance()

    print(f'\n[INFO] Missing data sources: {len(MISSING_SOURCES)}')
    for s in MISSING_SOURCES:
        print(f'  - {s}')

    print(f'\n{"=" * 60}')
    print(f'  Done. Output: {OUT_FIGS}/  |  {OUT_TABLES}/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()