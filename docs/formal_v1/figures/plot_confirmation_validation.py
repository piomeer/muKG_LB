"""Replot frozen confirmation validation records; does not train or read test ranks."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_SHA256 = '9f5d28ed4b398cad935fa5b6e138d7b9946f07509bc19f90a4ff9b12d2aa1fb8'
GROUPS = ('A', 'M00', 'M10-GPU', 'M11-GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    receipt = args.output.with_suffix('.json')
    if args.output.exists() or receipt.exists():
        raise FileExistsError('Choose new output paths; existing artifacts are preserved')
    raw = args.input.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != INPUT_SHA256:
        raise ValueError('Input does not match the verified confirmation-v1 records')
    records = json.loads(raw)
    quality = [row for row in records if row['task']['kind'] == 'Q']
    if len(quality) != 48:
        raise ValueError('Expected all 48 quality tasks')
    colors = ('#303030', '#D55E00', '#0072B2', '#009E73')
    fields = ('epoch', 'training_seconds', 'wall_seconds')
    titles = ('Training budget', 'Accumulated training time', 'Training-loop wall clock')
    labels = ('Epoch', 'Training time (hours)', 'Wall time (hours)')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    data_ranges = {}
    for group, color in zip(GROUPS, colors):
        rows = sorted((row for row in quality if row['task']['group'] == group),
                      key=lambda row: row['task']['seed'])
        if [row['task']['seed'] for row in rows] != list(range(501, 513)):
            raise ValueError('Missing or duplicated paired seed')
        for row in rows:
            history = row['history']
            if [int(point['epoch']) for point in history] != list(range(10, 1001, 10)):
                raise ValueError('Missing validation checkpoint')
            for field in fields:
                values = [float(point[field]) for point in history]
                if any(a >= b for a, b in zip(values, values[1:])):
                    raise ValueError('Non-increasing time or epoch')
        ys = [[float(point['mrr']) for point in row['history']] for row in rows]
        if any(not 0 <= value <= 1 for values in ys for value in values):
            raise ValueError('Invalid validation MRR')
        median_y = [statistics.median(values) for values in zip(*ys)]
        data_ranges[group] = {'seeds': [row['task']['seed'] for row in rows],
                              'validations': sum(len(row['history']) for row in rows)}
        for ax, field in zip(axes, fields):
            scale = 1 if field == 'epoch' else 3600
            xs = [[float(point[field])/scale for point in row['history']] for row in rows]
            for x, y in zip(xs, ys):
                ax.plot(x, y, color=color, alpha=.15, linewidth=.65)
            median_x = [statistics.median(values) for values in zip(*xs)]
            ax.plot(median_x, median_y, color=color, label=group,
                    linewidth=2.3 if group == 'M10-GPU' else 1.6,
                    linestyle='--' if group == 'M11-GPU' else '-')
    for ax, title, label in zip(axes, titles, labels):
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Full bidirectional filtered validation MRR')
    axes[0].legend(loc='lower right', frameon=False)
    fig.suptitle('Confirmation-v1: all 12 paired seeds per group, 1,000 epochs', fontsize=12)
    fig.text(.5, .015, 'Thin: individual seeds. Thick: coordinate-wise medians at each validation epoch. '
             'Lines connect observations; no time-to-quality inference.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    evidence = {'date': '2026-11-17', 'input_sha256': actual,
                'input': 'output/results/confirmation_v1/analysis-01/recomputed_tasks.json',
                'groups': data_ranges, 'points_per_panel': 4800,
                'axes': list(fields), 'aggregate': 'coordinate-wise median at each epoch',
                'wall_clock': 'training loop, including validation; excludes process setup and finalization',
                'selection': 'all successful frozen Q tasks; no test or best-checkpoint selection',
                'matplotlib': matplotlib.__version__,
                'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'quality_tasks': len(quality), 'points_per_panel': 4800,
                      'input_sha256': actual, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
