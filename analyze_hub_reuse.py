#!/usr/bin/env python3
"""
MuKG Phase 4 — Hub Reuse & Cache Feasibility Analysis
(Pure Python 3.8 stdlib, no numpy/matplotlib required)

分析 FB15k-237 训练过程中每个实体的出现次数，验证：
  Hub Entity → High Reuse → Cache Opportunity

输出:
  - output/results/hub_reuse_analysis.csv  (Task 1)
  - output/results/coverage_analysis.csv    (Task 2)
  - output/results/cache_feasibility.csv    (Task 3)
  - figs/entity_rank_vs_occurrence.svg      (Figure 1, SVG)
  - figs/topk_vs_cache_hit_rate.svg         (Figure 2, SVG)
"""

import csv
import os
import math
from collections import Counter

# ---------- paths ----------
DATA_DIR = "src/py/data/FB15K237"
TRAIN_PATH = os.path.join(DATA_DIR, "train2id.txt")
ENTITY_PATH = os.path.join(DATA_DIR, "entity2id.txt")

OUT_DIR = "output/results"
FIGS_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------- 1. Load data ----------
print("[INFO] Loading entity count...")
with open(ENTITY_PATH, "r") as f:
    n_entities = int(f.readline().strip())
print(f"  Total entities: {n_entities}")

print("[INFO] Loading training triples & counting entity occurrences...")
entity_counter = Counter()

with open(TRAIN_PATH, "r") as f:
    n_triples = int(f.readline().strip())
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        h, t = int(parts[0]), int(parts[1])
        entity_counter[h] += 1
        entity_counter[t] += 1

total_access_count = sum(entity_counter.values())
print(f"  Total triples: {n_triples}")
print(f"  Total entity accesses (head + tail): {total_access_count}")
print(f"  Unique entities accessed: {len(entity_counter)}")

# Sort by occurrence count descending
sorted_entities = entity_counter.most_common()  # list of (entity_id, count)

# ---------- 2. Task 1: hub_reuse_analysis.csv ----------
print("\n[Task 1] Building hub_reuse_analysis.csv ...")

rows_task1 = []
for rank, (eid, cnt) in enumerate(sorted_entities, start=1):
    rows_task1.append({
        "entity_id": eid,
        "degree": cnt,
        "occurrence_count": cnt,
        "rank": rank,
    })

with open(os.path.join(OUT_DIR, "hub_reuse_analysis.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["entity_id", "degree", "occurrence_count", "rank"])
    writer.writeheader()
    writer.writerows(rows_task1)

print(f"  Written {len(rows_task1)} rows to hub_reuse_analysis.csv")

# ---------- 3. Compute cumulative arrays (pure Python) ----------
occurrence_counts = [c for _, c in sorted_entities]
cumulative_occurrences = []
running = 0
for c in occurrence_counts:
    running += c
    cumulative_occurrences.append(running)

total_entities = len(sorted_entities)
total_occur = cumulative_occurrences[-1] if cumulative_occurrences else 0

# ---------- 4. Task 2: coverage_analysis.csv ----------
print("\n[Task 2] Building coverage_analysis.csv ...")

thresholds = [0.01, 0.05, 0.10]  # Top 1%, 5%, 10%

rows_task2 = []
for frac in thresholds:
    n_hubs = max(1, int(total_entities * frac))
    n_hubs_actual = min(n_hubs, len(cumulative_occurrences))
    occur_at_threshold = cumulative_occurrences[n_hubs_actual - 1]
    coverage_ratio = occur_at_threshold / total_occur if total_occur > 0 else 0.0

    rows_task2.append({
        "hub_threshold": "Top {:.0f}%".format(frac * 100),
        "entity_count": n_hubs_actual,
        "occurrence_count": int(occur_at_threshold),
        "coverage_ratio": round(coverage_ratio, 6),
    })
    print("  Top {:.0f}%: {} entities, {} occurrences, coverage = {:.4%}".format(
        frac * 100, n_hubs_actual, occur_at_threshold, coverage_ratio))

with open(os.path.join(OUT_DIR, "coverage_analysis.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["hub_threshold", "entity_count", "occurrence_count", "coverage_ratio"])
    writer.writeheader()
    writer.writerows(rows_task2)

print("  Written {} rows to coverage_analysis.csv".format(len(rows_task2)))

# ---------- 5. Task 3: cache_feasibility.csv ----------
print("\n[Task 3] Building cache_feasibility.csv ...")

top_k_values = [10, 50, 100, 500, 1000]

rows_task3 = []
for k in top_k_values:
    k_actual = min(k, len(cumulative_occurrences))
    cache_hit_count = int(cumulative_occurrences[k_actual - 1])
    cache_hit_rate = cache_hit_count / total_occur if total_occur > 0 else 0.0

    rows_task3.append({
        "top_k": k_actual,
        "cache_hit_count": cache_hit_count,
        "total_access_count": total_occur,
        "cache_hit_rate": round(cache_hit_rate, 6),
    })
    print("  Top {}: cache_hit_count = {}, cache_hit_rate = {:.4%}".format(
        k_actual, cache_hit_count, cache_hit_rate))

with open(os.path.join(OUT_DIR, "cache_feasibility.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["top_k", "cache_hit_count", "total_access_count", "cache_hit_rate"])
    writer.writeheader()
    writer.writerows(rows_task3)

print("  Written {} rows to cache_feasibility.csv".format(len(rows_task3)))

# ============================================================
# 6. SVG Chart Generation (pure Python, no external deps)
# ============================================================
print("\n[Task 4] Generating SVG figures ...")

def log10(x):
    """Safe log10, returns 0 for x <= 0."""
    return math.log10(x) if x > 0 else 0

def generate_svg_bar_chart(x_label, y_label, title, points, width=800, height=500,
                           margins=(60, 40, 50, 50), log_x=False, log_y=False,
                           annotate_points=None, point_labels=None):
    """
    Generate an SVG scatter/line chart.

    points: list of (x, y) tuples, already sorted.
    """
    ml, mr, mb, mt = margins  # margin left, right, bottom, top
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1

    # Padding
    if log_x:
        x_range = log10(x_max) - log10(x_min) if x_min > 0 else log10(x_max)
    else:
        x_range = x_max - x_min
    if log_y:
        y_range = log10(y_max) - log10(y_min) if y_min > 0 else log10(y_max)
    else:
        y_range = y_max - y_min

    x_pad = x_range * 0.05 if x_range > 0 else 1
    y_pad = y_range * 0.05 if y_range > 0 else 1

    def scale_x(val):
        if log_x:
            v = (log10(val) - (log10(x_min) - x_pad)) / (log10(x_max) - log10(x_min) + 2 * x_pad) if x_min > 0 else (log10(val) - log10(x_max) + x_range + x_pad) / (x_range + 2 * x_pad)
            return ml + v * plot_w
        else:
            v = (val - (x_min - x_pad)) / (x_max - x_min + 2 * x_pad)
            return ml + v * plot_w

    def scale_y(val):
        if log_y:
            v = (log10(val) - (log10(y_min) - y_pad)) / (log10(y_max) - log10(y_min) + 2 * y_pad) if y_min > 0 else (log10(val) - log10(y_max) + y_range + y_pad) / (y_range + 2 * y_pad)
            return mt + plot_h - v * plot_h
        else:
            v = (val - (y_min - y_pad)) / (y_max - y_min + 2 * y_pad)
            return mt + plot_h - v * plot_h

    # Build SVG
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" '
                 'width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height))
    lines.append('  <rect width="100%" height="100%" fill="white"/>')

    # Title
    lines.append('  <text x="{}" y="{}" text-anchor="middle" '
                 'font-size="16" font-weight="bold" fill="#333">{}</text>'.format(
                     width // 2, mt - 10, title))

    # Plot area border
    lines.append('  <rect x="{}" y="{}" width="{}" height="{}" fill="none" '
                 'stroke="#ccc" stroke-width="1"/>'.format(ml, mt, plot_w, plot_h))

    # Grid lines
    n_grid = 5
    for i in range(n_grid + 1):
        frac = i / n_grid
        gx = ml + plot_w * frac
        gy = mt + plot_h * (1 - frac)
        lines.append('  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#eee" stroke-width="1"/>'.format(
            ml, gy, ml + plot_w, gy))
        lines.append('  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#eee" stroke-width="1"/>'.format(
            gx, mt, gx, mt + plot_h))

        # Y axis labels
        if log_y:
            lbl = 10 ** (log10(y_min) * (1 - frac) + log10(y_max) * frac) if y_min > 0 else \
                  10 ** (log10(y_min + y_pad) * (1 - frac) + log10(y_max) * frac)
            lbl_str = "{:.0f}".format(lbl) if lbl >= 1 else "{:.2f}".format(lbl)
        else:
            lbl = y_min * (1 - frac) + y_max * frac
            lbl_str = "{:.0f}".format(lbl) if lbl >= 100 else "{:.1f}".format(lbl)
        lines.append('  <text x="{}" y="{}" text-anchor="end" font-size="10" fill="#666">{}</text>'.format(
            ml - 5, gy + 4, lbl_str))

        # X axis labels
        if log_x:
            lbl = 10 ** (log10(x_min) * (1 - frac) + log10(x_max) * frac) if x_min > 0 else \
                  10 ** (log10(x_min + x_pad) * (1 - frac) + log10(x_max) * frac)
            lbl_str = "{:.0f}".format(lbl) if lbl >= 1 else "{:.2f}".format(lbl)
        else:
            lbl = x_min * (1 - frac) + x_max * frac
            lbl_str = "{:.0f}".format(lbl) if lbl >= 100 else "{:.1f}".format(lbl)
        lines.append('  <text x="{}" y="{}" text-anchor="middle" font-size="10" fill="#666">{}</text>'.format(
            gx, mt + plot_h + 15, lbl_str))

    # Axis labels
    lines.append('  <text x="{}" y="{}" text-anchor="middle" font-size="13" fill="#333">{}</text>'.format(
        ml + plot_w // 2, height - 5, x_label))
    lines.append('  <text x="{}" y="{}" text-anchor="middle" font-size="13" fill="#333" '
                 'transform="rotate(-90, {}, {})">{}</text>'.format(
                     15, mt + plot_h // 2, 15, mt + plot_h // 2, y_label))

    # Data line
    if len(points) >= 2:
        path_data = []
        for i, (px, py) in enumerate(points):
            sx, sy = scale_x(px), scale_y(py)
            path_data.append("{}{:.1f},{:.1f}".format("M" if i == 0 else "L", sx, sy))
        lines.append('  <path d="{}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>'.format(" ".join(path_data)))

    # Data points
    for px, py in points:
        sx, sy = scale_x(px), scale_y(py)
        lines.append('  <circle cx="{:.1f}" cy="{:.1f}" r="2.5" fill="#1f77b4"/>'.format(sx, sy))

    # Annotations
    if annotate_points:
        for i, ap_idx in enumerate(annotate_points):
            if ap_idx < len(points):
                px, py = points[ap_idx]
                sx, sy = scale_x(px), scale_y(py)
                lbl = point_labels[i] if point_labels else "({:.0f},{:.0f})".format(px, py)
                # Arrow + label box
                ax = sx + 40
                ay = sy - 30
                lines.append('  <line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                             'stroke="red" stroke-width="1" marker-end="url(#arrow)"/>'.format(sx, sy, ax, ay))
                lines.append('  <rect x="{:.1f}" y="{:.1f}" width="{}" height="20" rx="3" '
                             'fill="lightyellow" stroke="red" stroke-width="1"/>'.format(
                                 ax, ay, len(lbl) * 7 + 10))
                lines.append('  <text x="{:.1f}" y="{:.1f}" font-size="10" fill="#333">{}</text>'.format(
                    ax + 5, ay + 14, lbl))

    lines.append('</svg>')
    return "\n".join(lines)


# --- Figure 1: Entity Rank vs Occurrence Count (log-log) ---
print("  Generating Figure 1: Entity Rank vs Occurrence Count...")
# Downsample for SVG: take first 2000 points + every Nth for the rest
max_svg_points = 5000
if len(occurrence_counts) > max_svg_points:
    step = max(1, len(occurrence_counts) // max_svg_points)
    fig1_points = [(i + 1, occurrence_counts[i]) for i in range(0, len(occurrence_counts), step)]
    # Always include the first point
    if fig1_points[0][0] != 1:
        fig1_points.insert(0, (1, occurrence_counts[0]))
else:
    fig1_points = [(i + 1, occurrence_counts[i]) for i in range(len(occurrence_counts))]

# Annotation indices in the downsampled list
annot_data = []
for k in [10, 100, 1000]:
    if k <= len(occurrence_counts):
        annot_data.append(k)

# Find closest index in fig1_points for each annotation
annot_indices = []
annot_labels = []
for k in [10, 100, 1000]:
    if k <= len(occurrence_counts):
        closest_idx = min(range(len(fig1_points)), key=lambda i: abs(fig1_points[i][0] - k))
        annot_indices.append(closest_idx)
        annot_labels.append("Top-{}\n({})".format(k, occurrence_counts[k - 1]))

svg1 = generate_svg_bar_chart(
    x_label="Entity Rank (log)",
    y_label="Occurrence Count (log)",
    title="Figure 1: Entity Rank vs Occurrence Count (FB15k-237)",
    points=fig1_points,
    log_x=True,
    log_y=True,
    annotate_points=annot_indices,
    point_labels=annot_labels,
)

with open(os.path.join(FIGS_DIR, "entity_rank_vs_occurrence.svg"), "w") as f:
    f.write(svg1)
print("  Saved: figs/entity_rank_vs_occurrence.svg")

# --- Figure 2: Top-K Hub Entities vs Cache Hit Rate ---
print("  Generating Figure 2: Top-K vs Cache Hit Rate...")
fig2_points = []
for k in top_k_values:
    if k <= len(cumulative_occurrences):
        hr = cumulative_occurrences[k - 1] / total_occur * 100
        fig2_points.append((k, hr))

# Simple SVG for figure 2 (linear scale, line chart with dots)
def generate_svg_figure2(x_label, y_label, title, points, width=800, height=500,
                          margins=(60, 50, 50, 60)):
    ml, mr, mb, mt = margins
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0, max(ys) * 1.15  # Start y at 0

    if x_min == x_max:
        x_max = x_min + 1

    x_pad = (x_max - x_min) * 0.05
    y_pad = y_max * 0.05

    def scale_x(val):
        v = (val - (x_min - x_pad)) / (x_max - x_min + 2 * x_pad)
        return ml + v * plot_w

    def scale_y(val):
        v = (val - (y_min - y_pad)) / (y_max - y_min + 2 * y_pad)
        return mt + plot_h - v * plot_h

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" '
                 'width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height))
    lines.append('  <rect width="100%" height="100%" fill="white"/>')

    # Title
    lines.append('  <text x="{}" y="{}" text-anchor="middle" '
                 'font-size="16" font-weight="bold" fill="#333">{}</text>'.format(
                     width // 2, mt - 10, title))
    # Plot border
    lines.append('  <rect x="{}" y="{}" width="{}" height="{}" fill="none" '
                 'stroke="#ccc" stroke-width="1"/>'.format(ml, mt, plot_w, plot_h))

    # Grid lines (horizontal only)
    n_grid = 5
    for i in range(n_grid + 1):
        frac = i / n_grid
        gy = mt + plot_h * (1 - frac)
        lines.append('  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#eee" stroke-width="1"/>'.format(
            ml, gy, ml + plot_w, gy))
        lbl = y_min * (1 - frac) + y_max * frac
        lbl_str = "{:.1f}".format(lbl)
        lines.append('  <text x="{}" y="{}" text-anchor="end" font-size="10" fill="#666">{}</text>'.format(
            ml - 5, gy + 4, lbl_str))

    # X axis labels
    for i, (px, py) in enumerate(points):
        sx = scale_x(px)
        lines.append('  <text x="{:.1f}" y="{}" text-anchor="middle" font-size="10" fill="#666">{}</text>'.format(
            sx, mt + plot_h + 15, int(px)))

    # Axis labels
    lines.append('  <text x="{}" y="{}" text-anchor="middle" font-size="13" fill="#333">{}</text>'.format(
        ml + plot_w // 2, height - 5, x_label))
    lines.append('  <text x="{}" y="{}" text-anchor="middle" font-size="13" fill="#333" '
                 'transform="rotate(-90, {}, {})">{}</text>'.format(
                     15, mt + plot_h // 2, 15, mt + plot_h // 2, y_label))

    # Data line
    if len(points) >= 2:
        path_data = []
        for i, (px, py) in enumerate(points):
            sx, sy = scale_x(px), scale_y(py)
            path_data.append("{}{:.1f},{:.1f}".format("M" if i == 0 else "L", sx, sy))
        lines.append('  <path d="{}" fill="none" stroke="green" stroke-width="2"/>'.format(" ".join(path_data)))

    # Data points + labels
    for px, py in points:
        sx, sy = scale_x(px), scale_y(py)
        lines.append('  <circle cx="{:.1f}" cy="{:.1f}" r="5" fill="red" stroke="white" stroke-width="1"/>'.format(
            sx, sy))
        # Label
        lines.append('  <rect x="{:.1f}" y="{:.1f}" width="55" height="18" rx="3" '
                     'fill="lightblue" stroke="#666" stroke-width="1"/>'.format(
                         sx - 27.5, sy - 30))
        lines.append('  <text x="{:.1f}" y="{:.1f}" text-anchor="middle" font-size="10" '
                     'fill="#333">{:.2f}%</text>'.format(sx, sy - 16, py))

    lines.append('</svg>')
    return "\n".join(lines)

svg2 = generate_svg_figure2(
    x_label="Top-K Hub Entities",
    y_label="Cache Hit Rate (%)",
    title="Figure 2: Top-K Hub Entities vs Theoretical Cache Hit Rate",
    points=fig2_points,
)

with open(os.path.join(FIGS_DIR, "topk_vs_cache_hit_rate.svg"), "w") as f:
    f.write(svg2)
print("  Saved: figs/topk_vs_cache_hit_rate.svg")

# ============================================================
# 7. Summary: Answer Research Questions
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Q1: Top 1% coverage
top_1pct_count = max(1, int(total_entities * 0.01))
top_1pct_actual = min(top_1pct_count, len(cumulative_occurrences))
top_1pct_occur = cumulative_occurrences[top_1pct_actual - 1]
top_1pct_ratio = top_1pct_occur / total_occur * 100
print("\nQ1: Top 1% Hub ({} entities) covers {}/{} accesses = {:.2f}%".format(
    top_1pct_actual, top_1pct_occur, total_occur, top_1pct_ratio))

# Q2: Top 100 coverage
top100_count = min(100, len(cumulative_occurrences))
top100_occur = cumulative_occurrences[top100_count - 1]
top100_ratio = top100_occur / total_occur * 100
print("Q2: Top 100 Hub covers {}/{} accesses = {:.2f}%".format(
    top100_occur, total_occur, top100_ratio))

# Q3: Max theoretical cache hit rate (Top 1000)
top1000_count = min(1000, len(cumulative_occurrences))
top1000_occur = cumulative_occurrences[top1000_count - 1]
top1000_ratio = top1000_occur / total_occur * 100
print("Q3: Top 1000 Hub max theoretical cache hit rate = {:.2f}%".format(top1000_ratio))

# Q4: Assessment
long_tail_count = sum(1 for c in occurrence_counts if c <= 10)
print("\nQ4: Hub-aware Cache Sampling Assessment")
print("    Total entities: {}".format(total_entities))
for k, hr in zip([k for k in top_k_values if k <= len(cumulative_occurrences)],
                  [cumulative_occurrences[k - 1] / total_occur * 100 for k in top_k_values if k <= len(cumulative_occurrences)]):
    print("    Top {} cache hit rate: {:.2f}%".format(k, hr))
print("    Long-tail entities (<=10 occurrences): {} / {}".format(long_tail_count, total_entities))

print("\n[INFO] All tasks completed.")