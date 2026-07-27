#!/usr/bin/env python3 -u
"""
Phase 6 - Node 4: Runtime Attribution Analysis
目的：验证 batch cost → negative sampling time → total step time 的因果链。
输出：
  - runtime_attribution.md  每个 batch 的分解时间 + batch weight
  - attribution_report.md   相关性矩阵与结论

使用方法：
    python src/py/experiments/runtime_attribution.py
"""

import argparse
import csv
import logging
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import (
    Scheduler, RandomSorter, CostSorter, ChunkPacker, FFDPacker,
)
from src.py.load.batch_provider import BatchProvider
from src.torch.kge_models.pytorch_dataloader import _deep_profiled_neg_sampling
from src.py.util.util import to_tensor_cpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runtime Attribution: causal chain Weight -> NegSampling -> Total"
    )
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dataset', default='FB15K237')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--neg_num', type=int, default=150)
    parser.add_argument('--embed_dim', type=int, default=400,
                        help='TransE embedding dimension')
    parser.add_argument('--output_dir',
                        default='output/results/runtime_attribution/')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('RuntimeAttribution')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, 'attribution.md'),
                             mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def load_dataset_and_build_cost(args, logger):
    dataset_name = args.dataset.lower()
    args_path = os.path.join(
        os.path.dirname(__file__), 'args_kge',
        f'transe_{dataset_name}_args.json'
    )
    logger.info(f"Loading args from: {args_path}")
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True

    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )
    num_entities = kgs.entities_num
    num_relations = kgs.relations_num
    train_triples = kgs.local_relation_triples_list

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"  Entities: {num_entities}, Relations: {num_relations}")
    logger.info(f"  Train triples: {len(train_triples)}")

    # Feature extraction
    logger.info("Extracting graph features...")
    extractor = FeatureExtractor(train_triples, num_entities)
    features = extractor.build()

    # Cost table
    logger.info("Building cost table...")
    cost_table = build_cost_table(features, neg_num=args.neg_num)
    logger.info(f"  Cost table: shape={cost_table.shape}, "
                f"range=[{cost_table.min():.2f}, {cost_table.max():.2f}]")

    # Entities list (for negative sampling)
    entities_list = list(range(num_entities))

    # All triples set (for collision check in neg sampling)
    all_triples_set = set(train_triples)

    return kgs, train_triples, cost_table, entities_list, all_triples_set


class SimpleTransE(torch.nn.Module):
    """Minimal TransE for runtime attribution (no loss/backward needed)"""
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(num_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(num_relations, dim)

    def forward(self, heads, rels, tails):
        # h + r - t (score for each triple)
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(rels)
        t = self.ent_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=-1)


def run_epoch_with_timing(train_triples, cost_table, entities_list,
                          all_triples_set, model, sorter, packer, args,
                          logger, config_label):
    """运行一个 epoch，记录每个 batch 的分解时间"""
    scheduler = Scheduler(sorter, packer)
    provider = BatchProvider(scheduler, cost_table, args.batch_size,
                             enable_logging=False)

    results = []
    model.train()

    # Warmup CUDA
    dummy = model(
        torch.zeros(1, dtype=torch.long).cuda(),
        torch.zeros(1, dtype=torch.long).cuda(),
        torch.zeros(1, dtype=torch.long).cuda(),
    )
    torch.cuda.synchronize()

    # Pre-build neighbor dict (empty = use full entities_list)
    neighbor = dict()

    batch_count = 0
    for batch_triples in provider.iterate(train_triples):
        # --- Batch Weight: avg cost from cost_table ---
        costs = []
        for h, r, t in batch_triples:
            hc = float(cost_table[h]) if h < len(cost_table) else 0.0
            tc = float(cost_table[t]) if t < len(cost_table) else 0.0
            costs.append(max(hc, tc))
        batch_weight = float(np.mean(costs)) if costs else 0.0

        # --- Stage 1: Negative Sampling (CPU) ---
        # Reuse the project's actual deep-profiled neg sampling function
        neg_start = time.perf_counter()
        neg_batch, retry_info = _deep_profiled_neg_sampling(
            type('obj', (object,), {}),  # dummy self (method is unbound)
            batch_triples, all_triples_set, entities_list,
            args.neg_num, neighbor, max_try=10,
        )
        neg_end = time.perf_counter()
        neg_sampling_time = neg_end - neg_start  # seconds

        # --- Stage 2: Tensor Construction (CPU→GPU transfer) ---
        tensor_start = time.perf_counter()
        heads = torch.tensor([t[0] for t in batch_triples], dtype=torch.long).cuda()
        rels  = torch.tensor([t[1] for t in batch_triples], dtype=torch.long).cuda()
        tails = torch.tensor([t[2] for t in batch_triples], dtype=torch.long).cuda()
        neg_heads = torch.tensor([n[0] for n in neg_batch], dtype=torch.long).cuda()
        neg_rels  = torch.tensor([n[1] for n in neg_batch], dtype=torch.long).cuda()
        neg_tails = torch.tensor([n[2] for n in neg_batch], dtype=torch.long).cuda()
        torch.cuda.synchronize()
        tensor_end = time.perf_counter()
        tensor_time = tensor_end - tensor_start

        # --- Stage 3: Forward (GPU) ---
        fwd_start = time.perf_counter()
        # Positive scores
        pos_scores = model(heads, rels, tails)
        # Negative scores
        neg_scores = model(neg_heads, neg_rels, neg_tails)
        torch.cuda.synchronize()
        fwd_end = time.perf_counter()
        forward_time = fwd_end - fwd_start

        total_time = neg_sampling_time + tensor_time + forward_time

        results.append({
            'config': config_label,
            'batch_idx': batch_count,
            'batch_weight': batch_weight,
            'neg_sampling_time': round(neg_sampling_time * 1000, 4),  # ms
            'tensor_time': round(tensor_time * 1000, 4),              # ms
            'forward_time': round(forward_time * 1000, 4),            # ms
            'total_time': round(total_time * 1000, 4),                # ms
        })

        batch_count += 1
        if batch_count % 20 == 0:
            logger.info(f"  [{config_label}] batch {batch_count}: "
                        f"weight={batch_weight:.2f}, "
                        f"neg={neg_sampling_time*1000:.1f}ms, "
                        f"fwd={forward_time*1000:.1f}ms")

    logger.info(f"  [{config_label}] Completed {batch_count} batches.")
    return results


def compute_correlation(x, y, label, f):
    """Compute Pearson r between x and y"""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return r


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("Runtime Attribution Analysis")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}, Batch size: {args.batch_size}, "
                f"Neg num: {args.neg_num}, Embed dim: {args.embed_dim}")

    # Load data
    kgs, train_triples, cost_table, entities_list, all_triples_set = \
        load_dataset_and_build_cost(args, logger)

    # Build model (TransE)
    logger.info("Building TransE model...")
    model = SimpleTransE(kgs.entities_num, kgs.relations_num, args.embed_dim).cuda()
    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters())}")

    # --- Baseline: RandomSorter + ChunkPacker ---
    sorter_base = RandomSorter(seed=args.seed)
    packer_base = ChunkPacker()
    base_results = run_epoch_with_timing(
        train_triples, cost_table, entities_list, all_triples_set,
        model, sorter_base, packer_base, args, logger, "Baseline",
    )

    # --- CBP: CostSorter + FFDPacker ---
    sorter_cbp = CostSorter()
    packer_cbp = FFDPacker()
    cbp_results = run_epoch_with_timing(
        train_triples, cost_table, entities_list, all_triples_set,
        model, sorter_cbp, packer_cbp, args, logger, "CBP",
    )

    all_results = base_results + cbp_results

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'runtime_attribution.md')
    fieldnames = ['config', 'batch_idx', 'batch_weight',
                  'neg_sampling_time', 'tensor_time', 'forward_time',
                  'total_time']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"CSV saved: {csv_path} ({len(all_results)} rows)")

    # --- Correlation analysis ---
    def extract(arr):
        return np.array([r[arr] for r in base_results]), \
               np.array([r[arr] for r in cbp_results])

    base_w, cbp_w = extract('batch_weight')
    base_neg, cbp_neg = extract('neg_sampling_time')
    base_tensor, cbp_tensor = extract('tensor_time')
    base_fwd, cbp_fwd = extract('forward_time')
    base_total, cbp_total = extract('total_time')

    pairs = [
        ('Weight vs Neg Sampling', base_w, base_neg, cbp_w, cbp_neg),
        ('Weight vs Tensor',       base_w, base_tensor, cbp_w, cbp_tensor),
        ('Weight vs Forward',      base_w, base_fwd, cbp_w, cbp_fwd),
        ('Weight vs Total',        base_w, base_total, cbp_w, cbp_total),
        ('Neg vs Total',           base_neg, base_total, cbp_neg, cbp_total),
        ('Tensor vs Total',        base_tensor, base_total, cbp_tensor, cbp_total),
        ('Forward vs Total',       base_fwd, base_total, cbp_fwd, cbp_total),
    ]

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("RUNTIME ATTRIBUTION ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Dataset: {args.dataset}")
    report_lines.append(f"Batch size: {args.batch_size}, Neg num: {args.neg_num}")
    report_lines.append(f"Baseline: RandomSorter + ChunkPacker")
    report_lines.append(f"CBP: CostSorter + FFDPacker")
    report_lines.append("")

    # Summary stats per config
    for label, arr in [('Baseline', base_results), ('CBP', cbp_results)]:
        ws = [r['batch_weight'] for r in arr]
        ns = [r['neg_sampling_time'] for r in arr]
        ts = [r['tensor_time'] for r in arr]
        fs = [r['forward_time'] for r in arr]
        tots = [r['total_time'] for r in arr]
        report_lines.append(f"--- {label} ---")
        report_lines.append(f"  Batch weight: mean={np.mean(ws):.4f}, std={np.std(ws):.4f}")
        report_lines.append(f"  Neg sampling: mean={np.mean(ns):.1f}ms, std={np.std(ns):.1f}ms")
        report_lines.append(f"  Tensor build: mean={np.mean(ts):.1f}ms, std={np.std(ts):.1f}ms")
        report_lines.append(f"  Forward:      mean={np.mean(fs):.1f}ms, std={np.std(fs):.1f}ms")
        report_lines.append(f"  Total step:   mean={np.mean(tots):.1f}ms, std={np.std(tots):.1f}ms")
        report_lines.append("")

    # Correlation table
    report_lines.append(f"{'Correlation':<30} {'Baseline r':>10} {'CBP r':>10}")
    report_lines.append("-" * 52)
    for label, bx, by, cx, cy in pairs:
        br = compute_correlation(bx, by, label, None)
        cr = compute_correlation(cx, cy, label, None)
        report_lines.append(f"{label:<30} {br:10.4f} {cr:10.4f}")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("Causal Chain Analysis")
    report_lines.append("=" * 60)

    # Check: Weight → Neg Sampling correlation
    br_wn = compute_correlation(base_w, base_neg, '', None)
    cr_wn = compute_correlation(cbp_w, cbp_neg, '', None)
    report_lines.append("")
    report_lines.append(f"1. Weight → Neg Sampling: Baseline r={br_wn:.4f}, CBP r={cr_wn:.4f}")
    if cr_wn < br_wn * 0.5:
        report_lines.append("   ✅ CBP significantly weakened the Weight→NegSampling link!")
    elif cr_wn < br_wn:
        report_lines.append("   ⚠️  CBP weakened the link but not dramatically.")
    else:
        report_lines.append("   ❌ CBP did NOT weaken the Weight→NegSampling link.")

    # Check: Neg Sampling → Total
    br_nt = compute_correlation(base_neg, base_total, '', None)
    cr_nt = compute_correlation(cbp_neg, cbp_total, '', None)
    report_lines.append("")
    report_lines.append(f"2. Neg Sampling → Total: Baseline r={br_nt:.4f}, CBP r={cr_nt:.4f}")
    report_lines.append("   Interpretation: If r > 0.9, neg sampling dominates total time.")
    if br_nt > 0.9 or cr_nt > 0.9:
        report_lines.append("   → Neg sampling IS the dominant factor in step time.")
    else:
        report_lines.append("   → Neg sampling is NOT the dominant factor.")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 60)

    # Overall verdict
    if cr_wn < br_wn * 0.5 and cr_nt > 0.9:
        report_lines.append(
            "CBP breaks the Weight→NegSampling correlation, but neg sampling "
            "still dominates total time.\n"
            "→ Total time variance reduction is limited by the neg sampling bottleneck.\n"
            "→ Recommendation: GPU-accelerate negative sampling for further gains."
        )
    elif cr_wn < br_wn:
        report_lines.append(
            "CBP partially reduces Weight→NegSampling correlation.\n"
            "→ Further tuning of cost model or packing strategy may improve."
        )
    else:
        report_lines.append(
            "CBP did NOT reduce Weight→NegSampling correlation.\n"
            "→ Cost model may not match actual runtime cost.\n"
            "→ Investigate cost_table vs actual neg sampling time mapping."
        )

    report_path = os.path.join(args.output_dir, 'attribution_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Report saved: {report_path}")

    # Print report
    for line in report_lines:
        print(line)

    return 0


if __name__ == '__main__':
    sys.exit(main())