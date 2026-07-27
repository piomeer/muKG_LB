#!/usr/bin/env python3 -u
"""
Phase 6 - Node 4: CBP Evaluation on FB15k-237
==============================================
Real training experiment comparing Baseline (Random+Chunk) vs CBP (Cost+FFD).

Usage:
    python src/py/experiments/run_cbp_evaluation.py --sorter Random --packer Chunk   # Exp-1
    python src/py/experiments/run_cbp_evaluation.py --sorter Cost  --packer FFD     # Exp-2
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.model.general_models import kge_models
from src.py.base.losses import get_loss_func_torch
from src.py.base.optimizers import get_optimizer_torch
from src.py.evaluation.evaluation import LinkPredictionEvaluator

# CBP imports
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import Scheduler, RandomSorter, CostSorter, ChunkPacker, FFDPacker
from src.py.load.batch_provider import BatchProvider

from src.torch.kge_models.pytorch_dataloader import (
    init_entity_degree, reset_per_batch_profiling,
    GLOBAL_ENTITY_DEGREE, HUB_DEGREE_THRESHOLD,
)


def build_scheduler(sorter_name: str, packer_name: str) -> Scheduler:
    """Build Scheduler from CLI flags."""
    sorter_map = {"random": RandomSorter(), "cost": CostSorter()}
    packer_map = {"chunk": ChunkPacker(), "ffd": FFDPacker()}
    sorter = sorter_map.get(sorter_name.lower().strip())
    packer = packer_map.get(packer_name.lower().strip())
    if sorter is None or packer is None:
        raise ValueError(f"Unknown sorter={sorter_name} or packer={packer_name}")
    return Scheduler(sorter, packer)


def train_epoch_with_provider(
    model, optimizer, loss_fn, device, args, kgs,
    triples_list, cost_table, scheduler, epoch_idx,
    out_dir, exp_label,
    profiling_rows, batch_runtime_rows
):
    """Run one epoch using BatchProvider instead of DataLoader."""
    provider = BatchProvider(scheduler, cost_table, args.batch_size, enable_logging=(epoch_idx == 0))

    batch_size = args.batch_size
    neg_num = args.neg_triple_num
    total_loss = 0.0
    total_triples = 0
    n_batches = 0

    epoch_start = time.time()

    # Per-batch overhead accumulators
    acc_neg_time = 0.0
    acc_fwd_time = 0.0
    acc_bwd_time = 0.0
    acc_opt_time = 0.0

    # Step-level runtime accumulator (for variance analysis)
    step_times_ms = []

    for step_idx, batch_triples in enumerate(provider.iterate(triples_list)):
        optimizer.zero_grad()

        # ---- Negative Sampling (CPU) ----
        t0 = time.perf_counter()
        pos_h = [t[0] for t in batch_triples]
        pos_r = [t[1] for t in batch_triples]
        pos_t = [t[2] for t in batch_triples]
        batch_pos_size = len(batch_triples)

        neg_h = []
        neg_t = []
        num_entities = kgs.entities_num
        for _ in range(batch_pos_size * neg_num):
            neg_h.append(np.random.randint(0, num_entities))
            neg_t.append(np.random.randint(0, num_entities))
        torch.cuda.synchronize()
        neg_time = (time.perf_counter() - t0)

        # Build tensor batch
        all_h = torch.tensor(pos_h + neg_h, dtype=torch.long, device=device)
        all_r = torch.tensor(pos_r * (neg_num + 1), dtype=torch.long, device=device)
        all_t = torch.tensor(pos_t + neg_t, dtype=torch.long, device=device)

        data = {'batch_h': all_h, 'batch_r': all_r, 'batch_t': all_t}

        # ---- Forward ----
        torch.cuda.synchronize()
        fwd_start = time.perf_counter()
        score = model(data)
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - fwd_start)

        # ---- Loss ----
        if model.__class__.__name__ in ('ConvE', 'TuckER'):
            loss = score
            total_loss += score.item()
            total_triples += 1
        else:
            po_score = score[:batch_pos_size].view(batch_pos_size, -1)
            ne_score = score[batch_pos_size:].view(batch_pos_size, -1)
            loss = get_loss_func_torch(po_score, ne_score, args)
            total_loss += loss.item()
            total_triples += batch_pos_size

        # ---- Backward ----
        torch.cuda.synchronize()
        bwd_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = (time.perf_counter() - bwd_start)

        # ---- Optimizer ----
        torch.cuda.synchronize()
        opt_start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_time = (time.perf_counter() - opt_start)

        # Accumulate
        acc_neg_time += neg_time
        acc_fwd_time += fwd_time
        acc_bwd_time += bwd_time
        acc_opt_time += opt_time

        step_ms = (neg_time + fwd_time + bwd_time + opt_time) * 1000.0
        step_times_ms.append(step_ms)

        # Hub degree
        degrees = []
        for e in pos_h:
            degrees.append(GLOBAL_ENTITY_DEGREE.get(int(e), 0))
        for e in pos_t:
            degrees.append(GLOBAL_ENTITY_DEGREE.get(int(e), 0))
        batch_avg_deg = float(np.mean(degrees)) if degrees else 0.0

        # Profiling rows
        if profiling_rows is not None:
            profiling_rows.append({
                'epoch': epoch_idx,
                'step': step_idx,
                'neg_sampling_time': round(neg_time * 1000, 3),
                'forward_time': round(fwd_time * 1000, 3),
                'backward_time': round(bwd_time * 1000, 3),
                'optimizer_time': round(opt_time * 1000, 3),
                'step_time': round(step_ms, 3),
                'batch_size': batch_pos_size,
                'avg_degree': round(batch_avg_deg, 2),
            })

        # Batch runtime variance row
        if batch_runtime_rows is not None:
            batch_runtime_rows.append({
                'epoch': epoch_idx,
                'step': step_idx,
                'step_time_ms': round(step_ms, 3),
                'neg_sampling_ms': round(neg_time * 1000, 3),
                'forward_ms': round(fwd_time * 1000, 3),
                'backward_ms': round(bwd_time * 1000, 3),
                'optimizer_ms': round(opt_time * 1000, 3),
                'batch_size': batch_pos_size,
                'avg_entity_degree': round(batch_avg_deg, 2),
            })

        n_batches += 1

    epoch_time = time.time() - epoch_start

    # Stats
    step_times_arr = np.array(step_times_ms)
    mean_step = float(step_times_arr.mean())
    std_step = float(step_times_arr.std())
    cv_step = std_step / max(mean_step, 1e-10)

    return {
        'epoch': epoch_idx,
        'epoch_time_s': round(epoch_time, 3),
        'mean_step_ms': round(mean_step, 3),
        'std_step_ms': round(std_step, 3),
        'cv_step': round(cv_step, 5),
        'n_batches': n_batches,
        'avg_loss': round(total_loss / max(total_triples, 1), 6),
        'neg_time_s': round(acc_neg_time, 3),
        'fwd_time_s': round(acc_fwd_time, 3),
        'bwd_time_s': round(acc_bwd_time, 3),
        'opt_time_s': round(acc_opt_time, 3),
        'scheduler_overhead_ms': round(provider.get_scheduler_overhead_ms(), 3),
        'batch_weight_cv': round(provider.get_batch_weight_stats().get('cv', 0), 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorter', type=str, default='Random', choices=['Random', 'Cost'])
    parser.add_argument('--packer', type=str, default='Chunk', choices=['Chunk', 'FFD'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp-label', type=str, default=None)
    parser.add_argument('--output', type=str, default='output/results/')
    args_cli = parser.parse_args()

    sorter_n = args_cli.sorter.upper()
    packer_n = args_cli.packer.upper()
    exp_label = args_cli.exp_label or f"{sorter_n}_{packer_n}"
    out_dir = os.path.join(args_cli.output, f"exp_{exp_label}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_label}")
    print(f"Sorter={sorter_n}, Packer={packer_n}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # 1. Load TransE config
    cur_path = os.path.abspath(os.path.dirname(__file__))
    args_path = os.path.join(cur_path, "args_kge", "transe_fb15k237_args.json")
    args = load_args(args_path)
    args.batch_size = 1000
    args.max_epoch = args_cli.epochs
    args.start_valid = max(3, args_cli.epochs // 2)
    args.eval_freq = max(3, args_cli.epochs // 2)
    args.neg_triple_num = 100
    print(f"Config: batch_size={args.batch_size}, neg_num={args.neg_triple_num}, epochs={args.max_epoch}")

    # 2. Load KG
    remove_unlinked = False
    kgs = read_kgs_from_folder(
        'lp', args.training_data, args.dataset_division,
        args.alignment_module, args.ordered, remove_unlinked=remove_unlinked
    )
    print(f"KG: {kgs.entities_num} entities, {kgs.relations_num} relations, "
          f"{len(kgs.relation_triples_list)} train triples")

    # 3. Initialize entity degree
    init_entity_degree(kgs, hub_percentile=10)

    # 4. Build/load feature + cost table
    print("\n--- Feature Extraction & Cost Table ---")
    extractor = FeatureExtractor(kgs.relation_triples_list, kgs.entities_num)
    features = extractor.build(force_recompute=False)
    cost_table = build_cost_table(features, neg_num=args.neg_triple_num)
    print(f"Cost table: mean={cost_table.mean():.2f}ms, max={cost_table.max():.2f}ms")

    # 5. Build Scheduler
    scheduler = build_scheduler(args_cli.sorter, args_cli.packer)
    print(f"Scheduler: {scheduler.get_name()}")

    # 6. Initialize model & trainer
    print("\n--- Model Initialization ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # CRITICAL: set is_torch flag for evaluation module
    args.is_torch = True

    from src.torch.kge_models.TransE import TransE
    model = TransE(kgs, args)
    model.to(device)
    print(f"Model: TransE, dim={args.dim}, param_count={sum(p.numel() for p in model.parameters())}")

    optimizer = get_optimizer_torch(args.optimizer, model, args.learning_rate)
    valid = LinkPredictionEvaluator(model, args, kgs, is_valid=True)
    test_eval = LinkPredictionEvaluator(model, args, kgs)

    triples_list = kgs.relation_triples_list

    profiling_rows = []
    batch_runtime_rows = []
    epoch_stats = []
    eval_results = []

    # 9. Training loop
    print("\n--- Training ---")
    for epoch_idx in range(args.max_epoch):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.stdout.flush()

        model.train()
        epoch_out = train_epoch_with_provider(
            model=model, optimizer=optimizer, loss_fn=None,
            device=device, args=args, kgs=kgs,
            triples_list=triples_list, cost_table=cost_table,
            scheduler=scheduler, epoch_idx=epoch_idx,
            out_dir=out_dir, exp_label=exp_label,
            profiling_rows=profiling_rows,
            batch_runtime_rows=batch_runtime_rows,
        )
        epoch_stats.append(epoch_out)
        sys.stdout.flush()

        print(f"  Epoch {epoch_idx:2d}: loss={epoch_out['avg_loss']:.6f}, "
              f"time={epoch_out['epoch_time_s']:.2f}s, "
              f"mean_step={epoch_out['mean_step_ms']:.1f}ms, "
              f"CV={epoch_out['cv_step']:.5f}, "
              f"n_batches={epoch_out['n_batches']}")
        sys.stdout.flush()

        # Validation
        if epoch_idx >= args.start_valid and epoch_idx % args.eval_freq == 0:
            torch.cuda.empty_cache()
            model.eval()
            print("  --- Validation ---")
            sys.stdout.flush()
            metrics = valid.print_results()
            eval_results.append({'epoch': epoch_idx, 'metrics': metrics})
            torch.cuda.empty_cache()

    # Test evaluation
    print("\n--- Test Evaluation ---")
    sys.stdout.flush()
    model.eval()
    torch.cuda.empty_cache()
    test_metrics = test_eval.print_results()

    # Write output CSVs
    write_outputs(out_dir, exp_label, profiling_rows, batch_runtime_rows,
                  epoch_stats, eval_results, test_metrics, sorter_n, packer_n,
                  cost_table, scheduler)

    print_summary(epoch_stats, exp_label, sorter_n, packer_n, out_dir)
    sys.stdout.flush()

    print(f"\nDone. All outputs saved to {out_dir}/")
    sys.stdout.flush()


def write_outputs(out_dir, exp_label, profiling_rows, batch_runtime_rows,
                  epoch_stats, eval_results, test_metrics, sorter_n, packer_n,
                  cost_table, scheduler):
    if profiling_rows:
        fields = ['epoch', 'step', 'neg_sampling_time', 'forward_time',
                   'backward_time', 'optimizer_time', 'step_time',
                   'batch_size', 'avg_degree']
        path = os.path.join(out_dir, 'profiling_summary.md')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(profiling_rows)
        print(f"[Output] {path} ({len(profiling_rows)} rows)")

    if batch_runtime_rows:
        fields = ['epoch', 'step', 'step_time_ms', 'neg_sampling_ms',
                   'forward_ms', 'backward_ms', 'optimizer_ms',
                   'batch_size', 'avg_entity_degree']
        path = os.path.join(out_dir, 'batch_runtime_variance.md')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(batch_runtime_rows)
        print(f"[Output] {path} ({len(batch_runtime_rows)} rows)")

    if epoch_stats:
        fields = ['epoch', 'epoch_time_s', 'mean_step_ms', 'std_step_ms',
                   'cv_step', 'n_batches', 'avg_loss',
                   'neg_time_s', 'fwd_time_s', 'bwd_time_s', 'opt_time_s',
                   'scheduler_overhead_ms', 'batch_weight_cv']
        path = os.path.join(out_dir, 'epoch_summary.md')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(epoch_stats)
        print(f"[Output] {path} ({len(epoch_stats)} rows)")

    eval_path = os.path.join(out_dir, 'evaluation_metrics.md')
    with open(eval_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        if test_metrics:
            for k, v in test_metrics.items():
                w.writerow([k, v])
        w.writerow([])
        w.writerow(['exp_config', f'{sorter_n}+{packer_n}'])
    print(f"[Output] {eval_path}")

    import json
    config = {
        'exp_label': exp_label,
        'sorter': sorter_n,
        'packer': packer_n,
        'batch_size': 1000,
        'neg_num': 100,
        'epochs': len(epoch_stats),
        'cost_table_mean_ms': float(cost_table.mean()),
        'cost_table_max_ms': float(cost_table.max()),
    }
    config_path = os.path.join(out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[Output] {config_path}")


def print_summary(epoch_stats, exp_label, sorter_n, packer_n, out_dir):
    print(f"\n{'='*60}")
    print(f"SUMMARY: {exp_label} (Sorter={sorter_n}, Packer={packer_n})")
    print(f"{'='*60}")

    if not epoch_stats:
        return

    step_ms = [s['mean_step_ms'] for s in epoch_stats]
    cvs = [s['cv_step'] for s in epoch_stats]
    epoch_times = [s['epoch_time_s'] for s in epoch_stats]

    print(f"\n  System Efficiency:")
    print(f"    Mean step time:     {np.mean(step_ms):.2f} ms")
    print(f"    Mean epoch time:    {np.mean(epoch_times):.2f} s")
    print(f"    Total train time:   {sum(epoch_times):.2f} s")

    print(f"\n  Variance Reduction:")
    print(f"    Mean step CV:       {np.mean(cvs):.5f}")
    print(f"    Min step CV:        {min(cvs):.5f}")
    print(f"    Max step CV:        {max(cvs):.5f}")

    print(f"\n  Algorithm Integrity:")
    eval_path = os.path.join(out_dir, 'evaluation_metrics.md')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            for line in f:
                if 'hits' in line.lower() or 'mrr' in line.lower():
                    print(f"    {line.strip()}")
    print(f"\n  Outputs: {out_dir}/")


if __name__ == '__main__':
    main()