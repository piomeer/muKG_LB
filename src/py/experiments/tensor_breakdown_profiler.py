#!/usr/bin/env python3 -u
"""Phase 7 Step 1: Tensor Construction Deep Profiling"""
import sys, os, time, csv, argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--neg_num', type=int, default=150)
    parser.add_argument('--embed_dim', type=int, default=400)
    parser.add_argument('--output_dir', default='output/results/tensor_breakdown/')
    parser.add_argument('--dataset', default='FB15K237')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_data(args):
    dataset_name = args.dataset.lower()
    args_path = os.path.join(
        os.path.dirname(__file__), 'args_kge',
        f'transe_{dataset_name}_args.json'
    )
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )
    train_triples = kgs.local_relation_triples_list
    num_entities = kgs.entities_num
    num_relations = kgs.relations_num

    extractor = FeatureExtractor(train_triples, num_entities)
    features = extractor.build()
    cost_table = build_cost_table(features, neg_num=args.neg_num)

    return kgs, train_triples, cost_table, features, num_entities, num_relations


class SimpleTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(num_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(num_relations, dim)

    def forward(self, heads, rels, tails):
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(rels)
        t = self.ent_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=-1)


def time_tensor_construction(batch_triples, neg_size, num_entities):
    """Time breakdown of tensor construction pipeline"""
    # T0-T1: Extract positive triples from list
    t0 = time.perf_counter()
    pos_heads = [t[0] for t in batch_triples]
    pos_rels  = [t[1] for t in batch_triples]
    pos_tails = [t[2] for t in batch_triples]
    t1 = time.perf_counter()

    # T1-T2: Convert to numpy
    pos_heads_np = np.array(pos_heads, dtype=np.int64)
    pos_rels_np  = np.array(pos_rels, dtype=np.int64)
    pos_tails_np = np.array(pos_tails, dtype=np.int64)
    t2 = time.perf_counter()

    # T2-T3: numpy -> CPU tensor (via from_numpy, zero-copy for pos)
    heads_t = torch.from_numpy(pos_heads_np)
    rels_t  = torch.from_numpy(pos_rels_np)
    tails_t = torch.from_numpy(pos_tails_np)
    t3 = time.perf_counter()

    # T3-T4: Generate negative samples directly as torch tensors
    neg_heads = torch.randint(0, num_entities, (neg_size,), dtype=torch.long)
    neg_tails = torch.randint(0, num_entities, (neg_size,), dtype=torch.long)
    t4 = time.perf_counter()

    # T4-T5: CPU -> GPU transfer (synchronized)
    heads_gpu = heads_t.cuda()
    rels_gpu  = rels_t.cuda()
    tails_gpu = tails_t.cuda()
    neg_heads_gpu = neg_heads.cuda()
    neg_tails_gpu = neg_tails.cuda()
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    # Run a minimal forward to keep GPU "warm" (realistic timing)
    _ = heads_gpu + rels_gpu + tails_gpu
    torch.cuda.synchronize()
    t6 = time.perf_counter()

    costs = {
        't1_extract_pos': (t1 - t0) * 1000,
        't2_numpy_convert': (t2 - t1) * 1000,
        't3_tensor_pos': (t3 - t2) * 1000,
        't4_neg_construct': (t4 - t3) * 1000,
        't5_gpu_transfer': (t5 - t4) * 1000,
        't6_gpu_warmup': (t6 - t5) * 1000,
        'tensor_total': (t6 - t0) * 1000,
    }
    return costs


def run_profiling(train_triples, cost_table, features, sorter, packer,
                  args, config_label, out_file, num_entities):
    scheduler = Scheduler(sorter, packer)
    provider = BatchProvider(scheduler, cost_table, args.batch_size,
                             enable_logging=False)
    writer = csv.DictWriter(out_file, fieldnames=[
        'step', 'config', 'batch_weight',
        't1_extract_pos', 't2_numpy_convert', 't3_tensor_pos',
        't4_neg_construct', 't5_gpu_transfer', 't6_gpu_warmup',
        'tensor_total',
    ])
    writer.writeheader()

    neg_total = args.batch_size * args.neg_num
    step = 0

    for batch_triples in provider.iterate(train_triples):
        # Batch weight
        costs = []
        for h, r, t in batch_triples:
            hc = float(cost_table[h]) if h < len(cost_table) else 0.0
            tc = float(cost_table[t]) if t < len(cost_table) else 0.0
            costs.append(max(hc, tc))
        batch_weight = float(np.mean(costs)) if costs else 0.0

        # Time tensor construction (no neg sampling - that's the point)
        costs_dict = time_tensor_construction(batch_triples, neg_total, num_entities)

        row = {
            'step': step,
            'config': config_label,
            'batch_weight': round(batch_weight, 4),
        }
        row.update({k: round(v, 4) for k, v in costs_dict.items()})
        writer.writerow(row)

        step += 1
        if step % 50 == 0:
            print(f"  [{config_label}] step {step}: tensor_total={costs_dict['tensor_total']:.2f}ms")

    print(f"  [{config_label}] Completed {step} steps")
    return step


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'tensor_breakdown.md')

    print("Loading data and building cost table...")
    kgs, train_triples, cost_table, features, num_entities, num_relations = load_data(args)
    print(f"  Entities: {num_entities}, Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples)}")
    print(f"  Cost table: shape={cost_table.shape}, range=[{cost_table.min():.2f}, {cost_table.max():.2f}]")

    with open(csv_path, 'w', newline='') as f:
        # Baseline
        print("\n--- Baseline (RandomSorter + ChunkPacker) ---")
        base_sorter = RandomSorter(seed=args.seed)
        base_packer = ChunkPacker()
        run_profiling(train_triples, cost_table, features, base_sorter, base_packer,
                      args, 'Baseline', f, num_entities)

        # CBP
        print("\n--- CBP (CostSorter + FFDPacker) ---")
        cbp_sorter = CostSorter()
        cbp_packer = FFDPacker()
        run_profiling(train_triples, cost_table, features, cbp_sorter, cbp_packer,
                      args, 'CBP', f, num_entities)

    total_lines = sum(1 for _ in open(csv_path)) - 1
    print(f"\nTensor breakdown saved to {csv_path}")
    print(f"Total steps: {total_lines}")


if __name__ == '__main__':
    main()
