"""
Phase 10 Step 2.5 — Sensitivity Experiments ONLY
==================================================
Re-runs EXP-4 (Batch Size Sensitivity) and EXP-5 (Neg Num Sensitivity)
to complete the validation results.
"""
import sys, os, time, csv, random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import (
    Scheduler, RandomSorter, ChunkPacker,
)
from src.py.load.batch_provider import BatchProvider
from src.py.load.gpu_sampler import GPUNegativeSampler


OUT_DIR = 'output/results/phase10_step2_5'
os.makedirs(OUT_DIR, exist_ok=True)


class SimpleTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.ent_embeddings = torch.nn.Embedding(num_entities, dim)
        self.rel_embeddings = torch.nn.Embedding(num_relations, dim)

    def forward(self, heads, rels, tails):
        h = self.ent_embeddings(heads)
        r = self.rel_embeddings(rels)
        t = self.ent_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=-1)


def run_single_epoch_timing(train_triples, all_triples_set,
                            n_entities, n_relations, cost_table,
                            batch_size, neg_num, seed=42):
    """Run 1 epoch with GPU sampler, return timing metrics."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleTransE(n_entities, n_relations, dim=400).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler_obj = GPUNegativeSampler(n_entities, neg_num)

    scheduler = Scheduler(RandomSorter(seed=42), ChunkPacker())
    provider = BatchProvider(scheduler, cost_table, batch_size, enable_logging=False)

    neg_times = []
    step_times = []
    t_start = time.time()
    n_batches = 0

    for step, batch in enumerate(provider.iterate(train_triples)):
        try:
            optimizer.zero_grad()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            neg_h, neg_t = sampler_obj.generate(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            neg_time = (t1 - t0) * 1000
            neg_times.append(neg_time)

            pos_h = torch.tensor([t[0] for t in batch], dtype=torch.long, device='cuda')
            pos_r = torch.tensor([t[1] for t in batch], dtype=torch.long, device='cuda')
            pos_t = torch.tensor([t[2] for t in batch], dtype=torch.long, device='cuda')

            pos_scores = model(pos_h, pos_r, pos_t)
            neg_scores = model(neg_h, pos_r.repeat_interleave(neg_num), neg_t)
            loss = torch.mean(torch.clamp(
                pos_scores[:, None] - neg_scores.view(-1, neg_num) + 1.0, min=0
            ))
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            step_time = (t2 - t0) * 1000
            step_times.append(step_time)
            n_batches += 1

            if step % 30 == 0:
                print(f"    S{step:3d} neg={neg_time:.1f}ms step={step_time:.1f}ms")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    ⚠️ OOM at step {step}, aborting this config")
                torch.cuda.empty_cache()
                return None
            raise

    epoch_time = time.time() - t_start
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()

    return {
        'epoch_time_s': epoch_time,
        'n_batches': n_batches,
        'mean_neg_ms': float(np.mean(neg_times)),
        'std_neg_ms': float(np.std(neg_times)),
        'mean_step_ms': float(np.mean(step_times)),
        'std_step_ms': float(np.std(step_times)),
        'gpu_mem_mb': gpu_mem,
    }


def csv_to_md(csv_path, md_path):
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        with open(md_path, 'w') as f:
            f.writelines(lines)
    except Exception:
        pass


def main():
    print("="*60)
    print("  Sensitivity Experiments (Batch Size + Neg Num)")
    print("="*60)

    # Load dataset
    print("\nLoading FB15k-237...")
    args_path = os.path.join(os.path.dirname(__file__), 'args_kge', 'transe_fb15k237_args.json')
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )

    train_triples = kgs.local_relation_triples_list
    n_entities = kgs.entities_num
    n_relations = kgs.relations_num
    all_triples_set = set(train_triples)

    random.seed(42)
    shuffled = list(train_triples)
    random.shuffle(shuffled)
    train_for_training = shuffled[5000:]

    extractor = FeatureExtractor(train_for_training, n_entities)
    features = extractor.build()
    cost_table = build_cost_table(features, neg_num=150)

    print(f"  Train: {len(train_for_training)}, Entities: {n_entities}")

    # ---- EXP-4: Batch Size Sensitivity ----
    print("\n" + "="*60)
    print("  EXP-4: Batch Size Sensitivity")
    print("="*60)

    batch_sizes = [1000, 2500, 5000, 10000]
    csv_path = os.path.join(OUT_DIR, 'batch_size_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_size', 'epoch_time_s', 'n_batches',
                         'mean_neg_ms', 'std_neg_ms',
                         'mean_step_ms', 'std_step_ms', 'gpu_mem_mb'])

        for bs in batch_sizes:
            print(f"\n  Batch size = {bs}")
            torch.cuda.empty_cache()
            result = run_single_epoch_timing(
                train_for_training, all_triples_set,
                n_entities, n_relations, cost_table,
                batch_size=bs, neg_num=150
            )
            if result is None:
                writer.writerow([bs, 'OOM', '', '', '', '', '', ''])
                print(f"    → OOM, skipping")
            else:
                writer.writerow([
                    bs,
                    f"{result['epoch_time_s']:.1f}",
                    result['n_batches'],
                    f"{result['mean_neg_ms']:.1f}",
                    f"{result['std_neg_ms']:.1f}",
                    f"{result['mean_step_ms']:.1f}",
                    f"{result['std_step_ms']:.1f}",
                    f"{result['gpu_mem_mb']:.0f}",
                ])
                print(f"    → epoch_time={result['epoch_time_s']:.1f}s, "
                      f"neg={result['mean_neg_ms']:.1f}ms, "
                      f"mem={result['gpu_mem_mb']:.0f}MB")
            f.flush()

    csv_to_md(csv_path, csv_path.replace('.csv', '.md'))
    print(f"\n  Saved → {csv_path}")

    # ---- EXP-5: Neg Num Sensitivity ----
    print("\n" + "="*60)
    print("  EXP-5: Neg Num Sensitivity")
    print("="*60)

    neg_nums = [10, 25, 50, 100, 150]
    csv_path = os.path.join(OUT_DIR, 'neg_num_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['neg_num', 'epoch_time_s', 'n_batches',
                         'mean_neg_ms', 'std_neg_ms',
                         'mean_step_ms', 'std_step_ms', 'gpu_mem_mb'])

        for nn in neg_nums:
            print(f"\n  Neg num = {nn}")
            torch.cuda.empty_cache()
            result = run_single_epoch_timing(
                train_for_training, all_triples_set,
                n_entities, n_relations, cost_table,
                batch_size=5000, neg_num=nn
            )
            if result is None:
                writer.writerow([nn, 'OOM', '', '', '', '', '', ''])
                print(f"    → OOM, skipping")
            else:
                writer.writerow([
                    nn,
                    f"{result['epoch_time_s']:.1f}",
                    result['n_batches'],
                    f"{result['mean_neg_ms']:.1f}",
                    f"{result['std_neg_ms']:.1f}",
                    f"{result['mean_step_ms']:.1f}",
                    f"{result['std_step_ms']:.1f}",
                    f"{result['gpu_mem_mb']:.0f}",
                ])
                print(f"    → epoch_time={result['epoch_time_s']:.1f}s, "
                      f"neg={result['mean_neg_ms']:.1f}ms, "
                      f"mem={result['gpu_mem_mb']:.0f}MB")
            f.flush()

    csv_to_md(csv_path, csv_path.replace('.csv', '.md'))
    print(f"\n  Saved → {csv_path}")

    print("\n" + "="*60)
    print("  SENSITIVITY EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()