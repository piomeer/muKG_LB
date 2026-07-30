"""
Phase 9 Step 4.5 — CPU Negative Sampling Variance Isolation
Measures pure neg-sampling time variance for BL and CBP,
excluding tensor build / optimizer / evaluation noise.
"""
import sys, os, time, csv, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import (
    Scheduler, RandomSorter, CostSorter, ChunkPacker, FFDPacker,
)
from src.py.load.batch_provider import BatchProvider


def original_cpu_neg_sampling(batch_triples, neg_num, n_entities, all_triples_set):
    """MuKG original: Bernoulli(0.5) head/tail + global all_triples_set collision"""
    heads_list, tails_list = [], []
    max_try = 10
    for h, r, t in batch_triples:
        for _ in range(neg_num):
            tries = 0
            if random.random() < 0.5:
                while tries < max_try:
                    cand_h = random.randint(0, n_entities - 1)
                    if (cand_h, r, t) not in all_triples_set and cand_h != h:
                        heads_list.append(cand_h); tails_list.append(t); break
                    tries += 1
                else:
                    heads_list.append(h); tails_list.append(t)
            else:
                while tries < max_try:
                    cand_t = random.randint(0, n_entities - 1)
                    if (h, r, cand_t) not in all_triples_set and cand_t != t:
                        heads_list.append(h); tails_list.append(cand_t); break
                    tries += 1
                else:
                    heads_list.append(h); tails_list.append(t)
    # Return list lengths for validation only (no tensor construction)
    return len(heads_list), len(tails_list)


def measure_config(label, sorter, packer, cost_table, train_triples,
                   all_triples_set, n_entities, neg_num, epochs):
    """Run multiple epochs of pure neg-sampling measurement."""
    print(f"\n{'='*50}")
    print(f"  Measuring: {label}")
    print(f"{'='*50}")

    random.seed(42)
    scheduler = Scheduler(sorter, packer)
    provider = BatchProvider(scheduler, cost_table, 5000, enable_logging=False)

    all_neg_times_ms = []
    epoch_stds = []

    for epoch in range(epochs):
        epoch_times = []
        for step, batch in enumerate(provider.iterate(train_triples)):
            time.sleep(0)  # yield GIL
            t0 = time.perf_counter()
            original_cpu_neg_sampling(batch, neg_num, n_entities, all_triples_set)
            t1 = time.perf_counter()
            neg_time_ms = (t1 - t0) * 1000
            epoch_times.append(neg_time_ms)
            all_neg_times_ms.append(neg_time_ms)

        epoch_std = float(np.std(epoch_times))
        epoch_mean = float(np.mean(epoch_times))
        epoch_stds.append(epoch_std)
        print(f"  Epoch {epoch}: mean={epoch_mean:.1f}ms, std={epoch_std:.1f}ms")

    overall_mean = float(np.mean(all_neg_times_ms))
    overall_std = float(np.std(all_neg_times_ms))
    print(f"  Overall: mean={overall_mean:.1f}ms, std={overall_std:.1f}ms "
          f"(epoch_stds: {[f'{s:.1f}' for s in epoch_stds]})")

    return {
        'label': label,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'epoch_stds': epoch_stds,
        'all_times': all_neg_times_ms,
    }


def main():
    out_dir = 'output/results/phase9_step4_5'
    os.makedirs(out_dir, exist_ok=True)

    print("Loading FB15k-237 dataset...")
    args_path = os.path.join(os.path.dirname(__file__), 'args_kge', 'transe_fb15k237_args.json')
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )

    train_triples = kgs.local_relation_triples_list
    n_entities = kgs.entities_num
    all_triples_set = set(train_triples)

    random.seed(42)
    shuffled = list(train_triples)
    random.shuffle(shuffled)
    train_for_training = shuffled[5000:]  # exclude eval subset

    print(f"  Train triples: {len(train_for_training)}")

    extractor = FeatureExtractor(train_for_training, n_entities)
    features = extractor.build()
    cost_table = build_cost_table(features, neg_num=150)
    print(f"  Cost table: mean={cost_table.mean():.2f}ms")

    neg_num = 150
    epochs = 3

    # Measure BL (Baseline: Random + Chunk)
    bl_result = measure_config(
        'BL', RandomSorter(seed=42), ChunkPacker(),
        cost_table, train_for_training, all_triples_set, n_entities, neg_num, epochs
    )

    # Measure CBP (Cost + FFD)
    cbp_result = measure_config(
        'CBP', CostSorter(), FFDPacker(),
        cost_table, train_for_training, all_triples_set, n_entities, neg_num, epochs
    )

    # Write detailed CSV
    detail_path = os.path.join(out_dir, 'neg_sampling_variance.csv')
    with open(detail_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'epoch', 'batch_idx', 'neg_time_ms'])
        batch_idx = 0
        for res in [bl_result, cbp_result]:
            config = res['label']
            epoch_size = len(res['all_times']) // epochs
            for e in range(epochs):
                for i in range(epoch_size):
                    idx = e * epoch_size + i
                    w.writerow([config, e, i, f"{res['all_times'][idx]:.2f}"])
    print(f"\n  Detail saved: {detail_path}")

    # Write summary CSV
    summary_path = os.path.join(out_dir, 'variance_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'mean_ms', 'std_ms', 'epoch_stds_ms'])
        for res in [bl_result, cbp_result]:
            w.writerow([
                res['label'],
                f"{res['overall_mean']:.1f}",
                f"{res['overall_std']:.1f}",
                ";".join(f"{s:.1f}" for s in res['epoch_stds'])
            ])
    print(f"  Summary saved: {summary_path}")

    # Comparison
    bl_std = bl_result['overall_std']
    cbp_std = cbp_result['overall_std']
    reduction = (1 - cbp_std / bl_std) * 100 if bl_std > 0 else 0

    print(f"\n{'='*50}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"  BL  std: {bl_std:.1f}ms")
    print(f"  CBP std: {cbp_std:.1f}ms")
    print(f"  Reduction: {reduction:.1f}% {'✅ CBP effective' if reduction > 20 else '⚠️ marginal'}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()